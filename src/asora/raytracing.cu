#include "raytracing.cuh"

#include "memory.h"
#include "octahedron.cuh"
#include "shortchar.cuh"
#include "utils.cuh"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>

namespace asora {

    // TODO: Generalize this function to take an arbitrary kernel as an argument.
    size_t max_resident_blocks(size_t block_size) {
        device::check_initialized();

        // Determine how many blocks of this kernel can be resident on a SM.
        int blocks_per_sm = 0;
        safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, asora::evolve0D_gpu, static_cast<int>(block_size), 0
        ));

        // This kernel can't be launched.
        if (blocks_per_sm == 0)
            throw std::runtime_error(
                std::format(
                    "no block of {} threads fits on a multiprocessor for this kernel",
                    block_size
                )
            );

        // Determine how many SMs are on the device.
        int num_sms = 0;
        safe_cuda(cudaDeviceGetAttribute(
            &num_sms, cudaDevAttrMultiProcessorCount, device::device_id()
        ));

        return static_cast<size_t>(blocks_per_sm) * num_sms;
    }

    __device__ double density_maps::get(size_t index) const {
        return ndens[index] * (1.0 - xHII[index]);
    }

}  // namespace asora

namespace {

    using namespace asora;

    // Resolve how many blocks to launch. This is called once everything else is
    // already allocated.
    size_t resolve_grid_size(
        size_t scratch_per_block, size_t num_src, size_t block_size
    ) {
        /* Three things bound the grid:
         * 1. how many blocks of this kernel fit in the device
         * 2. how many column density buffers the memory budget can accomodate
         * 3. how many sources there are
         */

        // Maximum occupancy on this device.
        auto by_occupancy = max_resident_blocks(block_size);

        // Memory required for each block.
        auto bytes_per_block = scratch_per_block * sizeof(double);
        auto free_bytes = device::available_memory();

        if (free_bytes <= bytes_per_block)
            throw std::runtime_error(
                std::format(
                    "column density scratch for a single block is {} MiB, but only "
                    "{} "
                    "MiB are available on the device",
                    bytes_per_block >> 20, free_bytes >> 20
                )
            );

        // Number of blocks that can fit in free memory.
        auto by_memory = free_bytes / bytes_per_block;

        return std::min({by_occupancy, by_memory, num_src});
    }

    // Index of the next source to be raytraced, shared by all the blocks of
    // evolve0D_gpu. It is set by the host before every launch.
    __device__ unsigned long long next_source;

    // Compute the photoionization rate for a given cell based on the incoming column
    // density and the pre-computed photoionization tables.
    __device__ void update_photo_rates(
        element_data &data_HI, size_t cd_index, size_t ph_index, double coldens_in,
        double nHI, double path, double scale, const photo_tables<> &ion_tables,
        const linspace<double> &logtau
    ) {
        // Compute outgoing column density and add to array for subsequent
        // interpolations
        auto &coldens_out = data_HI.column_density[cd_index];
        coldens_out = coldens_in + nHI * path;

        auto tau_in = coldens_in * data_HI.cross_section;
        auto tau_out = coldens_out * data_HI.cross_section;

        auto phion = asora::photo_table_lookup(tau_in, tau_out, ion_tables, logtau);
        // Rescale the photo-ionization rate by the flux strength normalized per volume
        // and per neutral density (part of the photon-conserving rate prescription) and
        // add it to the global array. Dividing by the product uses one FP64 division
        // instead of two; vol is a cell volume, so the product cannot underflow.
        atomicAdd(data_HI.photo_ionization + ph_index, phion * scale / nHI);
    }

    // Raytracing operation on a given cell, identified by (q, s). This is performed by
    // a single thread. Threads may call this function multiple times if required to
    // cover the full q-shell.
    __device__ void raytrace(
        const shortchar_lut &lut, size_t index, int q, int s, const int3 &pos,
        double scale, element_data &data_HI, double dr, double R_max,
        const density_maps &densities, size_t m1, const photo_tables<> &ion_tables,
        const linspace<double> &logtau
    ) {
        // Early return conditions only need the cell offsets.
        auto offset = lut.is_set() ? lut.offset(index) : shell2cart(q, s);
        const auto &[di, dj, dk] = offset;
        auto dist2 = di * di + dj * dj + dk * dk;

        // The LUT only holds cells within reach, inside the domain if periodic.
        // Check if the cell is within the simulated box.
        if (!lut.is_set() || !periodic_conditions)
            if (!in_domain(pos, offset, m1)) return;

        // Using integers for threshold check as it is more consistent.
        if (!lut.is_set() && dist2 > static_cast<int>(R_max * R_max)) return;

        // Get the short-characteristic interpolation information.
        auto info = lut.is_set()
                        ? lut.params(index)
                        : make_shortchar_interpolation_info(offset, lut.bank_size());

        auto coldens_in = shortchar_interpolation(
            info, data_HI.column_density, data_HI.cross_section
        );
        coldens_in *= shortchar_multiplier(offset);

        constexpr double max_coldens = 2e30;
        if (coldens_in > max_coldens) return;

        auto path = dr * info.path;

        // vol_ph = 4*pi * dist2 * path * dr^2, with path = dr * info.path. The
        // 4*pi*dr^3 factor does not depend on the cell and is hoisted to the caller.
        auto vol_ph = dist2 * info.path;

        // Get local ionization fraction & neutral hydrogen density in the cell
        const auto cd_index = lut.bank_size() * (q % num_banks) + s;
        const auto ph_index = ravel_index(pos + offset, m1);
        auto nHI = densities.get(ph_index);

        // Compute photoionization rates from column density.
        update_photo_rates(
            data_HI, cd_index, ph_index, coldens_in, nHI, path, scale / vol_ph,
            ion_tables, logtau
        );
    }

}  // namespace

namespace asora {

    void do_all_sources_gpu(
        double R, double sigma, double dr, double *phi_ion, size_t num_src, size_t m1,
        double minlogtau, double dlogtau, size_t num_tau, size_t block_size
    ) {
        // With no sources, there is nothing to do; just zero the output
        if (num_src == 0) {
            std::fill_n(phi_ion, m1 * m1 * m1, 0.0);
            return;
        }

        device::check_initialized();

        // Size of grid data
        auto n_cells = m1 * m1 * m1;

        namespace tag = resident_tag;
        const auto &number_density = device::resident<tag::number_density>();
        const auto &photo_ion_thin = device::resident<tag::photo_ion_thin>();
        const auto &photo_ion_thick = device::resident<tag::photo_ion_thick>();
        const auto &source_flux = device::resident<tag::source_flux>();
        const auto &source_position = device::resident<tag::source_position>();
        const auto &fraction_HII_avg = device::resident<tag::fraction_HII_avg>();
        check_array(number_density, n_cells, "number density");
        // Tables hold num_tau log-spaced entries plus the leading tau = 0 entry.
        // FIXME: Update when tau = 0 is dropped
        check_array(photo_ion_thin, num_tau + 1, "photo ionization thin table");
        check_array(photo_ion_thick, num_tau + 1, "photo ionization thick table");
        check_array(source_flux, num_src, "source flux");
        check_array(source_position, 3 * num_src, "source position");
        check_array(fraction_HII_avg, n_cells, "average ionized fraction");

        // The average ionized fraction is read but never written here. It is seeded
        // once per timestep by timestep_data_to_device and thereafter updated in
        // place by the chemistry pass, so no transfer is needed. Ranks that do not
        // run chemistry refresh it with average_fraction_to_device after the
        // broadcast.
        density_maps densities{number_density.data(), fraction_HII_avg.data()};

        // Collect the LUT for the raytracing kernel. If it cannot be built, it is
        // unset and the interpolation parameters are computed on the fly.
        auto lut_d = create_shortchar_interp_lut(R, m1);

        // Allocate and zero the output array for the photoionization rate.
        auto phion_HI = device::scratch<double>(n_cells);
        phion_HI.zero();

        // The LUT only keeps the cells within reach, so its banks can be smaller than
        // the largest shell. Determine how concurrent blocks can run on this device and
        // allocate scratch memory for each block.
        auto scratch_per_block = lut_d.bank_size() * num_banks;
        auto grid_size = resolve_grid_size(scratch_per_block, num_src, block_size);
        auto column_density_HI = device::scratch<double>(grid_size * scratch_per_block);

        // Each block starts from its own index, so the shared counter starts from
        // the first source no block has taken. Blocks then take sources as they
        // become free, so the work is balanced dynamically.
        {
            const unsigned long long zero = 0;
            safe_cuda(cudaMemcpyToSymbol(next_source, &zero, sizeof(zero)));
        }

        // Create helper data structures.
        element_data data_HI{phion_HI.data(), column_density_HI.data(), sigma};
        photo_tables ion_tables{photo_ion_thin.data(), photo_ion_thick.data()};
        linspace<double> logtau{minlogtau, dlogtau, static_cast<size_t>(num_tau)};

        std::cout << "ASORA: launching " << grid_size << " blocks of " << block_size
                  << " threads to raytrace " << num_src << " sources.\n";
        // Do a single kernel launch; the sources are iterated inside the kernel.
        evolve0D_gpu<<<grid_size, block_size>>>(
            lut_d, m1, dr, R, num_src, source_position.data(), source_flux.data(),
            data_HI, densities, ion_tables, logtau
        );

        safe_cuda(cudaPeekAtLastError());

        // Copy the accumulated ionization fraction back to the host.
        phion_HI.copy_to_host(phi_ion);
    }

    /* =================================================================
     * Raytracing kernel. Calculates column density from the source
     * and computes the photoionization rate for each cell within R_max.
     * =================================================================
     * The raytracing kernel proceeds as follows:
     * 1. Start from the source with the same index as the block
     * 2. Loop over the asora q-shells around the source, up to q_max
     * 3. For each shell, threads independently raytrace on all cells
     * 4. Before moving to the next q-shell, threads are synchronized to ensure
     * causality
     * 5. Take the next unprocessed source from the shared counter and go back to 2
     * until all the sources have been taken
     */
    /* The raytracing kernel proceeds as follows:
     * 1. Select the source based on the thread-block number
     * 2. Loop over the asora q-shells around the source, up to q_max
     * 3. For each shell, threads independently raytrace on all cells
     * 4. Before moving to the next q-shell, threads are synchronized to ensure
     * causality
     */
    __global__ void evolve0D_gpu(
        shortchar_lut lut, size_t m1, double dr, double R_max, size_t num_src,
        const int *__restrict__ src_pos, const double *__restrict__ src_flux,
        element_data data_HI, density_maps densities, photo_tables<double> ion_tables,
        linspace<double> logtau
    ) {
        // Offset pointer to the outgoing column density array used for
        // interpolation (each block works on its own array).
        const size_t cd_offset = blockIdx.x * lut.bank_size() * num_banks;
        data_HI.column_density += cd_offset;

        // Each block starts processing the sources from blockIdx.x and then
        // takes the next source (it starts from 0) from the shared counter
        // as soon as it is done with the current one.
        __shared__ size_t shared_ns;
        if (threadIdx.x == 0) shared_ns = atomicAdd(&next_source, 1ULL);
        __syncthreads();

        while (shared_ns < num_src) {
            // Get source properties.
            const auto *this_pos = src_pos + 3 * shared_ns;
            const int3 pos0{this_pos[0], this_pos[1], this_pos[2]};
            auto scale = src_flux[shared_ns] / (dr * dr * dr);

            // Calculate column density and photoionization rate for the source
            // cell. This is done separately from the main loop because to take
            // advantage of some simplifications.
            if (threadIdx.x == 0) {
                const auto index = ravel_index(pos0, m1);
                auto nHI = densities.get(index);
                update_photo_rates(
                    data_HI, 0, index, 0.0, nHI, 0.5 * dr, scale, ion_tables, logtau
                );
            }
            __syncthreads();

            // Adjust for non-source volumes
            scale /= 4 * c::pi<>;

            for (int q = 1; q <= lut.q_max(); ++q) {
                // Each thread can process multiple cells. Only the cells kept in the
                // LUT are visited; s is their position within the compacted shell.
                const auto begin = lut.shell_begin(q);
                const auto count = lut.shell_size(q);
                for (size_t s = threadIdx.x; s < count; s += blockDim.x) {
                    raytrace(
                        lut, begin + s, q, s, pos0, scale, data_HI, dr, R_max,
                        densities, m1, ion_tables, logtau
                    );
                }
                __syncthreads();
            }

            // Source is processed, take the next one.
            if (threadIdx.x == 0) shared_ns = atomicAdd(&next_source, 1ULL);
            __syncthreads();
        }
    }

}  // namespace asora
