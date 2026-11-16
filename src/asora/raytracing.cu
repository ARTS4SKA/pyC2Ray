#include "raytracing.cuh"
#include "utils.cuh"

#include "memory.h"
#include "octahedron.cuh"
#include "shortchar.cuh"
#include "utils.cuh"

#include <stdexcept>

namespace asora {

    __device__ double density_maps::get(size_t index) const {
        return ndens[index] * (1.0 - xHII[index]);
    }

}  // namespace asora

namespace {

    using namespace asora;

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
        double minlogtau, double dlogtau, size_t num_tau, size_t grid_size,
        size_t block_size
    ) {
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

        // Allocate and zero the output array for the photoionization rate
        auto phion_HI = device::scratch<double>(n_cells);
        phion_HI.zero();

        // Collect the LUT for the raytracing kernel. If it cannot be built, it is
        // unset and the interpolation parameters are computed on the fly.
        auto lut_d = create_shortchar_interp_lut(R, m1);

        // The LUT only keeps the cells within reach, so its banks can be smaller than
        // the largest shell. Allocate (if necessary) the column density array.
        auto column_density_HI =
            device::scratch<double>(grid_size * lut_d.bank_size() * num_banks);

        element_data data_HI{phion_HI.data(), column_density_HI.data(), sigma};
        photo_tables ion_tables{photo_ion_thin.data(), photo_ion_thick.data()};
        linspace<double> logtau{minlogtau, dlogtau, static_cast<size_t>(num_tau)};

        // Loop over batches of sources
        for (size_t ns = 0; ns < num_src; ns += grid_size) {
            // Raytrace the current batch of sources in parallel
            // Consecutive kernel launches are in the same stream and so are
            // serialized
            evolve0D_gpu<<<grid_size, block_size>>>(
                lut_d, m1, dr, R, ns, num_src, source_position.data(),
                source_flux.data(), data_HI, densities, ion_tables, logtau
            );

            safe_cuda(cudaPeekAtLastError());
        }

        // Copy the accumulated ionization fraction back to the host.
        // Memcpy blocks until last kernel has finished.
        phion_HI.copy_to_host(phi_ion);
    }

    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        shortchar_lut lut, size_t m1, double dr, double R_max, size_t ns_start,
        size_t num_src, const int *__restrict__ src_pos,
        const double *__restrict__ src_flux, element_data data_HI,
        density_maps densities, photo_tables<> ion_tables, linspace<double> logtau
    ) {
        /* The raytracing kernel proceeds as follows:
         * 1. Select the source based on the thread-block number
         * 2. Loop over the asora q-shells around the source, up to q_max
         * 3. For each shell, threads independently raytrace on all cells
         * 4. Before moving to the next q-shell, threads are synchronized to ensure
         * causality
         */

        // Source identifier: one source per thread-block.
        const size_t ns = ns_start + blockIdx.x;

        // Ensure the source index is valid.
        if (ns >= num_src) return;

        // Get source properties.
        src_pos += 3 * ns;
        const int3 pos0{src_pos[0], src_pos[1], src_pos[2]};
        auto scale = src_flux[ns] / (dr * dr * dr);

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block works on its own array).
        const size_t cd_offset = blockIdx.x * lut.bank_size() * num_banks;
        data_HI.column_density += cd_offset;

        // Calculate column density and photoionization rate for the source cell.
        // This is done separately from the main loop because to take advantage of
        // some simplifications.
        if (threadIdx.x == 0) {
            const auto index = ravel_index(pos0, m1);
            auto nHI = densities.get(index);
            update_photo_rates(
                data_HI, 0, index, 0.0, nHI, 0.5 * dr, scale, ion_tables, logtau
            );
        }
        __syncthreads();

        // Cell-independent part of the photon volume 4*pi * dist2 * path * dr^2.
        scale /= 4 * c::pi<>;

        for (int q = 1; q <= lut.q_max(); ++q) {
            // Each thread can process multiple cells. Only the cells kept in the LUT
            // are visited; s is their position within the compacted shell.
            const auto begin = lut.shell_begin(q);
            const auto count = lut.shell_size(q);
            for (size_t s = threadIdx.x; s < count; s += blockDim.x) {
                raytrace(
                    lut, begin + s, q, s, pos0, scale, data_HI, dr, R_max, densities,
                    m1, ion_tables, logtau
                );
            }
            __syncthreads();
        }
    }

}  // namespace asora
