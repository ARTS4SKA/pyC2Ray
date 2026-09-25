#include "raytracing.cuh"

#include "memory.h"
#include "utils.cuh"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>

namespace asora {

    __device__ void element_data::partition_column_density(int q) {
        /// Partition the column density array into 3 shared memory banks for easier
        /// interpolation
        shared_cdens = {
            column_density + cells_to_shell(q - 2),
            column_density + cells_to_shell(q - 3),
            column_density + cells_to_shell(q - 4)
        };
    }

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
        double nHI, double path, double strength, double vol,
        const photo_tables &ion_tables, const linspace<double> &logtau
    ) {
        // Compute outgoing column density and add to array for subsequent
        // interpolations
        auto &coldens_out = data_HI.column_density[cd_index];
        coldens_out = coldens_in + nHI * path;

        auto tau_in = coldens_in * data_HI.cross_section;
        auto tau_out = coldens_out * data_HI.cross_section;

#if defined(GREY_NOTABLES)
        auto phion = asora::photo_rates_test_gpu(tau_in, tau_out);
#else
        auto phion = asora::photo_rates_gpu(tau_in, tau_out, ion_tables, logtau);
#endif
        // Rescale the photo-ionization rate by the flux strength normalized per volume
        // and per neutral density (part of the photon-conserving rate prescription) and
        // add it to the global array
        atomicAdd(data_HI.photo_ionization + ph_index, phion * strength / vol / nHI);
    }

    // Raytracing operation on a given cell, identified by (q, s). This is performed by
    // a single thread. Threads may call this function multiple times if required to
    // cover the full q-shell.
    __device__ void raytrace(
        int q, int s, int i0, int j0, int k0, double strength, element_data &data_HI,
        double dr, double R_max, const density_maps &densities, size_t m1,
        const photo_tables &ion_tables, const linspace<double> &logtau
    ) {
        auto &&[di, dj, dk] = linthrd2cart(q, s);

        // Since the grid is periodic, we limit the maximum size of the raytraced
        // region to a cube as large as the mesh around the source. See line 93 of
        // evolve_source in C2Ray, this size will depend on if the mesh is even or
        // odd. Basically the idea is that you never touch a cell which is outside a
        // cube of length ~N centered on the source.
        // Only do cell if it is within the grid, shifted under periodicity
        // which means most ~N cells away from the source.
        int ll = -m1 / 2;
        int lr = m1 % 2 - 1 - ll;
        if ((di < ll) || (di > lr) || (dj < ll) || (dj > lr) || (dk < ll) || (dk > lr))
            return;

#if !defined(PERIODIC)
        // When not in periodic mode, only treat cell if its in the grid
        if (!in_box(i0 + di, j0 + dj, k0 + dk, m1)) return;
#endif
        // Using integers for threshold check as it is more consistent.
        auto dist2 = di * di + dj * dj + dk * dk;
        if (dist2 > static_cast<int>(R_max * R_max)) return;

        cell_interpolator interp{di, dj, dk};
        auto coldens_in =
            interp.interpolate(data_HI.shared_cdens, data_HI.cross_section);

        constexpr double max_coldens = 2e30;
        if (coldens_in > max_coldens) return;

        auto path = path_in_cell(di, dj, dk) * dr;
        auto vol_ph = 4 * c::pi<> * dist2 * path * dr * dr;

        // Get local ionization fraction & neutral hydrogen density in the cell
        const auto index = ravel_index(i0 + di, j0 + dj, k0 + dk, m1);
        const auto q_off = cells_to_shell(q - 1);
        auto nHI = densities.get(index);

        // Compute photoionization rates from column density.
        update_photo_rates(
            data_HI, q_off + s, index, coldens_in, nHI, path, strength, vol_ph,
            ion_tables, logtau
        );
    }

    /* @brief Resolve how many blocks to launch.
     *
     * Three things bound the grid: how many blocks of this kernel the device can
     * keep resident, how many per-block column density buffers the memory budget
     * can pay for, and how many sources there are. Blocks are persistent, so the
     * grid no longer has to grow with the source count.
     *
     * Call this once everything else this pass needs is already allocated: the
     * memory bound is read from what is left, so anything taken afterwards is
     * memory this grid has already been handed.
     *
     * @param[in] q_max Largest octahedral shell, which sets the per-block buffer
     * @param[in] num_src Number of sources
     * @param[in] block_size Threads per block, as launched
     *
     * @return Number of blocks to launch, never zero
     * @throw std::runtime_error if a single block's buffer does not fit
     */
    size_t resolve_grid_size(int q_max, size_t num_src, size_t block_size) {
        int blocks_per_sm = 0;
        safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, evolve0D_gpu, static_cast<int>(block_size), 0
        ));

        // Zero means the kernel cannot be launched at this block size at all,
        // usually because it exceeds the thread or register limits.
        if (blocks_per_sm == 0)
            throw std::runtime_error(
                std::format(
                    "no block of {} threads fits on a multiprocessor for this kernel",
                    block_size
                )
            );

        int num_sms = 0;
        safe_cuda(cudaDeviceGetAttribute(
            &num_sms, cudaDevAttrMultiProcessorCount, device::device_id()
        ));

        size_t free_bytes = 0, total_bytes = 0;
        safe_cuda(cudaMemGetInfo(&free_bytes, &total_bytes));

        // cudaMemGetInfo answers for the driver, which counts everything the pool
        // has reserved as used, including the part it is holding for reuse. That
        // part is what the next cudaMallocAsync is served from without touching the
        // driver at all, so it has to be added back or the grid shrinks as soon as
        // the pool has cached anything.
        if (auto pool = device::pool()) {
            uint64_t reserved = 0;
            uint64_t used = 0;
            safe_cuda(cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrReservedMemCurrent, &reserved
            ));
            safe_cuda(
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used)
            );
            free_bytes += reserved - used;
        }

        auto bytes_per_block = cells_to_shell(q_max) * sizeof(double);

        if (free_bytes <= bytes_per_block)
            throw std::runtime_error(
                std::format(
                    "column density scratch for a single block is {} MiB at q_max = "
                    "{}, "
                    "and only {} MiB are available on the device",
                    bytes_per_block >> 20, q_max, free_bytes >> 20
                )
            );

        auto by_memory = free_bytes / bytes_per_block;
        auto by_occupancy = static_cast<size_t>(blocks_per_sm) * num_sms;

        return std::min({by_occupancy, by_memory, num_src});
    }

}  // namespace

namespace asora {

    void do_all_sources_gpu(
        double R, double sigma, double dr, double *phi_ion, size_t num_src, size_t m1,
        double minlogtau, double dlogtau, size_t num_tau, size_t block_size
    ) {
        device::check_initialized();

        namespace tag = resident_tag;
        const auto &number_density = device::resident<tag::number_density>();
        const auto &photo_ion_thin = device::resident<tag::photo_ion_thin>();
        const auto &photo_ion_thick = device::resident<tag::photo_ion_thick>();
        const auto &source_flux = device::resident<tag::source_flux>();
        const auto &source_position = device::resident<tag::source_position>();
        const auto &fraction_HII_avg = device::resident<tag::fraction_HII_avg>();
        if (!number_density)
            throw std::runtime_error(
                "number density array must be allocated on the device before calling "
                "do_all_sources_gpu"
            );
        if (!photo_ion_thin || !photo_ion_thick)
            throw std::runtime_error(
                "photo ionization tables (thin, thick) must be allocated on the device "
                "before calling do_all_sources_gpu"
            );
        if (!source_flux || !source_position)
            throw std::runtime_error(
                "source data (flux, position) must be allocated on the device before "
                "calling do_all_sources_gpu"
            );
        if (!fraction_HII_avg)
            throw std::runtime_error(
                "average ionized fraction must be allocated on the device before "
                "calling do_all_sources_gpu"
            );

        // Size of grid data
        auto n_cells = m1 * m1 * m1;

        // The average ionized fraction is read but never written here. It is seeded
        // once per timestep by timestep_data_to_device and thereafter updated in
        // place by the chemistry pass, so no transfer is needed. Ranks that do not
        // run chemistry refresh it with average_fraction_to_device after the
        // broadcast.
        density_maps densities{number_density.data(), fraction_HII_avg.data()};

        // Allocate (if necessary) and zero the output array for the photoionization
        // rate
        auto phion_HI = device::scratch<double>(n_cells);
        phion_HI.zero();

        // With no sources there is nothing to accumulate, but the caller still
        // expects a zeroed rate field.
        if (num_src > 0) {
            // Determine how large the octahedron should be, based on the raytracing
            // radius. The radius equals the distance from the source to the middle
            // of the faces of the octahedron. To raytrace the whole volume, the
            // octahedron must be 1.5*N in size. Allocate (if necessary) the column
            // density array, one buffer per block.
            int q_max = std::ceil(c::sqrt3<> * std::min(R, c::sqrt3<> * m1 / 2.0));
            auto grid_size = resolve_grid_size(q_max, num_src, block_size);
            std::cout << std::format(
                "Launching {} blocks of {} threads for {} sources, q_max = {}\n",
                grid_size, block_size, num_src, q_max
            );
            auto column_density_HI =
                device::scratch<double>(cells_to_shell(q_max) * grid_size);

            // Create helper data structures: data_HI, ion_tables, logtau

            element_data data_HI{phion_HI.data(), column_density_HI.data(), sigma};
            photo_tables ion_tables{photo_ion_thin.data(), photo_ion_thick.data()};
            linspace<double> logtau{minlogtau, dlogtau, static_cast<size_t>(num_tau)};

            // A single launch: the blocks are persistent and share out the sources
            // between themselves, so there is no batch loop and no barrier between
            // batches for an early-finishing block to wait on.
            evolve0D_gpu<<<grid_size, block_size>>>(
                m1, dr, R, q_max, num_src, source_position.data(), source_flux.data(),
                data_HI, densities, ion_tables, logtau
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
        size_t m1, double dr, double R_max, int q_max, size_t num_src,
        const int *__restrict__ src_pos, const double *__restrict__ src_flux,
        element_data data_HI, density_maps densities, photo_tables ion_tables,
        linspace<double> logtau
    ) {
        /* The raytracing kernel proceeds as follows:
         * 1. Take the next source assigned to this block
         * 2. Loop over the asora q-shells around the source, up to q_max
         * 3. For each shell, threads independently raytrace on all cells
         * 4. Before moving to the next q-shell, threads are synchronized to ensure
         * causality
         * 5. Go back to 1 until the block's share of the sources is exhausted
         */

        // Offset pointer to the outgoing column density array used for
        // interpolation. Each block owns one buffer for the whole launch, so the
        // binding is static and no two blocks can ever share one: nothing has to be
        // synchronized to protect it.
        data_HI.column_density += blockIdx.x * cells_to_shell(q_max);

        // The blocks are persistent and walk the source list in strides. A block
        // that finishes a source early takes the next one straight away instead of
        // idling until the rest of the grid catches up, which is what a batch of
        // one-source-per-block would have made it do.
        for (size_t ns = blockIdx.x; ns < num_src; ns += gridDim.x) {
            // Get source properties.
            const auto i0 = src_pos[3 * ns + 0];
            const auto j0 = src_pos[3 * ns + 1];
            const auto k0 = src_pos[3 * ns + 2];
            const auto strength = src_flux[ns];

            // Calculate column density and photoionization rate for the source
            // cell. This is done separately from the main loop because to take
            // advantage of some simplifications.
            if (threadIdx.x == 0) {
                const auto index = ravel_index(i0, j0, k0, m1);
                auto nHI = densities.get(index);
                update_photo_rates(
                    data_HI, 0, index, 0.0, nHI, 0.5 * dr, strength, dr * dr * dr,
                    ion_tables, logtau
                );
            }
            __syncthreads();

            // Loop over q-shells and each thread peforms raytracing on one or more
            // cells. "s" is the index in the range [0, ..., 4q^2 + 2) that gets
            // mapped to the cells in the shell. (q, s) indices are mapped to
            // (i, j, k) indices via asora::linthrd2cart.
            for (int q = 1; q <= q_max; ++q) {
                // Prepare shared memory for column density interpolation for this
                // shell.
                data_HI.partition_column_density(q);

                // Each thread can process multiple cells.
                int s = threadIdx.x;
                while (static_cast<size_t>(s) < cells_in_shell(q)) {
                    raytrace(
                        q, s, i0, j0, k0, strength, data_HI, dr, R_max, densities, m1,
                        ion_tables, logtau
                    );
                    s += blockDim.x;
                }
                __syncthreads();
            }
        }
    }

}  // namespace asora
