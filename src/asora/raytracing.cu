#include "raytracing.cuh"

#include "memory.h"
#include "utils.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <exception>

namespace asora {

    __device__ void element_data::partition_column_density(int q) {
        /// Partition the column density array into 3 shared memory banks for easier
        /// interpolation
        shared_cdens = {
            column_density + asora::cells_to_shell(q - 2),
            column_density + asora::cells_to_shell(q - 3),
            column_density + asora::cells_to_shell(q - 4)
        };
    }

    __device__ double density_maps::get(size_t index) const {
        return ndens[index] * (1.0 - xHII[index]);
    }

}  // namespace asora

namespace {

    using namespace asora;

    template <typename T>
    T *get_data_view(asora::buffer_tag tag) {
        return asora::device::get(tag).data<T>();
    }

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
        cell_interpolator interp{di, dj, dk};
        auto coldens_in =
            interp.interpolate(data_HI.shared_cdens, data_HI.cross_section);

        constexpr double max_coldens = 2e30;
        if (coldens_in > max_coldens) return;

        auto dist2 =
            (dr * di) * (dr * di) + (dr * dj) * (dr * dj) + (dr * dk) * (dr * dk);
        // Reducing the following calculation changes the numerical precision of
        // the result, albeit the physical result doesn't.
        if (dist2 / (dr * dr) > R_max * R_max) return;

        auto path = path_in_cell(di, dj, dk) * dr;
        auto vol_ph = 4 * c::pi<> * dist2 * path;

        // Get local ionization fraction & neutral hydrogen density in the cell
        const auto index = ravel_index(i0 + di, j0 + dj, k0 + dk, m1);
        const auto q_off = cells_to_shell(q - 1);
        double nHI = densities.get(index);

        // Compute photoionization rates from column density.
        update_photo_rates(
            data_HI, q_off + s, index, coldens_in, nHI, path, strength, vol_ph,
            ion_tables, logtau
        );
    }

}  // namespace

namespace asora {

    void do_all_sources_gpu(
        double R, double sigma, double dr, const double *xh_av, double *phi_ion,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size
    ) {
        device::check_initialized();

        // Size of grid data
        auto n_cells = m1 * m1 * m1;

        // Allocate (if necessary) and copy the ionized fraction array to the device
        device::transfer<double>(buffer_tag::fraction_HII, xh_av, n_cells);

        // Number density array is not modified, it is assumed that it is already on the
        // device
        if (!device::contains(buffer_tag::number_density))
            throw std::runtime_error(
                "Number density array must be allocated on the device before calling "
                "do_all_sources_gpu"
            );
        density_maps densities{
            get_data_view<double>(buffer_tag::number_density),
            get_data_view<double>(buffer_tag::fraction_HII)
        };

        // Allocate (if necessary) and zero the output array for the photoionization
        // rate
        if (!device::contains(buffer_tag::photo_ionization_HI))
            device::add<double>(buffer_tag::photo_ionization_HI, n_cells);
        auto phi_buf = device::get(buffer_tag::photo_ionization_HI);
        auto phi_d = phi_buf.data<double>();
        safe_cuda(cudaMemset(phi_d, 0, phi_buf.size()));

        // Determine how large the octahedron should be, based on the raytracing
        // radius. The radius equals the distance from the source to the middle of the
        // faces of the octahedron. To raytrace the whole volume, the octahedron must
        // be 1.5*N in size. Allocate (if necessary) the column density array.
        int q_max = std::ceil(c::sqrt3<> * std::min(R, c::sqrt3<> * m1 / 2.0));
        if (!device::contains(buffer_tag::column_density_HI))
            device::add<double>(
                buffer_tag::column_density_HI, grid_size * cells_to_shell(q_max)
            );

        // Get source properties, assuming the arrays are already on the device.
        if (!device::contains(buffer_tag::source_flux) ||
            !device::contains(buffer_tag::source_position))
            throw std::runtime_error(
                "Source properties must be allocated on the device before calling "
                "do_all_sources_gpu"
            );
        auto src_flux_d = get_data_view<double>(buffer_tag::source_flux);
        auto src_pos_d = get_data_view<int>(buffer_tag::source_position);

        // Create helper data structures: data_HI, ion_tables, logtau

        element_data data_HI{
            phi_d, get_data_view<double>(buffer_tag::column_density_HI), sigma
        };

        photo_tables ion_tables{
            get_data_view<double>(buffer_tag::photo_ion_thin_table),
            get_data_view<double>(buffer_tag::photo_ion_thick_table)
        };

        linspace<double> logtau{minlogtau, dlogtau, static_cast<size_t>(num_tau)};

        // Loop over batches of sources
        for (size_t ns = 0; ns < num_src; ns += grid_size) {
            // Raytrace the current batch of sources in parallel
            // Consecutive kernel launches are in the same stream and so are serialized
            evolve0D_gpu<<<grid_size, block_size>>>(
                m1, dr, R, q_max, ns, num_src, src_pos_d, src_flux_d, data_HI,
                densities, ion_tables, logtau
            );

            safe_cuda(cudaPeekAtLastError());
        }

        // Copy the accumulated ionization fraction back to the host.
        // Memcpy blocks until last kernel has finished.
        phi_buf.copyToHost(phi_ion);
    }

    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        size_t m1, double dr, double R_max, int q_max, size_t ns_start, size_t num_src,
        int *src_pos, double *__restrict__ src_flux, element_data data_HI,
        density_maps densities, photo_tables ion_tables, linspace<double> logtau
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
        const auto i0 = src_pos[3 * ns + 0];
        const auto j0 = src_pos[3 * ns + 1];
        const auto k0 = src_pos[3 * ns + 2];
        const auto strength = src_flux[ns];

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block works on its own array).
        int cd_offset = blockIdx.x * cells_to_shell(q_max);
        data_HI.column_density += cd_offset;

        // Calculate column density and photoionization rate for the source cell.
        // This is done separately from the main loop because to take advantage of
        // some simplifications.
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
        // cells. "s" is the index in the range [0, ..., 4q^2 + 2) that gets mapped to
        // the cells in the shell. (q, s) indices are mapped to (i, j, k) indices via
        // asora::linthrd2cart.
        for (int q = 1; q <= q_max; ++q) {
            // Prepare shared memory for column density interpolation for this shell.
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

}  // namespace asora
