#include "raytracing.cuh"

#include "memory.h"
#include "utils.cuh"

#include <cuda_runtime.h>

#include <cuda/std/array>
#include <exception>

namespace asora {

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

    __device__ double cinterp(
        const raytracing_lut::entry &__restrict__ entry,
        const double *__restrict__ column_dens, double cross_section
    ) {
        // Reference optical depth from C2-Ray interpolation function.
        constexpr double tau_0 = 0.6;

        const cuda::std::array<double, 4> factors = {
            (1. - entry.dx) * (1. - entry.dy), (1. - entry.dy) * entry.dx,
            (1. - entry.dx) * entry.dy, entry.dx * entry.dy
        };

        // Column density at the crossing point is a weighted average.
        double cdens = 0.0;
        double wtot = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            auto c = column_dens[entry.indices[i]];
            auto w = factors[i] / max(tau_0, c * cross_section);

            cdens += w * c;
            wtot += w;
        }

        return cdens / wtot * entry.multiplier;
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
        const raytracing_lut::entry &entry, size_t cd_index, const int3 &pos,
        double strength, element_data &data_HI, double dr, double R_max,
        const density_maps &densities, size_t m1, const photo_tables &ion_tables,
        const linspace<double> &logtau, const int2 &limit
    ) {
        auto &&[di, dj, dk] = unpack_offset(entry.offset);

        if ((di < limit.x) || (di > limit.y) || (dj < limit.x) || (dj > limit.y) ||
            (dk < limit.x) || (dk > limit.y))
            return;

#if !defined(PERIODIC)
        // When not in periodic mode, only treat cell if its in the grid
        if (!in_box(pos.x + di, pos.y + dj, pos.z + dk, m1)) return;
#endif
        // Using integers for threshold check as it is more consistent.
        auto dist2 = di * di + dj * dj + dk * dk;
        if (dist2 > static_cast<int>(R_max * R_max)) return;

        auto coldens_in = cinterp(entry, data_HI.column_density, data_HI.cross_section);

        constexpr double max_coldens = 2e30;
        if (coldens_in > max_coldens) return;

        auto path = dr * entry.path;
        auto vol_ph = 4 * c::pi<> * dist2 * path * dr * dr;

        // Get local ionization fraction & neutral hydrogen density in the cell
        const auto ph_index = ravel_index(pos.x + di, pos.y + dj, pos.z + dk, m1);
        auto nHI = densities.get(ph_index);

        // Compute photoionization rates from column density.
        update_photo_rates(
            data_HI, cd_index, ph_index, coldens_in, nHI, path, strength, vol_ph,
            ion_tables, logtau
        );
    }

}  // namespace

namespace asora {

    void create_raytracing_lut(int q_max) {
        if (device::contains(buffer_tag::raylut_offsets)) return;

        auto lut = create_lut(q_max);

        auto upload = [](const auto &vec, buffer_tag tag) {
            device::ensure_transfer(tag, vec.data(), vec.size());
        };

        upload(lut.offsets, buffer_tag::raylut_offsets);
        upload(lut.multipliers, buffer_tag::raylut_multipliers);
        upload(lut.dxs, buffer_tag::raylut_dx);
        upload(lut.dys, buffer_tag::raylut_dy);
        upload(lut.paths, buffer_tag::raylut_path);
        upload(lut.indices, buffer_tag::raylut_indices);
    }

    void do_all_sources_gpu(
        double R, double sigma, double dr, const double *xh_av, double *phi_ion,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size
    ) {
        device::check_initialized();

        // Number density array is not modified, it is assumed that it is already on the
        // device.
        if (!device::contains(buffer_tag::number_density))
            throw std::runtime_error(
                "Number density array must be allocated on the device before calling "
                "do_all_sources_gpu"
            );

        // Determine how large the octahedron should be, based on the raytracing
        // radius. The radius equals the distance from the source to the middle of the
        // faces of the octahedron. To raytrace the whole volume, the octahedron must
        // be 1.5*N in size. Allocate (if necessary) the column density array.
        int q_max = std::ceil(c::sqrt3<> * std::min(R, c::sqrt3<> * m1 / 2.0));

        // Build the LUT for the raytracing kernel.
        raytracing_lut_ptr lut_d{
            get_data_view<uint32_t>(buffer_tag::raylut_offsets),
            get_data_view<double>(buffer_tag::raylut_multipliers),
            get_data_view<double>(buffer_tag::raylut_dx),
            get_data_view<double>(buffer_tag::raylut_dy),
            get_data_view<double>(buffer_tag::raylut_path),
            get_data_view<index4>(buffer_tag::raylut_indices)
        };

        // Size of grid data.
        auto n_cells = m1 * m1 * m1;

        // Allocate (if necessary) and copy the ionized fraction array to the device.
        device::ensure_transfer<double>(buffer_tag::fraction_HII, xh_av, n_cells);

        // Allocate (if necessary) and zero the output array for the photoionization
        // rate.
        device::ensure<double>(buffer_tag::photo_ionization_HI, n_cells);
        auto phi_buf = device::get(buffer_tag::photo_ionization_HI);
        auto phi_d = phi_buf.data<double>();
        safe_cuda(cudaMemset(phi_d, 0, phi_buf.size()));

        device::ensure<double>(
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

        // Create helper data structures: density maps, data_HI, ion_tables, logtau.
        density_maps densities{
            get_data_view<double>(buffer_tag::number_density),
            get_data_view<double>(buffer_tag::fraction_HII)
        };

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
                lut_d, m1, dr, R, q_max, ns, num_src, src_pos_d, src_flux_d, data_HI,
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
        raytracing_lut_ptr lut, size_t m1, double dr, double R_max, int q_max,
        size_t ns_start, size_t num_src, const int *__restrict__ src_pos,
        const double *__restrict__ src_flux, element_data data_HI,
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
        size_t cd_offset = blockIdx.x * cells_to_shell(q_max);
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

        int ll = -m1 / 2;
        int lr = m1 % 2 - 1 - ll;

        for (int q = 1; q <= q_max; ++q) {
            // Each thread can process multiple cells.
            size_t s = threadIdx.x;
            while (s < cells_in_shell(q)) {
                auto cd_index = cells_to_shell(q - 1) + s;
                auto entry = lut[cd_index];
                raytrace(
                    entry, cd_index, {i0, j0, k0}, strength, data_HI, dr, R_max,
                    densities, m1, ion_tables, logtau, {ll, lr}
                );

                s += blockDim.x;
            }
            __syncthreads();
        }
    }

}  // namespace asora
