#include "raytracing.cuh"

#include "memory.h"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <cuda/std/array>
#include <exception>

namespace asora {

    struct neutral_density {
        __device__ double operator()(double n, double x) const { return n * (1.0 - x); }
    };

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
        constexpr float tau_0 = 0.6f;

        cuda::std::array<float, 4> factors = {
            (1.f - entry.dx) * (1.f - entry.dy), (1.f - entry.dy) * entry.dx,
            (1.f - entry.dx) * entry.dy, entry.dx * entry.dy
        };

        // Column density at the crossing point is a weighted average.
        double cdens = 0.0;
        double wtot = 0.0;
#pragma unroll 4
        for (size_t i = 0; i < 4; ++i) {
            auto c = column_dens[entry.indices[i]];
            auto w = factors[i] / max(tau_0, static_cast<float>(c * cross_section));

            cdens += w * c;
            wtot += w;
        }

        return cdens / wtot * entry.multiplier;
    }

    // Compute the photoionization rate for a given cell based on the incoming column
    // density and the pre-computed photoionization tables.
    __device__ void update_photo_rates(
        element_data &data_HI, size_t cd_index, size_t ph_index, double coldens_in,
        double nHI, double path, double scale, const photo_tables &ion_tables,
        const linspace<double> &logtau
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
        // add it to the global array. Dividing by the product uses one FP64 division
        // instead of two; vol is a cell volume, so the product cannot underflow.
        atomicAdd(data_HI.photo_ionization + ph_index, phion * scale / nHI);
    }

    // Raytracing operation on a given cell, identified by (q, s). This is performed by
    // a single thread. Threads may call this function multiple times if required to
    // cover the full q-shell.
    __device__ void raytrace(
        const raytracing_lut::entry &entry, size_t cd_index, const int3 &pos,
        double scale, element_data &data_HI, double dr, double R_max,
        const density_maps &densities, size_t m1, const photo_tables &ion_tables,
        const linspace<double> &logtau, const int2 &limit
    ) {
        const auto &di = entry.di;
        const auto &dj = entry.dj;
        const auto &dk = entry.dk;

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

        // vol_ph = 4*pi * dist2 * path * dr^2, with path = dr * entry.path. The
        // 4*pi*dr^3 factor does not depend on the cell and is hoisted to the caller.
        auto vol_ph = dist2 * entry.path;

        // Get local ionization fraction & neutral hydrogen density in the cell
        const auto ph_index = ravel_index(pos.x + di, pos.y + dj, pos.z + dk, m1);
        const auto &nHI = densities.nHI[ph_index];

        // Compute photoionization rates from column density.
        update_photo_rates(
            data_HI, cd_index, ph_index, coldens_in, nHI, path, scale / vol_ph,
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

        if (!device::contains(buffer_tag::raylut_offsets))
            throw std::runtime_error(
                "Raytracing lookup table must be allocated on the device before "
                "calling do_all_sources_gpu; call setup_raytracing_lut_gpu(q_max) first"
            );

        // Number density array is not modified, it is assumed that it is already on the
        // device.
        if (!device::contains(buffer_tag::number_density))
            throw std::runtime_error(
                "Number density array must be allocated on the device before calling "
                "do_all_sources_gpu"
            );
        // Size of grid data
        auto n_cells = m1 * m1 * m1;

        // Allocate (if necessary) and copy the ionized fraction array to the device
        device::ensure_transfer<double>(buffer_tag::fraction_HII, xh_av, n_cells);

        // Transform fraction in place to get number density.
        auto ndens_d = get_data_view<double>(buffer_tag::number_density);
        auto xHII_d = get_data_view<double>(buffer_tag::fraction_HII);
        thrust::transform(
            thrust::device, ndens_d, ndens_d + n_cells, xHII_d, xHII_d,
            neutral_density{}
        );

        density_maps densities{xHII_d};

        // Allocate (if necessary) and zero the output array for the photoionization
        // rate
        device::ensure<double>(buffer_tag::photo_ionization_HI, n_cells);
        auto phi_buf = device::get(buffer_tag::photo_ionization_HI);
        auto phi_d = phi_buf.data<double>();
        safe_cuda(cudaMemset(phi_d, 0, phi_buf.size()));

        // Determine how large the octahedron should be, based on the raytracing
        // radius. The radius equals the distance from the source to the middle of the
        // faces of the octahedron. To raytrace the whole volume, the octahedron must
        // be 1.5*N in size. Allocate (if necessary) the column density array.
        int q_max = std::ceil(c::sqrt3<> * std::min(R, c::sqrt3<> * m1 / 2.0));
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

        element_data data_HI{
            phi_d, get_data_view<double>(buffer_tag::column_density_HI), sigma
        };

        photo_tables ion_tables{
            get_data_view<double>(buffer_tag::photo_ion_thin_table),
            get_data_view<double>(buffer_tag::photo_ion_thick_table)
        };

        linspace<double> logtau{minlogtau, dlogtau, static_cast<size_t>(num_tau)};

        // Collect the LUT for raytracing kernel.
        raytracing_lut lut_d{};

        // Loop over batches of sources
        for (size_t ns = 0; ns < num_src; ns += grid_size) {
            // Raytrace the current batch of sources in parallel
            // Consecutive kernel launches are in the same stream and so are serialized
            evolve0D_gpu<<<grid_size, block_size>>>(
                lut_d, m1, dr, R, q_max, ns, num_src, src_pos_d, src_flux_d, data_HI,
                densities, ion_tables, logtau
            );
        }
        safe_cuda(cudaGetLastError());

        // Copy the accumulated ionization fraction back to the host.
        // Memcpy blocks until last kernel has finished.
        phi_buf.copyToHost(phi_ion);
    }

    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        raytracing_lut lut, size_t m1, double dr, double R_max, int q_max,
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
        auto scale = src_flux[ns] / (dr * dr * dr);

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block works on its own array).
        size_t cd_offset = blockIdx.x * cells_to_shell(q_max);
        data_HI.column_density += cd_offset;

        // Calculate column density and photoionization rate for the source cell.
        // This is done separately from the main loop because to take advantage of
        // some simplifications.
        if (threadIdx.x == 0) {
            const auto index = ravel_index(i0, j0, k0, m1);
            const auto &nHI = densities.nHI[index];
            update_photo_rates(
                data_HI, 0, index, 0.0, nHI, 0.5 * dr, scale, ion_tables, logtau
            );
        }
        __syncthreads();

        int ll = -m1 / 2;
        int lr = m1 % 2 - 1 - ll;

        // Cell-independent part of the photon volume 4*pi * dist2 * path * dr^2.
        scale /= 4 * c::pi<>;

        for (int q = 1; q <= q_max; ++q) {
            // Each thread can process multiple cells.
            size_t s = threadIdx.x;
            while (s < cells_in_shell(q)) {
                auto cd_index = cells_to_shell(q - 1) + s;
                auto entry = lut[cd_index];
                raytrace(
                    entry, cd_index, {i0, j0, k0}, scale, data_HI, dr, R_max, densities,
                    m1, ion_tables, logtau, {ll, lr}
                );

                s += blockDim.x;
            }
            __syncthreads();
        }
    }

}  // namespace asora
