#include "raytracing.cuh"

#include "memory.h"
#include "rates.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>

#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <exception>

namespace {

    constexpr double max_coldensh = 2e30;

    // Fortran-type modulo function (C modulo is signed)
    __host__ __device__ int modulo(int a, int b) { return (a % b + b) % b; }

    // Flat-array index from 3D (i,j,k) indices
    __device__ size_t mem_offset(int i, int j, int k, size_t N) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

#if !defined(PERIODIC)
    // Check if point is in domain
    __device__ bool in_box(int i, int j, int k, int N) {
        return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
    }
#endif

    __device__ void update_col_dens_phi_ion(
        double &coldens_out, double &phi_ion, double coldens_in, double nHI_p,
        double path, double strength, double vol, double sigma,
        const asora::photo_tables &ion_tables, const asora::linspace<double> &logtau
    ) {
        // Compute outgoing column density and add to array for subsequent
        // interpolations
        coldens_out = coldens_in + nHI_p * path;

#if defined(GREY_NOTABLES)
        auto phi = asora::photoion_rates_test_gpu(coldens_in, coldens_out, sigma);
#else
        auto phi = asora::photoion_rates_gpu(
            coldens_in, coldens_out, sigma, ion_tables, logtau
        );
#endif
        // Rescale the photo-ionization rate by the flux strength normalized per volume
        // and per neutral density (part of the photon-conserving rate prescription) and
        // add it to the global array
        atomicAdd(&phi_ion, phi * strength / vol / nHI_p);
    }

    // Raytracing operation on a given cell, identified by (q, s). This is performed by
    // a single thread. Threads may call this function multiple times if required to
    // cover the full q-shell.
    __device__ void raytrace(
        int q, int s, int i0, int j0, int k0, double strength,
        const cuda::std::array<const double *, 3> &shared_cdens,
        double *__restrict__ coldensh_out, const double *__restrict__ ndens,
        const double *__restrict__ xh_av, double *__restrict__ phi_ion, size_t m1,
        double sigma, double dr, double R_max, const asora::photo_tables &ion_tables,
        const asora::linspace<double> &logtau, int ll, int lr
    ) {
        using namespace asora;

        auto q_off = cells_to_shell(q - 1);
        auto &&[di, dj, dk] = linthrd2cart(q, s);

        // Only do cell if it is within the grid, shifted under periodicity
        // which means most ~N cells away from the source
        if ((di < ll) || (di > lr) || (dj < ll) || (dj > lr) || (dk < ll) || (dk > lr))
            return;

#if !defined(PERIODIC)
        // When not in periodic mode, only treat cell if its in the grid
        if (!in_box(i0 + di, j0 + dj, k0 + dk, m1)) return;
#endif
        // Calculate column density to and path inside this cell.
        auto [coldensh_in, path] = cinterp_gpu(di, dj, dk, shared_cdens, sigma);
        path *= dr;

        auto dist2 =
            (dr * di) * (dr * di) + (dr * dj) * (dr * dj) + (dr * dk) * (dr * dk);
        auto vol_ph = 4 * c::pi<> * dist2 * path;

        // Get local ionization fraction & neutral hydrogen density in the cell
        const auto index = mem_offset(i0 + di, j0 + dj, k0 + dk, m1);
        double nHI_p = ndens[index] * (1.0 - xh_av[index]);

        if (coldensh_in > max_coldensh) return;

        // Reducing the following calculation changes the numerical precision of
        // the result, albeit the physical result doesn't.
        if (dist2 / (dr * dr) > R_max * R_max) return;

        // Compute photoionization rates from column density.
        update_col_dens_phi_ion(
            coldensh_out[q_off + s], phi_ion[index], coldensh_in, nHI_p, path, strength,
            vol_ph, sigma, ion_tables, logtau
        );
    }

}  // namespace

namespace asora {

    // ========================================================================
    // Raytrace all sources and add up ionization rates
    // ========================================================================
    void do_all_sources_gpu(
        double R, double sigma, double dr, const double *xh_av, double *phi_ion,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size
    ) {
        device::check_initialized();

        // Lazy allocation of memory that'll be used until the end of the application
        auto n_cells = m1 * m1 * m1;
        if (!device::contains(buffer_tag::photo_ionization))
            device::add<double>(buffer_tag::photo_ionization, n_cells);
        if (!device::contains(buffer_tag::hydrogen_fraction))
            device::add<double>(buffer_tag::hydrogen_fraction, n_cells);

        //  Determine how large the octahedron should be, based on the raytracing
        //  radius. Currently, this is set s.t. the radius equals the distance from
        //  the source to the middle of the faces of the octahedron. To raytrace the
        //  whole box, the octahedron bust be 1.5*N in size
        int q_max = std::ceil(c::sqrt3<> * min(R, c::sqrt3<> * m1 / 2.0));
        if (!device::contains(buffer_tag::column_density))
            device::add<double>(
                buffer_tag::column_density, grid_size * cells_to_shell(q_max)
            );

        // Here we fill the ionization rate array with zero before raytracing all
        // sources. The LOCALRATES flag is for debugging purposes and will be
        // removed later on
        auto phi_buf = device::get(buffer_tag::photo_ionization);
        auto phi_d = phi_buf.view<double>().data();
        safe_cuda(cudaMemset(phi_d, 0, phi_buf.size()));

        // density array is not modified, asora assumes that it has been copied to
        // the device before
        auto xh_buf = device::get(buffer_tag::hydrogen_fraction);
        auto xh_d = xh_buf.view<double>().data();
        xh_buf.copyFromHost(xh_av, xh_buf.size());

        // Since the grid is periodic, we limit the maximum size of the raytraced
        // region to a cube as large as the mesh around the source. See line 93 of
        // evolve_source in C2Ray, this size will depend on if the mesh is even or
        // odd. Basically the idea is that you never touch a cell which is outside a
        // cube of length ~N centered on the source

        auto src_flux_d = device::get(buffer_tag::source_flux).view<double>().data();
        auto src_pos_d = device::get(buffer_tag::source_position).view<int>().data();
        auto cdh_d = device::get(buffer_tag::column_density).view<double>().data();
        auto n_d = device::get(buffer_tag::number_density).view<double>().data();
        auto ph_thin_d =
            device::get(buffer_tag::photo_thin_table).view<double>().data();
        auto ph_thick_d =
            device::get(buffer_tag::photo_thick_table).view<double>().data();

        // Loop over batches of sources
        for (size_t ns = 0; ns < num_src; ns += grid_size) {
            // Raytrace the current batch of sources in parallel
            // Consecutive kernel launches are in the same stream and so are serialized
            evolve0D_gpu<<<grid_size, block_size>>>(
                R, q_max, ns, num_src, src_pos_d, src_flux_d, cdh_d, sigma, dr, n_d,
                xh_d, phi_d, m1, ph_thin_d, ph_thick_d, minlogtau, dlogtau, num_tau
            );

            safe_cuda(cudaPeekAtLastError());
        }

        // Copy the accumulated ionization fraction back to the host
        // Memcpy blocks until last kernel has finished
        phi_buf.copyToHost(phi_ion, phi_buf.size());
    }

    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        double R_max,
        int q_max,  // Is now the size of max q
        size_t ns_start, size_t num_src, int *__restrict__ src_pos,
        double *__restrict__ src_flux, double *__restrict__ coldensh_out, double sigma,
        double dr, const double *__restrict__ ndens, const double *__restrict__ xh_av,
        double *__restrict__ phi_ion, size_t m1,
        const double *__restrict__ photo_thin_table,
        const double *__restrict__ photo_thick_table, double minlogtau, double dlogtau,
        size_t num_tau
    ) {
        /* The raytracing kernel proceeds as follows:
            1. Select the source based on the block number (within the batch = the
           grid)
            2. Loop over the asora q-cells around the source, up to q_max
            3. Inside each shell, threads independently raytrace on all cells
            4. At the end, threads are synchronized to ensure causality
        */

        // Source idenfitifer; one source per block.
        const int ns = ns_start + blockIdx.x;

        // Ensure the source index is valid.
        if (ns >= num_src) return;

        asora::linspace<double> logtau{minlogtau, dlogtau, num_tau};
        asora::photo_tables ion_tables{photo_thin_table, photo_thick_table};

        // Get source properties.
        const auto i0 = src_pos[3 * ns + 0];
        const auto j0 = src_pos[3 * ns + 1];
        const auto k0 = src_pos[3 * ns + 2];
        const auto strength = src_flux[ns];

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block works on its own array).
        coldensh_out += blockIdx.x * cells_to_shell(q_max);

        if (threadIdx.x == 0) {
            const auto index = mem_offset(i0, j0, k0, m1);
            double nHI_p = ndens[index] * (1.0 - xh_av[index]);
            update_col_dens_phi_ion(
                coldensh_out[0], phi_ion[index], 0.0, nHI_p, 0.5 * dr, strength,
                dr * dr * dr, sigma, ion_tables, logtau
            );
        }
        __syncthreads();

        // Grid bounds, shifted under periodicity.
        int ll = -m1 / 2;
        int lr = m1 % 2 - 1 - ll;

        // Loop over ASORA q-shells and each thread does raytracing on one or more
        // cells. "s" is the index in the range [0, ..., 4q^2 + 1] that gets mapped to
        // the cells in the shell. The threads are usually fewer than the number of
        // cells, therefore they can do additional work. (q, s) indexing is mapped to
        // the (i, j, k) indexing of the cells via the mapping described in the paper.
        for (int q = 1; q <= q_max; ++q) {
            int num_cells = cells_in_shell(q);

            // Split column density in memory banks corresponding to shells q-1, q-2,
            // q-3.
            cuda::std::array<const double *, 3> shared_cdens = {
                &coldensh_out[cells_to_shell(q - 2)],
                &coldensh_out[cells_to_shell(q - 3)],
                &coldensh_out[cells_to_shell(q - 4)]
            };

            int s = threadIdx.x;
            while (s < num_cells) {
                raytrace(
                    q, s, i0, j0, k0, strength, shared_cdens, coldensh_out, ndens,
                    xh_av, phi_ion, m1, sigma, dr, R_max, ion_tables, logtau, ll, lr
                );
                s += blockDim.x;
            }
            __syncthreads();
        }
    }

    // dk is the largest delta.
    __device__ cuda::std::array<double, 5> geometric_factors(
        double di, double dj, double dk
    ) {
        auto path = sqrt(1.0 + (di * di + dj * dj) / (dk * dk));

        auto dx = abs(std::copysign(1.0, di) - di / std::abs(dk));
        auto dy = abs(std::copysign(1.0, dj) - dj / std::abs(dk));

        auto w1 = (1. - dx) * (1. - dy);
        auto w2 = (1. - dy) * dx;
        auto w3 = (1. - dx) * dy;
        auto w4 = dx * dy;

        return {path, w1, w2, w3, w4};
    }

    __device__ cuda::std::pair<double, double> cinterp_gpu(
        int di, int dj, int dk, const cuda::std::array<const double *, 3> &shared_cdens,
        double sigma_HI_at_ion_freq
    ) {
        // Degenerate case.
        if (di == 0 && dj == 0 && dk == 0) return {0.0, 0.5};

        int ai = std::abs(di);
        int aj = std::abs(dj);
        int ak = std::abs(dk);
        // Cartesian index offsets for short characteristic interpolation towards the
        // ray source.
        cuda::std::array<int, 4> offsets = {
            0,                                          //
            static_cast<int>(std::copysignf(1.0, di)),  //
            static_cast<int>(std::copysignf(1.0, dj)),  //
            static_cast<int>(std::copysignf(1.0, dk))
        };

        // Capture values before swaps take place.
        auto get_column_density = [&shared_cdens, di, dj, dk,
                                   q0 = abs(di) + abs(dj) +
                                        abs(dk)](int i_off, int j_off, int k_off) {
            auto &&[q, s] = cart2linthrd(di - i_off, dj - j_off, dk - k_off);
            // Select the memory bank for the right q-shell, e.g., q-1, q-2 or q-3.
            auto qlev = q0 - q - 1;
            if (qlev < 0 || qlev >= shared_cdens.size()) return 0.0;
            return shared_cdens[qlev][s];
        };

        // Offset index matrix for geometric factors w_i and cartesian coordinates (i,
        // j, k). Default case is for (ak >= aj && ak >= ai), otherwise rearrange such
        // that "k" is the largest dimension.
        cuda::std::array<size_t, 12> xi = {
            1, 2, 3,  //
            0, 2, 3,  //
            1, 0, 3,  //
            0, 0, 3   //
        };
        if (aj >= ai && aj >= ak) {
            xi = {
                1, 2, 3,  //
                0, 2, 3,  //
                1, 2, 0,  //
                0, 2, 0   //
            };
            cuda::std::swap(dj, dk);
            cuda::std::swap(aj, ak);

        } else if (ai >= aj && ai >= ak) {
            xi = {
                1, 2, 3,  //
                1, 0, 3,  //
                1, 2, 0,  //
                1, 0, 0   //
            };
            cuda::std::swap(di, dk);
            cuda::std::swap(ai, ak);
            cuda::std::swap(di, dj);
            cuda::std::swap(ai, aj);
        }

        const auto factors = geometric_factors(
            static_cast<double>(di), static_cast<double>(dj), static_cast<double>(dk)
        );

        // Weight function for C2Ray interpolation function.
        auto weightf = [sigma = sigma_HI_at_ion_freq](double cd) {
            constexpr double tau_0 = 0.6;
            return 1.0 / max(tau_0, cd * sigma);
        };

        // Column density at the crossing point is a weighted average.
        double cdens = 0.0;
        double wtot = 0.0;
        // Index matrix accessor.
        auto xa = xi.data();
        // Loop over geomtric factors and skip null ones: it helps avoid some expensive
        // memory reads.
        for (size_t idx = 1; idx < factors.size(); ++idx, xa += 3) {
            auto w = factors[idx];
            if (w <= 0.0) continue;

            auto c = get_column_density(offsets[xa[0]], offsets[xa[1]], offsets[xa[2]]);
            w *= weightf(c);
            cdens += w * c;
            wtot += w;
        }
        cdens /= wtot;

        // Take care of diagonals.
        if (ak == 1 && ai == 1 && aj == 1)
            cdens *= c::sqrt3<>;
        else if (ak == 1 && (ai == 1 || aj == 1))
            cdens *= c::sqrt2<>;

        auto &path = factors[0];
        return cuda::std::make_pair(cdens, path);
    }

}  // namespace asora
