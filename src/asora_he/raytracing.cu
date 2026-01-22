#include "raytracing.cuh"

#include "../asora/memory.h"
#include "../asora/utils.cuh"
#include "rates.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cuda/std/span>
#include <exception>

namespace asora {

    __device__ element_data::shared_cdens_t element_data::make_shared_cdens(
        int q
    ) const {
        return {
            column_density + asora::cells_to_shell(q - 2),
            column_density + asora::cells_to_shell(q - 3),
            column_density + asora::cells_to_shell(q - 4)
        };
    }

    __device__ cuda::std::array<double, 3> density_maps::get(size_t index) const {
        // TODO: need to expose this to parameters.yml
        constexpr double abu_he_mass = 0.2486;

        auto np = ndens[index];
        auto nHI = np * (1.0 - abu_he_mass) * (1.0 - xHII[index]);
        auto nHeI = np * abu_he_mass * (1.0 - xHeII[index] - xHeIII[index]);
        auto nHeII = np * abu_he_mass * xHeII[index];

        return {nHI, nHeI, nHeII};
    }

}  // namespace asora

namespace {

    using namespace asora;

    using cross_section_histogram = cuda::std::span<double>;

    template <typename T>
    T *get_data_view(asora::buffer_tag tag) {
        return asora::device::get(tag).view<T>().data();
    }

    element_data make_element_data(
        asora::buffer_tag ion, asora::buffer_tag heat, asora::buffer_tag cdens,
        asora::buffer_tag sigma, size_t first_bin
    ) {
        return {
            get_data_view<double>(ion),
            get_data_view<double>(heat),
            get_data_view<double>(cdens),
            get_data_view<double>(sigma),
            first_bin,
        };
    }

    density_maps make_density_maps() {
        return {
            get_data_view<double>(buffer_tag::number_density),
            get_data_view<double>(buffer_tag::fraction_HII),
            get_data_view<double>(buffer_tag::fraction_HeII),
            get_data_view<double>(buffer_tag::fraction_HeIII)
        };
    }

    // Fortran-type modulo function (C modulo is signed)
    __host__ __device__ int modulo(int a, int b) { return (a % b + b) % b; }

    // Flat-array index from 3D (i,j,k) indices
    __device__ int mem_offset(int i, int j, int k, int N) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

#if !defined(PERIODIC)
    __device__ bool in_box_gpu(const int &i, const int &j, const int &k, const int &N) {
        return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
    }
#endif

    __device__ void update_photo_rates(
        element_data &data_HI, element_data &data_HeI, element_data &data_HeII,
        size_t cd_index, size_t ph_index,
        const cuda::std::array<double, 3> &coldens_in,  // for HI, HeI, HeII
        const cuda::std::array<double, 3> &ndens_in,    // for HI, HeI, HeII
        double path, double strength, double vol, const photo_tables &ion_tables,
        const photo_tables &heat_tables, const linspace<double> &logtau, int num_freq
    ) {
        auto &&[cd_in_HI, cd_in_HeI, cd_in_HeII] = coldens_in;
        auto &&[nHI, nHeI, nHeII] = ndens_in;

        auto &cd_out_HI = data_HI.column_density[cd_index];
        auto &cd_out_HeI = data_HeI.column_density[cd_index];
        auto &cd_out_HeII = data_HeII.column_density[cd_index];

        cd_out_HI = cd_in_HI + nHI * path;
        cd_out_HeI = cd_in_HeI + nHeI * path;
        cd_out_HeII = cd_in_HeII + nHeII * path;

        // Default-initialized to {0, 0}
        using photo_rate = cuda::std::pair<double, double>;
        photo_rate rate_HI;
        photo_rate rate_HeI;
        photo_rate rate_HeII;

        using tau = cuda::std::pair<double, double>;
        auto get_tau = [](int nf, const element_data &data, double cd_in,
                          double cd_out) -> tau {
            if (nf < data.first_bin) return {0.0, 0.0};
            auto sigma = data.cross_section[nf];
            return {cd_in * sigma, cd_out * sigma};
        };

        // frequency loop
        for (size_t nf = 0; nf < num_freq; ++nf) {
            // Compute optical depths
            auto tau_HI = get_tau(nf, data_HI, cd_in_HI, cd_out_HI);
            auto tau_HeI = get_tau(nf, data_HeI, cd_in_HeI, cd_out_HeI);
            auto tau_HeII = get_tau(nf, data_HeII, cd_in_HeII, cd_out_HeII);

            tau tau_tot{
                tau_HI.first + tau_HeI.first + tau_HeII.first,
                tau_HI.second + tau_HeI.second + tau_HeII.second
            };

            // TODO: potentially a problem if the fraction value is close to
            // zero.
            auto norm = strength / vol / (tau_tot.second - tau_tot.first);
            auto mul_HI = (tau_HI.second - tau_HI.first) * norm;
            auto mul_HeI = (tau_HeI.second - tau_HeI.first) * norm;
            auto mul_HeII = (tau_HeII.second - tau_HeII.first) * norm;

            auto phi = asora::photo_rates_gpu(
                tau_tot.first, tau_tot.second, nf, ion_tables, logtau
            );
            // TODO: heating requires more tables than just one because of the
            // frequency dependency. Probably solution is to move this inside
            // the if else condition (look at the radiation_tables.f90 line 322)
            auto heat = asora::photo_rates_gpu(
                tau_tot.first, tau_tot.second, nf, heat_tables, logtau
            );

            // Assign the photo-ionization and heating rates to each element
            // (part of the photon-conserving rate prescription)
            rate_HI.first += phi * mul_HI;
            rate_HeI.first += phi * mul_HeI;
            rate_HeII.first += phi * mul_HeII;
            rate_HI.second += heat * mul_HI;
            rate_HeI.second += heat * mul_HeI;
            rate_HeII.second += heat * mul_HeII;
        }  // end loop freq

        // Add the computed ionization and heating rate to the array atomically
        atomicAdd(data_HI.photo_ionization + ph_index, rate_HI.first / nHI);
        atomicAdd(data_HeI.photo_ionization + ph_index, rate_HeI.first / nHeI);
        atomicAdd(data_HeII.photo_ionization + ph_index, rate_HeII.first / nHeII);
        atomicAdd(data_HI.photo_heating + ph_index, rate_HI.second / nHI);
        atomicAdd(data_HeI.photo_heating + ph_index, rate_HeI.second / nHeI);
        atomicAdd(data_HeII.photo_heating + ph_index, rate_HeII.second / nHeII);
    }

    __device__ void raytrace(
        int q, int s, int i0, int j0, int k0, double strength, element_data &data_HI,
        element_data &data_HeI, element_data &data_HeII, double dr, double R_max,
        const density_maps &densities, int m1, const photo_tables &ion_tables,
        const photo_tables &heat_tables, const linspace<double> &logtau, int num_freq
    ) {
        using namespace asora;

        auto &&[di, dj, dk] = linthrd2cart(q, s);

        // Since the grid is periodic, we limit the maximum size of the raytraced
        // region to a cube as large as the mesh around the source. See line 93 of
        // evolve_source in C2Ray, this size will depend on if the mesh is even or
        // odd. Basically the idea is that you never touch a cell which is outside a
        // cube of length ~N centered on the source
        // Only do cell if it is within the (shifted under periodicity)
        // grid, i.e. at most ~N cells away from the source
        int ll = -m1 / 2;
        int lr = m1 % 2 - 1 - ll;
        if ((di < ll) || (di > lr) || (dj < ll) || (dj > lr) || (dk < ll) || (dk > lr))
            return;

#if !defined(PERIODIC)
        // When not in periodic mode, only treat cell if its in the grid
        if (!in_box(i0 + di, j0 + dj, k0 + dk, m1)) return;
#endif

        // Split column density in memory banks corresponding to shells q-1,
        // q-2, q-3.
        // FIXME: This is the same for each s....
        auto shared_cdens_HI = data_HI.make_shared_cdens(q);
        auto shared_cdens_HeI = data_HeI.make_shared_cdens(q);
        auto shared_cdens_HeII = data_HeII.make_shared_cdens(q);

        // Column density of HI, HeI and HeII to the cell
        auto cd_in = cinterp_gpu(
            di, dj, dk, shared_cdens_HI, shared_cdens_HeI, shared_cdens_HeII,
            data_HI.cross_section[0], data_HI.cross_section[0], data_HI.cross_section[0]
        );

        // Compute photoionization rates from column density.
        // WARNING: for now this is limited to the grey-opacity
        // test case source
        constexpr double max_coldens = 2e30;
        if (cd_in[0] > max_coldens || cd_in[1] > max_coldens || cd_in[2] > max_coldens)
            return;

        auto dist2 =
            (dr * di) * (dr * di) + (dr * dj) * (dr * dj) + (dr * dk) * (dr * dk);
        if (dist2 / (dr * dr) > R_max * R_max) return;

        // Map to periodic grid
        const auto index = mem_offset(i0 + di, j0 + dj, k0 + dk, m1);
        const auto q_off = cells_to_shell(q - 1);

        // Get local number density of HI, HeI, and HeII
        auto ns = densities.get(index);

        auto path = path_in_cell(di, dj, dk) * dr;
        auto vol = 4 * c::pi<> * dist2 * path;

        update_photo_rates(
            data_HI, data_HeI, data_HeII, q_off + s, index, cd_in, ns, path, strength,
            vol, ion_tables, heat_tables, logtau, num_freq
        );
    }

}  // namespace

namespace asora {
    // ========================================================================
    // Raytrace all sources and add up ionization rates
    // ========================================================================
    void do_all_sources_gpu(
        double R, const double *sig_HI, const double *sig_HeI, const double *sig_HeII,
        int num_bin_1, int num_bin_2, int num_bin_3, int num_freq, double dr,
        const double *xHII_av, const double *xHeII_av, const double *xHeIII_av,
        double *phi_ion_HI, double *phi_ion_HeI, double *phi_ion_HeII,
        double *phi_heat_HI, double *phi_heat_HeI, double *phi_heat_HeII, int num_src,
        int m1, double minlogtau, double dlogtau, int num_tau, size_t grid_size,
        size_t block_size
    ) {
        device::check_initialized();

        // Size of grid data
        auto n_cells = m1 * m1 * m1;

        // Initialize and copy density data.
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::fraction_HII, xHII_av},
                 std::pair{buffer_tag::fraction_HeII, xHeII_av},
                 std::pair{buffer_tag::fraction_HeIII, xHeIII_av},
             }) {
            if (!device::contains(tag)) device::add<double>(tag, n_cells);
            auto buf = device::get(tag);
            buf.copyFromHost(data, buf.size());
        }

        // Initialize and set to zero photo rate data.
        for (auto tag : {
                 buffer_tag::photo_ionization_HI,
                 buffer_tag::photo_ionization_HeI,
                 buffer_tag::photo_ionization_HeII,
                 buffer_tag::photo_heating_HI,
                 buffer_tag::photo_heating_HeI,
                 buffer_tag::photo_heating_HeII,
             }) {
            if (!device::contains(tag)) device::add<double>(tag, n_cells);
            auto buf = device::get(tag);
            safe_cuda(cudaMemset(buf.view<double>().data(), 0, buf.size()));
        }

        // Initialize and copy cross section data.
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::cross_section_HI, sig_HI},
                 std::pair{buffer_tag::cross_section_HeI, sig_HeI},
                 std::pair{buffer_tag::cross_section_HeII, sig_HeII},
             }) {
            if (!device::contains(tag)) device::add<double>(tag, num_freq);
            auto buf = device::get(tag);
            buf.copyFromHost(data, buf.size());
        }

        // Determine how large the octahedron should be, based on the raytracing
        // radius. Currently, this is set s.t. the radius equals the distance from
        // the source to the middle of the faces of the octahedron. To raytrace the
        // whole box, the octahedron must be 1.5*N in size
        int q_max = std::ceil(c::sqrt3<> * min(R, c::sqrt3<> * m1 / 2.0));

        // Allocate memory for column density calculations.
        for (auto tag : {
                 buffer_tag::column_density_HI,
                 buffer_tag::column_density_HeI,
                 buffer_tag::column_density_HeII,
             }) {
            if (!device::contains(tag))
                device::add<double>(tag, grid_size * cells_to_shell(q_max));
        }

        // flag that indicated the frequency bin for: HI (value 0), HI+HeI (value 1)
        // and HI+HeI+HeII (value 2)
        // int freq_flag=0;

        auto src_flux_d = get_data_view<double>(buffer_tag::source_flux);
        auto src_pos_d = get_data_view<int>(buffer_tag::source_position);

        auto densities = make_density_maps();

        auto data_HI = make_element_data(
            buffer_tag::photo_ionization_HI, buffer_tag::photo_heating_HI,
            buffer_tag::column_density_HI, buffer_tag::cross_section_HI, 0
        );
        auto data_HeI = make_element_data(
            buffer_tag::photo_ionization_HeI, buffer_tag::photo_heating_HeI,
            buffer_tag::column_density_HeI, buffer_tag::cross_section_HeI, num_bin_1
        );
        auto data_HeII = make_element_data(
            buffer_tag::photo_ionization_HeII, buffer_tag::photo_heating_HeII,
            buffer_tag::column_density_HeII, buffer_tag::cross_section_HeII,
            num_bin_1 + num_bin_2
        );

        photo_tables ion_tables{
            get_data_view<double>(buffer_tag::photo_ion_thin_table),
            get_data_view<double>(buffer_tag::photo_ion_thick_table)
        };
        photo_tables heat_tables{
            get_data_view<double>(buffer_tag::photo_heat_thin_table),
            get_data_view<double>(buffer_tag::photo_heat_thick_table)
        };

        linspace<double> logtau{minlogtau, dlogtau, static_cast<size_t>(num_tau)};

        // Loop over batches of sources
        for (int ns = 0; ns < num_src; ns += grid_size) {
            // Raytrace the current batch of sources in parallel
            // Consecutive kernel launches are in the same stream and so are
            // serialized
            evolve0D_gpu<<<grid_size, block_size>>>(
                m1, dr, R, q_max, ns, num_src, src_pos_d, src_flux_d, data_HI, data_HeI,
                data_HeII, densities, ion_tables, heat_tables, logtau, num_freq
            );

            safe_cuda(cudaPeekAtLastError());
        }

        // Copy the accumulated ionization rates back to the host
        for (auto &&[tag, data] : {
                 std::pair{buffer_tag::photo_ionization_HI, phi_ion_HI},
                 std::pair{buffer_tag::photo_ionization_HeI, phi_ion_HeI},
                 std::pair{buffer_tag::photo_ionization_HeII, phi_ion_HeII},
                 std::pair{buffer_tag::photo_heating_HI, phi_heat_HI},
                 std::pair{buffer_tag::photo_heating_HeI, phi_heat_HeI},
                 std::pair{buffer_tag::photo_heating_HeII, phi_heat_HeII},
             }) {
            auto buf = device::get(tag);
            buf.copyToHost(data, buf.size());
        }
    }

    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        int m1, double dr, double R_max, int q_max, int ns_start, int num_src,
        int *src_pos, double *src_flux, element_data data_HI, element_data data_HeI,
        element_data data_HeII, density_maps densities, photo_tables ion_tables,
        photo_tables heat_tables, linspace<double> logtau, int num_freq
    ) {
        /* The raytracing kernel proceeds as follows:
        1. Select the source based on the block number (within the batch = the grid)
        2. Loop over the asora q-cells around the source, up to q_max (loop "A")
        3. Inside each shell, threads independently do all cells, possibly requiring
        multiple iterations if the block size is smaller than the number of cells in
        the shell (loop "B")
        4. After each shell, the threads are synchronized to ensure that causality
        is respected
        */

        // Source number = Start of batch + block number (each block does one
        // source)
        const int ns = ns_start + blockIdx.x;

        // Ensure the source index is valid
        if (ns >= num_src) return;

        // Get source properties.
        const auto i0 = src_pos[3 * ns + 0];
        const auto j0 = src_pos[3 * ns + 1];
        const auto k0 = src_pos[3 * ns + 2];
        const auto strength = src_flux[ns];

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block needs its own copy of the array)
        int cd_offset = blockIdx.x * cells_to_shell(q_max);

        data_HI.column_density += cd_offset;
        data_HeI.column_density += cd_offset;
        data_HeII.column_density += cd_offset;

        if (threadIdx.x == 0) {
            const auto index = mem_offset(i0, j0, k0, m1);
            auto ns = densities.get(index);
            update_photo_rates(
                data_HI, data_HeI, data_HeII, 0, index, {0.0, 0.0, 0.0}, ns, 0.5 * dr,
                strength, dr * dr * dr, ion_tables, heat_tables, logtau, num_freq
            );
        }
        __syncthreads();

        // Loop over ASORA q-shells and each thread does raytracing on one or more
        // cells. "s" is the index in the range [0, ..., 4q^2 + 1] that gets mapped
        // to the cells in the shell. The threads are usually fewer than the number
        // of cells, therefore they can do additional work. (q, s) indexing is
        // mapped to the (i, j, k) indexing of the cells via the mapping described
        // in the paper.
        for (int q = 1; q <= q_max; ++q) {
            // We figure out the number of cells in the shell and determine how many
            // passes the block needs to take to treat all of them
            int num_cells = cells_in_shell(q);

            int s = threadIdx.x;
            while (s < num_cells) {
                raytrace(
                    q, s, i0, j0, k0, strength, data_HI, data_HeI, data_HeII, dr, R_max,
                    densities, m1, ion_tables, heat_tables, logtau, num_freq
                );
                s += blockDim.x;
            }
            // IMPORTANT: Sync threads after each shell so that the next only begins
            // when all outgoing column densities of the current shell are available
            __syncthreads();
        }
    }

    __device__ double path_in_cell(int di, int dj, int dk) {
        if (di == 0 && dj == 0 && dk == 0) return 0.5;
        double di2 = di * di;
        double dj2 = dj * dj;
        double dk2 = dk * dk;
        auto delta_max = max(di2, max(dj2, dk2));
        return sqrt((di2 + dj2 + dk2) / delta_max);
    }

    // dk is the largest delta.
    __device__ cuda::std::array<double, 4> geometric_factors(int di, int dj, int dk) {
        assert(dk != 0 && abs(dk) >= abs(di) && abs(dk) >= abs(dj));
        auto dk_inv = 1.0 / abs(dk);
        auto dx = abs(copysign(1.0, static_cast<double>(di)) - di * dk_inv);
        auto dy = abs(copysign(1.0, static_cast<double>(dj)) - dj * dk_inv);

        auto w1 = (1. - dx) * (1. - dy);
        auto w2 = (1. - dy) * dx;
        auto w3 = (1. - dx) * dy;
        auto w4 = dx * dy;

        return {w1, w2, w3, w4};
    }

    __device__ cuda::std::array<double, 3> cinterp_gpu(
        int di, int dj, int dk, const element_data::shared_cdens_t &cd_HI,
        const element_data::shared_cdens_t &cd_HeI,
        const element_data::shared_cdens_t &cd_HeII, double sigma_HI, double sigma_HeI,
        double sigma_HeII
    ) {
        // Degenerate case.
        if (di == 0 && dj == 0 && dk == 0) return {0.0, 0.0, 0.0};

        // This lambda selects the memory bank for the right q-shell, e.g., q-1, q-2
        // or q-3. Defined here to capture values before swaps take place.
        auto get_qlevel = [di, dj, dk, q0 = abs(di) + abs(dj) + abs(dk)](
                              int i_off, int j_off, int k_off
                          ) -> cuda::std::array<int, 2> {
            auto &&[q, s] = cart2linthrd(di - i_off, dj - j_off, dk - k_off);
            auto qlev = q0 - q - 1;
            assert(qlev >= 0 && qlev < 3);
            return {qlev, s};
        };

        auto ai = abs(di);
        auto aj = abs(dj);
        auto ak = abs(dk);
        int si = copysignf(1.0, di);
        int sj = copysignf(1.0, dj);
        int sk = copysignf(1.0, dk);

        // Offset index matrix for geometric factors w_i and cartesian coordinates
        // (i, j, k). Depending on which delta is largest, some offsets are turned
        // off.
        cuda::std::array<int, 12> offsets;
        if (ak >= ai && ak >= aj) {
            offsets = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, 0,  sk,  //
                0,  0,  sk   //
            };
        } else if (aj >= ai && aj >= ak) {
            offsets = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, sj, 0,   //
                0,  sj, 0    //
            };
            cuda::std::swap(dj, dk);
        } else {  // if (ai >= aj && ai >= ak)
            offsets = {
                si, sj, sk,  //
                si, 0,  sk,  //
                si, sj, 0,   //
                si, 0,  0    //
            };
            cuda::std::swap(di, dk);
            cuda::std::swap(di, dj);
        }

        auto factors = geometric_factors(di, dj, dk);

        // Reference optical depth from C2Ray interpolation function.
        constexpr double tau_0 = 0.6;

        // Column density at the crossing point is a weighted average.
        auto avg = [&offsets, &factors,
                    &get_qlevel](const element_data::shared_cdens_t &cd, double sigma) {
            double cdens = 0.0;
            double wtot = 0.0;
            // Loop over geometric factors and skip null ones: it helps avoid some
            // reads.
#pragma unroll
            for (auto xa = offsets.data(); auto w : factors) {
                if (w > 0.0) {
                    auto &&[qlev, s] = get_qlevel(xa[0], xa[1], xa[2]);
                    auto c = cd[qlev][s];

                    // Rescale weight by optical path
                    w /= max(tau_0, c * sigma);
                    cdens += w * c;
                    wtot += w;
                }
                // Access next row of the offset matrix.
                xa += 3;
            }

            // At least one weight was valid.
            assert(wtot > 0.0);
            cdens /= wtot;

            return cdens;
        };

        auto cdens_HI = avg(cd_HI, sigma_HI);
        auto cdens_HeI = avg(cd_HeI, sigma_HeI);
        auto cdens_HeII = avg(cd_HeII, sigma_HeII);

        // Take care of diagonals for cells close to the source.
        if (ai <= 1 && aj <= 1 && ak <= 1) {
            auto fact = sqrt(static_cast<double>(ai + ak + aj));
            cdens_HI *= fact;
            cdens_HeI *= fact;
            cdens_HeII *= fact;
        }

        return {cdens_HI, cdens_HeI, cdens_HeII};
    }

}  // namespace asora
