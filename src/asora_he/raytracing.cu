#include "raytracing.cuh"

#include "../asora/memory.h"
#include "../asora/utils.cuh"
#include "rates.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <exception>

// ========================================================================
// Define macros. Could be passed as parameters but are kept as
// compile-time constants for now
// ========================================================================
#define FOURPI 12.566370614359172463991853874177  // 4π
#define INV4PI 0.079577471545947672804111050482   // 1/4π
#define SQRT3 1.73205080757                       // Square root of 3
#define MAX_COLDENSH 2e30    // Column density limit (rates are set to zero above this)
#define CUDA_BLOCK_SIZE 256  // Size of blocks used to treat sources

// ========================================================================
// Utility Device Functions
// ========================================================================

namespace {

    // TODO: later need to import this from parameters.ylm
    constexpr double abu_he_mass = 0.2486;

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

    template <typename T>
    T *get_data_view(asora::buffer_tag tag) {
        return asora::device::get(tag).view<T>().data();
    }

    __device__ void update_photo_rates(
        double &coldens_out_hi, double &coldens_out_hei, double &coldens_out_heii,
        double &phi_ion_HI, double &phi_ion_HeI, double &phi_ion_HeII,
        double &phi_heat_HI, double &phi_heat_HeI, double &phi_heat_HeII,
        double coldens_in_hi, double coldens_in_hei, double coldens_in_heii,
        double nHI_p, double nHeI_p, double nHeII_p, double path, double strength,
        double vol, const double *sig_hi, const double *sig_hei, const double *sig_heii,
        int num_bin_1, int num_bin_2, int num_bin_3, int num_freq,
        const double *photo_thin_table, const double *photo_thick_table,
        const double *heat_thin_table, const double *heat_thick_table, double minlogtau,
        double dlogtau, int num_tau
    ) {
        coldens_out_hi = coldens_in_hi + nHI_p * path;
        coldens_out_hei = coldens_in_hei + nHeI_p * path;
        coldens_out_heii = coldens_in_heii + nHeII_p * path;

        double phi_HI = 0.0;
        double phi_HeI = 0.0;
        double phi_HeII = 0.0;
        double heat_HI = 0.0;
        double heat_HeI = 0.0;
        double heat_HeII = 0.0;

        // frequency loop
        for (int nf = 0; nf < num_freq; ++nf) {
            double tau_in_hi = 0.0;
            double tau_in_hei = 0.0;
            double tau_in_heii = 0.0;
            double tau_out_hi = 0.0;
            double tau_out_hei = 0.0;
            double tau_out_heii = 0.0;

            // Compute optical depths
            if (nf >= 0) {
                // First frequency bin ionizes just HI
                tau_in_hi = coldens_in_hi * sig_hi[nf];
                tau_out_hi = coldens_out_hi * sig_hi[nf];
            }
            if (nf >= num_bin_1) {
                // Second frequency bin ionizes HI and HeI
                tau_in_hei = coldens_in_hei * sig_hei[nf];
                tau_out_hei = coldens_out_hei * sig_hei[nf];
            }
            if (nf >= num_bin_1 + num_bin_2) {
                // Third frequency bin ionizes HI, HeI and HeII
                tau_in_heii = coldens_in_heii * sig_heii[nf];
                tau_out_heii = coldens_out_heii * sig_heii[nf];
            }

            auto tau_in_tot = tau_in_hi + tau_in_hei + tau_in_heii;
            auto tau_out_tot = tau_out_hi + tau_out_hei + tau_out_heii;

            // TODO: potentially a problem if the fraction value is close to
            // zero.
            auto norm = 1.0 / (tau_out_tot - tau_in_tot);
            auto rate_hi = (tau_out_hi - tau_in_hi) * norm;
            auto rate_hei = (tau_out_hei - tau_in_hei) * norm;
            auto rate_heii = (tau_out_heii - tau_in_heii) * norm;

            auto phi = asora::photoion_rates_gpu(
                strength, tau_in_tot, tau_out_tot, nf, vol, photo_thin_table,
                photo_thick_table, minlogtau, dlogtau, num_tau, num_freq
            );
            // TODO: heating requires more tables than just one because of the
            // frequency dependency. Probably solution is to move this inside
            // the if else condition (look at the radiation_tables.f90 line 322)
            auto heat = asora::photoheat_rates_gpu(
                strength, tau_in_tot, tau_out_tot, nf, vol, heat_thin_table,
                heat_thick_table, minlogtau, dlogtau, num_tau, num_freq
            );

            // Assign the photo-ionization and heating rates to each element
            // (part of the photon-conserving rate prescription)
            phi_HI += phi * rate_hi;
            phi_HeI += phi * rate_hei;
            phi_HeII += phi * rate_heii;
            heat_HI += heat * rate_hi;
            heat_HeI += heat * rate_hei;
            heat_HeII += heat * rate_heii;

        }  // end loop freq

        // Add the computed ionization and heating rate/ to the array ATOMICALLY
        // since multiple blocks could be writing to the same cell at the same
        // time!
        atomicAdd(&phi_ion_HI, phi_HI / nHI_p);
        atomicAdd(&phi_ion_HeI, phi_HeI / nHeI_p);
        atomicAdd(&phi_ion_HeII, phi_HeII / nHeII_p);
        atomicAdd(&phi_heat_HI, heat_HI / nHI_p);
        atomicAdd(&phi_heat_HeI, heat_HeI / nHeI_p);
        atomicAdd(&phi_heat_HeII, heat_HeII / nHeII_p);
    }

    __device__ void raytrace(
        int q, int s, int i0, int j0, int k0, double strength,
        const asora::shared_cdens_t &shared_cdens_hi,
        const asora::shared_cdens_t &shared_cdens_hei,
        const asora::shared_cdens_t &shared_cdens_heii, double *coldens_out_hi,
        double *coldens_out_hei, double *coldens_out_heii, const double *sig_hi,
        const double *sig_hei, const double *sig_heii, double dr, double R_max,
        const double *ndens, const double *xHII_av, const double *xHeII_av,
        const double *xHeIII_av, double *phi_ion_HI, double *phi_ion_HeI,
        double *phi_ion_HeII, double *phi_heat_HI, double *phi_heat_HeI,
        double *phi_heat_HeII, int m1, const double *photo_thin_table,
        const double *photo_thick_table, const double *heat_thin_table,
        const double *heat_thick_table, double minlogtau, double dlogtau, int num_tau,
        int num_bin_1, int num_bin_2, int num_bin_3, int num_freq, int ll, int lr
    ) {
        using namespace asora;

        auto &&[di, dj, dk] = linthrd2cart(q, s);

        // Only do cell if it is within the (shifted under periodicity)
        // grid, i.e. at most ~N cells away from the source
        if ((di < ll) || (di > lr) || (dj < ll) || (dj > lr) || (dk < ll) || (dk > lr))
            return;

#if !defined(PERIODIC)
        // When not in periodic mode, only treat cell if its in the grid
        if (!in_box(i0 + di, j0 + dj, k0 + dk, m1)) return;
#endif
        // Column density of HI, HeI and HeII to the cell
        auto &&[coldens_in_hi, coldens_in_hei, coldens_in_heii] = cinterp_gpu(
            di, dj, dk, shared_cdens_hi, shared_cdens_hei, shared_cdens_heii, sig_hi[0],
            sig_hei[0], sig_heii[0]
        );

        auto dist2 =
            (dr * di) * (dr * di) + (dr * dj) * (dr * dj) + (dr * dk) * (dr * dk);

        // Compute photoionization rates from column density.
        // WARNING: for now this is limited to the grey-opacity
        // test case source
        if ((coldens_in_hi > MAX_COLDENSH) || (coldens_in_hei > MAX_COLDENSH) ||
            (coldens_in_heii > MAX_COLDENSH))
            return;
        if (dist2 / (dr * dr) > R_max * R_max) return;

        // Map to periodic grid
        const auto index = mem_offset(i0 + di, j0 + dj, k0 + dk, m1);
        const auto q_off = cells_to_shell(q - 1);

        // Get local number density of HI, HeI, and HeII
        auto np = ndens[index];
        auto nHI_p = np * (1.0 - abu_he_mass) * (1.0 - xHII_av[index]);
        auto nHeI_p = np * abu_he_mass * (1.0 - xHeII_av[index] - xHeIII_av[index]);
        auto nHeII_p = np * abu_he_mass * xHeII_av[index];

        auto path = path_in_cell(di, dj, dk) * dr;
        auto vol_ph = 4 * c::pi<> * dist2 * path;

        update_photo_rates(
            coldens_out_hi[q_off + s], coldens_out_hei[q_off + s],
            coldens_out_heii[q_off + s], phi_ion_HI[index], phi_ion_HeI[index],
            phi_ion_HeII[index], phi_heat_HI[index], phi_heat_HeI[index],
            phi_heat_HeII[index], coldens_in_hi, coldens_in_hei, coldens_in_heii, nHI_p,
            nHeI_p, nHeII_p, path, strength, vol_ph, sig_hi, sig_hei, sig_heii,
            num_bin_1, num_bin_2, num_bin_3, num_freq, photo_thin_table,
            photo_thick_table, heat_thin_table, heat_thick_table, minlogtau, dlogtau,
            num_tau
        );
    }

}  // namespace

namespace asora {
    // ========================================================================
    // Raytrace all sources and add up ionization rates
    // ========================================================================
    void do_all_sources_gpu(
        double R, const double *sig_hi, const double *sig_hei, const double *sig_heii,
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
                 std::pair{buffer_tag::cross_section_HI, sig_hi},
                 std::pair{buffer_tag::cross_section_HeI, sig_hei},
                 std::pair{buffer_tag::cross_section_HeII, sig_heii},
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

        auto cdHI_d = get_data_view<double>(buffer_tag::column_density_HI);
        auto cdHeI_d = get_data_view<double>(buffer_tag::column_density_HeI);
        auto cdHeII_d = get_data_view<double>(buffer_tag::column_density_HeII);

        auto sigHI_d = get_data_view<double>(buffer_tag::cross_section_HI);
        auto sigHeI_d = get_data_view<double>(buffer_tag::cross_section_HeI);
        auto sigHeII_d = get_data_view<double>(buffer_tag::cross_section_HeII);

        auto n_d = get_data_view<double>(buffer_tag::number_density);
        auto xHII_d = get_data_view<double>(buffer_tag::fraction_HII);
        auto xHeII_d = get_data_view<double>(buffer_tag::fraction_HeII);
        auto xHeIII_d = get_data_view<double>(buffer_tag::fraction_HeIII);

        auto phion_HI_d = get_data_view<double>(buffer_tag::photo_ionization_HI);
        auto phion_HeI_d = get_data_view<double>(buffer_tag::photo_ionization_HeI);
        auto phion_HeII_d = get_data_view<double>(buffer_tag::photo_ionization_HeII);
        auto pheat_HI_d = get_data_view<double>(buffer_tag::photo_heating_HI);
        auto pheat_HeI_d = get_data_view<double>(buffer_tag::photo_heating_HeI);
        auto pheat_HeII_d = get_data_view<double>(buffer_tag::photo_heating_HeII);

        auto phion_thin_d = get_data_view<double>(buffer_tag::photo_ion_thin_table);
        auto phion_thick_d = get_data_view<double>(buffer_tag::photo_ion_thick_table);
        auto pheat_thin_d = get_data_view<double>(buffer_tag::photo_heat_thin_table);
        auto pheat_thick_d = get_data_view<double>(buffer_tag::photo_heat_thick_table);

        // Loop over batches of sources
        for (int ns = 0; ns < num_src; ns += grid_size) {
            // Raytrace the current batch of sources in parallel
            // Consecutive kernel launches are in the same stream and so are serialized
            evolve0D_gpu<<<grid_size, block_size>>>(
                R, q_max, ns, num_src, src_pos_d, src_flux_d, cdHI_d, cdHeI_d, cdHeII_d,
                sigHI_d, sigHeI_d, sigHeII_d, dr, n_d, xHII_d, xHeII_d, xHeIII_d,
                phion_HI_d, phion_HeI_d, phion_HeII_d, pheat_HI_d, pheat_HeI_d,
                pheat_HeII_d, m1, phion_thin_d, phion_thick_d, pheat_thin_d,
                pheat_thick_d, minlogtau, dlogtau, num_tau, num_bin_1, num_bin_2,
                num_bin_3, num_freq
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
        double R_max,
        int q_max,  // Is now the size of max q
        int ns_start, int num_src, int *src_pos, double *src_flux,
        double *coldens_out_hi, double *coldens_out_hei, double *coldens_out_heii,
        const double *sig_hi, const double *sig_hei, const double *sig_heii, double dr,
        const double *ndens, const double *xHII_av, const double *xHeII_av,
        const double *xHeIII_av, double *phi_ion_HI, double *phi_ion_HeI,
        double *phi_ion_HeII, double *phi_heat_HI, double *phi_heat_HeI,
        double *phi_heat_HeII, int m1, const double *photo_thin_table,
        const double *photo_thick_table, const double *heat_thin_table,
        const double *heat_thick_table, double minlogtau, double dlogtau, int num_tau,
        int num_bin_1, int num_bin_2, int num_bin_3, int num_freq
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
        int cdh_offset = blockIdx.x * cells_to_shell(q_max);
        coldens_out_hi += cdh_offset;
        coldens_out_hei += cdh_offset;
        coldens_out_heii += cdh_offset;

        if (threadIdx.x == 0) {
            const auto index = mem_offset(i0, j0, k0, m1);
            auto np = ndens[index];
            auto nHI_p = np * (1.0 - abu_he_mass) * (1.0 - xHII_av[index]);
            auto nHeI_p = np * abu_he_mass * (1.0 - xHeII_av[index] - xHeIII_av[index]);
            auto nHeII_p = np * abu_he_mass * xHeII_av[index];
            update_photo_rates(
                coldens_out_hi[0], coldens_out_hei[0], coldens_out_heii[0],
                phi_ion_HI[index], phi_ion_HeI[index], phi_ion_HeII[index],
                phi_heat_HI[index], phi_heat_HeI[index], phi_heat_HeII[index], 0.0, 0.0,
                0.0, nHI_p, nHeI_p, nHeII_p, 0.5 * dr, strength, dr * dr * dr, sig_hi,
                sig_hei, sig_heii, num_bin_1, num_bin_2, num_bin_3, num_freq,
                photo_thin_table, photo_thick_table, heat_thin_table, heat_thick_table,
                minlogtau, dlogtau, num_tau
            );
        }
        __syncthreads();

        // Since the grid is periodic, we limit the maximum size of the raytraced
        // region to a cube as large as the mesh around the source. See line 93 of
        // evolve_source in C2Ray, this size will depend on if the mesh is even or
        // odd. Basically the idea is that you never touch a cell which is outside a
        // cube of length ~N centered on the source
        int ll = -m1 / 2;
        int lr = m1 % 2 - 1 - ll;

        // Loop over ASORA q-shells and each thread does raytracing on one or more
        // cells. "s" is the index in the range [0, ..., 4q^2 + 1] that gets mapped to
        // the cells in the shell. The threads are usually fewer than the number of
        // cells, therefore they can do additional work. (q, s) indexing is mapped to
        // the (i, j, k) indexing of the cells via the mapping described in the paper.
        for (int q = 1; q <= q_max; ++q) {
            // We figure out the number of cells in the shell and determine how many
            // passes the block needs to take to treat all of them
            int num_cells = cells_in_shell(q);

            // Split column density in memory banks corresponding to shells q-1, q-2,
            // q-3.
            shared_cdens_t shared_cdens_hi = {
                &coldens_out_hi[cells_to_shell(q - 2)],
                &coldens_out_hi[cells_to_shell(q - 3)],
                &coldens_out_hi[cells_to_shell(q - 4)]
            };
            shared_cdens_t shared_cdens_hei = {
                &coldens_out_hei[cells_to_shell(q - 2)],
                &coldens_out_hei[cells_to_shell(q - 3)],
                &coldens_out_hei[cells_to_shell(q - 4)]
            };
            shared_cdens_t shared_cdens_heii = {
                &coldens_out_heii[cells_to_shell(q - 2)],
                &coldens_out_heii[cells_to_shell(q - 3)],
                &coldens_out_heii[cells_to_shell(q - 4)]
            };

            int s = threadIdx.x;
            while (s < num_cells) {
                raytrace(
                    q, s, i0, j0, k0, strength, shared_cdens_hi, shared_cdens_hei,
                    shared_cdens_heii, coldens_out_hi, coldens_out_hei,
                    coldens_out_heii, sig_hi, sig_hei, sig_heii, dr, R_max, ndens,
                    xHII_av, xHeII_av, xHeIII_av, phi_ion_HI, phi_ion_HeI, phi_ion_HeII,
                    phi_heat_HI, phi_heat_HeI, phi_heat_HeII, m1, photo_thin_table,
                    photo_thick_table, heat_thin_table, heat_thick_table, minlogtau,
                    dlogtau, num_tau, num_bin_1, num_bin_2, num_bin_3, num_freq, ll, lr
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
        int di, int dj, int dk, const shared_cdens_t &shared_cdens_hi,
        const shared_cdens_t &shared_cdens_hei, const shared_cdens_t &shared_cdens_heii,
        double sigma_HI, double sigma_HeI, double sigma_HeII
    ) {
        // Degenerate case.
        if (di == 0 && dj == 0 && dk == 0) return {0.0, 0.0, 0.0};

        // This lambda selects the memory bank for the right q-shell, e.g., q-1, q-2 or
        // q-3. Defined here to capture values before swaps take place.
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

        // Offset index matrix for geometric factors w_i and cartesian coordinates (i,
        // j, k). Depending on which delta is largest, some offsets are turned off.
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
                    &get_qlevel](const shared_cdens_t &shared_cdens, double sigma) {
            double cdens = 0.0;
            double wtot = 0.0;
            // Loop over geometric factors and skip null ones: it helps avoid some
            // reads.
#pragma unroll
            for (auto xa = offsets.data(); auto w : factors) {
                if (w > 0.0) {
                    auto &&[qlev, s] = get_qlevel(xa[0], xa[1], xa[2]);
                    auto c = shared_cdens[qlev][s];

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

        auto cdens_hi = avg(shared_cdens_hi, sigma_HI);
        auto cdens_hei = avg(shared_cdens_hei, sigma_HeI);
        auto cdens_heii = avg(shared_cdens_heii, sigma_HeII);

        // Take care of diagonals for cells close to the source.
        if (ai <= 1 && aj <= 1 && ak <= 1) {
            auto fact = sqrt(static_cast<double>(ai + ak + aj));
            cdens_hi *= fact;
            cdens_hei *= fact;
            cdens_heii *= fact;
        }

        return {cdens_hi, cdens_hei, cdens_heii};
    }

}  // namespace asora
