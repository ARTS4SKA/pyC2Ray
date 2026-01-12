#include "raytracing.cuh"

#include "../asora/memory.h"
#include "../asora/utils.cuh"
#include "rates.cuh"

#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <string>

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

    // Fortran-type modulo function (C modulo is signed)
    __host__ __device__ int modulo(int a, int b) { return (a % b + b) % b; }

    // Sign function on the device
    __device__ int sign_gpu(const double &x) {
        if (x >= 0)
            return 1;
        else
            return -1;
    }

    // Flat-array index from 3D (i,j,k) indices
    __device__ int mem_offset_gpu(
        const int &i, const int &j, const int &k, const int &N
    ) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

    // Weight function for C2Ray interpolation function (see cinterp_gpu below)
    __device__ double weightf_gpu(const double &cd, const double &sig) {
        return 1.0 / max(0.6, cd * sig);
    }

#if !defined(PERIODIC)
    __device__ bool in_box_gpu(const int &i, const int &j, const int &k, const int &N) {
        return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
    }
#endif

    // Mapping from linear 1D indices to the cartesian coords of a q-shell in asora
    __device__ void linthrd2cart(const int &s, const int &q, int &i, int &j) {
        if (s == 0) {
            i = q;
            j = 0;
        } else {
            int b = (s - 1) / (2 * q);
            int a = (s - 1) % (2 * q);

            if (a + 2 * b > 2 * q) {
                a = a + 1;
                b = b - 1 - q;
            }
            i = a + b - q;
            j = b;
        }
    }

    template <typename T>
    T *get_data_view(asora::buffer_tag tag) {
        return asora::device::get(tag).view<T>().data();
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

        // Allocate memory for column density calculations.
        for (auto tag : {
                 buffer_tag::column_density_HI,
                 buffer_tag::column_density_HeI,
                 buffer_tag::column_density_HeII,
             }) {
            if (!device::contains(tag)) device::add<double>(tag, grid_size * n_cells);
        }

        // Determine how large the octahedron should be, based on the raytracing
        // radius. Currently, this is set s.t. the radius equals the distance from
        // the source to the middle of the faces of the octahedron. To raytrace the
        // whole box, the octahedron bust be 1.5*N in size
        int q_max = std::ceil(c::sqrt3<> * min(R, c::sqrt3<> * m1 / 2.0));

        // Since the grid is periodic, we limit the maximum size of the raytraced
        // region to a cube as large as the mesh around the source. See line 93 of
        // evolve_source in C2Ray, this size will depend on if the mesh is even or
        // odd. Basically the idea is that you never touch a cell which is outside a
        // cube of length ~N centered on the source
        int last_r = m1 / 2 - 1 + modulo(m1, 2);
        int last_l = -m1 / 2;

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
                num_bin_3, num_freq, last_l, last_r
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
        double Rmax_LLS,
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
        int num_bin_1, int num_bin_2, int num_bin_3, int num_freq, int last_l,
        int last_r
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

        // TODO: later need to import this from parameters.ylm
        double abu_he_mass = 0.2486;

        // Source number = Start of batch + block number (each block does one
        // source)
        int ns = ns_start + blockIdx.x;

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block needs its own copy of the array)
        int cdh_offset = blockIdx.x * m1 * m1 * m1;
        coldens_out_hi += cdh_offset;
        coldens_out_hei += cdh_offset;
        coldens_out_heii += cdh_offset;

        // Ensure the source index is valid
        if (ns >= num_src) return;

        // (A) Loop over ASORA q-shells
        for (int q = 0; q <= q_max; ++q) {
            // We figure out the number of cells in the shell and determine how many
            // passes the block needs to take to treat all of them
            int num_cells = 4 * q * q + 2;
            int Npass = num_cells / blockDim.x + 1;

            /* The threads have 1D indices 0,...,blocksize-1. We map these 1D
             * indices to the 3D positions of the cells inside the shell via the
             * mapping described in the paper. Since in general there are more cells
             * than threads, there is an additional loop here (B) so that all cells
             * are treated. */
            int s_end = q > 0 ? 4 * q * q + 2 : 1;
            int s_end_top = 2 * q * (q + 1) + 1;

            // (B) Loop over cells in the shell
            for (int ipass = 0; ipass < Npass; ipass++) {
                // "s" is the index in the 1D-range [0,...,4q^2 + 1] that gets
                // mapped to the cells in the shell
                int s = ipass * blockDim.x + threadIdx.x;
                int i, j, k;
                int sgn;

                // Ensure the thread maps to a valid cell
                if (s >= s_end) continue;

                // Determine if cell is in top or bottom part of the shell (the
                // mapping is slightly different due to the part that is on the
                // same z-plane as the source)
                if (s < s_end_top) {
                    sgn = 1;
                    ::linthrd2cart(s, q, i, j);
                } else {
                    sgn = -1;
                    ::linthrd2cart(s - s_end_top, q - 1, i, j);
                }
                k = sgn * q - sgn * (abs(i) + abs(j));

                // Only do cell if it is within the (shifted under periodicity)
                // grid, i.e. at most ~N cells away from the source
                if ((i < last_l) || (i > last_r) || (j < last_l) || (j > last_r) ||
                    (k < last_l) || (k > last_r))
                    continue;

                // Get source properties
                int i0 = src_pos[3 * ns + 0];
                int j0 = src_pos[3 * ns + 1];
                int k0 = src_pos[3 * ns + 2];
                double strength = src_flux[ns];

                // Center to source
                i += i0;
                j += j0;
                k += k0;

// When not in periodic mode, only treat cell if its in the grid
#if !defined(PERIODIC)
                if (!in_box_gpu(i, j, k, m1)) continue;
#endif
                // Map to periodic grid
                auto offset = mem_offset_gpu(i, j, k, m1);

                // Get local ionization fraction of HII, HeII and HeIII:
                auto xhii_av_p = xHII_av[offset];
                auto xhei_av_p = xHeII_av[offset];
                auto xheii_av_p = xHeIII_av[offset];

                // Get local number density of HI, HeI, and HeII
                auto nHI_p = ndens[offset] * (1.0 - abu_he_mass) * (1.0 - xhii_av_p);
                // Get local HeI number density
                auto nHeI_p =
                    ndens[offset] * abu_he_mass * (1.0 - xhei_av_p - xheii_av_p);
                // Get local HeII number density
                auto nHeII_p = ndens[offset] * abu_he_mass * xhei_av_p;

                // If its the source cell, just find path (no incoming
                // column density), otherwise if its another cell, do
                // interpolation to find incoming column density
                double path;
                double dist2;
                double vol_ph;
                double coldens_in_hi;    // HI Column density to the cell
                double coldens_in_hei;   // HeI Column density to the cell
                double coldens_in_heii;  // HeII Column density to the cell
                if (i == i0 && j == j0 && k == k0) {
                    coldens_in_hi = 0.0;
                    coldens_in_hei = 0.0;
                    coldens_in_heii = 0.0;
                    path = 0.5 * dr;
                    // vol_ph = dr*dr*dr / (4*M_PI);
                    vol_ph = dr * dr * dr;
                    dist2 = 0.0;
                } else {
                    cinterp_gpu(
                        i, j, k, i0, j0, k0, coldens_in_hi, path, coldens_out_hi,
                        sig_hi[0], m1
                    );
                    cinterp_gpu(
                        i, j, k, i0, j0, k0, coldens_in_hei, path, coldens_out_hei,
                        sig_hei[0], m1
                    );
                    cinterp_gpu(
                        i, j, k, i0, j0, k0, coldens_in_heii, path, coldens_out_heii,
                        sig_heii[0], m1
                    );

                    path *= dr;
                    // Find the distance to the source
                    auto xs = dr * (i - i0);
                    auto ys = dr * (j - j0);
                    auto zs = dr * (k - k0);
                    dist2 = xs * xs + ys * ys + zs * zs;
                    // vol_ph = dist2 * path;
                    vol_ph = dist2 * path * FOURPI;
                }

                double cdo_hi = coldens_in_hi + nHI_p * path;
                double cdo_hei = coldens_in_hei + nHeI_p * path;
                double cdo_heii = coldens_in_heii + nHeII_p * path;

                // Compute outgoing column density and add to array for
                // subsequent interpolations
                coldens_out_hi[offset] = cdo_hi;
                coldens_out_hei[offset] = cdo_hei;
                coldens_out_heii[offset] = cdo_heii;

                // Compute photoionization rates from column density.
                // WARNING: for now this is limited to the grey-opacity
                // test case source
                if ((coldens_in_hi > MAX_COLDENSH) || (coldens_in_hei > MAX_COLDENSH) ||
                    (coldens_in_heii > MAX_COLDENSH))
                    continue;
                if (dist2 / (dr * dr) > Rmax_LLS * Rmax_LLS) continue;

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
                        tau_out_hi = cdo_hi * sig_hi[nf];
                    }
                    if (nf >= num_bin_1) {
                        // Second frequency bin ionizes HI and HeI
                        tau_in_hei = coldens_in_hei * sig_hei[nf];
                        tau_out_hei = cdo_hei * sig_hei[nf];
                    }
                    if (nf >= num_bin_1 + num_bin_2) {
                        // Third frequency bin ionizes HI, HeI and HeII
                        tau_in_heii = coldens_in_heii * sig_heii[nf];
                        tau_out_heii = cdo_heii * sig_heii[nf];
                    }

                    double tau_in_tot = tau_in_hi + tau_in_hei + tau_in_heii;
                    double tau_out_tot = tau_out_hi + tau_out_hei + tau_out_heii;
                    double phi = photoion_rates_gpu(
                        strength, tau_in_tot, tau_out_tot, nf, vol_ph, photo_thin_table,
                        photo_thick_table, minlogtau, dlogtau, num_tau, num_freq
                    );
                    // TODO: heating requires more tables than just one because of the
                    // frequency dependency. Probably solution is to move this inside
                    // the if else condition (look at the radiation_tables.f90 line 322)
                    double heat = photoheat_rates_gpu(
                        strength, tau_in_tot, tau_out_tot, nf, vol_ph, heat_thin_table,
                        heat_thick_table, minlogtau, dlogtau, num_tau, num_freq
                    );

                    // Assign the photo-ionization and heating rates
                    // to each element (part of the
                    // photon-conserving rate prescription)
                    // TODO: potentially a problem if the fraction
                    // value is close to zero.
                    double phi_HI = phi * (tau_out_hi - tau_in_hi) /
                                    (tau_out_tot - tau_in_tot) / nHI_p;
                    double phi_HeI = phi * (tau_out_hei - tau_in_hei) /
                                     (tau_out_tot - tau_in_tot) / nHeI_p;
                    double phi_HeII = phi * (tau_out_heii - tau_in_heii) /
                                      (tau_out_tot - tau_in_tot) / nHeII_p;
                    double heat_HI = heat * (tau_out_hi - tau_in_hi) /
                                     (tau_out_tot - tau_in_tot) / nHI_p;
                    double heat_HeI = heat * (tau_out_hei - tau_in_hei) /
                                      (tau_out_tot - tau_in_tot) / nHeI_p;
                    double heat_HeII = heat * (tau_out_heii - tau_in_heii) /
                                       (tau_out_tot - tau_in_tot) / nHeII_p;

                    // Add the computed ionization and heating rate
                    // to the array ATOMICALLY since multiple blocks
                    // could be writing to the same cell at the same
                    // time!
                    atomicAdd(phi_ion_HI + offset, phi_HI);
                    atomicAdd(phi_ion_HeI + offset, phi_HeI);
                    atomicAdd(phi_ion_HeII + offset, phi_HeII);
                    atomicAdd(phi_heat_HI + offset, heat_HI);
                    atomicAdd(phi_heat_HeI + offset, heat_HeI);
                    atomicAdd(phi_heat_HeII + offset, heat_HeII);

                }  // end loop freq
            }
            // IMPORTANT: Sync threads after each shell so that the next only begins
            // when all outgoing column densities of the current shell are available
            __syncthreads();
        }
    }

    // ========================================================================
    // Short-characteristics interpolation function
    // ========================================================================
    __device__ void cinterp_gpu(
        int i, int j, int k, int i0, int j0, int k0, double &cdensi, double &path,
        double *coldensh_out, double sigma_at_freq, int m1
    ) {
        int idel, jdel, kdel;
        int idela, jdela, kdela;
        int im, jm, km;
        int sgni, sgnj, sgnk;
        double alam, xc, yc, zc, dx, dy, dz, s1, s2, s3, s4;
        double c1, c2, c3, c4;
        double w1, w2, w3, w4;
        double di, dj, dk;

        // calculate the distance between the source point (i0,j0,k0) and the
        // destination point (i,j,k)
        idel = i - i0;
        jdel = j - j0;
        kdel = k - k0;
        idela = abs(idel);
        jdela = abs(jdel);
        kdela = abs(kdel);

        // Find coordinates of points closer to source
        sgni = sign_gpu(idel);
        sgnj = sign_gpu(jdel);
        sgnk = sign_gpu(kdel);
        im = i - sgni;
        jm = j - sgnj;
        km = k - sgnk;
        di = double(idel);
        dj = double(jdel);
        dk = double(kdel);

        // Z plane (bottom and top face) crossing
        // we find the central (c) point (xc,xy) where the ray crosses the z-plane
        // below or above the destination (d) point, find the column density there
        // through interpolation, and add the contribution of the neutral material
        // between the c-point and the destination point.
        if (kdela >= jdela && kdela >= idela) {
            // alam is the parameter which expresses distance along the line s to d
            // add 0.5 to get to the interface of the d cell.
            alam = (double(km - k0) + sgnk * 0.5) / dk;

            xc = alam * di + double(i0);  // x of crossing point on z-plane
            yc = alam * dj + double(j0);  // y of crossing point on z-plane

            dx =
                2.0 * abs(xc - (double(im) + 0.5 * sgni));  // distances from c-point to
            dy = 2.0 * abs(yc - (double(jm) + 0.5 * sgnj));  // the corners.

            s1 = (1. - dx) * (1. - dy);  // interpolation weights of
            s2 = (1. - dy) * dx;         // corner points to c-point
            s3 = (1. - dx) * dy;
            s4 = dx * dy;

            c1 = coldensh_out[mem_offset_gpu(im, jm, km, m1)];
            c2 = coldensh_out[mem_offset_gpu(i, jm, km, m1)];
            c3 = coldensh_out[mem_offset_gpu(im, j, km, m1)];
            c4 = coldensh_out[mem_offset_gpu(i, j, km, m1)];

            // extra weights for better fit to analytical solution
            w1 = s1 * weightf_gpu(c1, sigma_at_freq);
            w2 = s2 * weightf_gpu(c2, sigma_at_freq);
            w3 = s3 * weightf_gpu(c3, sigma_at_freq);
            w4 = s4 * weightf_gpu(c4, sigma_at_freq);

            // column density at the crossing point
            cdensi = (c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) / (w1 + w2 + w3 + w4);

            // Take care of diagonals
            if (kdela == 1 && (idela == 1 || jdela == 1)) {
                if (idela == 1 && jdela == 1) {
                    cdensi = 1.73205080757 * cdensi;
                } else {
                    cdensi = 1.41421356237 * cdensi;
                }
            }

            // Path length from c through d to other side cell.
            path = sqrt((di * di + dj * dj) / (dk * dk) + 1.0);
        } else if (jdela >= idela && jdela >= kdela) {
            alam = (double(jm - j0) + sgnj * 0.5) / dj;
            zc = alam * dk + double(k0);
            xc = alam * di + double(i0);
            dz = 2.0 * abs(zc - (double(km) + 0.5 * sgnk));
            dx = 2.0 * abs(xc - (double(im) + 0.5 * sgni));
            s1 = (1. - dx) * (1. - dz);
            s2 = (1. - dz) * dx;
            s3 = (1. - dx) * dz;
            s4 = dx * dz;

            c1 = coldensh_out[mem_offset_gpu(im, jm, km, m1)];
            c2 = coldensh_out[mem_offset_gpu(i, jm, km, m1)];
            c3 = coldensh_out[mem_offset_gpu(im, jm, k, m1)];
            c4 = coldensh_out[mem_offset_gpu(i, jm, k, m1)];

            // extra weights for better fit to analytical solution
            w1 = s1 * weightf_gpu(c1, sigma_at_freq);
            w2 = s2 * weightf_gpu(c2, sigma_at_freq);
            w3 = s3 * weightf_gpu(c3, sigma_at_freq);
            w4 = s4 * weightf_gpu(c4, sigma_at_freq);

            cdensi = (c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) / (w1 + w2 + w3 + w4);

            // Take care of diagonals
            if (jdela == 1 && (idela == 1 || kdela == 1)) {
                if (idela == 1 && kdela == 1) {
                    cdensi = 1.73205080757 * cdensi;
                } else {
                    cdensi = 1.41421356237 * cdensi;
                }
            }
            path = sqrt((di * di + dk * dk) / (dj * dj) + 1.0);
        } else {
            alam = (double(im - i0) + sgni * 0.5) / di;
            zc = alam * dk + double(k0);
            yc = alam * dj + double(j0);
            dz = 2.0 * abs(zc - (double(km) + 0.5 * sgnk));
            dy = 2.0 * abs(yc - (double(jm) + 0.5 * sgnj));
            s1 = (1. - dz) * (1. - dy);
            s2 = (1. - dz) * dy;
            s3 = (1. - dy) * dz;
            s4 = dy * dz;

            c1 = coldensh_out[mem_offset_gpu(im, jm, km, m1)];
            c2 = coldensh_out[mem_offset_gpu(im, j, km, m1)];
            c3 = coldensh_out[mem_offset_gpu(im, jm, k, m1)];
            c4 = coldensh_out[mem_offset_gpu(im, j, k, m1)];

            // extra weights for better fit to analytical solution
            w1 = s1 * weightf_gpu(c1, sigma_at_freq);
            w2 = s2 * weightf_gpu(c2, sigma_at_freq);
            w3 = s3 * weightf_gpu(c3, sigma_at_freq);
            w4 = s4 * weightf_gpu(c4, sigma_at_freq);

            cdensi = (c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) / (w1 + w2 + w3 + w4);

            if (idela == 1 && (jdela == 1 || kdela == 1)) {
                if (jdela == 1 && kdela == 1) {
                    cdensi = 1.73205080757 * cdensi;
                } else {
                    cdensi = 1.41421356237 * cdensi;
                }
            }
            path = sqrt(1.0 + (dj * dj + dk * dk) / (di * di));
        }
    }

}  // namespace asora
