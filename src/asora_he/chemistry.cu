#include "chemistry.cuh"

#include "../asora/memory.h"
#include "../asora/utils.cuh"

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <iostream>

namespace {

    using namespace asora;

    __host__ __device__ double get_energy(
        double temp, double ndens, double gamma = 5. / 3.
    ) {
        return c::k_B<> * temp * ndens / (gamma - 1.0);
    }

    __host__ __device__ double get_temperature(
        double energy, double ndens, double gamma = 5. / 3.
    ) {
        return energy * (gamma - 1.0) / (c::k_B<> * ndens);
    }

    __host__ __device__ double cooling_rate(
        double temp, double ndens_atom, double ndens_elec,
        const cuda::std::array<double, 3>& xh, const cooling_tables& rates,
        const linspace<double>& logtemp, double abu_h, double abu_he
    ) {
        auto&& [i0, i1, p] = log_table_index(temp, logtemp);
        auto q = 1 - p;

        auto& [xHI, xHeI, xHeII] = xh;
        auto xHII = 1.0 - xHI;
        auto xHeIII = 1.0 - xHeI - xHeII;

        auto rHI = xHI * (rates.HI[i0] * q + rates.HI[i1] * p);
        auto rHII = xHII * (rates.HII[i0] * q + rates.HII[i1] * p);
        auto rHeI = xHeI * (rates.HeI[i0] * q + rates.HeI[i1] * p);
        auto rHeII = xHeII * (rates.HeII[i0] * q + rates.HeII[i1] * p);
        auto rHeIII = xHeIII * (rates.HeIII[i0] * q + rates.HeIII[i1] * p);

        return ndens_atom * ndens_elec *
               ((rHI + rHII) * abu_h + (rHeI + rHeII + rHeIII) * abu_he);
    }

    __host__ __device__ double cosmo_cooling_rate(double energy, double Hz) {
        return 2.0 * energy * Hz;
    }

    __device__ double electron_density(
        double ndens, const double3& xh, double abu_h, double abu_he, double abu_c
    ) {
        return ndens * (abu_h * xh.x + abu_he * (xh.y + 2.0 * xh.z) + abu_c);
    }

}  // namespace

namespace asora {

    __host__ __device__ cuda::std::array<double, 2> thermal(
        double dt, double temp, double ndens_elec, double ndens_atom, double heating,
        double Hz, const cuda::std::array<double, 3>& xh, const cooling_tables& rates,
        const linspace<double>& logtemp, const parameters& p, double min_temp,
        size_t max_iterations
    ) {
        // Thermal process only if temperature > min_temp
        if (temp <= min_temp) return {temp, temp};

        // Find initial internal energy
        auto ui = get_energy(temp, ndens_atom + ndens_elec, p.gamma);
        double tot_time = 0.0;
        double temp_av = 0.0;

        // Exit conditions: reached dt (with tolerance like Fortran)
        while (max_iterations > 0 && tot_time < dt * (1.0 - 1e-6)) {
            --max_iterations;

            auto rate = heating - cosmo_cooling_rate(ui, Hz);
            if (!p.cosmo_only)
                rate -= cooling_rate(
                    temp, ndens_atom, ndens_elec, xh, rates, logtemp, p.abu_h, p.abu_he
                );

            // Thermal time scale. Limit energy change to fraction relative_denergy
            // Don't integrate longer than remaining time.
            auto subdt =
                min(p.relative_denergy * ui / max(1e-50, abs(rate)), dt - tot_time);

            ui += rate * subdt;
            temp_av += 0.5 * temp * subdt;

            temp = get_temperature(ui, ndens_atom + ndens_elec, p.gamma);
            temp_av += 0.5 * temp * subdt;

            tot_time += subdt;

            // Enforce minimum temperature
            if (temp < min_temp) {
                ui = get_energy(min_temp, ndens_atom + ndens_elec, p.gamma);
                temp = min_temp;
                break;
            }
        }

        temp_av /= dt;
        return {temp, temp_av};
    }

}  // namespace asora

namespace {

    // Convergence criteria constants.
    constexpr double minimum_fractional_change = 1.0e-3;
    constexpr double minimum_fraction_of_atoms = 1.0e-8;

    // Compute recombination rates for HII, HeII, and HeIII based on their ionization
    // temperatures.
    __device__ double2
    recombination_rates(double temp, double temp0, double aA, double aB) {
        auto lambda = 2.0 * temp0 / temp;
        aA *= std::pow(lambda, 1.503) /
              std::pow(1.0 + std::pow(lambda / 0.522, 0.470), 1.923);
        aB *= std::pow(lambda, 1.500) /
              std::pow(1.0 + std::pow(lambda / 2.740, 0.407), 2.242);
        return {aA, aB};
    }

    // Create the solution for the chemistry equations: the same structure is used for
    // both the analytical solution and the time-averaged solution, just with different
    // weights.
    __device__ double3 compose_solution(
        const double3& psol, const double3& weights, const double3& x1,
        const double3& x2, const double3& x3
    ) {
        constexpr double res_tol = 1e-20;
        auto xHII = psol.x + x1.x * weights.x + x2.x * weights.y + x3.x * weights.z;
        auto xHeII = psol.y + x1.y * weights.x + x2.y * weights.y + x3.y * weights.z;
        auto xHeIII = psol.z + x1.z * weights.x + x2.z * weights.y + x3.z * weights.z;
        auto xHeI = 1.0 - xHeII - xHeIII;

        // Add minimum tolerance to avoid too small components and renormalize helium
        // part.
        xHII = max(xHII, res_tol);
        xHeII = max(xHeII, res_tol);
        xHeIII = max(xHeIII, res_tol);

        if (auto he_norm = xHeI + xHeII + xHeIII; he_norm > 1.0) {
            xHeII /= he_norm;
            xHeIII /= he_norm;
        }

        return {xHII, xHeII, xHeIII};
    }

    __device__ bool check_convergence_local(double new_value, double old_value) {
        bool cond1 =
            abs(new_value - old_value) / (1 - new_value) < minimum_fractional_change;
        bool cond2 = 1 - new_value < minimum_fraction_of_atoms;
        return cond1 || cond2;
    }

    __device__ bool check_convergence_global(double new_value, double old_value) {
        auto cond1 = abs(new_value - old_value) > minimum_fractional_change;
        auto cond2 =
            abs((new_value - old_value) / (1 - old_value)) > minimum_fractional_change;
        auto cond3 = (1 - old_value) > minimum_fraction_of_atoms;

        return cond1 && cond2 && cond3;
    }

    // Create the first row of the matrix A in the chemistry equations.
    __device__ double3 make_row1(
        const double2& alpha_HII, const double2& alpha_HeII, const double2& alpha_HeIII,
        double beta_HeIII, double nu, double n_e, double yy, double y2a, double y2b,
        double zz, const double3& phi, const parameters& p, double f_lya
    ) {
        auto rHII_HI = -alpha_HII.y;
        auto rHeII_HI = yy * (alpha_HeII.x - alpha_HeII.y) + p.p_rec * alpha_HeII.y;
        auto rHeIII_HI =
            (1 - y2a - y2b) * (alpha_HeIII.x - alpha_HeIII.y) + beta_HeIII +
            (nu * (p.l_dec - p.m_dec + p.m_dec * yy) + (1 - nu) * f_lya * zz) *
                alpha_HeIII.y;
        auto frac = (p.abu_he / p.abu_h) * n_e;
        return {
            -phi.x + rHII_HI * n_e,  // A11
            frac * rHeII_HI,         // A12
            frac * rHeIII_HI         // A13
        };
    }

    // Create the second row of the matrix A in the chemistry equations.
    __device__ double3 make_row2(
        const double2& alpha_HeII, const double2& alpha_HeIII, double nu, double n_e,
        double yy, double y2a, double y2b, double zz, const double3& phi,
        const parameters& p, double f_lya
    ) {
        auto rHeII_HeI = (1 - yy) * (alpha_HeII.x - alpha_HeII.y) - alpha_HeII.x;
        auto rHeIII_HeI =
            (y2b - y2a) * (alpha_HeIII.x - alpha_HeIII.y) +
            (nu * p.m_dec * (1 - yy) + f_lya * (1 - nu) * (1 - zz)) * alpha_HeIII.y +
            alpha_HeIII.x;
        return {
            0.0,                               // A21
            -phi.y - phi.z + rHeII_HeI * n_e,  // A22
            -phi.y + rHeIII_HeI * n_e          // A23
        };
    }

    // Create the third row of the matrix A in the chemistry equations.
    __device__ double3
    make_row3(const double2& alpha_HeIII, double n_e, double y2a, const double3& phi) {
        auto rHeIII_HeII = y2a * (alpha_HeIII.x - alpha_HeIII.y) - alpha_HeIII.x;
        return {
            0.0,               // A31
            phi.z,             // A32
            rHeIII_HeII * n_e  // A33
        };
    }

    __device__ cuda::std::array<double, 4> optical_depth_ratios(
        const double3& ndens, const parameters& p
    ) {
        // Optical depths normalized by dr
        auto tau_H_at_HeI = ndens.x * p.sigma_H_at_HeI;
        auto tau_HeI_at_HeI = ndens.y * p.sigma_HeI_at_HeI;

        auto tau_H_at_HeLya = ndens.x * p.sigma_H_at_HeLya;
        auto tau_He_at_HeLya = ndens.y * p.sigma_HeI_at_HeLya;

        auto tau_H_at_HeII = ndens.x * p.sigma_H_at_HeII;
        auto tau_HeI_at_HeII = ndens.y * p.sigma_HeI_at_HeII;
        auto tau_HeII_at_HeII = ndens.z * p.sigma_HeII_at_HeII;

        return {
            tau_H_at_HeI / (tau_H_at_HeI + tau_HeI_at_HeI),       // yy
            tau_H_at_HeLya / (tau_H_at_HeLya + tau_He_at_HeLya),  // zz
            tau_HeII_at_HeII /
                (tau_H_at_HeII + tau_HeI_at_HeII + tau_HeII_at_HeII),  // y2a
            tau_HeI_at_HeII /
                (tau_H_at_HeII + tau_HeI_at_HeII + tau_HeII_at_HeII)  // y2b
        };
    }

}  // namespace

namespace asora {

    /* double3 map
     *
     * x -> HI
     * y -> HeI
     * z -> HeII
     */

    // Numerically stable version of (exp(lambda) - 1) / lambda
    __device__ double expm1x(double lambda) {
        constexpr double exp_tol = 1e-50;
        if (abs(lambda) < exp_tol) return 1.0 + lambda / 2.0;
        return std::expm1(lambda) / lambda;
    }

    __device__ cuda::std::array<double3, 2> friedrich(
        double dt, double temp, double n_e, const double3& xh, const double3& phion,
        [[maybe_unused]] const double3& pheat, const double3& ndens,
        [[maybe_unused]] double clumping, const parameters& p
    ) {
        constexpr double ev2K = 1.0 / 8.617e-05;
        constexpr double etHI = 13.598;    // eV
        constexpr double etHeI = 24.587;   // eV
        constexpr double etHeII = 54.416;  // eV
        constexpr double tempHI = etHI * ev2K;
        constexpr double tempHeI = etHeI * ev2K;
        constexpr double tempHeII = etHeII * ev2K;

        // Recombination rate of HII (Eq. 2.12 and 2.13)
        auto alpha_HII = recombination_rates(temp, tempHI, 1.269e-13, 2.753e-14);

        // Recombination rate of HeII (Eq. 2.14-17)
        double2 alpha_HeII = alpha_HII;
        if (temp >= 9.0e3) {
            auto dielec = 1.9e-3 * std::pow(temp, -1.5) * std::exp(-4.7e5 / temp) *
                          (1.0 + 0.3 * std::exp(-9.4e4 / temp));
            auto lambda = 2.0 * tempHeI / temp;
            alpha_HeII = {
                3.000e-14 * std::pow(lambda, 0.654) + dielec,
                1.260e-14 * std::pow(lambda, 0.750) + dielec
            };
        }

        // Recombination rate of HeIII (Eq. 2.18-20) [confirmed by Garrelt (13.10.24)]
        auto alpha_HeIII = recombination_rates(temp, tempHeII, 2.538e-13, 5.506e-14);
        auto beta_HeIII = 3.4e-13 * std::pow(temp / 1.0e4, -0.6);

        // Two photons emission from recombination of HeIII
        auto nu = 0.285 * std::pow(temp / 1.0e4, 0.119);

        // Clip f_lya based on neutral fraction
        auto f_lya = min(max(10.0 * xh.x, p.f_lya_range.first), p.f_lya_range.second);

        // Ratios between optical depths
        auto&& [yy, zz, y2a, y2b] = optical_depth_ratios(ndens, p);

        // Collisional ionization (Eq. 2.21-23)
        constexpr double colHI = 1.3e-8 * 0.83 * 1.0 / (etHI * etHI);
        constexpr double colHeI = 1.3e-8 * 0.63 * 2.0 / (etHeI * etHeI);
        constexpr double colHeII = 1.3e-8 * 1.30 * 1.0 / (etHeII * etHeII);
        auto sqrtT = std::sqrt(temp);
        double3 col{
            colHI * sqrtT * std::exp(-tempHI / temp),     // cHI
            colHeI * sqrtT * std::exp(-tempHeI / temp),   // cHeI
            colHeII * sqrtT * std::exp(-tempHeII / temp)  // cHeII
        };

        // Photo-ionization rates (Eq. 2.27-29)
        constexpr double phi_tol = 1e-200;
        double3 phi{
            max(phion.x + col.x * n_e, phi_tol),  // uHI
            max(phion.y + col.y * n_e, phi_tol),  // uHeI
            max(phion.z + col.z * n_e, phi_tol)   // uHeII
        };

        // Matrix elements with recombination rates (Eq. 2.30-35)
        auto A1 = make_row1(
            alpha_HII, alpha_HeII, alpha_HeIII, beta_HeIII, nu, n_e, yy, y2a, y2b, zz,
            phi, p, f_lya
        );
        auto A2 = make_row2(
            alpha_HeII, alpha_HeIII, nu, n_e, yy, y2a, y2b, zz, phi, p, f_lya
        );
        auto A3 = make_row3(alpha_HeIII, n_e, y2a, phi);

        // Some useful coefficients
        auto B = A3.z - A2.y;
        auto S = std::sqrt(B * B + 4.0 * A3.y * A2.z);
        auto K = 1.0 / (A2.z * A3.y - A3.z * A2.y);
        auto R = 2.0 * A3.y * (A3.z * phi.y * K - xh.y);
        auto T = -A3.y * phi.y * K - xh.z;

        // Eigen-values
        double3 lambda{A1.x, (A3.z + A2.y - S) / 2.0, (A3.z + A2.y + S) / 2.0};

        // Particular solution
        double3 psol{
            -(phi.x + (A3.z * A1.y - A3.y * A1.z) * phi.y * K) / A1.x,  // p1
            A3.z * phi.y * K,                                           // p2
            -A3.y * phi.y * K                                           // p3
        };

        // Useful eigen vectors components
        double3 x1{1.0, 0.0, 0.0};
        double3 x2{
            (-2.0 * A3.y * A1.z + A1.y * (B + S)) / (2.0 * A3.y * (A1.x - lambda.y)),
            (-A3.z + A2.y - S) / (2.0 * A3.y), 1.0
        };
        double3 x3{
            (-2.0 * A3.y * A1.z + A1.y * (B - S)) / (2.0 * A3.y * (A1.x - lambda.z)),
            (-A3.z + A2.y + S) / (2.0 * A3.y), 1.0
        };

        // Boundary condition coefficients
        double2 raast = {R + (B - S) * T, R + (B + S) * T};
        double3 coeff{
            xh.x - psol.x - (raast.x * x2.x - raast.y * x3.x) / (2.0 * S),  // c1
            raast.x / (2.0 * S),                                            // c2
            -raast.y / (2.0 * S)                                            // c3
        };

        lambda.x *= dt;
        lambda.y *= dt;
        lambda.z *= dt;

        // Analytical solution
        double3 ws{
            coeff.x * std::exp(lambda.x),  // HII
            coeff.y * std::exp(lambda.y),  // HeI
            coeff.z * std::exp(lambda.z)   // HeII
        };
        auto res = compose_solution(psol, ws, x1, x2, x3);

        // Time average solution
        double3 ws_av{
            coeff.x * expm1x(lambda.x),  // HII
            coeff.y * expm1x(lambda.y),  // HeI
            coeff.z * expm1x(lambda.z)   // HeII
        };
        auto res_av = compose_solution(psol, ws_av, x1, x2, x3);

        return {res, res_av};
    }

    __device__ cuda::std::array<double3, 2> do_chemistry(
        double dt, double Hz, double temp, double ndens, const double3& xh,
        double3 xh_av, const double3& phi_ion, const double3& phi_heat, double clump,
        const cooling_tables& tables, const linspace<double>& logtemp,
        const parameters& p, size_t max_iterations
    ) {
        auto heating = phi_heat.x + phi_heat.y + phi_heat.z;

        // At each loop iteration, the counter is decreased until 0 unless convergence
        // is reached before.
        double3 xh_new, xh_av_new;
        double temp_av = temp;
        ++max_iterations;  // to match fortran code
        while (max_iterations > 0) {
            double3 ndens_species{
                ndens * p.abu_h * (1 - xh_av.x),             // HI
                ndens * p.abu_he * (1 - xh_av.y - xh_av.z),  // HeI
                ndens * p.abu_he * xh_av.y                   // HeII
            };

            // Determine electron density.
            auto ndens_elec =
                electron_density(ndens, xh_av, p.abu_h, p.abu_he, p.abu_c);

            // Update ionizattion fractions according to the chemistry equations.
            cuda::std::tie(xh_new, xh_av_new) = friedrich(
                dt, temp, ndens_elec, xh, phi_ion, phi_heat, ndens_species, clump, p
            );

            // Update electron density based on the fractions for thermal evolution.
            ndens_elec = electron_density(ndens, xh_av_new, p.abu_h, p.abu_he, p.abu_c);

            // Update temperature.
            auto&& [temp_new, temp_av_new] = thermal(
                dt, temp, ndens_elec, ndens, heating, Hz,
                {xh_av_new.x, xh_av_new.y, xh_av_new.z}, tables, logtemp, p
            );

            if (check_convergence_local(xh_av_new.x, xh_av.x) &&  // HI
                check_convergence_local(xh_av_new.y, xh_av.y) &&  // HeI
                check_convergence_local(xh_av_new.z, xh_av.z) &&  // HeII
                abs(temp_av_new - temp_av) / temp_av_new < minimum_fractional_change)
                max_iterations = 0;
            else
                --max_iterations;

            // Update xh_av for the next iteration.
            xh_av = xh_av_new;
            temp = temp_new;
            temp_av = temp_av_new;
        }

        // Return  xHII, HeI, HeII, xHII_av, HeI_av, HeII_av */
        return {xh_new, xh_av_new};
    }

    // Global pass kernel
    __global__ void evolve0D_gpu(
        double dt, double Hz, double* __restrict__ temp, double* __restrict__ ndens,
        double3ptr xh, double3ptr xh_av, double3ptr xh_int, double3ptr phi_ion,
        double3ptr phi_heat, const double* __restrict__ clump, cooling_tables tables,
        linspace<double> logtemp, bool* conv_flag, parameters p, size_t size
    ) {
        auto idx = threadIdx.x + blockDim.x * blockIdx.x;

        // Thread can process more than one cell.
        while (idx < size) {
            // Get average fraction value as a reference: it will be updated later.
            double3 xh_p = {xh.x[idx], xh.y[idx], xh.z[idx]};
            double3 xh_av_p = {xh_av.x[idx], xh_av.y[idx], xh_av.z[idx]};

            auto&& [xh_int_new, xh_av_new] = do_chemistry(
                dt, Hz, temp[idx], ndens[idx], xh_p, xh_av_p,
                {phi_ion.x[idx], phi_ion.y[idx], phi_ion.z[idx]},
                {phi_heat.x[idx], phi_heat.y[idx], phi_heat.z[idx]}, clump[idx], tables,
                logtemp, p
            );

            conv_flag[idx] = check_convergence_global(xh_av_new.x, xh_av_p.x) &&
                             check_convergence_global(xh_av_new.y, xh_av_p.y) &&
                             check_convergence_global(xh_av_new.z, xh_av_p.z);

            // Update the results in global memory.
            xh_int.x[idx] = xh_int_new.x;
            xh_int.y[idx] = xh_int_new.y;
            xh_int.z[idx] = xh_int_new.z;
            xh_av.x[idx] = xh_av_new.x;
            xh_av.y[idx] = xh_av_new.y;
            xh_av.z[idx] = xh_av_new.z;

            idx += blockDim.x * gridDim.x;
        }
    }

    device_buffer allocate_and_copy(size_t n_cells, const double* src) {
        auto buf = device_buffer(n_cells * sizeof(double));
        buf.copyFromHost(src);
        return buf;
    }

    // Host function to call global_pass
    size_t global_pass(
        double dt, double Hz, const double* __restrict__ temp,
        const double* __restrict__ ndens, double3ptr xh, double3ptr xh_av,
        double3ptr xh_int, const double3ptr& phi_ion, const double3ptr& phi_heat,
        const double* __restrict__ clump, const linspace<double>& logtemp,
        const parameters& p, size_t n_cells, size_t block_size
    ) {
        // Initialize and copy const data.
        for (auto&& [tag, data] : {
                 std::pair{buffer_tag::number_density, ndens},
                 std::pair{buffer_tag::temperature, temp},
                 std::pair{buffer_tag::clumping_factor, clump},
                 std::pair{buffer_tag::fraction_HII, xh_av.cx()},
                 std::pair{buffer_tag::fraction_HeII, xh_av.cy()},
                 std::pair{buffer_tag::fraction_HeIII, xh_av.cz()},
                 std::pair{buffer_tag::photo_ionization_HI, phi_ion.cx()},
                 std::pair{buffer_tag::photo_ionization_HeI, phi_ion.cy()},
                 std::pair{buffer_tag::photo_ionization_HeII, phi_ion.cz()},
                 std::pair{buffer_tag::photo_heating_HI, phi_heat.cx()},
                 std::pair{buffer_tag::photo_heating_HeI, phi_heat.cy()},
                 std::pair{buffer_tag::photo_heating_HeII, phi_heat.cz()},
             }) {
            device::ensure_transfer<double>(tag, data, n_cells);
        }

        // Initialize and copy non-const data.
        auto xHII_buf = allocate_and_copy(n_cells, xh.x);
        auto xHII_int_buf = allocate_and_copy(n_cells, xh_int.x);
        auto xHeII_buf = allocate_and_copy(n_cells, xh.y);
        auto xHeII_int_buf = allocate_and_copy(n_cells, xh_int.y);
        auto xHeIII_buf = allocate_and_copy(n_cells, xh.z);
        auto xHeIII_int_buf = allocate_and_copy(n_cells, xh_int.z);

        device_buffer conv_flag(n_cells);
        auto conv_flag_d = conv_flag.view<bool>().data();

        auto temp_d = get_data_view<double>(buffer_tag::temperature);
        auto ndens_d = get_data_view<double>(buffer_tag::number_density);
        auto clump_d = get_data_view<double>(buffer_tag::clumping_factor);

        double3ptr xh_d = {
            xHII_buf.data<double>(), xHeII_buf.data<double>(), xHeIII_buf.data<double>()
        };
        double3ptr xh_av_d = {
            get_data_view<double>(buffer_tag::fraction_HII),
            get_data_view<double>(buffer_tag::fraction_HeII),
            get_data_view<double>(buffer_tag::fraction_HeIII)
        };
        double3ptr xh_int_d = {
            xHII_int_buf.data<double>(), xHeII_int_buf.data<double>(),
            xHeIII_int_buf.data<double>()
        };
        double3ptr phi_ion_d = {
            get_data_view<double>(buffer_tag::photo_ionization_HI),
            get_data_view<double>(buffer_tag::photo_ionization_HeI),
            get_data_view<double>(buffer_tag::photo_ionization_HeII)
        };
        double3ptr phi_heat_d = {
            get_data_view<double>(buffer_tag::photo_heating_HI),
            get_data_view<double>(buffer_tag::photo_heating_HeI),
            get_data_view<double>(buffer_tag::photo_heating_HeII)
        };

        cooling_tables tables{};
        if (!p.cosmo_only) {
            tables = cooling_tables{
                get_data_view<double>(buffer_tag::cooling_HI_table),
                get_data_view<double>(buffer_tag::cooling_HII_table),
                get_data_view<double>(buffer_tag::cooling_HeI_table),
                get_data_view<double>(buffer_tag::cooling_HeII_table),
                get_data_view<double>(buffer_tag::cooling_HeIII_table)
            };
        }

        // Launch kernel, divide by 2 so that threads do more work.
        size_t grid_size = std::ceil(static_cast<float>(n_cells) / block_size / 2);
        evolve0D_gpu<<<grid_size, block_size>>>(
            dt, Hz, temp_d, ndens_d, xh_d, xh_av_d, xh_int_d, phi_ion_d, phi_heat_d,
            clump_d, tables, logtemp, conv_flag_d, p, n_cells
        );

        // Check for errors.
        safe_cuda(cudaPeekAtLastError());

        // Reduction kernel to count non-zero elements.
        auto convergence =
            thrust::count(thrust::device, conv_flag_d, conv_flag_d + n_cells, true);

        device::get(buffer_tag::fraction_HII).copyToHost(xh_av.x);
        device::get(buffer_tag::fraction_HeII).copyToHost(xh_av.y);
        device::get(buffer_tag::fraction_HeIII).copyToHost(xh_av.z);
        xHII_int_buf.copyToHost(xh_int.x);
        xHeII_int_buf.copyToHost(xh_int.y);
        xHeIII_int_buf.copyToHost(xh_int.z);

        return convergence;
    }

}  // namespace asora
