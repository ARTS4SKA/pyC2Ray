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

    [[deprecated("Must be replaced with correct cooling rate tables.")]]
    __host__ __device__ double cooling_rate(
        double temp, double ndens_atom, double ndens_elec
    ) {
        return ndens_atom * ndens_elec * temp;
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
        double dt, double temp_start, double ndens_elec, double ndens_atom,
        double heating, double Hz, const parameters& p, double min_temp,
        size_t max_iterations
    ) {
        // Thermal process only if temperature > min_temp
        if (temp_start <= min_temp) return {temp_start, temp_start};

        // Find initial internal energy
        auto u0 = get_energy(temp_start, ndens_atom + ndens_elec, p.gamma);
        auto umin = get_energy(min_temp, ndens_atom + ndens_elec, p.gamma);
        auto ui = u0;
        auto ui_av = u0;

        double tot_time = 0.0;
        auto temp_end = temp_start;

        // Exit conditions: reached dt (with tolerance like Fortran)
        while (max_iterations > 0 && tot_time < dt * (1.0 - 1e-6)) {
            --max_iterations;

            auto rate = heating - cooling_rate(temp_end, ndens_atom, ndens_elec) -
                        cosmo_cooling_rate(ui, Hz);

            // Thermal time scale. Limit energy change to fraction relative_denergy
            // Don't integrate longer than remaining time.
            auto subdt =
                min(p.relative_denergy * ui / max(1e-50, abs(rate)), dt - tot_time);

            ui += rate * subdt;
            ui_av += rate * subdt * subdt / dt;

            temp_end = get_temperature(ui, ndens_atom + ndens_elec, p.gamma);
            tot_time += subdt;

            // Enforce minimum temperature
            if (ui < umin) {
                ui = umin;
                temp_end = min_temp;
                break;
            }
        }

        // Final temperature
        auto temp_avg = get_temperature(ui_av, ndens_atom + ndens_elec, p.gamma);
        return {temp_end, temp_avg};
    }

}  // namespace asora

namespace {

    // Convergence criteria constants.
    constexpr double minimum_fractional_change = 1.0e-3;
    constexpr double minimum_fraction_of_atoms = 1.0e-8;

    __device__ double recombination_rate(
        double temp, const cuda::std::array<double, 5>& fit_p
    ) {
        // from c1 * pow(c2 / temp, c3) / pow(1.0 + pow(c4 / temp, c5), c6)
        // with the following conversion:
        // a = c1 * pow(c2, c3)
        // b = -c3 + c5 * c6
        // c = c5
        // d = pow(c4, c5)
        // e = c6
        auto&& [a, b, c, d, e] = fit_p;
        return a * std::pow(temp, b) / std::pow(std::pow(temp, c) + d, e);
    }

    __device__ double3
    compose_solution(const double3& weights, const double2& x2, const double2& x3) {
        return {
            weights.x + x2.x * weights.y + x3.x * weights.z,  // HII
            x2.y * weights.y + x3.y * weights.z,              // HeI
            weights.y + weights.z                             // HeII
        };
    }

    __device__ bool check_convergence_local(
        const double3& xh_new, const double3& xh_old, double temp_new, double temp_old
    ) {
        bool cond1 =
            abs(xh_new.x - xh_old.x) / (1 - xh_new.x) < minimum_fractional_change;
        bool cond2 = 1 - xh_new.x < minimum_fraction_of_atoms;
        bool cond3 = abs(temp_new - temp_old) / temp_new < minimum_fractional_change;
        return (cond1 || cond2) && cond3;
    }

    __device__ bool check_convergence_global(double new_value, double old_value) {
        auto cond1 = abs(new_value - old_value) > minimum_fractional_change;
        auto cond2 =
            abs((new_value - old_value) / (1 - old_value)) > minimum_fractional_change;
        auto cond3 = (1 - old_value) > minimum_fraction_of_atoms;

        return cond1 && cond2 && cond3;
    }

    __device__ double3 make_row1(
        const double2& alpha_HeII, const double2& alpha_HII, const double2& alpha_HeIII,
        double beta_HeIII, double nu, double n_e, double yy, double y2a, double y2b,
        double zz, const double3& phi, const parameters& p
    ) {
        auto rHII_HI = -alpha_HII.y;
        auto rHeII_HI = p.p_rec * alpha_HeII.x + yy * (alpha_HeIII.x - alpha_HeIII.y);
        auto rHeIII_HI =
            (1 - y2a - y2b) * (alpha_HeIII.x - alpha_HeIII.y) + beta_HeIII +
            (nu * (p.l_dec - p.m_dec + p.m_dec * yy) + (1 - nu) * p.f_lya * zz) *
                alpha_HeIII.y;
        auto mul = (p.abu_he / p.abu_h) * n_e;
        return {
            -phi.x + rHII_HI,  // A11
            mul * rHeII_HI,    // A12
            mul * rHeIII_HI    // A13
        };
    }

    __device__ double3 make_row2(
        const double2& alpha_HeII, const double2& alpha_HeIII, const double2& alpha_HII,
        double nu, double n_e, double yy, double y2a, double y2b, double zz,
        const double3& phi, const parameters& p
    ) {
        auto rHeII_HeI = (1 - yy) * (alpha_HII.x - alpha_HII.y) - alpha_HeII.x;
        auto rHeIII_HeI =
            (y2b - y2a) * (alpha_HeIII.x - alpha_HeIII.y) +
            (nu * p.m_dec * (1 - yy) + p.f_lya * (1 - nu) * (1 - zz)) * alpha_HeIII.y +
            alpha_HeIII.x;
        return {
            0.0,                               // A21
            -phi.y - phi.z + rHeII_HeI * n_e,  // A22
            -phi.y + rHeIII_HeI * n_e          // A23
        };
    }

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
        auto tau_HeI_at_ion_freq = ndens.y * p.sigma_HeI_at_ion_freq;

        auto tau_H_at_HeLya = ndens.x * p.sigma_H_at_HeLya;
        auto tau_He_at_HeLya = ndens.y * p.sigma_HeI_at_HeLya;

        auto tau_H_at_HeII = ndens.x * p.sigma_H_at_HeII;
        auto tau_HeI_at_HeII = ndens.y * p.sigma_HeI_at_HeII;
        auto tau_HeII_at_ion_freq = ndens.z * p.sigma_HeII_at_ion_freq;

        return {
            tau_H_at_HeI / (tau_H_at_HeI + tau_HeI_at_ion_freq),  // yy
            tau_H_at_HeLya / (tau_H_at_HeLya + tau_He_at_HeLya),  // zz
            tau_HeII_at_ion_freq /
                (tau_H_at_HeII + tau_HeI_at_HeII + tau_HeII_at_ion_freq),  // y2a
            tau_HeI_at_HeII /
                (tau_H_at_HeII + tau_HeI_at_HeII + tau_HeII_at_ion_freq)  // y2b
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

    __device__ cuda::std::array<double3, 2> friedrich(
        double dt, [[maybe_unused]] double dr, double temp, double n_e,
        const double3& xh, const double3& phion, [[maybe_unused]] const double3& pheat,
        const double3& ndens, [[maybe_unused]] double clumping, const parameters& p
    ) {
        // Recombination rate of HII (Eq. 2.12 and 2.13)
        // NOTE: 1.5 vs 1.503 and 0.47 vs 0.407 are suspicious, but they match the paper
        double2 alpha_HII = {
            // alphaA {1.269e-13, 315608.0, 1.503, 604613.0, 0.470, 1.923}
            recombination_rate(temp, {2.33712e-05, -0.599190, 0.470, 521.548, 1.923}),
            // alphaB {2.753e-14, 315608.0, 1.500, 115185.0, 0.407, 2.242}
            recombination_rate(temp, {4.88122e-06, -0.587506, 0.407, 114.812, 2.242})
        };

        // Recombination rate of HeII (Eq. 2.14-17)
        double2 alpha_HeII;
        // double alphaB_HeII;
        if (temp < 9.0e3) {
            alpha_HeII = {
                // alphaA {1.269e-13, 570662.0, 1.503, 1093222.0, 0.470, 1.923}
                recombination_rate(
                    temp, {5.69245e-05, -0.599190, 0.470, 688.958, 1.923}
                ),
                // alphaB temp, {2.753e-14, 570662.0, 1.500, 208271.0, 0.407, 2.242}
                /* this element is unused, but it would be:
                recombination_rate(temp, {1.18679e-05, -0.587506, 0.407,
                146.111, 2.242})
                */
                0.0
            };
        } else {
            auto alpha = 1.9e-3 * std::pow(temp, -1.5) * std::exp(-4.7e5 / temp) *
                         (1.0 + 0.3 * std::exp(-9.4e4 / temp));
            alpha_HeII = {
                3.0e-14 * std::pow(570662.0 / temp, 0.654) + alpha,
                /* this element is unused, but it would be:
                1.26e-14 * std::pow(570662.0 / temp, 0.75) + alpha
                */
                0.0
            };
        }

        // Recombination rate of HeIII (Eq. 2.18-20) [confirmed by Garrelt (13.10.24)]
        double2 alpha_HeIII = {
            // alphaA_HeIII {2.538e-13, 1262990.0, 1.503, 2419521.0, 1.923, 1.923}
            recombination_rate(temp, {0.000375747, 2.19493, 1.923, 1.88761e+12, 1.923}),
            // alphaB_HeIII {5.506e-14, 1262990.0, 1.500, 460945.0, 0.407, 2.242}
            recombination_rate(temp, {7.81513e-05, -0.587506, 0.407, 201.886, 2.242})
        };
        auto beta_HeIII = 8.54e-11 * std::pow(temp, -0.6);

        // Two photons emission from recombination of HeIII
        auto nu = 0.285 * std::pow(temp / 1.0e4, 0.119);

        // Ratios between optical depths
        auto&& [yy, zz, y2a, y2b] = optical_depth_ratios(ndens, p);

        // Collisional ionization (Eq. 2.21-23)
        auto sqrtT = std::sqrt(temp);
        double3 col{
            5.835e-11 * sqrtT * std::exp(-157804.0 / temp),  // cHI
            2.710e-11 * sqrtT * std::exp(-285331.0 / temp),  // cHeI
            5.707e-12 * sqrtT * std::exp(-631495.0 / temp)   // cHeII
        };

        // Photo-ionization rates (Eq. 2.27-29)
        double3 phi{
            phion.x + col.x * n_e,  // uHI
            phion.y + col.y * n_e,  // uHeI
            phion.z + col.z * n_e   // uHeII
        };

        // Matrix elements with recombination rates (Eq. 2.30-35)
        auto A1 = make_row1(
            alpha_HeII, alpha_HII, alpha_HeIII, beta_HeIII, nu, n_e, yy, y2a, y2b, zz,
            phi, p
        );
        auto A2 = make_row2(
            alpha_HeII, alpha_HeIII, alpha_HII, nu, n_e, yy, y2a, y2b, zz, phi, p
        );
        auto A3 = make_row3(alpha_HeIII, n_e, y2a, phi);

        // Some useful coefficients
        auto S = std::sqrt(
            A3.z * A3.z - 2.0 * A3.z * A2.y + A2.y * A2.y + 4.0 * A3.y * A2.z
        );
        auto K = 1.0 / (A2.z * A3.y - A3.z * A2.y);
        auto R = 2.0 * A2.z * (A3.z * phi.y * K - xh.y);
        auto T = -A3.y * phi.y * K - xh.z;

        // Eigen-values
        double3 lambda{A1.x, (A3.z - A2.y - S) / 2.0, (A3.z - A2.y + S) / 2.0};

        // Particular solution
        double3 psol{
            -(phi.x + (A3.z * A1.y - A3.y * A1.z) * phi.y * K) / A1.x,  // p1
            A3.z * phi.y * K,                                           // p2
            -A3.y * phi.y * K                                           // p3
        };

        // Useful eigen vectors components
        // double2 x1{1.0, 0.0};
        double2 x2{
            (-2.0 * A3.y * A1.z + A1.y * (A3.z - A2.y + S)) /
                (2.0 * A3.y * (A1.x - lambda.y)),
            (-A3.z + A2.y - S) / (2.0 * A3.y)
        };

        double2 x3{
            (-2.0 * A3.y * A1.z + A1.y * (A3.z - A2.y - S)) /
                (2.0 * A3.y * (A1.x - lambda.z)),
            (-A3.z + A2.y + S) / (2.0 * A3.y)
        };

        // Boundary condition coefficients
        double3 coeff{
            xh.x - psol.x + T * (x3.x + x2.x) / 2 +
                (R + (A3.z - A2.y) * T) * (x3.x - x2.x) / (2.0 * S),  // c1
            (R + (A3.z - A2.y - S) * T) / (2.0 * S),                  // c2
            -(R + (A3.z - A2.y + S) * T) / (2.0 * S)                  // c3
        };

        // Analytical solution
        double3 ws{
            coeff.x * std::exp(lambda.x * dt),  // HII
            coeff.y * std::exp(lambda.y * dt),  // HeI
            coeff.z * std::exp(lambda.z * dt)   // HeII
        };
        auto res = compose_solution(ws, x2, x3);

        // Time average solution
        double3 ws_av{
            coeff.x / (lambda.x * dt) * std::expm1(lambda.x * dt),  // HII
            coeff.y / (lambda.y * dt) * std::expm1(lambda.y * dt),  // HeI
            coeff.z / (lambda.z * dt) * std::expm1(lambda.z * dt)   // HeII
        };
        auto res_av = compose_solution(ws_av, x2, x3);

        return {
            {{res.x + psol.x, res.y + psol.y, res.z + psol.z},
             {res_av.x, res_av.y, res_av.z}}
        };
    }

    __device__ cuda::std::array<double3, 2> do_chemistry(
        double dt, double dr, double Hz, double temp_start, double ndens,
        const double3& xh, double3 xh_av, const double3& phi_ion,
        const double3& phi_heat, double clump, const parameters& p,
        size_t max_iterations
    ) {
        auto temp_end = temp_start;
        auto heating = phi_heat.x + phi_heat.y + phi_heat.z;

        // At each loop iteration, the counter is decreased until 0 unless convergence
        // is reached before.
        double3 xh_new, xh_av_new;
        double temp_prev = temp_end;
        while (max_iterations > 0) {
            double3 ndens_species{
                ndens * p.abu_h * (1 - xh_av.x),             // HI
                ndens * p.abu_he * (1 - xh_av.y - xh_av.z),  // HeI
                ndens * p.abu_he * xh_av.z                   // HeII
            };

            auto ndens_elec =
                electron_density(ndens, xh_av, p.abu_h, p.abu_he, p.abu_c);

            cuda::std::tie(xh_new, xh_av_new) = friedrich(
                dt, dr, temp_end, ndens_elec, xh, phi_ion, phi_heat, ndens_species,
                clump, p
            );

            // Update average solution value
            ndens_elec = electron_density(ndens, xh_av_new, p.abu_h, p.abu_he, p.abu_c);

            // Update temperature
            cuda::std::tie(temp_end, cuda::std::ignore) =
                thermal(dt, temp_end, ndens_elec, ndens, heating, Hz, p);

            if (check_convergence_local(xh_av_new, xh_av, temp_end, temp_prev))
                max_iterations = 0;
            else
                --max_iterations;

            // Update xh_av for the next iteration.
            xh_av = xh_av_new;
            temp_prev = temp_end;
        }

        // Return  xHII, HeI, HeII, xHII_av, HeI_av, HeII_av */
        return {xh_new, xh_av_new};
    }

    // Global pass kernel
    __global__ void evolve0D_gpu(
        double dt, double dr, double Hz, double* __restrict__ temp,
        double* __restrict__ ndens, double3ptr xh, double3ptr xh_av, double3ptr xh_int,
        double3ptr phi_ion, double3ptr phi_heat, const double* __restrict__ clump,
        bool* conv_flag, parameters p, size_t size
    ) {
        auto idx = threadIdx.x + blockDim.x * blockIdx.x;

        // Thread can process more than one cell.
        while (idx < size) {
            // Get average fraction value as a reference: it will be updated later.
            auto& xHI_p = xh.x[idx];
            auto& xHeI_p = xh.y[idx];
            auto& xHeII_p = xh.z[idx];
            auto& xHI_av_p = xh_av.x[idx];
            auto& xHeI_av_p = xh_av.y[idx];
            auto& xHeII_av_p = xh_av.z[idx];

            auto&& [xh_int_new, xh_av_new] = do_chemistry(
                dt, dr, Hz, temp[idx], ndens[idx], {xHI_p, xHeI_p, xHeII_p},
                {xHI_av_p, xHeI_av_p, xHeII_av_p},
                {phi_ion.x[idx], phi_ion.y[idx], phi_ion.z[idx]},
                {phi_heat.x[idx], phi_heat.y[idx], phi_heat.z[idx]}, clump[idx], p
            );

            conv_flag[idx] = check_convergence_global(xh_av_new.x, xHI_av_p) &&
                             check_convergence_global(xh_av_new.y, xHeI_av_p) &&
                             check_convergence_global(xh_av_new.z, xHeII_av_p);

            xh_int.x[idx] = xh_int_new.x;
            xh_int.y[idx] = xh_int_new.y;
            xh_int.z[idx] = xh_int_new.z;
            xHI_av_p = xh_av_new.x;
            xHeI_av_p = xh_av_new.y;
            xHeII_av_p = xh_av_new.z;

            idx += blockDim.x * gridDim.x;
        }
    }

}  // namespace asora
