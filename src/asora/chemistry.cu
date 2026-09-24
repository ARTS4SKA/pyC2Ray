#include "chemistry.h"

#include "memory.h"
#include "utils.cuh"

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <iostream>

namespace {

    // Convergence criteria constants.
    constexpr double minimum_fractional_change = 1.0e-3;
    constexpr double minimum_fraction_of_atoms = 1.0e-8;
    constexpr double epsilon = 1e-14;

    // Compute the main components of the chemistry solution: the equilibrium fraction
    // and characteristic rate, used like xh(t) = eqxh + (xh(0) - eqxh) * exp(-deltht)
    __device__ cuda::std::array<double, 2> compute_doric_components(
        double dt, double xh, double phi, double ndens, double abu_c, double col_ion,
        double rec_coeff
    ) {
        // NOTE: fma(a, b, c) = a * b + c is the fused multiply-add operation.
        double rhe = ndens * (xh + abu_c);
        auto aih0 = fma(rhe, col_ion, phi);
        auto delth = fma(rhe, rec_coeff, aih0);

        auto eqxh = aih0 / delth;
        auto deltht = delth * dt;

        return {eqxh, deltht};
    }

    __device__ bool check_convergence_local(double new_value, double old_value) {
        bool cond1 =
            abs(new_value - old_value) / (1 - new_value) < minimum_fractional_change;
        bool cond2 = 1 - new_value < minimum_fraction_of_atoms;
        // cond3 = (temp - temp_prev) / temp < minimum_fractional_change is not
        // needed because temperature is not updated in the loop.

        return cond1 || cond2;
    }

    __device__ bool check_convergence_global(double new_value, double old_value) {
        auto cond1 = abs(new_value - old_value) > minimum_fractional_change;
        auto cond2 =
            abs((new_value - old_value) / (1 - old_value)) > minimum_fractional_change;
        auto cond3 = (1 - old_value) > minimum_fraction_of_atoms;

        return cond1 && cond2 && cond3;
    }

    // Device function for chemistry calculations
    __device__ cuda::std::array<double, 2> do_chemistry(
        double xh, double xh_av, double temp, double ndens, double phi_ion,
        double clump, double dt, double bh00, double albpow, double colh0,
        double temph0, double abu_c, size_t max_iterations = 400
    ) {
        // These factors are constant for a given cell and can be computed once.
        double col_ion = colh0 * sqrt(temp) * exp(-temph0 / temp);
        double rec_coeff = clump * bh00 * pow(temp / 1e4, albpow);

        // At each loop iteration, the counter is decreased until 0 unless convergence
        // is reached before.
        double eqxh, deltht;
        while (max_iterations > 0) {
            // Compute equilibrium fraction and deltht(?) needed for xh_av and xh_int
            cuda::std::tie(eqxh, deltht) = compute_doric_components(
                dt, xh_av, phi_ion, ndens, abu_c, col_ion, rec_coeff
            );

            // Compute the average fraction.
            auto avg = (deltht < 1.0e-8) ? 1.0 : (1.0 - exp(-deltht)) / deltht;
            double xh_av_new = max(fma(xh - eqxh, avg, eqxh), epsilon);

            if (check_convergence_local(xh_av_new, xh_av))
                max_iterations = 0;
            else
                --max_iterations;

            // Update xh_av for the next iteration.
            xh_av = xh_av_new;
        }

        // xh_int is not needed for convergence.
        auto xh_int = max(fma(xh - eqxh, exp(-deltht), eqxh), epsilon);

        return {xh_int, xh_av};
    }

    // Global pass kernel
    __global__ void evolve0D_gpu(
        double* __restrict__ xh, double* __restrict__ xh_av,
        double* __restrict__ xh_int, double* __restrict__ temp,
        const double* __restrict__ ndens, const double* __restrict__ phi_ion,
        const double* __restrict__ clump, bool* conv_flag, double dt, double bh00,
        double albpow, double colh0, double temph0, double abu_c, size_t size
    ) {
        auto idx = threadIdx.x + blockDim.x * blockIdx.x;

        // Thread can process more than one cell.
        while (idx < size) {
            // Get average fraction value as a reference: it will be updated later.
            auto& xh_av_p = xh_av[idx];

            auto&& [xh_int_new, xh_av_new] = do_chemistry(
                xh[idx], xh_av_p, temp[idx], ndens[idx], phi_ion[idx], clump[idx], dt,
                bh00, albpow, colh0, temph0, abu_c
            );

            conv_flag[idx] = check_convergence_global(xh_av_new, xh_av_p);
            xh_int[idx] = xh_int_new;
            xh_av_p = xh_av_new;

            idx += blockDim.x * gridDim.x;
        }
    }

}  // namespace

namespace asora {

    // Host function to call global_pass
    size_t global_pass(
        double* xh, double* xh_avg, double* xh_int, const double* temp,
        const double* phi_ion, const double* clump, double dt, double bh00,
        double albpow, double colh0, double temph0, double abu_c, size_t n_cells,
        size_t block_size
    ) {
        const auto& ndens = device::resident().number_density;
        if (!ndens)
            throw std::runtime_error(
                "number density array must be allocated on the device before calling "
                "do_all_sources_gpu"
            );
        // Allocate (if necessary) and copy the average ionized fraction array to the
        // device. This array is also used by raytracing.
        // TODO: make phion_HI and fraction_HII_avg resident data
        auto fraction_HII = device::scratch<double>(n_cells);
        auto fraction_HII_avg = device::scratch<double>(n_cells);
        auto fraction_HII_int = device::scratch<double>(n_cells);

        fraction_HII.copy_from_host(xh);
        fraction_HII_avg.copy_from_host(xh_avg);
        fraction_HII_int.copy_from_host(xh_int);

        auto phion_HI = device::scratch<double>(n_cells);
        auto temp_d = device::scratch<double>(n_cells);
        auto clump_d = device::scratch<double>(n_cells);

        phion_HI.copy_from_host(phi_ion);
        temp_d.copy_from_host(temp);
        clump_d.copy_from_host(clump);

        auto conv_flag = device::scratch<bool>(n_cells);

        // Launch kernel, divide by 2 so that threads do more work.
        size_t grid_size = std::ceil(static_cast<float>(n_cells) / block_size / 2);
        evolve0D_gpu<<<grid_size, block_size>>>(
            fraction_HII.data(), fraction_HII_avg.data(), fraction_HII_int.data(),
            temp_d.data(), ndens.data(), phion_HI.data(), clump_d.data(),
            conv_flag.data(), dt, bh00, albpow, colh0, temph0, abu_c, n_cells
        );

        // Check for errors.
        safe_cuda(cudaPeekAtLastError());

        // Reduction kernel to count non-zero elements.
        // TODO: set thrust to the correct stream
        auto convergence = thrust::count(
            thrust::device, conv_flag.data(), conv_flag.data() + n_cells, true
        );

        fraction_HII_avg.copy_to_host(xh_avg);
        fraction_HII_int.copy_to_host(xh_int);
        return convergence;
    }

}  // namespace asora
