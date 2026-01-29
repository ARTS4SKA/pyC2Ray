#include "chemistry.h"

#include "memory.h"
#include "utils.cuh"

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <iostream>

namespace {

    constexpr double minimum_fractional_change = 1.0e-3;
    constexpr double minimum_fraction_of_atoms = 1.0e-8;

    // Device function for doric calculations
    __device__ cuda::std::array<double, 2> doric(
        double xh_old, double dt, double temp, double rhe, double phi, double bh00,
        double albpow, double colh0, double temph0, double clumping
    ) {
        constexpr double epsilon = 1e-14;

        auto aih0 = phi + rhe * colh0 * sqrt(temp) * exp(-temph0 / temp);
        auto delth = aih0 + rhe * clumping * bh00 * pow(temp / 1e4, albpow);

        auto deltht = delth * dt;
        auto ee = exp(-deltht);
        auto avg = (deltht < 1.0e-8) ? 1.0 : (1.0 - ee) / deltht;

        auto eqxh = aih0 / delth;
        xh_old -= eqxh;

        auto xh = max(eqxh + xh_old * ee, epsilon);
        auto xh_av = max(eqxh + xh_old * avg, epsilon);

        return {xh, xh_av};
    }

    // Device function for chemistry calculations
    __device__ cuda::std::array<double, 2> do_chemistry(
        double xh, double xh_av, double temp, double ndens, double phi_ion,
        double clump, double dt, double bh00, double albpow, double colh0,
        double temph0, double abu_c, size_t max_iterations = 400
    ) {
        size_t niter = max_iterations;

        double xh_int;
        while (niter > 0) {
            double xh_av_prev = xh_av;
            double de = ndens * (xh_av + abu_c);

            cuda::std::tie(xh_int, xh_av) =
                doric(xh, dt, temp, de, phi_ion, bh00, albpow, colh0, temph0, clump);

            // Convergence check
            bool cond1 =
                abs(xh_av - xh_av_prev) / (1 - xh_av) < minimum_fractional_change;
            bool cond2 = 1 - xh_av < minimum_fraction_of_atoms;

            if (cond1 || cond2)
                niter = 0;
            else
                --niter;
        }
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

        while (idx < size) {
            auto& xh_av_prev = xh_av[idx];

            auto&& [xh_int_new, xh_av_new] = do_chemistry(
                xh[idx], xh_av_prev, temp[idx], ndens[idx], phi_ion[idx], clump[idx],
                dt, bh00, albpow, colh0, temph0, abu_c
            );

            auto cond1 = abs(xh_av_new - xh_av_prev) > minimum_fractional_change;
            auto cond2 = abs((xh_av_new - xh_av_prev) / (1 - xh_av_prev)) >
                         minimum_fractional_change;
            auto cond3 = (1 - xh_av_prev) > minimum_fraction_of_atoms;

            xh_int[idx] = xh_int_new;
            xh_av_prev = xh_av_new;
            conv_flag[idx] = cond1 && cond2 && cond3;

            idx += blockDim.x * gridDim.x;
        }
    }

}  // namespace

namespace asora {

    // Host function to call global_pass
    size_t global_pass(
        double* xh, double* xh_av, double* xh_int, const double* temp,
        const double* ndens, const double* phi_ion, const double* clump, double dt,
        double bh00, double albpow, double colh0, double temph0, double abu_c,
        size_t n_cells, size_t block_size
    ) {
        // Initialize and copy non-const data.
        // FIXME: This is inefficient becuse memory is allocated every time.
        auto xh_buf = device_buffer(n_cells * sizeof(double));
        auto xh_av_buf = device_buffer(n_cells * sizeof(double));
        auto xh_int_buf = device_buffer(n_cells * sizeof(double));
        xh_buf.copyFromHost(xh);
        xh_av_buf.copyFromHost(xh_av);
        xh_int_buf.copyFromHost(xh_int);

        // Initialize and copy const data.
        for (auto&& [tag, data] : {
                 std::pair{buffer_tag::number_density, ndens},
                 std::pair{buffer_tag::photo_ionization_HI, phi_ion},
                 std::pair{buffer_tag::temperature, temp},
                 std::pair{buffer_tag::clumping_factor, clump},
             }) {
            device::transfer<double>(tag, data, n_cells);
        }

        device_buffer conv_flag(n_cells);
        auto conv_flag_d = conv_flag.view<bool>().data();

        auto xh_d = xh_buf.data<double>();
        auto xh_av_d = xh_av_buf.data<double>();
        auto xh_int_d = xh_int_buf.data<double>();

        auto temp_d = device::get(buffer_tag::temperature).data<double>();
        auto ndens_d = device::get(buffer_tag::number_density).data<double>();
        auto phi_ion_d = device::get(buffer_tag::photo_ionization_HI).data<double>();
        auto clump_d = device::get(buffer_tag::clumping_factor).data<double>();

        // Launch kernel, divide by 2 so that threads do more work
        size_t grid_size = std::ceil(static_cast<float>(n_cells) / block_size / 2);
        evolve0D_gpu<<<grid_size, block_size>>>(
            xh_d, xh_av_d, xh_int_d, temp_d, ndens_d, phi_ion_d, clump_d, conv_flag_d,
            dt, bh00, albpow, colh0, temph0, abu_c, n_cells
        );

        // Check for errors
        safe_cuda(cudaGetLastError());

        // Reduction kernel to count non-zero elements
        auto convergence =
            thrust::count(thrust::device, conv_flag_d, conv_flag_d + n_cells, true);

        // xh_buf.copyToHost(xh, xh_buf.size());
        xh_av_buf.copyToHost(xh_av, xh_av_buf.size());
        xh_int_buf.copyToHost(xh_int, xh_int_buf.size());
        return convergence;
    }

}  // namespace asora
