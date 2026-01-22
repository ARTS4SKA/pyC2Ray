#pragma once

namespace asora {

    template <typename T>
    struct linspace {
        T start;
        T step;
        size_t num;

        __host__ __device__ T stop() const { return start + num * step; }
    };

    struct photo_tables {
        const double *thin;
        const double *thick;
    };

    // Photoionization rate from tables
    __device__ double photo_rates_gpu(
        double tau_in, double tau_out, int nf, const photo_tables &ion_tables,
        const linspace<double> &logtau
    );

    // Photoionization rates from analytical expression (grey-opacity)
    __device__ double photo_rates_test_gpu(double tau_in, double tau_out);

}  // namespace asora
