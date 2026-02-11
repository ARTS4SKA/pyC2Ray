#pragma once

#include "rates.cuh"

#include <cuda/std/array>

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(
        double R, double sig, double dr, const double *xh_av, double *phi_ion,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size = 256
    );

    struct element_data {
        double *__restrict__ photo_ionization;
        double *__restrict__ column_density;
        double cross_section;
        cuda::std::array<const double *__restrict__, 3> shared_cdens = {};

        // Prepare shared column density memory banks for cell interpolation
        __device__ void partition_column_density(int q);
    };

    struct density_maps {
        const double *__restrict__ ndens;
        const double *__restrict__ xHII;

        __device__ double get(size_t index) const;
    };

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(
        size_t m1, double dr, double R_max, int q_max, size_t ns_start, size_t num_src,
        int *src_pos, double *src_flux, element_data data_HI, density_maps densities,
        photo_tables ion_tables, linspace<double> logtau
    );

}  // namespace asora
