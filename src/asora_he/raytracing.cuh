#pragma once

#include "../asora/rates.cuh"

#include <cuda/std/array>

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(
        double R, const double *sig_HI, const double *sig_HeI, const double *sig_HeII,
        const double *heat_factors, size_t num_freq, double dr, const double *xHII_av,
        const double *xHeII_av, const double *xHeIII_av, double *phion_HI,
        double *phion_HeI, double *phion_HeII, double *pheat, size_t num_src, size_t m1,
        double minlogtau, double dlogtau, size_t num_tau, size_t grid_size,
        size_t block_size = 256
    );

    struct element_data {
        double *__restrict__ photo_ionization;
        double *__restrict__ column_density;
        const double *__restrict__ cross_section;

        cuda::std::array<const double *__restrict__, 3> shared_cdens = {};

        // Prepare shared column density memory banks for cell interpolation
        __device__ void partition_column_density(int q);
    };

    struct density_maps {
        const double *__restrict__ ndens;
        const double *__restrict__ xHII;
        const double *__restrict__ xHeII;
        const double *__restrict__ xHeIII;

        __device__ double3 get(size_t index) const;
    };

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(
        size_t m1, double dr, double R_max, int q_max, size_t ns_start, size_t num_src,
        int *src_pos, double *__restrict__ src_flux, element_data data_HI,
        element_data data_HeI, element_data data_HeII,
        double *__restrict__ photo_heating, double *__restrict__ heat_factors,
        density_maps densities, photo_tables<> ion_tables, photo_tables<> heat_tables,
        linspace<double> logtau, size_t num_freq
    );

}  // namespace asora
