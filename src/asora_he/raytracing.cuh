#pragma once

#include "../asora/rates.cuh"

#include <cuda/std/array>

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(
        double R, const double *sig_HI, const double *sig_HeI, const double *sig_HeII,
        size_t num_bin_1, size_t num_bin_2, size_t num_freq, double dr,
        const double *xHI_av, const double *xHeI_av, const double *xHeII_av,
        double *phi_ion_HI, double *phi_ion_HeI, double *phi_ion_HeII,
        double *phi_heat_HI, double *phi_heat_HeI, double *phi_heat_HeII,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size = 256
    );

    struct element_data {
        double *__restrict__ photo_ionization;
        double *__restrict__ photo_heating;
        double *__restrict__ column_density;
        const double *__restrict__ cross_section;
        size_t first_bin;
        cuda::std::array<const double *__restrict__, 3> shared_cdens;

        // Prepare shared column density memory banks for cell interpolation
        __device__ void partition_column_density(int q);
    };

    struct density_maps {
        const double *__restrict__ ndens;
        const double *__restrict__ xHII;
        const double *__restrict__ xHeII;
        const double *__restrict__ xHeIII;

        __device__ cuda::std::array<double, 3> get(size_t index) const;
    };

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(
        size_t m1, double dr, double R_max, int q_max, size_t ns_start, size_t num_src,
        int *src_pos, double *src_flux, element_data data_HI, element_data data_HeI,
        element_data data_HeII, density_maps densities, photo_tables ion_tables,
        photo_tables heat_tables, linspace<double> logtau, size_t num_freq
    );

}  // namespace asora
