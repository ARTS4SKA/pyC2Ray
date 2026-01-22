#pragma once

#include "rates.cuh"

#include <cuda/std/array>

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(
        double R, const double *sig_HI, const double *sig_HeI, const double *sig_HeII,
        int num_bin_1, int num_bin_2, int num_bin_3, int num_freq, double dr,
        const double *xHI_av, const double *xHeI_av, const double *xHeII_av,
        double *phi_ion_HI, double *phi_ion_HeI, double *phi_ion_HeII,
        double *phi_heat_HI, double *phi_heat_HeI, double *phi_heat_HeII, int num_src,
        int m1, double minlogtau, double dlogtau, int num_tau, size_t grid_size,
        size_t block_size = 256
    );

    struct element_data {
        double *photo_ionization;
        double *photo_heating;
        double *column_density;
        const double *cross_section;
        size_t first_bin;

        // Shared column density relevant for cinterp.
        __device__ cuda::std::array<const double *, 3> make_shared_cdens(int q) const;
    };

    struct density_maps {
        const double *ndens;
        const double *xHII;
        const double *xHeII;
        const double *xHeIII;

        __device__ cuda::std::array<double, 3> get(size_t index) const;
    };

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(
        int m1, double dr, double R_max, int q_max, int ns_start, int num_src,
        int *src_pos, double *src_flux, element_data data_HI, element_data data_HeI,
        element_data data_HeII, density_maps densities, photo_tables ion_tables,
        photo_tables heat_tables, linspace<double> logtau, int num_freq
    );

}  // namespace asora
