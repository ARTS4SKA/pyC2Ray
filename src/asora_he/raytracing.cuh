#pragma once

// ========================================================================
// Header file for OCTA raytracing library.
// Functions defined and documented in raytracing_gpu.cu
// ========================================================================

#include <cuda/std/utility>

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(
        double R, const double *sig_hi, const double *sig_hei, const double *sig_heii,
        int num_bin_1, int num_bin_2, int num_bin_3, int num_freq, double dr,
        const double *xHI_av, const double *xHeI_av, const double *xHeII_av,
        double *phi_ion_HI, double *phi_ion_HeI, double *phi_ion_HeII,
        double *phi_heat_HI, double *phi_heat_HeI, double *phi_heat_HeII, int num_src,
        int m1, double minlogtau, double dlogtau, int num_tau, size_t grid_size,
        size_t block_size = 256
    );

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(
        double R_max, int q_max, int ns_start, int num_src, int *src_pos,
        double *src_flux, double *coldens_out_hi, double *coldens_out_hei,
        double *coldens_out_heii, const double *sig_hi, const double *sig_hei,
        const double *sig_heii, double dr, const double *ndens, const double *xHII_av,
        const double *xHeII_av, const double *xHeIII_av, double *phi_ion_HI,
        double *phi_ion_HeI, double *phi_ion_HeII, double *phi_heat_HI,
        double *phi_heat_HeI, double *phi_heat_HeII, int m1,
        const double *photo_thin_table, const double *photo_thick_table,
        const double *heat_thin_table, const double *heat_thick_table, double minlogtau,
        double dlogtau, int num_tau, int num_bin_1, int num_bin_2, int num_bin_3,
        int num_freq
    );

    // Path inside the cell
    __device__ double path_in_cell(int di, int dj, int dk);

    using shared_cdens_t = cuda::std::array<const double *, 3>;

    // Short-characteristics interpolation function
    __device__ cuda::std::array<double, 3> cinterp_gpu(
        int di, int dj, int dk, const shared_cdens_t &shared_cdens_hi,
        const shared_cdens_t &shared_cdens_hei, const shared_cdens_t &shared_cdens_heii,
        double sigma_HI, double sigma_HeI, double sigma_HeII
    );

}  // namespace asora
