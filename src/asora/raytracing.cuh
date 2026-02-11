#pragma once

#include <cuda/std/utility>

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(
        double R, double sig, double dr, const double *xh_av, double *phi_ion,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size = 256
    );

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(
        double R_max, int q, size_t ns_start, size_t num_src, int *src_pos,
        double *src_flux, double *coldensh_out, double sig, double dr,
        const double *ndens, const double *xh_av, double *phi_ion, size_t m1,
        const double *photo_thin_table, const double *photo_thick_table,
        double minlogtau, double dlogtau, size_t num_tau
    );

    // Path inside the cell; dk is the largest delta.
    __device__ double path_in_cell(int di, int dj, int dk);

    // Short-characteristics interpolation function
    __device__ double cinterp_gpu(
        int di, int dj, int dk, const cuda::std::array<const double *, 3> &shared_cdens,
        double sigma
    );

}  // namespace asora
