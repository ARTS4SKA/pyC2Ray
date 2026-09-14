#pragma once

#include "rates.cuh"
#include "raytracing_lut.cuh"

#include <cuda/std/array>

namespace asora {
    /* @brief Raytrace all sources and compute photoionization rates
     *
     * Performs GPU-accelerated raytracing for all radiation sources to calculate
     * photoionization rates across the simulated volume.
     *
     * @param R Maximum propagation radius for photons from the source
     * @param sig Ionization cross section
     * @param dr Co-moving dimension of one grid cell
     * @param xh_av Array of average neutral hydrogen fractions
     * @param phi_ion Output array for computed photoionization rates
     * @param num_src Number of radiation sources
     * @param m1 Grid dimension size for a cubic domain (total grid points = m1^3)
     * @param minlogtau Minimum log10(optical depth)
     * @param dlogtau Step size in log10(optical depth)
     * @param num_tau Number of optical depth bins
     * @param grid_size GPU grid size for kernel launch
     * @param block_size GPU block size for kernel launch (default: 256)
     */
    void do_all_sources_gpu(
        double R, double sig, double dr, const double *xh_av, double *phi_ion,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size = 256
    );

    /* @brief Data structure for chemical element properties used in raytracing
     */
    struct element_data {
        /// Output array for photoionization rates
        double *__restrict__ photo_ionization;

        /// Column density along rays
        double *__restrict__ column_density;

        /// Photoionization cross section
        double cross_section;
    };

    /* @brief Read-only maps of number and fractional densities
     */
    struct density_maps {
        /// Number density data
        const double *__restrict__ ndens;

        /// Ionized hydrogen fraction data
        const double *__restrict__ xHII;

        /// Get hydrogen density value at the specified index
        __device__ double get(size_t index) const;
    };

    /* @brief GPU kernel for raytracing and photoionization evolution
     *
     * CUDA kernel that performs the core raytracing computation, propagating rays
     * from all sources through the computational domain and computing photoionization
     * rates based on optical depth and cross sections.
     *
     * @param m1 Grid dimension size
     * @param dr Co-moving dimension of a grid pixel
     * @param R_max Maximum propagation radius for photons from the source
     * @param q_max Maximum octahedral q-shell for raytracing
     * @param ns_start Starting source index for this kernel invocation
     * @param num_src Number of sources to process
     * @param src_pos Array of source positions (integer grid coordinates)
     * @param src_flux Array of source luminosities/flux values
     * @param data_HI Element data structure for neutral hydrogen
     * @param densities Density maps containing ndens and ionization fractions
     * @param ion_tables Lookup tables for photoionization cross sections
     * @param logtau Logarithmically-spaced optical depth grid
     */
    __global__ void evolve0D_gpu(
        raytracing_lut lut, size_t m1, double dr, double R_max, int q_max,
        size_t ns_start, size_t num_src, const int *__restrict__ src_pos,
        const double *__restrict__ src_flux, element_data data_HI,
        density_maps densities, photo_tables ion_tables, linspace<double> logtau
    );

}  // namespace asora
