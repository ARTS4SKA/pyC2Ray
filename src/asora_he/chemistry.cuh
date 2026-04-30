#pragma once

#include <cuda/std/array>

/* @file chemistry.cuh
 * @brief Routines for the chemistry ODE solver.
 */

namespace asora {

    /* @brief Runtime/user-adjustable parameters for the chemistry + thermal solver.
     *
     * All members default to the values that were previously hard-coded as constexpr
     * variables in chemistry.cu.
     */
    struct parameters {
        // HI cross-section at HeI ionization frequency
        double sigma_H_at_HeI = 1.238e-18;
        // HI cross-section at HeII ionization frequency
        double sigma_H_at_HeII = 1.230695924714239e-19;
        // HI cross-section at HeI Lya frequency (h\nu = 40.8 eV)
        double sigma_H_at_HeLya = 9.907e-22;
        // HeI cross section at its ionzing frequency
        double sigma_HeI_at_ion_freq = 7.430e-18;
        // HeI cross-section at HeII ionization threshold
        double sigma_HeI_at_HeII = 1.690780687052975e-18;
        // HeI cross-section at HeI Lya frequency (h\nu = 40.8 eV)
        double sigma_HeI_at_HeLya = 1.301e-20;
        // HeII cross section at its ionzing frequency
        double sigma_HeII_at_ion_freq = 1.589e-18;

        // Fraction of photons from recombination of HeII that ionize HeI
        // (p. 32 of Kai Yan Lee's thesis)
        double p_rec = 0.96;
        // Fraction of photons from 2-photon decay, energetic enough to ionize hydrogen
        double l_dec = 1.425;
        // Fraction of photons from 2-photon decay, energetic enough to ionize neutral
        // helium
        double m_dec = 0.737;
        // Escape fraction of Ly α photons, it depends on the neutral fraction
        double f_lya = 1.0;

        // Cosmological abundances
        double abu_he = 0.074;
        double abu_h = 0.926;
        double abu_c = 7.1e-7;

        // Relative energy change threshold controlling iteration/convergence
        double relative_denergy = 0.1;
        // Adiabatic index
        double gamma = 5. / 3.;
    };

    /* @brief Integrate thermal evolution over a timestep.
     *
     * Updates the gas temperature by integrating a thermal energy equation over the
     * time interval `dt`, given external heating and expansion/cooling terms.
     *
     * @param dt Timestep size
     * @param temp_start Temperature at the beginning of the step
     * @param ndens_elec Electron number density
     * @param ndens_atom Atomic number density
     * @param heating Heating rate
     * @param Hz Hubble parameter in cgs
     * @param p Parameter set (cross sections, abundances, etc.)
     * @param min_temp Floor temperature (default: 1.0)
     * @param relative_denergy Relative energy change threshold controlling
     * iteration/convergence
     * @param max_iterations Maximum number of iterations allowed
     *
     * @return {end temperature, average temperature}
     */
    __host__ __device__ cuda::std::array<double, 2> thermal(
        double dt, double temp_start, double ndens_elec, double ndens_atom,
        double heating, double Hz, const parameters& p = {}, double min_temp = 1.0,
        size_t max_iterations = 10'000
    );

    /* @brief Chemistry solution.
     *
     * @param dt Timestep size
     * @param dr Cell size (currently unused)
     * @param temp Gas temperature
     * @param n_e Electron number density
     * @param xh Current ionization fractions
     * @param phion Photoionization rates
     * @param pheat Photoheating rates (currently unused)
     * @param ndens Number densities
     * @param clumping Clumping factor (currently unused)
     * @param p Parameter set (cross sections, abundances, etc.)
     *
     * @return {ionization fractions, average ionization fractions}
     */
    __device__ cuda::std::array<double3, 2> friedrich(
        double dt, [[maybe_unused]] double dr, double temp, double n_e,
        const double3& xh, const double3& phion, [[maybe_unused]] const double3& pheat,
        const double3& ndens, [[maybe_unused]] double clumping, const parameters& p = {}
    );

    /* @brief Chemistry and temperature evolution on a single cell.
     *
     * @param dt Timestep size
     * @param dr Co-moving dimension of one grid cell
     * @param Hz Hubble parameter
     * @param temp_start Temperature at the beginning of the step
     * @param ndens Hydrogen number density for the cell
     * @param xh Current ionization fractions
     * @param xh_av Current average ionization fractions
     * @param phi_ion Photo-ionization rates
     * @param phi_heat Photo-heating rates
     * @param clump Clumping factor
     * @param p Parameter set (cross sections, abundances, etc.)
     * @param max_iterations Maximum number of chemistry iterations
     *
     * @return {ionization fractions, average ionization fractions}
     */

    __device__ cuda::std::array<double3, 2> do_chemistry(
        double dt, double dr, double Hz, double temp_start, double ndens,
        const double3& xh, double3 xh_av, const double3& phi_ion,
        const double3& phi_heat, double clump, const parameters& p = {},
        size_t max_iterations = 400
    );

    /* @brief Convenience structure holding 3 component-wise pointers.
     *
     * Used to represent three separate arrays (x, y, z) that together store a
     * `double3`-like quantity in structure-of-arrays layout on the GPU.
     */
    struct double3ptr {
        double* __restrict__ x;
        double* __restrict__ y;
        double* __restrict__ z;
    };

    /* @brief CUDA kernel: evolve chemistry/thermal state independently per cell.
     *
     * @param dt Timestep size
     * @param dr Co-moving dimension of one grid cell
     * @param Hz Hubble parameter
     * @param temp Temperature array
     * @param ndens Number density array
     * @param xh Ionization fractions
     * @param xh_av Average ionization fractions
     * @param xh_int Intermediate ionization fractions
     * @param phi_ion Photo-ionization rates
     * @param phi_heat Photo-heating rates
     * @param clump Clumping factors
     * @param conv_flag Per-cell convergence flags (output)
     * @param p Parameter set (cross sections, abundances, etc.)
     * @param size Number of cells
     */
    __global__ void evolve0D_gpu(
        double dt, double dr, double Hz, double* __restrict__ temp,
        double* __restrict__ ndens, double3ptr xh, double3ptr xh_av, double3ptr xh_int,
        double3ptr phi_ion, double3ptr phi_heat, const double* __restrict__ clump,
        bool* conv_flag, parameters p, size_t size
    );

}  // namespace asora
