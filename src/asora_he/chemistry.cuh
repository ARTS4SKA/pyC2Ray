#pragma once

#include "../asora/rates.cuh"

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
        double sigma_HeI_at_HeI = 7.430e-18;
        // HeI cross-section at HeII ionization threshold
        double sigma_HeI_at_HeII = 1.690780687052975e-18;
        // HeI cross-section at HeI Lya frequency (h\nu = 40.8 eV)
        double sigma_HeI_at_HeLya = 1.301e-20;
        // HeII cross section at its ionzing frequency
        double sigma_HeII_at_HeII = 1.589e-18;

        // Fraction of photons from recombination of HeII that ionize HeI
        // (p. 32 of Kai Yan Lee's thesis)
        double p_rec = 0.96;
        // Fraction of photons from 2-photon decay, energetic enough to ionize hydrogen
        double l_dec = 1.425;
        // Fraction of photons from 2-photon decay, energetic enough to ionize neutral
        // helium
        double m_dec = 0.737;
        // Escape fraction range of Ly α photons, it depends on the neutral fraction
        std::pair<double, double> f_lya_range = {0.01, 1.0};

        // Cosmological abundances
        double abu_h = 0.926;
        double abu_he = 0.074;
        double abu_c = 7.1e-7;

        // Relative energy change threshold controlling iteration/convergence
        double relative_denergy = 0.1;
        // Adiabatic index
        double gamma = 5. / 3.;

        // Flag to enable/disable non-cosmological cooling in the thermal solver.
        bool cosmo_only = false;
    };

    /* @brief Container for cooling rate lookup tables.
     *
     * Holds pointers to data-fitted tables for cooling rates of different gas species.
     * Both tables must be allocated in device memory before use.
     */
    struct cooling_tables {
        const double* __restrict__ HI = nullptr;
        const double* __restrict__ HII = nullptr;
        const double* __restrict__ HeI = nullptr;
        const double* __restrict__ HeII = nullptr;
        const double* __restrict__ HeIII = nullptr;
    };

    /* @brief Integrate thermal evolution over a timestep.
     *
     * Updates the gas temperature by integrating a thermal energy equation over the
     * time interval `dt`, given external heating and expansion/cooling terms.
     *
     * @param dt Timestep size
     * @param temp_start Temperature at the beginning of the step
     * @param ndens Atomic number density
     * @param heating Heating rate
     * @param Hz Hubble parameter in cgs
     * @param xh Current ionization fractions
     * @param xh_av Current average ionization fractions
     * @param xh_int Current intermediate ionization fractions
     * @param p Parameter set (cross sections, abundances, etc.)
     * @param min_temp Floor temperature (default: 1.0)
     * @param max_iterations Maximum number of iterations allowed
     *
     * @return {end temperature, average temperature}
     */
    __host__ __device__ double2 thermal(
        double dt, double temp_start, double ndens, double heating, double Hz,
        const double3& xh, const double3& xh_av, const double3& xh_int,
        const cooling_tables& rates, const linspace<double>& logtemp,
        const parameters& p = {}, double min_temp = 1.0, size_t max_iterations = 10'000
    );

    /* @brief Compute optical depth ratios for photoionization rates.
     *
     * @param ndens Number densities of H, HeI, HeII
     * @param p Parameter set (cross sections, abundances, etc.)
     *
     * @return Optical depth ratios for use in friedrich
     */
    __host__ __device__ cuda::std::array<double, 4> optical_depth_ratios(
        const double3& ndens, const parameters& p = {}
    );

    /* @brief Chemistry solution.
     *
     * @param dt Timestep size
     * @param temp Gas temperature
     * @param n_e Electron number density
     * @param xh Current ionization fractions
     * @param phion Photoionization rates
     * @param opt_depth_ratios Optical depth ratios
     * @param clumping Clumping factor (currently unused)
     * @param p Parameter set (cross sections, abundances, etc.)
     *
     * @return {ionization fractions, average ionization fractions}
     */
    __host__ __device__ cuda::std::array<double3, 2> friedrich(
        double dt, double temp, double n_e, const double3& xh, double xHII_int,
        const double3& phion, const cuda::std::array<double, 4>& opt_depth_ratios,
        double clumping = 1.0, const parameters& p = {}
    );

    /* @brief Chemistry and temperature evolution on a single cell.
     *
     * @param dt Timestep size
     * @param Hz Hubble parameter
     * @param temp Temperature at the beginning of the step
     * @param ndens Hydrogen number density for the cell
     * @param xh Current ionization fractions
     * @param xh_av Current average ionization fractions
     * @param phion Photo-ionization rates
     * @param heating Photo-heating rate
     * @param clump Clumping factor
     * @param tables Cooling rate lookup tables
     * @param logtemp Log-scale temperature range for cooling rate interpolation
     * @param p Parameter set (cross sections, abundances, etc.)
     * @param max_iterations Maximum number of chemistry iterations
     *
     * @return {ionization fractions, average ionization fractions, temperature}
     *         where temperature is {temp, temp_av}.
     */
    __device__ cuda::std::tuple<double3, double3, double2> do_chemistry(
        double dt, double Hz, double temp, double temp_av, double ndens,
        const double3& xh, double3 xh_av, const double3& phion, double heating,
        double clump, const cooling_tables& rates, const linspace<double>& logscale,
        const parameters& p = {}, size_t max_iterations = 400
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

        /// Const-corrected accessor methods for the component pointers.
        const double* cx() const { return x; }
        const double* cy() const { return y; }
        const double* cz() const { return z; }
    };

    /* @brief CUDA kernel: evolve chemistry/thermal state independently per cell.
     *
     * @param dt Timestep size
     * @param Hz Hubble parameter
     * @param temp Temperature arrays: initial, average and result
     * @param ndens Number density array
     * @param xh Ionization fractions
     * @param xh_av Average ionization fractions
     * @param xh_int Intermediate ionization fractions
     * @param phion Photo-ionization rates
     * @param pheat Photo-heating rates
     * @param clump Clumping factors
     * @param conv_flag Per-cell convergence flags (output)
     * @param tables Cooling rate lookup tables
     * @param logtemp Log-scale temperature range for cooling rate interpolation
     * @param p Parameter set (cross sections, abundances, etc.)
     * @param size Number of cells
     */
    __global__ void evolve0D_gpu(
        double dt, double Hz, double3ptr temp, const double* __restrict__ ndens,
        double3ptr xh, double3ptr xh_av, double3ptr xh_int, double3ptr phion,
        const double* __restrict__ pheat, const double* __restrict__ clump,
        cooling_tables tables, linspace<double> logtemp, bool* conv_flag, parameters p,
        size_t size
    );

    size_t global_pass(
        double dt, double Hz, double3ptr temp, double3ptr xh, double3ptr xh_av,
        double3ptr xh_int, const double3ptr& phion, const double* __restrict__ pheat,
        const double* __restrict__ clump, const linspace<double>& logtemp,
        const parameters& p, size_t n_cells, size_t block_size
    );

}  // namespace asora
