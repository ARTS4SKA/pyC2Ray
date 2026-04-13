#pragma once

#include <concepts>

/* @file rates.cuh
 * @brief Photoionization rate calculations for radiative transfer on GPU
 *
 * Provides:
 * - Linear space specification for logarithmic optical depth grids
 * - Data structures for photoionization lookup tables
 * - GPU device functions for computing photoionization rates
 */

namespace asora {

    /* @brief Linear space specification for logarithmically-spaced lookup tables.
     *
     * Collect parameters to describe a linearly-spaced grid used to index
     * logarithmically-spaced lookup tables.
     */
    template <std::floating_point T>
    struct linspace {
        /// Starting value of the linear space
        T start;

        /// Step size between consecutive points
        T step;

        /// Number of points in the linear space
        size_t num;

        /// Calculate the end value of the linear space.
        __host__ __device__ T stop() const { return start + num * step; }
    };

    /// Optical depth threshold to distinguish between optically thin and thick cells.
    constexpr double tau_photo_limit = 1.e-7;

    /* @brief Container for photoionization lookup tables.
     *
     * Holds pointers to pre-computed tables for optically thin and thick regimes.
     * Both tables must be allocated in device memory before use.
     *
     * The tables store values of the integral ∫L_ν*e^(-τ_ν)/hν computed over
     * the source spectrum.
     */
    struct photo_tables {
        const double *__restrict__ thin;
        const double *__restrict__ thick;
    };

    // Photoionization rate from tables

    /* @brief Compute photoionization rate from optical depths using lookup tables.
     *
     * Calculates the photoionization rate for a ray segment through a cell by
     * interpolating pre-computed tables. The method automatically selects between
     * optically thin and thick approximations based on the optical depth difference:
     * - Thin regime (|τ_out - τ_in| ≤ 10^-7): Uses linear approximation
     * - Thick regime (|τ_out - τ_in| > 10^-7): Uses difference of cumulative integrals
     *
     * @param[in] tau_in   Optical depth at ray entry into the cell
     * @param[in] tau_out  Optical depth at ray exit from the cell
     * @param[in] tables   Structure containing pointers to thin and thick lookup tables
     * @param[in] logtau   Linear space specification for the logarithmic τ-grid
     *
     * @return Photoionization rate for this cell segment
     */
    __device__ double photo_rates_gpu(
        double tau_in, double tau_out, const photo_tables &tables,
        const linspace<double> &logtau
    );

#ifdef GREY_NOTABLES
    // Reference ionizing flux normalization factor.
    constexpr double s_star_ref = 1e48;

    /* @brief Analytical photoionization rate for grey-opacity test cases.
     *
     * Computes photoionization rate using an analytical formula that assumes
     * monochromatic radiation. This version bypasses lookup tables and is used for code
     * validation and testing
     *
     * @param[in] tau_in   Optical depth at ray entry into the cell
     * @param[in] tau_out  Optical depth at ray exit from the cell
     *
     * @return Photoionization rate for this cell segment
     *
     * @note Uses s_star_ref as reference flux normalization
     */
    __device__ double photo_rates_test_gpu(double tau_in, double tau_out);
#endif  // GREY_NOTABLES

}  // namespace asora
