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
    template <std::floating_point T = double>
    struct linspace {
        /// Starting value of the linear space
        T start = 0;

        /// Step size between consecutive points
        T step = 1;

        /// Number of points in the linear space
        size_t num = 1;

        /// Calculate the end value of the linear space.
        __host__ __device__ T stop() const { return start + num * step; }
    };

    /* @brief Container for photoionization lookup tables.
     *
     * Holds pointers to pre-computed tables for optically thin and thick regimes.
     * Both tables must be allocated in device memory before use.
     *
     * The tables store values of the integral ∫L_ν*e^(-τ_ν)/hν computed over
     * the source spectrum.
     */
    template <std::floating_point T = double>
    struct photo_tables {
        const T *__restrict__ thin;
        const T *__restrict__ thick;
    };

    /* @brief Structure to hold interpolation indices and weights for log-scale tables.
     *
     * Contains the indices of the two nearest data points in a lookup table and
     * the interpolation weight for linear interpolation between them.
     */
    template <std::floating_point T = double>
    struct tau_pos {
        size_t i0;
        size_t i1;
        T p;

        __host__ __device__ T interp(const T *table) const {
            return (1 - p) * table[i0] + p * table[i1];
        }
    };

    /* @brief Compute interpolation indices and weights for log-scale tables.
     *
     * Given a value x, this function computes the indices of the two nearest
     * data points in a logarithmically-spaced lookup table and the interpolation
     * weight for linear interpolation between them.
     *
     * @param[in] x        Value to interpolate (e.g., optical depth)
     * @param[in] logscale Linear space specification for the logarithmic τ-grid
     * @param[in] base     Base of the logarithm used for the grid (default: 10)
     *
     * @return Structure containing indices and interpolation weight
     */
    template <std::floating_point T = double>
    __host__ __device__ tau_pos<T> log_table_index(
        T x, const asora::linspace<T> &logscale, T base = 10.0
    ) {
        // Clamp the log(tau) to be within the table range
        auto lx = max(logscale.start, log2(x) / log2(base));

        // Map lx to its position in the table
        auto interp =
            min(static_cast<T>(logscale.num),
                max(static_cast<T>(0.0), (lx - logscale.start) / logscale.step));

        // Split the continuous index into integer and fractional parts
        // integral = floor of the index, used for table lookup
        // residual = fractional part, used for interpolation weight
        T integral;
        auto residual = modf(interp, &integral);

        // Determine the two table indices for linear interpolation and perform the
        // interpolation
        auto i0 = static_cast<size_t>(integral);
        auto i1 = min(logscale.num, i0 + 1);

        return {i0, i1, residual};
    }

    /* @brief Compute photo rate from optical depths using lookup tables.
     *
     * Calculates the photo rate for a ray segment through a cell by
     * interpolating pre-computed tables. The method automatically selects between
     * optically thin and thick approximations based on the optical depth
     * difference:
     * - Thin regime (|τ_out - τ_in| ≤ 10^-7): Uses linear approximation
     * - Thick regime (|τ_out - τ_in| > 10^-7): Uses difference of cumulative
     * integrals
     *
     * @param[in] tpos_in   Optical depth at ray entry into the cell
     * @param[in] tpos_out  Optical depth at ray exit from the cell
     * @param[in] tau_cell  Optical depth difference (tau_out - tau_in) across the cell
     * @param[in] tables   Structure containing pointers to thin and thick lookup
     * tables
     * @param[in] logtau   Linear space specification for the logarithmic τ-grid
     *
     * @return Photo rate for this cell segment
     */
    template <std::floating_point T = double>
    __host__ __device__ T photo_table_lookup(
        const tau_pos<T> &tpos_in, const tau_pos<T> &tpos_out, T tau_cell,
        const photo_tables<T> &tables
    ) {
        /// Optical depth threshold to distinguish between thin and thick cells.
        constexpr T tau_photo_limit = 1.e-7;

        // Check if the cell is optically thin - simplified calculation
        if (abs(tau_cell) <= tau_photo_limit)
            return tau_cell * tpos_out.interp(tables.thin);

        // Cell is optically thick - use both tables
        auto rate_in = tpos_in.interp(tables.thick);
        auto rate_out = tpos_out.interp(tables.thick);
        return rate_in - rate_out;
    }

    /* @see photo_table_lookup(const tau_pos<T> &, const tau_pos<T> &, T, const
     * photo_tables<T> &)
     *
     * This overload computes the interpolation indices and weights for the
     * input and output optical depths before calling the main lookup function.
     */
    template <std::floating_point T = double>
    __host__ __device__ T photo_table_lookup(
        T tau_in, T tau_out, const photo_tables<T> &tables, const linspace<T> &logtau
    ) {
        return photo_table_lookup(
            log_table_index(tau_in, logtau), log_table_index(tau_out, logtau),
            tau_out - tau_in, tables
        );
    }

}  // namespace asora
