#pragma once

#include <cuda_runtime.h>
#include <cuda/std/array>

#include <concepts>
#include <source_location>

/* @file utils.cuh
 * @brief Utility functions and constants for ASORA GPU raytracing
 *
 * Provides:
 * - CUDA error checking wrapper
 * - Mathematical constants
 * - Octahedral coordinate system transformations
 * - Short-characteristics interpolation for radiative transfer
 */

namespace asora {

    /* @brief Check CUDA error and throw exception with source location on failure.
     *
     * @param[in] err CUDA error code to check
     * @param[in] loc Source location for error reporting (auto-captured)
     * @throw std::runtime_error if err != cudaSuccess
     */
    void safe_cuda(
        cudaError_t err,
        const std::source_location &loc = std::source_location::current()
    );

    /// Common mathematical constants
    namespace c {

        /// Pi constant
        template <std::floating_point F = double>
        constexpr F pi = F(3.1415926535897932385L);

        /// Square root of 3
        template <std::floating_point F = double>
        constexpr F sqrt3 = F(1.7320508075688772L);

        /// Square root of 2
        template <std::floating_point F = double>
        constexpr F sqrt2 = F(1.4142135623730951L);

    }  // namespace c

    /* @brief Fortran/Python-style modulo operation (always non-negative).
     *
     * Reminder: "%" in C is the remainder operator which preserves sign.
     *
     * @param[in] a Dividend
     * @param[in] b Divisor
     * @return Modulo result in range [0, b)
     */
    __host__ __device__ int modulo(int a, int b);

    /* @brief Convert 3D grid indices to flat array index.
     *
     * @param[in] i X-index
     * @param[in] j Y-index
     * @param[in] k Z-index
     * @param[in] N Grid size (assumed cubic)
     * @return Flat index into 1D array
     */
    __device__ size_t ravel_index(int i, int j, int k, int N);

    /* @brief Check if grid indices are within bounds.
     *
     * @param[in] i X-index
     * @param[in] j Y-index
     * @param[in] k Z-index
     * @param[in] N Grid size
     * @return True if (i,j,k) is inside [0,N)^3
     */
    __device__ bool in_box(int i, int j, int k, int N);

    /* @brief Convert octahedral (q,s) coordinates to Cartesian (i,j,k).
     *
     * The mapping is such that for a given q shell, the s-index enumerates the cells
     * ordered by ravel_index(i, j, k). This should partially help with memory access
     * patterns when processing the cells in a shell.
     *
     * @see cart2shell() for the backward transformation.
     *
     * @param[in] q Shell index (distance from source)
     * @param[in] s Position index within shell q
     * @return Array containing {i, j, k} offsets
     */
    __host__ __device__ int3 shell2cart(int q, int s);

    /* @brief Convert Cartesian (i,j,k) coordinates to octahedral (q,s).
     *
     * @see shell2cart() for the forward transformation.
     *
     * @param[in] i X-offset
     * @param[in] j Y-offset
     * @param[in] k Z-offset
     * @return Array containing {q, s}
     */
    __host__ __device__ int2 cart2shell(int i, int j, int k);

    /* @brief Calculate geometric path length of a ray through a cell.
     *
     * @param[in] di X-component of direction
     * @param[in] dj Y-component of direction
     * @param[in] dk Z-component of direction
     * @return Path length in units of cell size
     */
    [[deprecated("Not used anymore")]]
    __host__ __device__ double path_in_cell(int di, int dj, int dk);

    /* @brief Compute interpolation weights for 4 adjacent upstream cells.
     *
     * Used in short-characteristics method to weight contributions from
     * neighboring cells. dk must be the largest delta component.
     *
     * @param[in] di X-component of direction
     * @param[in] dj Y-component of direction
     * @param[in] dk Z-component of direction
     * @return Array of 4 geometric weighting factors
     */
    [[deprecated("Not used anymore")]]
    __host__ __device__
        cuda::std::array<double, 4> geometric_factors(int di, int dj, int dk);

    /* @brief Short-characteristics interpolator for radiative transfer.
     *
     * Implements the short-characteristics method for computing column densities
     * along rays. Interpolates values from 4 upstream cells using geometric weights
     * based on ray direction.
     */
    class [[deprecated("Not used anymore")]] cell_interpolator {
       public:
        /* @brief Construct interpolator for the cell at position (di, dj, dk) for
         * a ray coming from the origin of the coordinate system.
         * The constructor pre-computes the interpolation weights and cell offsets.
         *
         * @param[in] di X-component of the cell
         * @param[in] dj Y-component of the cell
         * @param[in] dk Z-component of the cell
         */
        __device__ cell_interpolator(int di, int dj, int dk);

        /* @brief Interpolate column density from upstream cells.
         *
         * Combines column densities from 4 adjacent cells in the direction of the
         * incoming ray using pre-computed geometric weights.
         *
         * @param[in] coldens Array of column density pointers for the three previous
         *                    q-shells (q-1, q-2, q-3)
         * @param[in] sigma Photoionization cross-section
         * @return Interpolated column density value
         * @see element_data::partition_column_density() for preparing the shared memory
         *      banks of column densities.
         */
        __device__ double interpolate(
            const cuda::std::array<const double *__restrict__, 3> &coldens, double sigma
        );

       private:
        /// Position of the cell with respect to the source
        int _di, _dj, _dk;

        /// Current shell level
        int _q0;

        /// Path length multiplier for cells close to the source (q <= 1)
        double _mul;

        /// Memory offsets for 4 neighbors
        cuda::std::array<int, 12> _offsets;

        /// Interpolation weights
        cuda::std::array<double, 4> _factors;

        /// Get octahedral (q-q0-1, s) coordinates for given offset.
        inline __device__ cuda::std::array<int, 2> get_qlevel(
            int i_off, int j_off, int k_off
        );

        /// Check if interpolator points at source origin.
        inline __device__ bool is_origin();
    };

}  // namespace asora
