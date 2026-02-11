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
    __device__ int ravel_index(int i, int j, int k, int N);

    /* @brief Check if grid indices are within bounds.
     *
     * @param[in] i X-index
     * @param[in] j Y-index
     * @param[in] k Z-index
     * @param[in] N Grid size
     * @return True if (i,j,k) is inside [0,N)^3
     */
    __device__ bool in_box(const int &i, const int &j, const int &k, const int &N);

    /* @brief Convert octahedral (q,s) coordinates to Cartesian (i,j,k).
     *
     * Maps from shell index q and position s within shell to 3D grid offsets
     * relative to the source position.
     * The ASORA algorithm uses a coordinate system based on nested octahedral shells
     * around each source. Each q-shell contains cells indexed by s.
     * Here follows an example of this mapping for the top part (k >= 0) and bottom part
     * (k < 0) of the q = 3 shell. The cells are projected in the (i, j) plane and
     * numbers correspond to their s-index.
     *
     *           k >= 0                      k < 0
     *              3
     *           2  6 10                       27
     *        1  5  9 13 17                 26 29 32
     *     0  4  8 12 16 20 24           25 28 31 34 37
     *        7 11 15 19 22                 30 33 36
     *          14 18 22                       35
     *             21
     *
     * Note that the bottom part (k < 0) is equivalent to the top part of the
     * 2-shell with s-index shifted by (q+1)² + q².
     *
     * @see cart2linthrd() for the backward transformation.
     *
     * @param[in] q Shell index (distance from source)
     * @param[in] s Position index within shell q
     * @return Array containing {i, j, k} offsets
     */
    __host__ __device__ cuda::std::array<int, 3> linthrd2cart(int q, int s);

    /* @brief Convert Cartesian (i,j,k) coordinates to octahedral (q,s).
     *
     * Inverse of linthrd2cart. Maps 3D grid offsets to octahedral shell coordinates.
     *
     * @see linthrd2cart() for the forward transformation.
     *
     * @param[in] i X-offset
     * @param[in] j Y-offset
     * @param[in] k Z-offset
     * @return Array containing {q, s}
     */
    __host__ __device__ cuda::std::array<int, 2> cart2linthrd(int i, int j, int k);

    /// Get number of cells in octahedral shell q.
    __host__ __device__ size_t cells_in_shell(int q);

    /// Get cumulative number of cells up to and including shell q.
    __host__ __device__ size_t cells_to_shell(int q);

    /* @brief Calculate geometric path length of a ray through a cell.
     *
     * @param[in] di X-component of direction
     * @param[in] dj Y-component of direction
     * @param[in] dk Z-component of direction
     * @return Path length in units of cell size
     */
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
    __host__ __device__ cuda::std::array<double, 4> geometric_factors(
        int di, int dj, int dk
    );

    /* @brief Short-characteristics interpolator for radiative transfer.
     *
     * Implements the short-characteristics method for computing column densities
     * along rays. Interpolates values from 4 upstream cells using geometric weights
     * based on ray direction.
     */
    class cell_interpolator {
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
