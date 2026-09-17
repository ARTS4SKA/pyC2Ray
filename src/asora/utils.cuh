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
 * - Fortran/Python-style modulo operation
 * - 3D grid index raveling and bounds checking
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
    __device__ inline int modulo(int a, int b) { return (a % b + b) % b; }

    /* @brief Convert 3D grid indices to flat array index.
     *
     * @param[in] i X-index
     * @param[in] j Y-index
     * @param[in] k Z-index
     * @param[in] N Grid size (assumed cubic)
     * @return Flat index into 1D array
     */
    __device__ inline size_t ravel_index(int i, int j, int k, int N) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

    /* @brief Check if grid indices are within bounds.
     *
     * @param[in] i X-index
     * @param[in] j Y-index
     * @param[in] k Z-index
     * @param[in] N Grid size
     * @return True if (i,j,k) is inside [0,N)^3
     */
    __device__ inline bool in_box(int i, int j, int k, int N) {
        return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
    }

}  // namespace asora
