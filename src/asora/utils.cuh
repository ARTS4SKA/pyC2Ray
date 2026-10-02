#pragma once

#include <cuda_runtime.h>

#include <cassert>
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
     * @param a Dividend
     * @param b Divisor
     * @return Modulo result in range [0, b)
     */
    __device__ inline int modulo(int a, int b) { return (a % b + b) % b; }

    /* @brief Convert 3D grid indices to flat array index.
     *
     * @param i X-index
     * @param j Y-index
     * @param k Z-index
     * @param N Grid size (assumed cubic)
     * @return Flat index into 1D array
     */
    __device__ inline size_t ravel_index(int i, int j, int k, int N) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

    /* @brief Check if the indices are strictly within the box.
     *
     * Negative values to unsigned wrap around and become >= N, so one unsigned compare
     * checks both bounds, if i,j,k and N are small (less than 2^31).
     *
     * @param i X-index
     * @param j Y-index
     * @param k Z-index
     * @param N Grid size
     * @return True if inside [0,N)^3
     */
    __host__ __device__ inline bool in_box(int i, int j, int k, size_t N) {
        return static_cast<size_t>(i) < N && static_cast<size_t>(j) < N &&
               static_cast<size_t>(k) < N;
    }

#ifdef PERIODIC
    constexpr bool periodic_conditions = true;
#else
    constexpr bool periodic_conditions = false;
#endif

    /* @brief Check if an offset from the source lies within the box.
     *
     * If periodic boundary conditions are enabled, the offset is checked against the
     * periodic box [-N/2, N/2] for odd N or [-N/2, N/2) for even N. Shifting by N/2
     * maps both cases onto [0, N) (because odd: N/2 + N/2 = N - 1, but even N/2 + N/2 =
     * N).
     *
     * Without periodic boundary conditions, the offset is added to the source position
     * and simply checked against the box [0, N).
     *
     * @param pos The source cell position
     * @param off Offset from the source cell
     * @param N Grid size
     * @return True if the offset is within the domain boundaries
     */
    __host__ __device__ inline bool in_domain(
        [[maybe_unused]] const int3 &pos, const int3 &off, size_t N
    ) {
        if constexpr (periodic_conditions)
            return in_box(off.x + N / 2, off.y + N / 2, off.z + N / 2, N);
        else
            return in_box(pos.x + off.x, pos.y + off.y, pos.z + off.z, N);
    }

}  // namespace asora
