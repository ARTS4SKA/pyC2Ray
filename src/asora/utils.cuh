#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <concepts>
#include <format>
#include <source_location>
#include <stdexcept>

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

    /* @brief Check a condition and throw an exception with formatted message if false.
     *
     * In device code, this will trigger an assertion failure. In host code, it will
     * throw a std::runtime_error with the provided formatted message.
     * Use this function if you want to throw on a condition in a host AND device
     * function.
     *
     * @param condition The condition to check
     * @param msg Format string for the error message
     * @param args Arguments for the format string
     * @throw std::runtime_error if condition is false (host code)
     */
#ifdef __CUDA_ARCH__
    template <typename... Args>
    __host__ __device__ void check(
        [[maybe_unused]] bool condition, const char *, const Args &...
    ) {
        assert(condition);
    }
#else
    template <typename... Args>
    __host__ __device__ void check(
        bool condition, std::format_string<const Args &...> msg, const Args &...args
    ) {
        if (!condition) [[unlikely]]
            throw std::runtime_error(std::format(msg, args...));
    }
#endif

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

    /* @brief Wrap an index into [0, N) with one compare-and-add per side.
     *
     * Equivalent to Fortran module, but cheaper. Valid only for a in [-N, 2N).
     *
     * @param a Index
     * @param N Grid size
     * @return Wrapped index in range [0, N)
     */
    __device__ inline int wrap(int a, size_t N) {
        const auto n = static_cast<int>(N);
        assert(-n <= a && a < 2 * n);
        if (a < 0) return a + n;
        if (a >= n) return a - n;
        return a;
    }

    /* @brief Convert 3D grid indices to flat array index.
     *
     * @param pos Grid indices, in [-N, 2N) and wrapped periodically into [0, N)
     * @param N Grid size (assumed cubic)
     * @return Flat index into 1D array
     */
    __device__ inline size_t ravel_index(const int3 &pos, size_t N) {
        auto i = static_cast<size_t>(wrap(pos.x, N));
        auto j = static_cast<size_t>(wrap(pos.y, N));
        auto k = static_cast<size_t>(wrap(pos.z, N));
        return (i * N + j) * N + k;
    }

    /// Component-wise sum of two integer vectors.
    __host__ __device__ inline int3 operator+(const int3 &a, const int3 &b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    /* @brief Check if the indices are strictly within the box.
     *
     * Negative values to unsigned wrap around and become >= N, so one unsigned compare
     * checks both bounds, if the indices and N are small (less than 2^31).
     *
     * @param pos Grid indices
     * @param N Grid size
     * @return True if inside [0,N)^3
     */
    __host__ __device__ inline bool in_box(const int3 &pos, size_t N) {
        return static_cast<size_t>(pos.x) < N && static_cast<size_t>(pos.y) < N &&
               static_cast<size_t>(pos.z) < N;
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
        if constexpr (periodic_conditions) {
            auto h = static_cast<int>(N / 2);
            return in_box(off + make_int3(h, h, h), N);
        } else {
            return in_box(pos + off, N);
        }
    }

}  // namespace asora
