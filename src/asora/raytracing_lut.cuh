#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cuda/std/array>
#include <map>
#include <span>
#include <unordered_map>
#include <vector>

namespace asora {

    /// Maximum allowed q-shell index, included.
    static constexpr int Q_MAX = 512;

    // 2^10 = 1024, possible values in range [-512, 512)
    constexpr uint32_t OFFSET_BITS = 10;
    constexpr uint32_t OFFSET_MASK = (1u << OFFSET_BITS) - 1;

    /* @brief Pack the integer coordinates (di, dj, dk) of a cell in the q-shell into a
     * single 32bit value.
     *
     * The coordinates are offset by Q_MAX to ensure they are non-negative. The packed
     * value uses 30 bits to store the three coordinates, leaving 2 bits unused. The
     * remaining 2 bits are used to represent the 3 missing points:
     *
     * di = Q_MAX, dj = 0, dk = 0 -> 11...
     * di = 0, dj = Q_MAX, dk = 0 -> 10...
     * di = 0, dj = 0, dk = Q_MAX -> 01...
     *            everything else -> 00...
     */
    uint32_t pack_offset(int di, int dj, int dk);
    __host__ __device__ cuda::std::array<int, 3> unpack_offset(uint32_t offset);

    struct alignas(16) index4 {
        uint32_t d[4];

        __host__ __device__ uint32_t operator[](size_t i) const { return d[i]; }
        __host__ __device__ uint32_t &operator[](size_t i) { return d[i]; }
    };

    /// Structure of arrays for creating the raytracing lookup table.
    struct raytracing_lut {
        struct entry {
            /// Packed cell offset (di, dj, dk).
            uint32_t offset;

            /// Geometric factors.
            double multiplier;
            double dx;
            double dy;
            double path;

            /// Offset indices for short-characteristic interpolation.
            index4 indices;
        };

        /// Packed cell offsets (N, ).
        std::vector<uint32_t> offsets;

        /// Geometric factors (N, ).
        std::vector<double> multipliers;
        std::vector<double> dxs;
        std::vector<double> dys;
        std::vector<double> paths;

        /// Offset indices for short-characteristic interpolation (N, 4).
        std::vector<index4> indices;

        explicit raytracing_lut(size_t n = 0)
            : offsets(n), multipliers(n, 1.0), dxs(n), dys(n), paths(n), indices(n) {}

        entry operator[](size_t i) const {
            assert(i < offsets.size());
            return {offsets[i], multipliers[i], dxs[i], dys[i], paths[i], indices[i]};
        }

        size_t size() const { return offsets.size(); }
    };

    /// Structure of arrays for the LUT on device.
    struct raytracing_lut_ptr {
        const uint32_t *__restrict__ offsets;
        const double *__restrict__ multipliers;
        const double *__restrict__ dx;
        const double *__restrict__ dy;
        const double *__restrict__ path;
        const index4 *__restrict__ indices;

        __device__ raytracing_lut::entry operator[](size_t i) const {
            return {offsets[i], multipliers[i], dx[i], dy[i], path[i], indices[i]};
        }
    };

    /// Create a lookup table for all cells in the q-shells up to q_max.
    raytracing_lut create_lut(int q_max);

}  // namespace asora
