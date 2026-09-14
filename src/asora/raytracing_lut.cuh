#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

namespace asora {

    /// Maximum allowed q-shell index, included.
    constexpr int Q_MAX = 512;

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
    __host__ __device__ uint32_t pack_offset(int3 pos);
    __host__ __device__ int3 unpack_offset(uint32_t offset);

    struct alignas(16) index4 {
        uint32_t d[4];

        __host__ __device__ uint32_t operator[](size_t i) const { return d[i]; }
        __host__ __device__ uint32_t &operator[](size_t i) { return d[i]; }
    };

    /// Structure of arrays for the LUT on device.
    struct raytracing_lut {
        uint32_t *__restrict__ offsets;
        double *__restrict__ multipliers;
        double *__restrict__ dxs;
        double *__restrict__ dys;
        double *__restrict__ paths;
        index4 *__restrict__ indices;

        raytracing_lut();
        raytracing_lut(const raytracing_lut &) = default;
        raytracing_lut &operator=(const raytracing_lut &) = default;

        // FIXME: add constructor that unpacks offset
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

        __device__ entry operator[](size_t i) const {
            return {offsets[i], multipliers[i], dxs[i], dys[i], paths[i], indices[i]};
        }
    };

    /* @brief Create a lookup table for all cells in the q-shells up to q_max.
     *
     * @param q_max Maximum q-shell index, included.
     * @return The number of cells in the q-shells up to q_max.
     */
    size_t create_raytracing_lut(int q_max);

    using raytracing_lut_entries = std::vector<asora::raytracing_lut::entry>;
    raytracing_lut_entries copy_lut_to_host();

}  // namespace asora
