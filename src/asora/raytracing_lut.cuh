#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

namespace asora {

    /// Maximum allowed q-shell index, included.
    constexpr int Q_MAX = 512;

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
    __host__ __device__ uint32_t pack_offset(const int3 &pos);
    __host__ __device__ int3 unpack_offset(uint32_t offset);

    /* @brief Structure to hold a fixed-size array of indices, aligned to 16 bytes.
     *
     * This structure is used to store the short-characteristic indices with element
     * access by index. The alignment ensures that the structure can be safely used in
     * device code.
     */
    struct alignas(16) index4 {
        uint32_t d[4];

        __host__ __device__ uint32_t operator[](size_t i) const { return d[i]; }
        __host__ __device__ uint32_t &operator[](size_t i) { return d[i]; }
    };

    /// Get number of cells in octahedral shell q.
    __host__ __device__ size_t cells_in_shell(int q);

    /// Get cumulative number of cells up to and including shell q.
    __host__ __device__ size_t cells_to_shell(int q);

    /// Initialize lookup tables for octahedral indexing.
    void setup_cells_to_shell_luts();

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

        struct entry {
            /// Cell offsets.
            int di;
            int dj;
            int dk;

            /// Geometric factors.
            double multiplier;
            double dx;
            double dy;
            double path;

            /// Offset indices for short-characteristic interpolation.
            index4 indices;
        };

        __device__ entry operator[](size_t i) const {
            auto &&[di, dj, dk] = unpack_offset(offsets[i]);
            return {di, dj, dk, multipliers[i], dxs[i], dys[i], paths[i], indices[i]};
        }
    };

    /* @brief Create a lookup table for all cells in the q-shells up to q_max.
     *
     * @param q_max Maximum q-shell index, included.
     * @return The number of cells in the q-shells up to q_max.
     */
    size_t create_raytracing_lut(int q_max);

    using raytracing_lut_entries = std::vector<asora::raytracing_lut::entry>;

    /* @brief Copy the raytracing lookup table from device to host.
     *
     * @param q_max Maximum q-shell index, included.
     * @return A vector of raytracing_lut::entry structures containing the lookup table
     *         data on the host.
     */
    raytracing_lut_entries copy_lut_to_host(int q_max);

}  // namespace asora
