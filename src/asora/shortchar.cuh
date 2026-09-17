#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

namespace asora {

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

    struct shortchar_info {
        /// Cell offsets (di, dj, dk)
        int3 pos;

        /// Geometric factors.
        float multiplier;
        float dx;
        float dy;
        float path;

        /// Offset indices for short-characteristic interpolation.
        index4 indices;
    };
    /* @brief Compute the geometric factors for short-characteristic interpolation.
     *
     * The geometric factors, namely dx, dy, and path, are symmetric under octahedral
     * symmetries. It is assumed that the largest offset is in the k-direction:
     * (|dk| >= |di| and |dk| >= |dj|).
     *
     * @param di Offset in the i-direction
     * @param dj Offset in the j-direction
     * @param dk Offset in the k-direction
     * @return A float3 containing {dx, dy, path} geometric factors.
     */
    __host__ __device__ float3 compute_shortchar_factors(int di, int dj, int dk);

    /* @brief Create a shortchar_info structure from cell coordinates.
     *
     * @param pos Integer coordinates of the cell (di, dj, dk).
     * @return A shortchar_interp_data structure containing the cell's properties.
     */
    __device__ shortchar_info make_shortchar_interpolation_info(const int3 &pos);

    /// Structure of arrays for a short-characteristic interpolation LUT on device.
    struct shortchar_lut {
        uint32_t *__restrict__ offsets = nullptr;
        float *__restrict__ multipliers = nullptr;
        float *__restrict__ dxs = nullptr;
        float *__restrict__ dys = nullptr;
        float *__restrict__ paths = nullptr;
        index4 *__restrict__ indices = nullptr;

        shortchar_lut();
        shortchar_lut(const shortchar_lut &) = default;
        shortchar_lut &operator=(const shortchar_lut &) = default;

        __device__ bool is_set() const {
            return offsets && multipliers && dxs && dys && paths && indices;
        }

        __device__ shortchar_info operator[](size_t i) const {
            return {unpack_offset(offsets[i]),
                    multipliers[i],
                    dxs[i],
                    dys[i],
                    paths[i],
                    indices[i]};
        }
    };

    /* @brief Create a lookup table for all cells in the q-shells up to q_max.
     *
     * @param q_max Maximum q-shell index, included.
     * @return The number of cells in the q-shells up to q_max.
     */
    size_t create_shortchar_interp_lut(int q_max);

    // Array of structure for a short-characteristic interpolation LUT on host.
    using shortchar_entries = std::vector<shortchar_info>;

    /* @brief Copy the short-characteristic interpolation LUT from device to host.
     *
     * @param q_max Maximum q-shell index, included.
     * @return A vector of shortchar_interp_lut::entry structures containing the lookup
     * table data on the host.
     */
    shortchar_entries copy_lut_to_host(int q_max);

    /* @brief Perform short-characteristic interpolation for a given cell.
     *
     * @param info Short-characteristic interpolation information for the cell.
     * @param column_dens Array of column densities for the neighboring cells.
     * @param cross_section Cross-section value for the interpolation.
     * @return Interpolated value based on the geometric factors and column densities.
     */
    __device__ double shortchar_interpolation(
        const shortchar_info &__restrict__ info, const double *__restrict__ column_dens,
        double cross_section
    );

}  // namespace asora
