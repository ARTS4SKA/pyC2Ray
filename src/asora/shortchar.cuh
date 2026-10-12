#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <vector>

namespace asora {

    constexpr size_t num_banks = 4;

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
    __device__ shortchar_info
    make_shortchar_interpolation_info(int q, int s, size_t bank_size);

    /// Structure of arrays for a short-characteristic interpolation LUT on device.
    struct shortchar_lut {
        shortchar_lut() = default;
        shortchar_lut(const shortchar_lut &) = default;
        shortchar_lut &operator=(const shortchar_lut &) = default;

        /// Check if the LUT has been assigned and is ready for use.
        __host__ __device__ bool is_set() const { return _assigned; }

        /// Get the short-characteristic parameters.
        __device__ shortchar_info get(size_t index) const {
            return {
                unpack_offset(_offsets[index]), _dxs[index], _dys[index], _paths[index],
                _indices[index]
            };
        }

        /// Set the short-characteristic parameters.
        __device__ void set(size_t index, const shortchar_info &info) {
            _offsets[index] = pack_offset(info.pos);
            _dxs[index] = info.dx;
            _dys[index] = info.dy;
            _paths[index] = info.path;
            _indices[index] = info.indices;
        }

       private:
        uint32_t *__restrict__ _offsets = nullptr;
        float *__restrict__ _dxs = nullptr;
        float *__restrict__ _dys = nullptr;
        float *__restrict__ _paths = nullptr;
        index4 *__restrict__ _indices = nullptr;

        bool _assigned = false;

        /// Construct a shortchar lut and allocate device memory if needed.
        shortchar_lut(size_t n_cells);

        /// Mark the LUT as ready to use.
        void set_assigned() { _assigned = true; }

        /// Helpfer function to create a shortchar_lut for a given q_max.
        friend shortchar_lut create_shortchar_interp_lut(int q_max);
    };

    /* @brief Create a lookup table for all cells in the q-shells up to q_max.
     *
     * @param q_max Maximum q-shell index, included.
     * @return The number of cells in the q-shells up to q_max.
     */
    shortchar_lut create_shortchar_interp_lut(int q_max);

    // Array of structure for a short-characteristic interpolation LUT on host.
    using shortchar_entries = std::vector<shortchar_info>;

    /* @brief Copy the short-characteristic interpolation LUT from device to host.
     *
     * The user must call create_shortchar_interp_lut(q_max) first to allocate and fill
     * the LUT on the device.
     *
     * @return A vector of shortchar_interp_lut::entry structures containing the lookup
     * table data on the host.
     */
    shortchar_entries copy_shortchar_lut_to_host();

    __device__ inline float shortchar_multiplier(int di, int dj, int dk) {
        auto ai = std::abs(di);
        auto aj = std::abs(dj);
        auto ak = std::abs(dk);
        if (ai <= 1 && aj <= 1 && ak <= 1)
            return std::sqrt(static_cast<float>(ai + aj + ak));
        return 1.f;
    }

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
