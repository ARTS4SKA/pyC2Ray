#pragma once

#include "octahedron.cuh"
#include "resident_tag.h"

#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <vector>

namespace asora {

    constexpr size_t num_banks = 4;

    /* @brief Pack the integer coordinates (di, dj, dk) of a cell in the q-shell into a
     * single 32bit value.
     *
     * The offsets are shifted by Q_MAX to make them non-negative and summed as the
     * digits of a base-2^10 number, which corresponds to the cube [-Q_MAX, Q_MAX)^3.
     * A coordinate of +Q_MAX overflows and carries into the next one, so
     * (0, 0, Q_MAX) is mapped to (0, 1, -Q_MAX) and (0, Q_MAX, 0) to (1, -Q_MAX, 0),
     * which both have q = Q_MAX + 1 and cannot be packed. (Q_MAX, 0, 0) instead
     * overflows to the 30th bit. The packed values of the octahedron are therefore
     * unique and in lexicographic order.
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

    namespace resident_tag::lut {

        struct shells : base<uint32_t> {};   ///< first LUT index of each q-shell
        struct offsets : base<uint32_t> {};  ///< di, dj, dk offsets
        struct dx : base<float> {};          ///< dx weight component
        struct dy : base<float> {};          ///< dy weight component
        struct path : base<float> {};        ///< path length through each cell
        struct indices : base<index4> {};    ///< interpolation indices

    }  // namespace resident_tag::lut

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

    struct shortchar_info {
        /// Geometric factors.
        float dx;
        float dy;
        float path;

        /// Offset indices for short-characteristic interpolation.
        index4 indices;
    };

    /* @brief Create a shortchar_info structure from cell offsets already in hand.
     *
     * The caller must handle the source cell (pos = {0, 0, 0}) separately.
     *
     * @param pos Integer offsets of the cell (di, dj, dk) from the source.
     * @param bank_size Number of cells in the largest q-shell.
     * @return A shortchar_interp_data structure containing the cell's properties.
     */
    __device__ shortchar_info
    make_shortchar_interpolation_info(const int3 &pos, size_t bank_size);

    /// Structure of arrays for a short-characteristic interpolation LUT on device.
    struct shortchar_lut {
        shortchar_lut() = default;
        shortchar_lut(const shortchar_lut &) = default;
        shortchar_lut &operator=(const shortchar_lut &) = default;

        /// Check if the LUT has been assigned and is ready for use.
        __host__ __device__ bool is_set() const { return _assigned; }

        /// Get the cell offsets alone. Enough to reject a cell before paying for
        /// the remaining 28 bytes of the entry.
        __device__ int3 offset(size_t index) const {
            return unpack_offset(_offsets[index]);
        }

        /// Get the interpolation parameters for a cell whose offsets are in hand.
        __device__ shortchar_info params(size_t index) const {
            return {_dxs[index], _dys[index], _paths[index], _indices[index]};
        }

        /// Set the short-characteristic parameters. The offsets are uploaded by the
        /// host beforehand.
        __device__ void set(size_t index, const shortchar_info &info) {
            _dxs[index] = info.dx;
            _dys[index] = info.dy;
            _paths[index] = info.path;
            _indices[index] = info.indices;
        }

        /// First LUT index of shell q. Without a LUT, the full shells are assumed.
        __device__ size_t shell_begin(int q) const {
            return _shells ? _shells[q] : cells_to_shell(q - 1);
        }

        /// Number of cells kept in shell q. Without a LUT, the full shells are assumed.
        __device__ size_t shell_size(int q) const {
            return _shells ? _shells[q + 1] - _shells[q] : cells_in_shell(q);
        }

        /// Stride between column-density banks, i.e. the largest shell kept.
        __host__ __device__ size_t bank_size() const { return _bank_size; }

        /// Outermost shell, included.
        __host__ __device__ int q_max() const { return _q_max; }

       private:
        const uint32_t *__restrict__ _shells = nullptr;
        uint32_t *__restrict__ _offsets = nullptr;
        float *__restrict__ _dxs = nullptr;
        float *__restrict__ _dys = nullptr;
        float *__restrict__ _paths = nullptr;
        index4 *__restrict__ _indices = nullptr;

        size_t _bank_size = 0;
        int _q_max = 0;
        bool _assigned = false;

        /// Construct a shortchar lut and allocate device memory if needed.
        shortchar_lut(size_t n_cells, int q_max);

        /// Mark the LUT as ready to use.
        void set_assigned() { _assigned = true; }

        /// Helper function to create a shortchar_lut for a given radius.
        friend shortchar_lut create_shortchar_interp_lut(double R, size_t N);
    };

    /* @brief Create a lookup table for the cells a source can reach.
     *
     * Only the cells of the octahedron up to get_q_max(R, N) that are inside the sphere
     * of radius R (and inside the domain with periodic conditions) are kept. Shells are
     * contiguous and cells are lexicographically sorted. If the LUT cannot be built,
     * an unset LUT is returned and the interpolation parameters are computed on the
     * fly.
     *
     * @param R Raytracing radius, in cells.
     * @param N Number of cells along a side of the box.
     * @return The LUT, or the one already on the device for the same R and N.
     */
    shortchar_lut create_shortchar_interp_lut(double R, size_t N);

    // Array of structure for a short-characteristic interpolation LUT on host.
    struct shortchar_entry {
        int3 pos;
        shortchar_info info;
    };

    using shortchar_entries = std::vector<shortchar_entry>;

    /* @brief Copy the short-characteristic interpolation LUT from device to host.
     *
     * The user must call create_shortchar_interp_lut(R, N) first to allocate and fill
     * the LUT on the device.
     *
     * @return A vector of shortchar_interp_lut::entry structures containing the lookup
     * table data on the host.
     */
    shortchar_entries copy_shortchar_lut_to_host();

    /// Compute a multiplicative correction term for short-characteristic interpolation.
    __device__ inline float shortchar_multiplier(const int3 &pos) {
        auto ai = abs(pos.x);
        auto aj = abs(pos.y);
        auto ak = abs(pos.z);
        if (ai > 1 || aj > 1 || ak > 1) return 1.f;
        switch (ai + aj + ak) {
            case 2:
                return c::sqrt2<float>;
            case 3:
                return c::sqrt3<float>;
            default:
                return 1.f;
        }
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
