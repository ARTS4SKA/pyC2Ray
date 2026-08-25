#pragma once

#include <array>
#include <cstdint>
#include <map>
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

    /* @brief Lookup table entry for a given cell in the q-shell.
     *
     * Each entry contains the cell's integer coordinates (di, dj, dk), packed as a
     * single 32bit value, the fractional distances (dx, dy) to the nearest neighbor
     * cells, and the path length from the origin to the cell.
     * The indices array contains the indices of the four nearest neighbor cells in
     * the lookup table.
     */
    struct lut_entry {
        /// Packed cell offset (di, dj, dk)
        uint32_t offset = 0;

        /// Geometric factors.
        double dx = 0.0;
        double dy = 0.0;
        double path = 0.5;

        /// Offset indices for short-characteristic interpolation.
        std::array<uint32_t, 4> indices = {0, 0, 0, 0};

        /// Unpack the offset to (di, dj, dk). See pack_offset() for details.
        std::array<int, 3> dijk() const;
    };

    /// Create a lookup table for all cells in the q-shells up to q_max.
    std::vector<lut_entry> create_lut(int q_max);

}  // namespace asora
