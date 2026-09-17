#pragma once

#include <cuda_runtime.h>
#include <cuda/std/array>

#include <algorithm>
#include <cmath>
#include <ranges>

/* @file octahedron.cuh
 * @brief Octahedral coordinate system transformations for ASORA GPU raytracing
 *
 * Provides:
 * - Maximum q-shell index constant for LUTs
 * - Functions to compute number of cells in octahedral shells
 * - Functions to convert between octahedral (q,s) and Cartesian (i,j,k) coordinates
 */

namespace asora {

    /// Get number of cells in octahedral shell q.
    __host__ __device__ inline size_t cells_in_shell(int q) {
        // Defined also for negative q to avoid bound checking.
        if (q < 0) return 0;
        if (q == 0) return 1;
        return 4 * q * q + 2;
    }

    /// Get cumulative number of cells up to and including shell q.
    __host__ __device__ inline size_t cells_to_shell(int q) {
        // This formula comes from the series sum of cells_in_shell(p) for p = 0 to q.
        if (q < 0) return 0;
        return (1 + 2 * q) * (3 + 2 * q * (1 + q)) / 3;
    }

    /* @brief Convert octahedral (q,s) coordinates to Cartesian (i,j,k).
     *
     * The mapping is such that for a given q shell, the s-index enumerates the cells
     * ordered by ravel_index(i, j, k). This should partially help with memory access
     * patterns when processing the cells in a shell.
     * Shell q is the surface of an octahedron, |i| + |j| + |k| = q. Slicing it by i
     * gives flat square rings: layer i is a ring of radius m = q - |i| holding 4m
     * cells, so the layers hold 1, 4, 8, ..., 4q, ..., 8, 4, 1 cells bottom to top.
     * Along a ring j sweeps -m to m and |k| = m - |j| is whatever is left over, so
     * every j gives two cells (k negative, then positive) except the ends j = -m and
     * j = m, where k = 0 and there is only one. Both mappings below are nothing but
     * "which layer" plus "where along the ring". Example, q = 2, layer i = 0 (m = 2,
     * whose ring occupies slots 5 to 12):
     *
     *   s | 5   6   7   8   9  10  11  12
     *   j |-2  -1  -1   0   0   1   1   2
     *   k | 0  -1  +1  -2  +2  -1  +1   0
     *
     *
     * @see cart2shell() for the backward transformation.
     *
     * @param[in] q Shell index (distance from source)
     * @param[in] s Position index within shell q
     * @return Array containing {i, j, k} offsets
     */
    __host__ __device__ inline int3 shell2cart(int q, int s) {
        // The numbering is centrally symmetric: s and 4q^2 + 1 - s hold opposite cells.
        // Folding the upper half down leaves only the lower layers to solve, which are
        // then negated. This also absorbs q == 0.
        int sign = -1;
        if (s >= 2 * q * q + 1) {
            s = 4 * q * q + 1 - s;
            sign = 1;
        }
        // Bottom tip: the lone cell of the m = 0 layer, which no ring formula covers.
        if (s == 0) return {sign * q, 0, 0};

        // The cells preceding the ring of radius m number 1, 5, 13, 25, ... = 2m(m - 1)
        // + 1, so recovering m means inverting a quadratic.
        // The bound 2m(m - 1) + 1 <= s is equivalent to (2m - 1)^2 <= 2s < (2m + 1)^2.
        // The radicand is an exact integer and the truncation can only be off by one:
        // the two correction predicates are mutually exclusive.
        auto m = static_cast<int>(0.5f * (1.f + std::sqrt(2.f * s)));
        assert(m > 0);
        auto low = 2 * m * (m - 1) + 1;
        auto up = low + 4 * m;
        m += (up <= s) - (low > s);

        // r is the position within it a layer/ring, less one: halving says how
        // far j has walked in from the end, and its parity picks which of the two k
        // twins. The offset of one is what lets the ends (a single cell with k = 0)
        // share the expression, since '>> 1' floors towards -inf.
        auto r = s - (2 * m * (m - 1) + 1) - 1;
        auto j = m - 1 - (r >> 1);
        auto k = (r & 1) ? abs(j) - m : m - abs(j);

        return {sign * (q - m), sign * j, sign * k};
    }

    /* @brief Convert Cartesian (i,j,k) coordinates to octahedral (q,s).
     *
     * It counts the layers below and step along the ring, two slots per j plus one more
     * for the positive k twin.
     *
     * @see shell2cart() for the forward transformation.
     *
     * @param[in] i X-offset
     * @param[in] j Y-offset
     * @param[in] k Z-offset
     * @return Array containing {q, s}
     */
    __host__ __device__ inline int2 cart2shell(int i, int j, int k) {
        int q = abs(i) + abs(j) + abs(k);
        if (q == 0) return {0, 0};

        // Cells of shell q in the layers below layer i, i.e. where its ring starts.
        auto row_offset = [](int q, int i) {
            if (i == -q) return 0;

            auto x = 1 + 2 * (q + i) * (q + i) - 2 * q;
            if (i <= 0) return x - 2 * i;
            return x + 2 * i - 4 * i * i;
        };

        int m = q - abs(i);
        int P = row_offset(q, i);
        int Q = (m == 0 || j == -m) ? 0 : 1 + 2 * (j + m - 1);

        int s = P + Q + (k > 0);
        return {q, s};
    }

}  // namespace asora
