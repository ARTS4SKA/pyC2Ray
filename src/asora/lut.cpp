#include "lut.h"

#include "utils.cuh"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <map>
#include <vector>

namespace {

    struct array_hash {
        array_hash(size_t N) : N(N), N2(N * N) {}

        size_t operator()(const std::array<int, 3>& a) const {
            return N2 * a[0] + N * a[1] + a[2];
        }

       private:
        size_t N;
        size_t N2;
    };

    using inverse_lut_t = std::unordered_map<std::array<int, 3>, size_t, array_hash>;

    asora::lut_entry make_lut_entry(
        int di, int dj, int dk, const inverse_lut_t& inverse_lut
    ) {
        using namespace asora;

        // Boundary checks.
        assert(abs(di) + abs(dj) + abs(dk) > 0);
        assert(abs(di) + abs(dj) + abs(dk) <= Q_MAX);

        lut_entry entry;
        entry.offset = pack_offset(di, dj, dk);

        // Compute geometric factors.
        auto ai = std::abs(di);
        auto aj = std::abs(dj);
        auto ak = std::abs(dk);
        auto max_delta = std::max(ai, std::max(aj, ak));

        entry.dx = abs(std::copysignf(1.0, di) - di / max_delta);
        entry.dy = abs(std::copysignf(1.0, dj) - dj / max_delta);
        entry.path = std::sqrt((di * di + dj * dj + dk * dk) / (max_delta * max_delta));

        // Compute interpolation indices using inverse_lut.
        int si = (di > 0) - (di < 0);
        int sj = (dj > 0) - (dj < 0);
        int sk = (dk > 0) - (dk < 0);

        std::array<int, 12> shifts;
        if (ak >= ai && ak >= aj) {
            shifts = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, 0,  sk,  //
                0,  0,  sk   //
            };
        } else if (aj >= ai && aj >= ak) {
            shifts = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, sj, 0,   //
                0,  sj, 0    //
            };
        } else {  // if (ai >= aj && ai >= ak)
            shifts = {
                si, sj, sk,  //
                si, 0,  sk,  //
                si, sj, 0,   //
                si, 0,  0    //
            };
        }

#if defined(__clang__)
#pragma unroll 4
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (size_t idx = 0, x = 0; idx < 4; ++idx, x += 3) {
            auto it = inverse_lut.find(
                {di - shifts[x + 0], dj - shifts[x + 1], dk - shifts[x + 2]}
            );
            assert(it != inverse_lut.end());
            entry.indices[idx] = it->second;
        }

        return entry;
    }

    // Helper struct to hold cell coordinates and slot index for parallel work.
    struct cell {
        int i;
        int j;
        int k;
        size_t slot;
    };

}  // namespace

namespace asora {

    // With OFFSET_BITS = 10 and Q_MAX = 512, not every point is representable. The
    // maximum representable point is (511, 511, 511). The minimum representable point
    // is (-512, -512, -512). We use the remaining 2 bits to represent the 3 missing
    // points:
    //
    // di = Q_MAX, dj = 0, dk = 0 -> 11...
    // di = 0, dj = Q_MAX, dk = 0 -> 10...
    // di = 0, dj = 0, dk = Q_MAX -> 01...
    //            everything else -> 00xxxxx
    //
    uint32_t pack_offset(int di, int dj, int dk) {
        using namespace asora;

        if (di == Q_MAX) {
            assert(dj == 0 && dk == 0);
            return 3 << (3 * OFFSET_BITS);
        }

        if (dj == Q_MAX) {
            assert(di == 0 && dk == 0);
            return 2 << (3 * OFFSET_BITS);
        }

        if (dk == Q_MAX) {
            assert(di == 0 && dj == 0);
            return 1 << (3 * OFFSET_BITS);
        }

        // 3 x 10 bits = 30 bits used, 2 bits spare in the uint32.
        auto pi = static_cast<uint32_t>(di + Q_MAX) & OFFSET_MASK;
        auto pj = static_cast<uint32_t>(dj + Q_MAX) & OFFSET_MASK;
        auto pk = static_cast<uint32_t>(dk + Q_MAX) & OFFSET_MASK;
        return (pi << 2 * OFFSET_BITS) | (pj << OFFSET_BITS) | pk;
    }

    std::array<int, 3> lut_entry::dijk() const {
        switch (offset >> (3 * OFFSET_BITS)) {
            case 3:
                return {Q_MAX, 0, 0};
            case 2:
                return {0, Q_MAX, 0};
            case 1:
                return {0, 0, Q_MAX};
            default:
                break;
        }
        auto di = static_cast<int>((offset >> 2 * OFFSET_BITS) & OFFSET_MASK) - Q_MAX;
        auto dj = static_cast<int>((offset >> OFFSET_BITS) & OFFSET_MASK) - Q_MAX;
        auto dk = static_cast<int>(offset & OFFSET_MASK) - Q_MAX;
        return {di, dj, dk};
    }

    std::vector<lut_entry> create_lut(int q_max) {
        auto n_cells = asora::cells_to_shell(q_max);

        std::vector<lut_entry> lut;
        lut.reserve(n_cells);

        // Inverse LUT for interpolation indices.
        array_hash hash(2 * q_max + 1);
        inverse_lut_t inverse_lut(n_cells, hash);
        inverse_lut.try_emplace({0, 0, 0}, 0);

        // Add q = 0 entry.
        {
            lut_entry first;
            first.offset = pack_offset(0, 0, 0);
            lut.push_back(std::move(first));
        }

        // Work is parallelized over each q-shell.
        for (int q = 1; q <= q_max; ++q) {
            std::vector<cell> cells;
            cells.reserve(asora::cells_in_shell(q));

            // Collect cell slots.
            auto slot = lut.size();
            for (int i = -q; i <= q; ++i) {
                auto ai = std::abs(i);
                for (int j = ai - q; j <= q - ai; ++j) {
                    int k = q - ai - abs(j);
                    cells.push_back({i, j, -k, slot++});
                    if (k != 0) cells.push_back({i, j, k, slot++});
                }
            }

            // Perform parallel work.
            lut.resize(slot);

#pragma omp parallel for schedule(static)
            for (size_t c = 0; c < cells.size(); ++c) {
                const auto& cell = cells[c];
                lut[cell.slot] = make_lut_entry(cell.i, cell.j, cell.k, inverse_lut);
            }

            // Update inverse LUT.
            for (const auto& cell : cells)
                inverse_lut.try_emplace({cell.i, cell.j, cell.k}, cell.slot);
        }

        return lut;
    }

}  // namespace asora
