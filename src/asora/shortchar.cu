#include "shortchar.cuh"

#include "memory.h"
#include "octahedron.cuh"
#include "utils.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda/std/array>
#include <format>
#include <iostream>
#include <optional>
#include <vector>

namespace {

    // 2^10 = 1024 values per digit, [-512, 512); +512 carries into the next digit.
    constexpr uint32_t OFFSET_BITS = 10;
    constexpr uint32_t OFFSET_MASK = (1u << OFFSET_BITS) - 1;

    /// Maximum allowed q-shell index, included.
    constexpr int Q_MAX = 512;

    __device__ cuda::std::array<float, 4> make_weights(float dx, float dy) {
        // More precise formula for dx,dy = 0 or 1.
        auto ex = 1.f - dx;
        auto ey = 1.f - dy;
        return {dx * dy, ex * dy, dx * ey, ex * ey};
    }

    /// Kept cells on host, laid out as in the LUT.
    struct kept_cells {
        /// Packed offsets, shell by shell, lexicographic within a shell.
        std::vector<uint32_t> offsets;

        /// First index of each shell, plus the total as sentinel.
        std::vector<uint32_t> start;

        /// Largest number of cells kept in a shell.
        size_t bank_size = 0;
    };

    /// Collect the octahedral cells inside the sphere of squared radius radius2 and
    /// inside the domain (only periodic conditions).
    kept_cells find_shortchar_offsets(
        int radius2, [[maybe_unused]] size_t N, int q_max
    ) {
        kept_cells kept;
        kept.start.reserve(q_max + 2);
        kept.start.push_back(0);

        for (int q = 0; q <= q_max; ++q) {
            auto n_cells = asora::cells_in_shell(q);
            size_t n_kept = 0;

            for (size_t s = 0; s < n_cells; ++s) {
                auto pos = asora::shell2cart(q, static_cast<int>(s));

                auto dist2 = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
                if (dist2 > radius2) continue;

                if constexpr (asora::periodic_conditions) {
                    if (!asora::in_domain(int3{}, pos, N)) continue;
                }

                kept.offsets.push_back(asora::pack_offset(pos));
                ++n_kept;
            }

            kept.start.push_back(static_cast<uint32_t>(kept.offsets.size()));
            kept.bank_size = std::max(kept.bank_size, n_kept);
        }

        return kept;
    }

    /// Bank index of a cell for interpolation.
    __device__ size_t bank_index(
        const int3& pos, size_t bank_size,
        const uint32_t* __restrict__ shell_start = nullptr,
        const uint32_t* __restrict__ offsets = nullptr
    ) {
        assert(bool(shell_start) == bool(offsets));
        int2 qs;
        if (!shell_start) {
            qs = asora::cart2shell(pos);
        } else {
            qs.x = abs(pos.x) + abs(pos.y) + abs(pos.z);
            // Binary search where pos sits in the packed offsets.
            // Codes are in lexicographic order within a shell.
            auto key = asora::pack_offset(pos);
            auto lo = shell_start[qs.x];
            auto hi = shell_start[qs.x + 1];
            while (lo < hi) {
                auto mid = lo + (hi - lo) / 2;
                if (offsets[mid] < key)
                    lo = mid + 1;
                else
                    hi = mid;
            }

            assert(lo < shell_start[qs.x + 1] && offsets[lo] == key);
            qs.y = lo - shell_start[qs.x];
        }
        return bank_size * (qs.x % asora::num_banks) + qs.y;
    }

    __device__ asora::shortchar_info make_info(
        const int3& pos, size_t bank_size,
        const uint32_t* __restrict__ shell_start = nullptr,
        const uint32_t* __restrict__ offsets = nullptr
    ) {
        using namespace asora;

        const auto& [di, dj, dk] = pos;
        if (di == 0 && dj == 0 && dk == 0) return {0.f, 0.f, 0.5f, {0, 0, 0, 0}};

        auto ai = std::abs(di);
        auto aj = std::abs(dj);
        auto ak = std::abs(dk);

        assert(ai + aj + ak > 0);

        // Compute geometric factors and interpolation indices.
        int si = (di > 0) - (di < 0);
        int sj = (dj > 0) - (dj < 0);
        int sk = (dk > 0) - (dk < 0);

        cuda::std::array<int, 12> shifts;
        float3 dd;
        if (ak >= ai && ak >= aj) {
            shifts = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, 0,  sk,  //
                0,  0,  sk   //
            };
            dd = compute_shortchar_factors(di, dj, dk);
        } else if (aj >= ai && aj >= ak) {
            shifts = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, sj, 0,   //
                0,  sj, 0    //
            };
            dd = compute_shortchar_factors(di, dk, dj);
        } else {  // if (ai >= aj && ai >= ak)
            shifts = {
                si, sj, sk,  //
                si, 0,  sk,  //
                si, sj, 0,   //
                si, 0,  0    //
            };
            dd = compute_shortchar_factors(dj, dk, di);
        }

        auto weights = make_weights(dd.x, dd.y);
        size_t index = 0;
        index4 indices;
        auto sx = shifts.data();
#pragma unroll 4
        for (size_t k = 0; k < 4; ++k) {
            if (weights[k] > 0.0) {
                const int3 corner{di - sx[0], dj - sx[1], dk - sx[2]};
                index = bank_index(corner, bank_size, shell_start, offsets);
            }
            indices[k] = index;
            sx += 3;
        }

        return {dd.x, dd.y, dd.z, std::move(indices)};
    }

    /* @brief Owner of the last look-up table built.
     *
     * The LUT is a pure function of the squared radius and the box size, so repeating
     * them hands the same one back. This memo holds the LUT and decides whether
     * rebuilding is necessary.
     */
    class lut_memo {
       public:
        /// The table built for this radius and box, if it is still on the device.
        std::optional<asora::shortchar_lut> get(
            int radius2, size_t N, int q_max
        ) const {
            if (radius2 != _radius2 || N != _N || q_max != _lut.q_max() || !alive())
                return std::nullopt;
            return _lut;
        }

        /// Remember a freshly built table.
        void put(int radius2, size_t N, const asora::shortchar_lut& lut) {
            _radius2 = radius2;
            _N = N;
            _lut = lut;
            _offsets = pool_offsets();
        }

       private:
        /// The buffer the pool currently holds the cell offsets in, if any.
        static const uint32_t* pool_offsets() {
            using namespace asora;
            if (!device::contains(buffer_tag::raylut_offsets)) return nullptr;
            return device::get(buffer_tag::raylut_offsets).data<uint32_t>();
        }

        bool alive() const { return _lut.is_set() && _offsets == pool_offsets(); }

        int _radius2 = -1;
        size_t _N = 0;
        const uint32_t* _offsets = nullptr;
        asora::shortchar_lut _lut;
    };

}  // namespace

namespace asora {

    // The offsets are shifted by Q_MAX and summed as base-2^10 digits. Every point of
    // the cube [-Q_MAX, Q_MAX)^3 fits. A coordinate of +Q_MAX overflows its digit
    // and carries into the next:
    //
    // (0, 0, Q_MAX) -> code of (0, 1, -Q_MAX)
    // (0, Q_MAX, 0) -> code of (1, -Q_MAX, 0)
    // (Q_MAX, 0, 0) -> bit 30
    //
    // The two special codes have q = Q_MAX + 1, so they are never part of the
    // octahedron. These codes are unique and lexicographically ordered over the
    // octahedron and the cube.
    __host__ __device__ uint32_t pack_offset(const int3& pos) {
        auto&& [di, dj, dk] = pos;

        constexpr const char* msg = "({}, {}, {}) cannot be packed in 32 bits";

        if (di == Q_MAX || dj == Q_MAX || dk == Q_MAX) {
            // Only the tips of the octahedron reach +Q_MAX.
            check(std::abs(di) + std::abs(dj) + std::abs(dk) == Q_MAX, msg, di, dj, dk);
        } else {
            check(-Q_MAX <= di && di < Q_MAX, msg, di, dj, dk);
            check(-Q_MAX <= dj && dj < Q_MAX, msg, di, dj, dk);
            check(-Q_MAX <= dk && dk < Q_MAX, msg, di, dj, dk);
            // These share their code with a carried tip.
            check(!(di == 0 && dj == 1 && dk == -Q_MAX), msg, di, dj, dk);
            check(!(di == 1 && dj == -Q_MAX && dk == 0), msg, di, dj, dk);
        }

        auto pi = static_cast<uint32_t>(di + Q_MAX);
        auto pj = static_cast<uint32_t>(dj + Q_MAX);
        auto pk = static_cast<uint32_t>(dk + Q_MAX);
        return (pi << 2 * OFFSET_BITS) + (pj << OFFSET_BITS) + pk;
    }

    __host__ __device__ int3 unpack_offset(uint32_t offset) {
        /// Codes of (0, 0, Q_MAX) and (0, Q_MAX, 0), whose +Q_MAX carried over.
        constexpr uint32_t uq_max = static_cast<uint32_t>(Q_MAX);
        constexpr uint32_t K_TIP =
            (uq_max << 2 * OFFSET_BITS) + (uq_max << OFFSET_BITS) + 2 * uq_max;
        constexpr uint32_t J_TIP =
            (uq_max << 2 * OFFSET_BITS) + (2 * uq_max << OFFSET_BITS) + uq_max;

        if (offset == K_TIP) return {0, 0, Q_MAX};
        if (offset == J_TIP) return {0, Q_MAX, 0};

        auto di = static_cast<int>(offset >> 2 * OFFSET_BITS) - Q_MAX;
        auto dj = static_cast<int>((offset >> OFFSET_BITS) & OFFSET_MASK) - Q_MAX;
        auto dk = static_cast<int>(offset & OFFSET_MASK) - Q_MAX;
        return {di, dj, dk};
    }

    // Return dx, dy, path
    __host__ __device__ float3 compute_shortchar_factors(int di, int dj, int dk) {
        check(
            std::abs(dk) >= std::abs(di) && std::abs(dk) >= std::abs(dj) && dk != 0,
            "Invalid short-characteristic offsets: |dk| must be the largest and "
            "non-zero"
        );
        auto inv_dk = 1.f / dk;
        auto xi = std::abs(di * inv_dk);
        auto xj = std::abs(dj * inv_dk);
        return {xi, xj, std::sqrt(1.f + xi * xi + xj * xj)};
    }

    __device__ shortchar_info
    make_shortchar_interpolation_info(const int3& pos, size_t bank_size) {
        return make_info(pos, bank_size);
    }

    // This constructor is private.
    shortchar_lut::shortchar_lut(size_t n_cells, int q_max) : _q_max{q_max} {
        device::ensure<uint32_t>(buffer_tag::raylut_shells, q_max + 2);
        device::ensure<uint32_t>(buffer_tag::raylut_offsets, n_cells);
        device::ensure<float>(buffer_tag::raylut_dx, n_cells);
        device::ensure<float>(buffer_tag::raylut_dy, n_cells);
        device::ensure<float>(buffer_tag::raylut_path, n_cells);
        device::ensure<index4>(buffer_tag::raylut_indices, n_cells);

        _shells = device::get(buffer_tag::raylut_shells).data<uint32_t>();
        _offsets = device::get(buffer_tag::raylut_offsets).data<uint32_t>();
        _dxs = device::get(buffer_tag::raylut_dx).data<float>();
        _dys = device::get(buffer_tag::raylut_dy).data<float>();
        _paths = device::get(buffer_tag::raylut_path).data<float>();
        _indices = device::get(buffer_tag::raylut_indices).data<index4>();
    }

    /* Differently from raytracing, the LUT fill has no ordering constraints
     * between shells: every entry is written exactly once and never read back.
     */
    __global__ void fill_lut_kernel(
        shortchar_lut sclut, size_t n_cells, size_t bank_size,
        const uint32_t* __restrict__ shell_start, const uint32_t* __restrict__ offsets
    ) {
        auto start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;

        for (auto index = start; index < n_cells; index += stride) {
            auto pos = sclut.offset(index);
            auto info = make_info(pos, bank_size, shell_start, offsets);
            sclut.set(index, info);
        }
    }

    shortchar_lut create_shortchar_interp_lut(double R, size_t N) {
        auto q_max = get_q_max(R, N);

        try {
            if (q_max < 0 || q_max > Q_MAX)
                throw std::invalid_argument("q_max must be in the range [0, Q_MAX]");
            // Same integer threshold as the raytracing kernel.
            auto radius2 = static_cast<int>(R * R);

            // The table only depends on radius2, N and q_max, so an identical one that
            // is still on the device is handed straight back.
            static lut_memo memo;
            if (auto cached = memo.get(radius2, N, q_max)) return *cached;

            auto&& [offsets, start, bank_size] =
                find_shortchar_offsets(radius2, N, q_max);
            auto n_cells = offsets.size();

            shortchar_lut lut_d(n_cells, q_max);
            lut_d._bank_size = bank_size;

            // The offsets are uploaded directly: the fill only computes the parameters.
            device::get(buffer_tag::raylut_shells)
                .copyFromHost(start.data(), sizeof(uint32_t) * start.size());
            device::get(buffer_tag::raylut_offsets)
                .copyFromHost(offsets.data(), sizeof(uint32_t) * n_cells);

#ifdef NDEBUG
            constexpr int block_size = 1024;
#else
            constexpr int block_size = 128;
#endif
            // Launch kernel to fill lut
            auto grid_size = (n_cells + block_size - 1) / block_size;
            fill_lut_kernel<<<grid_size, block_size>>>(
                lut_d, n_cells, lut_d._bank_size, lut_d._shells, lut_d._offsets
            );
            safe_cuda(cudaDeviceSynchronize());
            safe_cuda(cudaGetLastError());

            // Mark the LUT as usable.
            lut_d.set_assigned();
            memo.put(radius2, N, lut_d);
            return lut_d;
        } catch (const std::exception& e) {
            std::cerr << "Error creating raytracing lookup table: " << e.what()
                      << "; calculating cinterp on the fly\n";
            // Clear memory
            device::clear(buffer_tag::raylut_shells);
            device::clear(buffer_tag::raylut_offsets);
            device::clear(buffer_tag::raylut_dx);
            device::clear(buffer_tag::raylut_dy);
            device::clear(buffer_tag::raylut_path);
            device::clear(buffer_tag::raylut_indices);
        }

        // Unset LUT over the full shells: the parameters are computed on the fly.
        shortchar_lut lut;
        lut._q_max = q_max;
        lut._bank_size = cells_in_shell(q_max);
        return lut;
    }

    shortchar_entries copy_shortchar_lut_to_host() {
        if (!device::contains(buffer_tag::raylut_offsets)) {
            throw std::runtime_error(
                "LUT not found on device. Call create_shortchar_interp_lut(R, N) first"
            );
        }

        auto offsets = device::get(buffer_tag::raylut_offsets);
        auto dxs = device::get(buffer_tag::raylut_dx);
        auto dys = device::get(buffer_tag::raylut_dy);
        auto paths = device::get(buffer_tag::raylut_path);
        auto indices = device::get(buffer_tag::raylut_indices);

        auto n_cells = offsets.size<uint32_t>();

        std::vector<uint32_t> offsets_h(n_cells);
        std::vector<float> dxs_h(n_cells);
        std::vector<float> dys_h(n_cells);
        std::vector<float> paths_h(n_cells);
        std::vector<index4> indices_h(n_cells);

        offsets.copyToHost(offsets_h.data(), sizeof(uint32_t) * n_cells);
        dxs.copyToHost(dxs_h.data(), sizeof(float) * n_cells);
        dys.copyToHost(dys_h.data(), sizeof(float) * n_cells);
        paths.copyToHost(paths_h.data(), sizeof(float) * n_cells);
        indices.copyToHost(indices_h.data(), sizeof(index4) * n_cells);

        shortchar_entries lut;
        lut.reserve(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            auto&& [di, dj, dk] = unpack_offset(offsets_h[i]);
            lut.push_back(
                {{di, dj, dk}, {dxs_h[i], dys_h[i], paths_h[i], indices_h[i]}}
            );
        }

        return lut;
    }

    __device__ double shortchar_interpolation(
        const shortchar_info& __restrict__ info, const double* __restrict__ column_dens,
        double cross_section
    ) {
        // Reference optical depth from C2-Ray interpolation function.
        constexpr float tau_0 = 0.6f;

        // Column density at the crossing point is a weighted average.
        auto weights = make_weights(info.dx, info.dy);
        double cdens = 0.0;
        double wtot = 0.0;
#pragma unroll 4
        for (size_t i = 0; i < 4; ++i) {
            auto c = column_dens[info.indices[i]];
            // Fast division: error cancels out in the weighted average.
            auto w = __fdividef(
                weights[i], max(tau_0, static_cast<float>(c * cross_section))
            );

            cdens += w * c;
            wtot += w;
        }

        return cdens / wtot;
    }

}  // namespace asora
