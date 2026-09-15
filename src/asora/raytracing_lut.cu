#include "raytracing_lut.cuh"

#include "memory.h"
#include "utils.cuh"

#include <cassert>
#include <cmath>
#include <cuda/std/array>
#include <format>
#include <ranges>
#include <vector>

#ifdef __CUDA_ARCH__
#define assert_or_throw(condition, pos) assert(condition)
#else
#define assert_or_throw(condition, pos)                              \
    if (!(condition))                                                \
        throw std::runtime_error(                                    \
            std::format("({}, {}, {}) is out of bounds", di, dj, dk) \
        );
#endif

namespace {

    constexpr size_t LUT_SIZE = asora::Q_MAX + 1;
    __device__ __constant__ size_t cells_in_shell_cache[LUT_SIZE];
    __device__ __constant__ size_t cells_to_shell_cache[LUT_SIZE];

    // 2^10 = 1024, possible values in range [-512, 512)
    constexpr uint32_t OFFSET_BITS = 10;
    constexpr uint32_t OFFSET_MASK = (1u << OFFSET_BITS) - 1;

    // Cells in shell q preceding row i.
    __host__ __device__ inline int row_offset(int q, int i) {
        if (i == -q) return 0;

        auto x = 1 + 2 * (q + i) * (q + i) - 2 * q;
        if (i <= 0) return x - 2 * i;
        return x + 2 * i - 4 * i * i;
    }

    __host__ __device__ size_t cart2slot(const int3& pos) {
        auto&& [i, j, k] = pos;
        int q = abs(i) + abs(j) + abs(k);
        if (q == 0) return 0;

        int m = q - abs(i);
        int P = row_offset(q, i);
        int Q = (m == 0 || j == -m) ? 0 : 1 + 2 * (j + m - 1);

        return asora::cells_to_shell(q - 1) + P + Q + (k > 0);
    }

    // Return dx, dy, path
    __device__ double3 compute_geometric_factors(int di, int dj, int dk) {
        assert(std::abs(dk) >= std::abs(di) && std::abs(dk) >= std::abs(dj) && dk != 0);
        auto inv_dk = 1.0 / std::abs(dk);
        auto xi = di * inv_dk;
        auto xj = dj * inv_dk;
        return {
            1.0 - std::abs(xi), 1.0 - std::abs(xj), std::sqrt(1.0 + xi * xi + xj * xj)
        };
    }

    __device__ void fill_lut_entry(
        asora::raytracing_lut& raylut, size_t idx, const int3& pos
    ) {
        // Check that q > 0.
        assert(std::abs(pos.x) + std::abs(pos.y) + std::abs(pos.z) > 0);

        using namespace asora;

        raylut.offsets[idx] = pack_offset(pos);

        auto&& [di, dj, dk] = pos;
        auto ai = std::abs(di);
        auto aj = std::abs(dj);
        auto ak = std::abs(dk);

        if (ai <= 1 && aj <= 1 && ak <= 1)
            raylut.multipliers[idx] = sqrt(static_cast<double>(ai + ak + aj));
        else
            raylut.multipliers[idx] = 1.0;

        // Compute geometric factors and interpolation indices using inverse_lut.
        int si = (di > 0) - (di < 0);
        int sj = (dj > 0) - (dj < 0);
        int sk = (dk > 0) - (dk < 0);

        cuda::std::array<int, 12> shifts;
        double3 dd;
        if (ak >= ai && ak >= aj) {
            shifts = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, 0,  sk,  //
                0,  0,  sk   //
            };
            dd = compute_geometric_factors(di, dj, dk);
        } else if (aj >= ai && aj >= ak) {
            shifts = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, sj, 0,   //
                0,  sj, 0    //
            };
            dd = compute_geometric_factors(di, dk, dj);
        } else {  // if (ai >= aj && ai >= ak)
            shifts = {
                si, sj, sk,  //
                si, 0,  sk,  //
                si, sj, 0,   //
                si, 0,  0    //
            };
            dd = compute_geometric_factors(dj, dk, di);
        }
        raylut.dxs[idx] = dd.x;
        raylut.dys[idx] = dd.y;
        raylut.paths[idx] = dd.z;

        const cuda::std::array<double, 4> weights = {
            (1. - dd.x) * (1. - dd.y), (1. - dd.y) * dd.x, (1. - dd.x) * dd.y,
            dd.x * dd.y
        };

        size_t index = 0;
        auto& indices = raylut.indices[idx];
        auto sx = shifts.data();
#pragma unroll 4
        for (size_t k = 0; k < 4; ++k) {
            if (weights[k] > 0.0)
                index = cart2slot({di - sx[0], dj - sx[1], dk - sx[2]});
            indices[k] = index;
            sx += 3;
        }
    }

}  // namespace

namespace asora {

    // With OFFSET_BITS = 10 and Q_MAX = 512, not every point is representable. The
    // maximum representable point is (511, 511, 511). The minimum representable point
    // is (-512, -512, -512). We use the remaining 2 bits to represent the following 3
    // missing points, which are valid octahedral offsets:
    //
    // di = Q_MAX, dj = 0, dk = 0 -> 11...
    // di = 0, dj = Q_MAX, dk = 0 -> 10...
    // di = 0, dj = 0, dk = Q_MAX -> 01...
    //            everything else -> 00xxxxx
    //
    __host__ __device__ uint32_t pack_offset(const int3& pos) {
        using namespace asora;

        auto&& [di, dj, dk] = pos;

        if (di == Q_MAX) {
            assert_or_throw(dj == 0 && dk == 0, pos);
            return uint32_t(3) << (3 * OFFSET_BITS);
        }

        if (dj == Q_MAX) {
            assert_or_throw(di == 0 && dk == 0, pos);
            return uint32_t(2) << (3 * OFFSET_BITS);
        }

        if (dk == Q_MAX) {
            assert_or_throw(di == 0 && dj == 0, pos);
            return uint32_t(1) << (3 * OFFSET_BITS);
        }

        assert_or_throw(-Q_MAX <= di && di < Q_MAX, pos);
        assert_or_throw(-Q_MAX <= dj && dj < Q_MAX, pos);
        assert_or_throw(-Q_MAX <= dk && dk < Q_MAX, pos);

        // 3 x 10 bits = 30 bits used, 2 bits spare in the uint32.
        auto pi = static_cast<uint32_t>(di + Q_MAX) & OFFSET_MASK;
        auto pj = static_cast<uint32_t>(dj + Q_MAX) & OFFSET_MASK;
        auto pk = static_cast<uint32_t>(dk + Q_MAX) & OFFSET_MASK;
        return (pi << 2 * OFFSET_BITS) | (pj << OFFSET_BITS) | pk;
    }

    __host__ __device__ int3 unpack_offset(uint32_t offset) {
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

    void setup_cells_to_shell_luts() {
        auto fill_and_load_lut = [](auto func, auto& cache) {
            std::array<size_t, LUT_SIZE> host_lut;
            std::ranges::copy(
                std::views::iota(0ul, LUT_SIZE) | std::views::transform(func),
                host_lut.begin()
            );
            safe_cuda(
                cudaMemcpyToSymbol(cache, host_lut.data(), LUT_SIZE * sizeof(size_t))
            );
        };

        fill_and_load_lut(cells_in_shell, cells_in_shell_cache);
        fill_and_load_lut(cells_to_shell, cells_to_shell_cache);
    }

    __host__ __device__ size_t cells_in_shell(int q) {
        // Defined also for negative q to avoid bound checking.
        if (q < 0) return 0;
#ifdef __CUDA_ARCH__
        if (static_cast<size_t>(q) <= Q_MAX) return cells_in_shell_cache[q];
#else
        if (q == 0) return 1;
#endif
        return 4 * q * q + 2;
    }

    __host__ __device__ size_t cells_to_shell(int q) {
        // This formula comes from the series sum of cells_in_shell(p) for p = 0 to q.
        if (q < 0) return 0;
#ifdef __CUDA_ARCH__
        if (static_cast<size_t>(q) <= Q_MAX) return cells_to_shell_cache[q];
#endif
        return (1 + 2 * q) * (3 + 2 * q * (1 + q)) / 3;
    }

    raytracing_lut::raytracing_lut() {
        offsets = device::get(buffer_tag::raylut_offsets).data<uint32_t>();
        multipliers = device::get(buffer_tag::raylut_multipliers).data<double>();
        dxs = device::get(buffer_tag::raylut_dx).data<double>();
        dys = device::get(buffer_tag::raylut_dy).data<double>();
        paths = device::get(buffer_tag::raylut_path).data<double>();
        indices = device::get(buffer_tag::raylut_indices).data<index4>();
    }

    // Performed by a single block for now
    __global__ void fill_lut_kernel(
        raytracing_lut raylut, int q_max, const int3* __restrict__ cells
    ) {
        // q = 0:
        if (threadIdx.x == 0) {
            raylut.offsets[0] = pack_offset({0, 0, 0});
            raylut.multipliers[0] = 1.0;
            raylut.dxs[0] = 0.0;
            raylut.dys[0] = 0.0;
            raylut.paths[0] = 0.5;
            raylut.indices[0] = {0, 0, 0, 0};
        }
        __syncthreads();

        for (int q = 1; q <= q_max; ++q) {
            // Each thread can process multiple cells.
            for (size_t s = cells_to_shell(q - 1) + threadIdx.x; s < cells_to_shell(q);
                 s += blockDim.x) {
                fill_lut_entry(raylut, s, cells[s]);
            }
            __syncthreads();
        }
    }

    size_t create_raytracing_lut(int q_max) {
        if (q_max < 0 || q_max > Q_MAX) {
            throw std::invalid_argument("q_max must be in the range [0, Q_MAX]");
        }

        auto n_cells = asora::cells_to_shell(q_max);

        // Check first if the LUT already exists and has the correct size.
        if (device::contains(buffer_tag::raylut_offsets)) {
            auto existing_size =
                device::get(buffer_tag::raylut_offsets).size<uint32_t>();
            if (existing_size >= n_cells) return n_cells;
        }

        // Create array of cells to process
        std::vector<int3> cells;
        cells.reserve(n_cells);

        for (int q = 0; q <= q_max; ++q) {
            for (int i = -q; i <= q; ++i) {
                auto ai = std::abs(i);
                for (int j = ai - q; j <= q - ai; ++j) {
                    int k = q - ai - abs(j);
                    if (k != 0) cells.push_back({i, j, -k});
                    cells.push_back({i, j, k});
                }
            }
        }

        // Create device buffer and move cells array
        device_buffer cells_buffer(n_cells * sizeof(int3));
        cells_buffer.copyFromHost(cells.data(), n_cells * sizeof(int3));

        // Allocate lut memory on device
        device::ensure<uint32_t>(buffer_tag::raylut_offsets, n_cells);
        device::ensure<double>(buffer_tag::raylut_multipliers, n_cells);
        device::ensure<double>(buffer_tag::raylut_dx, n_cells);
        device::ensure<double>(buffer_tag::raylut_dy, n_cells);
        device::ensure<double>(buffer_tag::raylut_path, n_cells);
        device::ensure<index4>(buffer_tag::raylut_indices, n_cells);

        // Launch kernel to fill lut
        raytracing_lut lut_d{};

        fill_lut_kernel<<<1, 256>>>(lut_d, q_max, cells_buffer.data<int3>());
        safe_cuda(cudaDeviceSynchronize());

        return n_cells;
    }

    raytracing_lut_entries copy_lut_to_host(int q_max) {
        auto offsets = device::get(buffer_tag::raylut_offsets);
        auto multipliers = device::get(buffer_tag::raylut_multipliers);
        auto dxs = device::get(buffer_tag::raylut_dx);
        auto dys = device::get(buffer_tag::raylut_dy);
        auto paths = device::get(buffer_tag::raylut_path);
        auto indices = device::get(buffer_tag::raylut_indices);

        auto n_cells = asora::cells_to_shell(q_max);
        if (n_cells > offsets.size<uint32_t>())
            throw std::runtime_error(
                "Requested a larger LUT than what available. Call "
                "create_raytracing_lut(q_max) with the correct q_max value first"
            );

        std::vector<uint32_t> offsets_h(n_cells);
        std::vector<double> multipliers_h(n_cells);
        std::vector<double> dxs_h(n_cells);
        std::vector<double> dys_h(n_cells);
        std::vector<double> paths_h(n_cells);
        std::vector<index4> indices_h(n_cells);

        offsets.copyToHost(offsets_h.data(), sizeof(uint32_t) * n_cells);
        multipliers.copyToHost(multipliers_h.data(), sizeof(double) * n_cells);
        dxs.copyToHost(dxs_h.data(), sizeof(double) * n_cells);
        dys.copyToHost(dys_h.data(), sizeof(double) * n_cells);
        paths.copyToHost(paths_h.data(), sizeof(double) * n_cells);
        indices.copyToHost(indices_h.data(), sizeof(index4) * n_cells);

        raytracing_lut_entries lut;
        lut.reserve(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            auto&& [di, dj, dk] = unpack_offset(offsets_h[i]);
            lut.push_back(
                {di, dj, dk, multipliers_h[i], dxs_h[i], dys_h[i], paths_h[i],
                 indices_h[i]}
            );
        }

        return lut;
    }

}  // namespace asora
