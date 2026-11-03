#include "shortchar.cuh"

#include "memory.h"
#include "octahedron.cuh"
#include "utils.cuh"

#include <cassert>
#include <cmath>
#include <cuda/std/array>
#include <format>
#include <optional>

#ifdef __CUDA_ARCH__
#define assert_or_throw(condition, msg) assert(condition)
#else
#define assert_or_throw(condition, msg) \
    if (!(condition)) throw std::runtime_error(msg);
#endif

namespace {

    // 2^10 = 1024, possible values in range [-512, 512)
    constexpr uint32_t OFFSET_BITS = 10;
    constexpr uint32_t OFFSET_MASK = (1u << OFFSET_BITS) - 1;
    /// Maximum allowed q-shell index, included.
    constexpr int Q_MAX = 512;

    __device__ cuda::std::array<float, 4> make_weights(float dx, float dy) {
        auto dxdy = dx * dy;
        return {dxdy, dy - dxdy, dx - dxdy, 1.f - dx - dy + dxdy};
    }

    /* @brief Host-side memo of the last look-up table built.
     *
     * The LUT is a pure function of q_max, so repeating a q_max hands the same one
     * back. This memo holds the LUT and decides whether rebuilding is necessary:
     * closing the device, or a different q_max, makes the pool hand out a different
     * offsets buffer, so comparing that pointer invalidates the memo on its own.
     */
    class lut_memo {
       public:
        /// The table built for this q_max, if it is still on the device.
        std::optional<asora::shortchar_lut> get(int q_max) const {
            if (q_max != _q_max || !alive()) return std::nullopt;
            return _lut;
        }

        /// Remember a freshly built table.
        void put(int q_max, const asora::shortchar_lut& lut) {
            _q_max = q_max;
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

        bool alive() const {
            return _lut.is_set() && _offsets != nullptr && _offsets == pool_offsets();
        }

        int _q_max = -1;
        const uint32_t* _offsets = nullptr;
        asora::shortchar_lut _lut;
    };

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

#ifdef __CUDA_ARCH__
        [[maybe_unused]] constexpr const char* msg = "";
#else
        std::string msg = std::format("({}, {}, {}) is out of bounds", di, dj, dk);
#endif

        if (di == Q_MAX) {
            assert_or_throw(dj == 0 && dk == 0, msg);
            return uint32_t(3) << (3 * OFFSET_BITS);
        }

        if (dj == Q_MAX) {
            assert_or_throw(di == 0 && dk == 0, msg);
            return uint32_t(2) << (3 * OFFSET_BITS);
        }

        if (dk == Q_MAX) {
            assert_or_throw(di == 0 && dj == 0, msg);
            return uint32_t(1) << (3 * OFFSET_BITS);
        }

        assert_or_throw(-Q_MAX <= di && di < Q_MAX, msg);
        assert_or_throw(-Q_MAX <= dj && dj < Q_MAX, msg);
        assert_or_throw(-Q_MAX <= dk && dk < Q_MAX, msg);

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

    // Return dx, dy, path
    __host__ __device__ float3 compute_shortchar_factors(int di, int dj, int dk) {
        assert_or_throw(
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
        const auto& [di, dj, dk] = pos;
        if (di == 0 && dj == 0 && dk == 0) return {0.f, 0.f, 0.5f, {0, 0, 0, 0}};

        auto ai = std::abs(di);
        auto aj = std::abs(dj);
        auto ak = std::abs(dk);

        assert(ai + aj + ak > 0);

        // Compute geometric factors and interpolation indices using inverse_lut.
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
                auto&& [q, s] = cart2shell(di - sx[0], dj - sx[1], dk - sx[2]);
                index = bank_size * (q % num_banks) + s;
            }
            indices[k] = index;
            sx += 3;
        }

        return {dd.x, dd.y, dd.z, std::move(indices)};
    }

    // This constructor if private.
    shortchar_lut::shortchar_lut(size_t n_cells) {
        device::ensure<uint32_t>(buffer_tag::raylut_offsets, n_cells);
        device::ensure<float>(buffer_tag::raylut_dx, n_cells);
        device::ensure<float>(buffer_tag::raylut_dy, n_cells);
        device::ensure<float>(buffer_tag::raylut_path, n_cells);
        device::ensure<index4>(buffer_tag::raylut_indices, n_cells);

        _offsets = device::get(buffer_tag::raylut_offsets).data<uint32_t>();
        _dxs = device::get(buffer_tag::raylut_dx).data<float>();
        _dys = device::get(buffer_tag::raylut_dy).data<float>();
        _paths = device::get(buffer_tag::raylut_path).data<float>();
        _indices = device::get(buffer_tag::raylut_indices).data<index4>();
    }

    /* Every entry is written exactly once and none is ever read back, so the fill has
     * no ordering between shells to respect. That is what the raytracing kernel needs
     * its barrier for - it interpolates shell q from the column densities of the
     * shells below it - and why this one needs none and can run as a flat grid-stride
     * loop over the whole table.
     */
    __global__ void fill_lut_kernel(shortchar_lut sclut, int q_max) {
        auto bank_size = cells_in_shell(q_max);
        auto n_cells = cells_to_shell(q_max);

        auto start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;

        for (auto index = start; index < n_cells; index += stride) {
            auto&& [q, s] = index2shell(index);
            auto pos = shell2cart(q, s);
            auto info = make_shortchar_interpolation_info(pos, bank_size);
            sclut.set(index, pos, info);
        }
    }

    shortchar_lut create_shortchar_interp_lut(int q_max) {
        if (q_max < 0 || q_max > Q_MAX) {
            throw std::invalid_argument("q_max must be in the range [0, Q_MAX]");
        }

        // The table only depends on q_max, so an identical one that is still on the
        // device is handed straight back.
        static lut_memo memo;
        if (auto cached = memo.get(q_max)) {
            return *cached;
        }

        auto n_cells = asora::cells_to_shell(q_max);
        shortchar_lut lut_d(n_cells);

#ifdef NDEBUG
        constexpr int block_size = 1024;
#else
        constexpr int block_size = 128;
#endif
        // Launch kernel to fill lut
        auto grid_size = (n_cells + block_size - 1) / block_size;
        fill_lut_kernel<<<grid_size, block_size>>>(lut_d, q_max);
        safe_cuda(cudaDeviceSynchronize());
        safe_cuda(cudaGetLastError());

        // Mark the LUT as usable.
        lut_d.set_assigned();
        memo.put(q_max, lut_d);
        return lut_d;
    }

    shortchar_entries copy_shortchar_lut_to_host() {
        if (!device::contains(buffer_tag::raylut_offsets)) {
            throw std::runtime_error(
                "LUT not found on device. Call create_shortchar_interp_lut(q_max) first"
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
            auto w = weights[i] / max(tau_0, static_cast<float>(c * cross_section));

            cdens += w * c;
            wtot += w;
        }

        return cdens / wtot;
    }

}  // namespace asora
