#include "utils.cuh"

#include <cassert>
#include <exception>
#include <format>
#include <iostream>

namespace {

    __host__ __device__ cuda::std::array<int, 2> divmod(int x, int y) {
        return {x / y, x % y};
    }

}  // namespace

namespace asora {

    void safe_cuda(cudaError_t err, const std::source_location &loc) {
        if (err != cudaSuccess) {
            auto msg = std::format(
                "CUDA Error {}: {}. At {} in {}:{}", cudaGetErrorName(err),
                cudaGetErrorString(err), loc.function_name(), loc.file_name(),
                loc.line()
            );
            std::cerr << msg << "\n";
            throw std::runtime_error(msg);
        }
    }

    __host__ __device__ int modulo(int a, int b) { return (a % b + b) % b; }

    __device__ int mem_offset(int i, int j, int k, int N) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

#if !defined(PERIODIC)
    __device__ bool in_box_gpu(const int &i, const int &j, const int &k, const int &N) {
        return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
    }
#endif

    __host__ __device__ cuda::std::array<int, 3> linthrd2cart(int q, int s) {
        auto s_top = (q + 1) * (q + 1) + q * q;
        auto [t, sh] = divmod(s, s_top);
        auto qh = q - t;

        auto [m, n] = divmod(sh, 2 * qh + 1);
        auto [o, p] = divmod(n, qh + 1);

        auto i = p + m + o - qh;
        auto j = p - m;
        auto k = (1 - 2 * t) * (q - abs(i) - abs(j));

        return {i, j, k};
    }

    __host__ __device__ cuda::std::array<int, 2> cart2linthrd(int i, int j, int k) {
        auto q = abs(i) + abs(j) + abs(k);
        auto t = int(k < 0);
        auto qh = q - t;

        auto s_top = (q + 1) * (q + 1) + q * q;
        auto s = s_top * t + (qh + 1) * (qh + i) - qh * j;
        return {q, s};
    }

    __host__ __device__ size_t cells_in_shell(int q) {
        if (q < 0) return 0;
        if (q == 0) return 1;
        return 4 * q * q + 2;
    }

    __host__ __device__ size_t cells_to_shell(int q) {
        if (q < 0) return 0;
        return (1 + 2 * q) * (3 + 2 * q * (1 + q)) / 3;
    }

    __device__ double path_in_cell(int di, int dj, int dk) {
        if (di == 0 && dj == 0 && dk == 0) return 0.5;
        double di2 = di * di;
        double dj2 = dj * dj;
        double dk2 = dk * dk;
        auto delta_max = max(di2, max(dj2, dk2));
        return sqrt((di2 + dj2 + dk2) / delta_max);
    }

    // dk is the largest delta.
    __device__ cuda::std::array<double, 4> geometric_factors(int di, int dj, int dk) {
        assert(dk != 0 && abs(dk) >= abs(di) && abs(dk) >= abs(dj));
        auto dk_inv = 1.0 / abs(dk);
        auto dx = abs(copysign(1.0, static_cast<double>(di)) - di * dk_inv);
        auto dy = abs(copysign(1.0, static_cast<double>(dj)) - dj * dk_inv);

        auto w1 = (1. - dx) * (1. - dy);
        auto w2 = (1. - dy) * dx;
        auto w3 = (1. - dx) * dy;
        auto w4 = dx * dy;

        return {w1, w2, w3, w4};
    }

    __device__ cell_interpolator::cell_interpolator(int di, int dj, int dk)
        : _di(di), _dj(dj), _dk(dk), _q0(0), _mul(1.0) {
        if (is_origin()) return;

        auto ai = abs(di);
        auto aj = abs(dj);
        auto ak = abs(dk);

        _q0 = ai + aj + ak;
        if (ai <= 1 && aj <= 1 && ak <= 1)
            _mul = sqrt(static_cast<double>(ai + ak + aj));

        int si = copysignf(1.0, di);
        int sj = copysignf(1.0, dj);
        int sk = copysignf(1.0, dk);

        // Offset index matrix for geometric factors w_i and cartesian coordinates
        // (i, j, k). Depending on which delta is largest, some offsets are turned
        // off. At the same time, swap local variables
        if (ak >= ai && ak >= aj) {
            _offsets = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, 0,  sk,  //
                0,  0,  sk   //
            };
        } else if (aj >= ai && aj >= ak) {
            _offsets = {
                si, sj, sk,  //
                0,  sj, sk,  //
                si, sj, 0,   //
                0,  sj, 0    //
            };
            cuda::std::swap(dj, dk);
        } else {  // if (ai >= aj && ai >= ak)
            _offsets = {
                si, sj, sk,  //
                si, 0,  sk,  //
                si, sj, 0,   //
                si, 0,  0    //
            };
            cuda::std::swap(di, dk);
            cuda::std::swap(di, dj);
        }

        _factors = geometric_factors(di, dj, dk);
    }

    inline __device__ bool cell_interpolator::is_origin() {
        return _di == 0 && _dj == 0 && _dk == 0;
    }

    inline __device__ cuda::std::array<int, 2> cell_interpolator::get_qlevel(
        int i_off, int j_off, int k_off
    ) {
        auto &&[q, s] = cart2linthrd(_di - i_off, _dj - j_off, _dk - k_off);
        auto qlev = _q0 - q - 1;
        assert(qlev >= 0 && qlev < 3);

        return {qlev, s};
    }

    __device__ double cell_interpolator::interpolate(
        const cuda::std::array<const double *__restrict__, 3> &coldens, double sigma
    ) {
        // Degenerate case.
        if (is_origin()) return 0.0;

        // Reference optical depth from C2-Ray interpolation function.
        constexpr double tau_0 = 0.6;

        // Column density at the crossing point is a weighted average.
        double cdens = 0.0;
        double wtot = 0.0;

        // Loop over geometric factors and skip null ones: it helps avoid some reads.
#pragma unroll
        for (auto xa = _offsets.data(); auto w : _factors) {
            if (w > 0.0) {
                auto &&[qlev, s] = get_qlevel(xa[0], xa[1], xa[2]);
                auto c = coldens[qlev][s];

                // Rescale weight by optical path
                w /= max(tau_0, c * sigma);
                cdens += w * c;
                wtot += w;
            }
            // Access next row of the offset matrix.
            xa += 3;
        }

        // At least one weight was valid.
        assert(wtot > 0.0);

        return _mul * cdens / wtot;
    }

}  // namespace asora
