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
            // Clear the error state.
            cudaGetLastError();

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

    __device__ int ravel_index(int i, int j, int k, int N) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

    __device__ bool in_box(const int &i, const int &j, const int &k, const int &N) {
        return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
    }

    __host__ __device__ cuda::std::array<int, 3> linthrd2cart(int q, int s) {
        // Determine if we are in the top (k >= 0) or bottom (k < 0) part of the shell
        auto s_top = (q + 1) * (q + 1) + q * q;
        auto [t, sh] = divmod(s, s_top);
        // If we are in the bottom part, the same formula applies but with q-1
        auto qh = q - t;

        /* These formulae are derived from purely geometric considerations, by rotating
         * the (i, j) plane by 45 degrees and then applying divmod to determine the
         * position within the shell. Using the example in utils.cuh (q = 3, k >= 0):
         *
         *
         *         n = 0   1   2   3
         *               4   5   6
         *         p = 0   1   2   3
         * m   o         0   1   2
         * =   =  i < 0 ---------- j > 0
         * 0   0    |  0   1   2   3 |
         *     1    |    4   5   6   |
         * 1   0    |  7   8   9  10 |
         *     1    |   11  12  13   |
         * 2   0    | 14  15  16  17 |
         *     1    |   18  19  20   |
         * 3   0    | 21  22  23  24 |
         *        j < 0 ---------- i > 0
         */
        auto [m, n] = divmod(sh, 2 * qh + 1);
        auto [o, p] = divmod(n, qh + 1);

        auto i = p + m + o - qh;
        auto j = p - m;

        // Thanks to the relation q = |i| + |j| + |k|, we can derive k.
        auto k = (1 - 2 * t) * (q - abs(i) - abs(j));

        return {i, j, k};
    }

    __host__ __device__ cuda::std::array<int, 2> cart2linthrd(int i, int j, int k) {
        // Starting from q = |i| + |j| + |k|, the formulae of the forward mapping are
        // simply inverted to get s.
        auto q = abs(i) + abs(j) + abs(k);
        auto t = int(k < 0);
        auto qh = q - t;

        auto s_top = (q + 1) * (q + 1) + q * q;
        auto s = s_top * t + (qh + 1) * (qh + i) - qh * j;
        return {q, s};
    }

    __host__ __device__ size_t cells_in_shell(int q) {
        // Defined also for negative q to avoid bound checking.
        if (q < 0) return 0;
        if (q == 0) return 1;
        return 4 * q * q + 2;
    }

    __host__ __device__ size_t cells_to_shell(int q) {
        // This formula comes from the series sum of cells_in_shell(p) for p = 0 to q.
        if (q < 0) return 0;
        return (1 + 2 * q) * (3 + 2 * q * (1 + q)) / 3;
    }

    __host__ __device__ double path_in_cell(int di, int dj, int dk) {
        /* If, for example, Δk > Δj and Δk > Δi, the path length through the cell along
         * the ray direction crossing the cell from the z-plane is:
         *
         * path² = 1 + (Δi² + Δj²) / Δk²
         *
         * The following formula is the generalization independent of the incoming
         * direction.
         */
        if (di == 0 && dj == 0 && dk == 0) return 0.5;
        double di2 = di * di;
        double dj2 = dj * dj;
        double dk2 = dk * dk;
        auto delta_max = max(di2, max(dj2, dk2));
        return sqrt((di2 + dj2 + dk2) / delta_max);
    }

    // dk is the largest delta.
    __host__ __device__ cuda::std::array<double, 4> geometric_factors(
        int di, int dj, int dk
    ) {
        /* The geometric factors are computed as follows (see C2-Ray paper for details):
         * Assuming Δk = k - k0 is the largest offset, the distance along the line of
         * the ray s to the interface of the cell d is:
         *
         * a = (Δk - σk / 2) / Δk,
         *
         * where σk = sign(Δk) is the direction of the ray along k. Then, the crossing
         * point of the ray on the z-plane has the following (x, y) coordinates:
         *
         * xc = a * Δi + i0, yc = a * Δj + j0,
         *
         * where (i0, j0) are the coordinates of the ray origin.
         * Finally the distances from the projection on the z-plane c to the cell's
         * corners are:
         *
         *  Δx = 2 * |xc - (i - σi / 2)|
         *  Δy = 2 * |yc - (j - σj / 2)|
         *
         *  Putting all together, the equations for Δx and Δy simplify considerably:
         *
         *  Δx = |(Δk * σi - Δi * σk) / Δk|
         *  Δy = |(Δk * σj - Δj * σk) / Δk|
         */
        assert(dk != 0 && abs(dk) >= abs(di) && abs(dk) >= abs(dj));
        auto dk_inv = 1.0 / abs(dk);
        auto dx = abs(copysignf(1.0, di) - di * dk_inv);
        auto dy = abs(copysignf(1.0, dj) - dj * dk_inv);

        auto w1 = (1. - dx) * (1. - dy);
        auto w2 = (1. - dy) * dx;
        auto w3 = (1. - dx) * dy;
        auto w4 = dx * dy;

        return {w1, w2, w3, w4};
    }

    __device__ cell_interpolator::cell_interpolator(int di, int dj, int dk)
        : _di(di), _dj(dj), _dk(dk), _q0(0), _mul(1.0) {
        // If the cell is the origin, we can skip all the calculations.
        if (is_origin()) return;

        auto ai = abs(di);
        auto aj = abs(dj);
        auto ak = abs(dk);

        // Determine the octahedral shell index q0 and the path length multiplier for
        // rays close to the origin.
        _q0 = ai + aj + ak;
        // The multiplier is either sqrt(2) or sqrt(3) close to the source, or 1
        // otherwise
        if (ai <= 1 && aj <= 1 && ak <= 1)
            _mul = sqrt(static_cast<double>(ai + ak + aj));

        int si = copysignf(1.0, di);
        int sj = copysignf(1.0, dj);
        int sk = copysignf(1.0, dk);

        // Offset index matrix for geometric factors w_i and cartesian coordinates
        // (i, j, k). The first weight w_0 always corresponds to the cell that is
        // at (Δk - σi, Δk - σj, Δk - σk). The other three weights correspond to
        // adjacent cells depending from which direction the ray crosses the cell:
        // some offsets are turned off. At the same time, swap local variables because
        // geometric_factors(...) assumes dk is the largest delta.
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
        // Given the offset of a neighbor cell, we can compute its octahedral
        // coordinates (q, s) and so determine which column density pointer to use.
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
            // If the weight is zero, the ray does not cross the cell.
            if (w > 0.0) {
                // Compute which cell and so q-shell correspond to the current weight
                // and read the column density.
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
