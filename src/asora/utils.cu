#include "utils.cuh"

#include <exception>
#include <format>
#include <iostream>

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

    namespace {

        __host__ __device__ cuda::std::array<int, 2> divmod(int x, int y) {
            return {x / y, x % y};
        }

    }  // namespace

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

    // Return the cumulative number of cells up to the shell.
    __host__ __device__ size_t cells_to_shell(int q) {
        if (q < 0) return 0;
        return (1 + 2 * q) * (3 + 2 * q * (1 + q)) / 3;
    }

}  // namespace asora
