#include "tests.cuh"

#include "memory.h"
#include "raytracing.cuh"
#include "utils.cuh"

#include <numeric>
#include <vector>

namespace asoratest {

    namespace {

        __global__ void cinterp_gpu_kernel(
            double *coldens_data, double *path_data, const double *dens_data
        ) {
            int di = blockIdx.x - gridDim.x / 2;
            int dj = threadIdx.x - blockDim.x / 2;
            int dk = threadIdx.y - blockDim.y / 2;
            auto q0 = abs(di) + abs(dj) + abs(dk);
            cuda::std::array<const double *, 3> shared_cdens = {
                &dens_data[asora::cells_to_shell(q0 - 2)],
                &dens_data[asora::cells_to_shell(q0 - 3)],
                &dens_data[asora::cells_to_shell(q0 - 4)]
            };
            auto &&[cdens, path] = asora::cinterp_gpu(di, dj, dk, shared_cdens, 1.0);

            auto idx =
                threadIdx.y + blockDim.y * (threadIdx.x + blockDim.x * blockIdx.x);
            coldens_data[idx] = cdens;
            path_data[idx] = path;
        }

    }  // namespace

    // Arrays are host pointers:
    void cinterp_gpu(
        double *coldens_data, double *path_data, double *dens_data,
        const std::array<size_t, 3> &shape
    ) {
        size_t size = std::accumulate(
            shape.begin(), shape.end(), sizeof(double), std::multiplies<>()
        );
        asora::device_buffer coldens_dev(size);
        asora::device_buffer path_dev(size);
        asora::device_buffer dens_dev(size);
        dens_dev.copyFromHost(dens_data, dens_dev.size());

        uint3 gs = {static_cast<unsigned int>(shape[0]), 1, 1};
        uint3 ts = {
            static_cast<unsigned int>(shape[1]), static_cast<unsigned int>(shape[2]), 1
        };

        cinterp_gpu_kernel<<<gs, ts>>>(
            coldens_dev.view<double>().data(),  //
            path_dev.view<double>().data(),     //
            dens_dev.view<double>().data()
        );

        asora::safe_cuda(cudaPeekAtLastError());
        coldens_dev.copyToHost(coldens_data, coldens_dev.size());
        path_dev.copyToHost(path_data, path_dev.size());
    }

    std::array<int, 3> linthrd2cart(int q, int s) {
        auto [i, j, k] = asora::linthrd2cart(q, s);
        return {i, j, k};
    }

    std::array<int, 2> cart2linthrd(int i, int j, int k) {
        auto [q, s] = asora::cart2linthrd(i, j, k);
        return {q, s};
    }

}  // namespace asoratest
