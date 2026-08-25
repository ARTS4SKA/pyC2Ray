#include "tests.cuh"

#include "lut.h"
#include "memory.h"
#include "utils.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace asoratest {

    namespace {

        __global__ void cell_interpolator_kernel(
            double *coldens_data, const double *shared_cdens_data
        ) {
            int di = blockIdx.x - gridDim.x / 2;
            int dj = threadIdx.x - blockDim.x / 2;
            int dk = threadIdx.y - blockDim.y / 2;
            auto q0 = abs(di) + abs(dj) + abs(dk);
            cuda::std::array<const double *__restrict__, 3> shared_cdens = {
                shared_cdens_data + asora::cells_to_shell(q0 - 2),
                shared_cdens_data + asora::cells_to_shell(q0 - 3),
                shared_cdens_data + asora::cells_to_shell(q0 - 4)
            };

            auto idx =
                threadIdx.y + blockDim.y * (threadIdx.x + blockDim.x * blockIdx.x);
            coldens_data[idx] =
                asora::cell_interpolator(di, dj, dk).interpolate(shared_cdens, 1.0);
        }

        __global__ void geometric_factors_kernel(double *factors_data) {
            int di = blockIdx.x - gridDim.x / 2;
            int dj = threadIdx.x - blockDim.x / 2;
            int dk = threadIdx.y - blockDim.y / 2;

            auto swap = [](int &a, int &b) {
                if (abs(a) > abs(b)) cuda::std::swap(a, b);
            };

            swap(di, dj);
            swap(di, dk);
            swap(dj, dk);

            auto idx =
                threadIdx.y + blockDim.y * (threadIdx.x + blockDim.x * blockIdx.x);
            factors_data += 4 * idx;

            if (dk == 0) {
                factors_data[0] = 0.0;
                factors_data[1] = 0.0;
                factors_data[2] = 0.0;
                factors_data[3] = 0.0;
                return;
            }

            auto factors = asora::geometric_factors(di, dj, dk);
            factors_data[0] = factors[0];
            factors_data[1] = factors[1];
            factors_data[2] = factors[2];
            factors_data[3] = factors[3];
        }

        __global__ void path_in_cell_kernel(double *path_data) {
            int di = blockIdx.x - gridDim.x / 2;
            int dj = threadIdx.x - blockDim.x / 2;
            int dk = threadIdx.y - blockDim.y / 2;

            auto idx =
                threadIdx.y + blockDim.y * (threadIdx.x + blockDim.x * blockIdx.x);
            path_data[idx] = asora::path_in_cell(di, dj, dk);
        }

    }  // namespace

    // Arrays are host pointers:
    void cell_interpolator(
        double *coldens_data, double *dens_data, const std::array<size_t, 3> &shape
    ) {
        asora::setup_luts();
        size_t size =
            std::accumulate(shape.begin(), shape.end(), 1u, std::multiplies<>());

        // Create an array that simulates a column density octahedral array:
        auto n_max = *std::max_element(shape.begin(), shape.end());
        // Size of the octahedron
        size_t dens_size = asora::cells_to_shell(int(std::ceil(1.5 * n_max)));
        std::vector<double> dens_data_vec(dens_size, 0.0);
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    int di = i - shape[0] / 2;
                    int dj = j - shape[1] / 2;
                    int dk = k - shape[2] / 2;
                    auto &&[q, s] = asora::cart2linthrd(di, dj, dk);
                    auto i_off = k + shape[2] * (j + shape[1] * i);
                    auto q_off = asora::cells_to_shell(q - 1) + s;
                    dens_data_vec[q_off] = dens_data[i_off];
                }
            }
        }

        // This is the input:
        asora::device_buffer shared_cdens_dev(dens_size * sizeof(double));
        shared_cdens_dev.copyFromHost(dens_data_vec.data());

        // This is the output:
        asora::device_buffer coldens_dev(size * sizeof(double));

        uint3 gs = {static_cast<unsigned int>(shape[0]), 1, 1};
        uint3 ts = {
            static_cast<unsigned int>(shape[1]), static_cast<unsigned int>(shape[2]), 1
        };

        cell_interpolator_kernel<<<gs, ts>>>(
            coldens_dev.view<double>().data(),  //
            shared_cdens_dev.view<double>().data()
        );

        asora::safe_cuda(cudaPeekAtLastError());
        coldens_dev.copyToHost(coldens_data, coldens_dev.size());
    }

    // Arrays are host pointers:
    void geometric_factors(double *fact_data, const std::array<size_t, 3> &shape) {
        size_t size =
            4 * std::accumulate(
                    shape.begin(), shape.end(), sizeof(double), std::multiplies<>()
                );
        asora::device_buffer fact_dev(size);

        uint3 gs = {static_cast<unsigned int>(shape[0]), 1, 1};
        uint3 ts = {
            static_cast<unsigned int>(shape[1]), static_cast<unsigned int>(shape[2]), 1
        };

        geometric_factors_kernel<<<gs, ts>>>(fact_dev.view<double>().data());

        asora::safe_cuda(cudaPeekAtLastError());
        fact_dev.copyToHost(fact_data, fact_dev.size());
    }

    // Arrays are host pointers:
    void path_in_cell(double *path_data, const std::array<size_t, 3> &shape) {
        size_t size = std::accumulate(
            shape.begin(), shape.end(), sizeof(double), std::multiplies<>()
        );
        asora::device_buffer path_dev(size);

        uint3 gs = {static_cast<unsigned int>(shape[0]), 1, 1};
        uint3 ts = {
            static_cast<unsigned int>(shape[1]), static_cast<unsigned int>(shape[2]), 1
        };

        path_in_cell_kernel<<<gs, ts>>>(path_dev.view<double>().data());

        asora::safe_cuda(cudaPeekAtLastError());
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

    std::vector<asora::lut_entry> lut_edge_cases() {
        using namespace asora;

        lut_entry entry;
        std::vector<lut_entry> lut;

        entry.offset = pack_offset(Q_MAX, 0, 0);
        lut.push_back(entry);
        entry.offset = pack_offset(0, Q_MAX, 0);
        lut.push_back(entry);
        entry.offset = pack_offset(0, 0, Q_MAX);
        lut.push_back(entry);
        entry.offset = pack_offset(-Q_MAX, 0, 0);
        lut.push_back(entry);
        entry.offset = pack_offset(0, -Q_MAX, 0);
        lut.push_back(entry);
        entry.offset = pack_offset(0, 0, -Q_MAX);
        lut.push_back(entry);

        return lut;
    }

}  // namespace asoratest
