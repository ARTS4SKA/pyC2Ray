#pragma once

#include <cuda_runtime.h>
#include <cuda/std/array>

#include <concepts>
#include <source_location>

namespace asora {

    // Throw exception for CUDA errors.
    void safe_cuda(
        cudaError_t err,
        const std::source_location &loc = std::source_location::current()
    );

    // Namespace for common constants.
    namespace c {

        template <std::floating_point F = double>
        constexpr F pi = F(3.1415926535897932385L);

        template <std::floating_point F = double>
        constexpr F sqrt3 = F(1.7320508075688772L);

        template <std::floating_point F = double>
        constexpr F sqrt2 = F(1.4142135623730951L);

    }  // namespace c

    // Fortran-type modulo function (C modulo is signed)
    __host__ __device__ int modulo(int a, int b);

    // Flat-array index from 3D (i,j,k) indices
    __device__ int mem_offset(int i, int j, int k, int N);

#if !defined(PERIODIC)
    __device__ bool in_box(const int &i, const int &j, const int &k, const int &N);
#endif

    // Mapping from octahedral shells (q, s) to 3D cartesian coordinates (i, j, k) and
    // back.
    __host__ __device__ cuda::std::array<int, 3> linthrd2cart(int q, int s);
    __host__ __device__ cuda::std::array<int, 2> cart2linthrd(int i, int j, int k);

    // Return the number of cells in the shell.
    __host__ __device__ size_t cells_in_shell(int q);

    // Return the cumulative number of cells up to the shell.
    __host__ __device__ size_t cells_to_shell(int q);

    // Path inside the cell
    __host__ __device__ double path_in_cell(int di, int dj, int dk);

    // Compute the geometric factors of the 4 adjacent cells; dk is the largest delta
    __host__ __device__ cuda::std::array<double, 4> geometric_factors(
        int di, int dj, int dk
    );

    // Short-characteristics interpolator
    class cell_interpolator {
       public:
        __device__ cell_interpolator(int di, int dj, int dk);

        // Interpolate the column density values from the previous cells.
        __device__ double interpolate(
            const cuda::std::array<const double *__restrict__, 3> &coldens, double sigma
        );

       private:
        int _di, _dj, _dk;
        int _q0;
        double _mul;
        cuda::std::array<int, 12> _offsets;
        cuda::std::array<double, 4> _factors;

        // Get the corresponding q-level of the provided offset.
        inline __device__ cuda::std::array<int, 2> get_qlevel(
            int i_off, int j_off, int k_off
        );

        // True if interpolator is pointing at the origin.
        inline __device__ bool is_origin();
    };

}  // namespace asora
