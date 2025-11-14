#pragma once

#include "utils.cuh"

namespace asora {

    // Allocate grid memory
    void device_init(int, int, int, int);

    // Deallocate grid memory
    void device_close();

    template <typename T>
    void array_to_device(T *&dst, const T *src, size_t nbytes) {
        try {
            safe_cuda(cudaMalloc(&dst, nbytes));
            safe_cuda(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
        } catch (const std::exception &) {
        }
    }

    // Pointers to device memory
    extern double *cdh_dev;
    extern double *n_dev;
    extern double *x_dev;
    extern double *phi_dev;
    extern double *photo_thin_table_dev;
    extern double *photo_thick_table_dev;
    extern int *src_pos_dev;
    extern double *src_flux_dev;

    // Number of sources done in parallel ("source batch size")
    extern int NUM_SRC_PAR;

}  // namespace asora
