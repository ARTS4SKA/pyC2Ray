#include "memory.h"
#include "utils.cuh"

#include <cstdint>
#include <iostream>
#include <limits>

namespace asora {

    void device::initialize(unsigned int rank) {
        if (is_initialized()) return;

        // Map MPI rank to available GPUs using modulo and select the device
        int device_count;
        safe_cuda(cudaGetDeviceCount(&device_count));
        if (device_count <= 0)
            throw std::runtime_error("no CUDA capable device is available");

        int gpu_id = static_cast<int>(rank % static_cast<unsigned int>(device_count));
        safe_cuda(cudaSetDevice(gpu_id));

        // TODO: use a dedicated stream

        auto &self = instance();

        // Per-call scratch buffers are served by the stream-ordered allocator where the
        // platform supports it. Leaving _pool null there makes scratch() fall back to
        // plain device allocations.
        int pools_supported = 0;
        safe_cuda(cudaDeviceGetAttribute(
            &pools_supported, cudaDevAttrMemoryPoolsSupported, gpu_id
        ));

        if (pools_supported) {
            safe_cuda(cudaDeviceGetDefaultMemPool(&self._pool, gpu_id));

            // Set maximum threshold to retain everything. Without this the pool hands
            // memory back to the OS at every synchronization point.
            uint64_t threshold = std::numeric_limits<uint64_t>::max();
            safe_cuda(cudaMemPoolSetAttribute(
                self._pool, cudaMemPoolAttrReleaseThreshold, &threshold
            ));
        } else
            std::cerr
                << "Warning: CUDA memory pools not supported on device " << gpu_id
                << "; falling back to plain device allocations for scratch buffers\n";

        setup_luts();

        // is_initialized() flips here, so a throw above leaves the
        // singleton untouched rather than half-initialized.
        self._gpu_id = gpu_id;
    }

    void device::close() { instance().release(); }

    // Runs on the explicit close() path and again when the singleton is destroyed at
    // exit, so it must be idempotent. It must not throw, release failures are not
    // actionable, so their status is discarded rather than checked.
    void device::release() noexcept {
        // Release the resident arrays and wait for any pending stream-ordered
        // frees, so that the pool has nothing in flight left to release.
        _resident = {};
        if (_pool) {
            cudaDeviceSynchronize();
            cudaMemPoolTrimTo(_pool, 0);
            _pool = nullptr;
        }

        // TODO: nothing else to destroy while _stream is the legacy default stream.
        _gpu_id = -1;
    }

    resident_data &device::resident() {
        check_initialized();
        return instance()._resident;
    }

    // Thread-safe singleton by C++11 standard. Construction is lazy, on the
    // first call, so the instance registers for destruction after the CUDA
    // runtime's own statics and is therefore torn down before them.
    device &device::instance() noexcept {
        static device self;
        return self;
    }

    void device::check_initialized(const std::source_location &loc) {
        if (!is_initialized()) {
            auto msg = std::format(
                "device not initialized at {} in {}:{}; call "
                "asora::device::initialize(...) before",
                loc.function_name(), loc.file_name(), loc.line()
            );
            throw std::runtime_error(msg);
        }
    }

}  // namespace asora
