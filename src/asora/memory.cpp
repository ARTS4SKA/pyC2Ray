#include "memory.h"
#include "utils.cuh"

#include <format>
#include <iostream>
#include <span>

namespace asora {

    device_buffer::device_buffer(size_t nbytes) : _nbytes(nbytes) {
        std::byte *ptr;
        safe_cuda(cudaMalloc(&ptr, _nbytes));

        // Custom deleter ensures cudaFree is called on destruction
        // The shared_ptr deleter can't throw, so we ignore any exceptions
        _ptr.reset(ptr, [](std::byte *ptr) {
            try {
                safe_cuda(cudaFree(ptr));
            } catch (const std::exception &) {
            }
        });
    }

    void swap(device_buffer &lhs, device_buffer &rhs) noexcept {
        std::swap(lhs._ptr, rhs._ptr);
        std::swap(lhs._nbytes, rhs._nbytes);
    }

    void device_buffer::copyFromHost(const void *src) {
        safe_cuda(cudaMemcpy(data(), src, size(), cudaMemcpyHostToDevice));
    }

    void device_buffer::copyToHost(void *dst) const {
        safe_cuda(cudaMemcpy(dst, data(), size(), cudaMemcpyDeviceToHost));
    }

    void device_buffer::copyFromHost(const void *src, size_t nbytes) {
        if (size() < nbytes)
            throw std::invalid_argument(
                std::format(
                    "copyFromHost size mismatch: device buffer has {} bytes, requested "
                    "{} bytes",
                    size(), nbytes
                )
            );
        safe_cuda(cudaMemcpy(data(), src, nbytes, cudaMemcpyHostToDevice));
    }

    void device_buffer::copyToHost(void *dst, size_t nbytes) const {
        if (size() < nbytes)
            throw std::invalid_argument(
                std::format(
                    "copyToHost size mismatch: device buffer has {} bytes, requested "
                    "{} bytes",
                    size(), nbytes
                )
            );
        safe_cuda(cudaMemcpy(dst, data(), nbytes, cudaMemcpyDeviceToHost));
    }

    device &device::initialize(unsigned int rank) {
        // TODO: add log
        auto &self = instance();
        if (is_initialized()) return self;

        // Map MPI rank to available GPUs using modulo and select the device
        int device_count;
        safe_cuda(cudaGetDeviceCount(&device_count));
        self._gpu_id = rank % device_count;
        safe_cuda(cudaSetDevice(self._gpu_id));
        setup_luts();
        return self;
    }

    void device::close() {
        auto &self = instance();
        self._gpu_id = -1;
        self._memory_pool.clear();
    }

    device_buffer &device::get(buffer_tag tag) {
        check_initialized();
        return instance()._memory_pool.at(tag);
    }

    bool device::contains(buffer_tag tag) {
        return is_initialized() && instance()._memory_pool.contains(tag);
    }

    // Thread-safe singleton by C++11 standard
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

    void device::allocate_or_copy(
        buffer_tag tag, size_t nbytes, const void *src, bool ensure,
        bool force_matching_size
    ) {
        check_initialized();

        auto &&[it, inserted] = _memory_pool.try_emplace(tag, nbytes);

        if (!inserted) {
            if (ensure) {
                // Ensure-mode is idempotent: keep existing buffer if sizing
                // policy is satisfied, otherwise replace it.
                const bool needs_realloc = force_matching_size
                                               ? (it->second.size() != nbytes)
                                               : (it->second.size() < nbytes);
                if (needs_realloc) {
                    it->second = device_buffer(nbytes);
                }
            } else if (!src) {
                throw std::runtime_error("tag already in use");
            }
        }

        // Copy host data whenever a source pointer is provided.
        if (src) it->second.copyFromHost(src, nbytes);
    }

    void enable_persistent_L2_memory(
        void *ptr, size_t nbytes, cudaStream_t stream, float hit_ratio
    ) {
        // Query GPU properties.
        cudaDeviceProp prop;
        safe_cuda(cudaGetDeviceProperties(&prop, static_cast<int>(device::get_id())));

        if (prop.major < 8) {
            throw std::runtime_error(
                "Persistent L2 memory is only supported with compute capability >= 8.0"
            );
        }

        // CHECK: cudaDeviceSetLimit clamps the size to the maximum allowed value.
        // Set the required size of persisting L2 cache memory.
        size_t size =
            std::min(nbytes, static_cast<size_t>(prop.persistingL2CacheMaxSize));
        safe_cuda(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size));

        // Set access policy window for the desired memory span.
        cudaStreamAttrValue attr;

        attr.accessPolicyWindow.base_ptr = ptr;
        attr.accessPolicyWindow.num_bytes =
            std::min(prop.accessPolicyMaxWindowSize, static_cast<int>(nbytes));
        attr.accessPolicyWindow.hitRatio = hit_ratio;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

        safe_cuda(
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)
        );
    }

    void disable_persistent_L2_memory(cudaStream_t stream) {
        cudaDeviceProp prop;
        safe_cuda(cudaGetDeviceProperties(&prop, static_cast<int>(device::get_id())));

        if (prop.major < 8) return;

        // Disable by setting null window size.
        cudaStreamAttrValue attr;
        safe_cuda(
            cudaStreamGetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)
        );

        attr.accessPolicyWindow.num_bytes = 0;
        safe_cuda(
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)
        );

        // Remove any persistent lines in L2
        safe_cuda(cudaCtxResetPersistingL2Cache());
        safe_cuda(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
    }

}  // namespace asora
