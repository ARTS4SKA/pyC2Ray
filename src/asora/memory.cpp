#include "memory.h"
#include "utils.cuh"

#include <format>

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
        if (size() != nbytes)
            throw std::invalid_argument("this device buffer is not large enough");
        safe_cuda(cudaMemcpy(data(), src, nbytes, cudaMemcpyHostToDevice));
    }

    void device_buffer::copyToHost(void *dst, size_t nbytes) const {
        if (size() != nbytes)
            throw std::invalid_argument("the destination buffer is not large enough");
        safe_cuda(cudaMemcpy(dst, data(), nbytes, cudaMemcpyDeviceToHost));
    }

    void memory_pool::activate() { safe_cuda(cudaSetDevice(_gpu_id)); }

    void memory_pool::free() { _pool.clear(); }

    bool memory_pool::is_active() const {
        int cur_id;
        safe_cuda(cudaGetDevice(&cur_id));
        return static_cast<id_t>(cur_id) == _gpu_id;
    }

    void memory_pool::check_active() const {
        if (!is_active())
            throw std::runtime_error(
                std::format("device {} is not selected as the current device", _gpu_id)
            );
    }

    device_buffer &memory_pool::operator[](buffer_tag tag) { return _pool.at(tag); }

    const device_buffer &memory_pool::operator[](buffer_tag tag) const {
        return _pool.at(tag);
    }

    bool memory_pool::contains(buffer_tag tag) const { return _pool.contains(tag); }

    void memory_pool::allocate_or_copy(buffer_tag tag, size_t nbytes, const void *src) {
        activate();

        auto &&[it, success] = _pool.try_emplace(tag, nbytes);

        // Throw if tag exists but no copy requested, otherwise copy data
        if (!success && !src) throw std::runtime_error("tag already in use");
        if (src) it->second.copyFromHost(src, nbytes);
    }

    memory_pool &get_device_pool(memory_pool::id_t id) {
        static std::unordered_map<memory_pool::id_t, memory_pool> devices;

        auto &&[it, success] = devices.emplace(id, memory_pool{id});
        it->second.activate();

        return it->second;
    }

}  // namespace asora
