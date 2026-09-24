#pragma once

#include "utils.cuh"

#include <cuda_runtime.h>

#include <format>
#include <memory>
#include <source_location>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

/* @file memory.h
 * @brief CUDA device and device memory management for ASORA raytracing library
 *
 * Provides:
 * - RAII wrapper for typed device arrays, backed either by a plain device
 *   allocation or by the stream-ordered memory pool.
 * - Aggregate holding the arrays whose contents must outlive a single library
 *   call.
 * - Singleton class managing device initialization, the memory pool and the
 *   stream that orders pool operations.
 *
 */

namespace asora {

    /* @brief Backing allocator of a device array.
     *
     * - 'device' uses cudaMalloc/cudaFree and is meant for long-lived data: such
     * allocations would otherwise pin memory in the middle of the pool for the whole
     * run and defeat its trimming.
     * - 'pooled' uses cudaMallocAsync/cudaFreeAsync and is meant for the per-call
     * working set, which is allocated and released on every entry point and is
     * therefore served from cached pool memory.
     */
    enum class allocation {
        device,  ///< cudaMalloc / cudaFree
        pooled,  ///< cudaMallocAsync / cudaFreeAsync, stream-ordered
    };

    /* @brief RAII-managed typed array in device memory.
     *
     * The array is the unique owner of its allocation, which is released when
     * it goes out of scope. It is movable but not copyable: observers take a
     * raw pointer through data(), and arrays that several subsystems need to
     * read live in resident_data. A pooled array records the stream its
     * allocation is ordered on; it must only be accessed by work ordered on
     * that same stream.
     *
     * @tparam T Element type
     * @warning Do not access the underlying memory from host code!
     */
    template <typename T>
    class device_array {
        static_assert(
            std::is_trivially_copyable_v<T>,
            "device_array elements must be trivially copyable"
        );

        /* @brief Releases an allocation through the allocator that made it.
         *
         * The allocator and stream are held here rather than in the enclosing
         * array so that the deleter never depends on the lifetime of that
         * object or of the device singleton. A deleter is installed even when
         * nothing is allocated, so that an empty array still remembers how it
         * is meant to grow.
         */
        struct deleter {
            allocation alloc = allocation::device;
            cudaStream_t stream = nullptr;

            /// Deleters must not throw, so the release status is discarded.
            void operator()(T *ptr) const noexcept {
                if (alloc == allocation::pooled)
                    cudaFreeAsync(ptr, stream);
                else
                    cudaFree(ptr);
            }
        };

       public:
        using value_type = T;

        /// Default constructor creates an empty array
        device_array() = default;

        /* @brief Allocate `items` elements with cudaMalloc.
         * @param[in] items Number of elements
         */
        explicit device_array(size_t items)
            : _ptr(nullptr, deleter{allocation::device, nullptr}), _items(items) {
            if (_items == 0) return;
            allocate();
        }

        /* @brief Allocate `items` elements from the stream-ordered memory pool.
         * @param[in] items Number of elements
         * @param[in] stream Stream the allocation and its release are ordered on
         */
        device_array(size_t items, cudaStream_t stream)
            : _ptr(nullptr, deleter{allocation::pooled, stream}), _items(items) {
            if (_items == 0) return;
            allocate();
        }

        /// Device arrays own their allocation uniquely and cannot be copied.
        device_array(const device_array &other) = delete;
        device_array &operator=(const device_array &other) = delete;

        /* Moving transfers the allocation and leaves the source empty. The
         * element count is reset explicitly so that a moved-from array does not
         * report a size it no longer backs.
         */
        device_array(device_array &&other) noexcept
            : _ptr(std::move(other._ptr)), _items(std::exchange(other._items, 0)) {}

        device_array &operator=(device_array &&other) noexcept {
            if (this != &other) {
                _ptr = std::move(other._ptr);
                _items = std::exchange(other._items, 0);
            }
            return *this;
        }

        /// True if the array owns an allocation
        explicit operator bool() const noexcept { return bool(_ptr); }

        /* @brief Get raw pointer to device memory.
         *
         * @return Pointer to device memory
         * @warning Do not dereference this from host code!
         */
        T *data() noexcept { return _ptr.get(); }

        /// Const version of data()
        const T *data() const noexcept { return _ptr.get(); }

        /* @brief Get a view of the device array.
         *
         * @return Span view over the device memory
         * @warning Do not dereference this from host code!
         */
        std::span<T> view() noexcept { return {data(), _items}; }

        /// Const version of view()
        std::span<const T> view() const noexcept { return {data(), _items}; }

        /// Get the number of elements in the array
        size_t size() const noexcept { return _items; }

        /// Get the size of the array in bytes
        size_t bytes() const noexcept { return _items * sizeof(T); }

        /// Get the stream this array is ordered on, or 0 for a plain allocation
        cudaStream_t stream() const noexcept { return _ptr.get_deleter().stream; }

        /// Get the backing allocator of this array
        allocation allocator() const noexcept { return _ptr.get_deleter().alloc; }

        /* @brief Ensure the array can hold `items` elements.
         *
         * Reallocation keeps the backing allocator and stream of the current
         * array. A default-constructed array therefore grows into a plain
         * device allocation.
         *
         * @param[in] items Required number of elements
         * @param[in] exact If true the size must match exactly, otherwise a
         * larger existing array is left untouched.
         * @warning Reallocation does not preserve the existing contents.
         */
        void ensure(size_t items, bool exact = false) {
            if (exact ? (_items == items) : (_items >= items)) return;
            // Read the allocator out before assigning: the replacement is built
            // from the deleter of the array it is about to replace.
            auto [alloc, stream] = _ptr.get_deleter();
            *this = (alloc == allocation::pooled) ? device_array(items, stream)
                                                  : device_array(items);
        }

        /* @brief Copy data from host to device.
         * @param[in] src Host memory source pointer
         * @param[in] items Number of elements to copy
         * @throw std::invalid_argument if the array is too small
         */
        void copy_from_host(const T *src, size_t items) {
            check_fits(items, "copy_from_host");
            safe_cuda(
                cudaMemcpy(data(), src, items * sizeof(T), cudaMemcpyHostToDevice)
            );
        }

        /// Like copy_from_host but all elements of the array are copied
        void copy_from_host(const T *src) { copy_from_host(src, _items); }

        /* @brief Copy data from device to host.
         * @param[out] dst Host memory destination pointer
         * @param[in] items Number of elements to copy
         * @throw std::invalid_argument if the array is too small
         */
        void copy_to_host(T *dst, size_t items) const {
            check_fits(items, "copy_to_host");
            safe_cuda(
                cudaMemcpy(dst, data(), items * sizeof(T), cudaMemcpyDeviceToHost)
            );
        }

        /// Like copy_to_host but all elements of the array are copied
        void copy_to_host(T *dst) const { copy_to_host(dst, _items); }

        /* @brief Resize if needed, then copy data from host to device.
         * @param[in] src Host memory source pointer
         * @param[in] items Number of elements to copy
         * @param[in] exact Forwarded to ensure()
         */
        void assign(const T *src, size_t items, bool exact = false) {
            ensure(items, exact);
            copy_from_host(src, items);
        }

       private:
        /// Allocate _items elements through the allocator recorded in the deleter
        void allocate() {
            auto [alloc, stream] = _ptr.get_deleter();
            auto nbytes = _items * sizeof(T);

            T *ptr = nullptr;
            if (alloc == allocation::pooled)
                safe_cuda(cudaMallocAsync(&ptr, nbytes, stream));
            else
                safe_cuda(cudaMalloc(&ptr, nbytes));

            // reset() keeps the deleter installed by the constructor.
            _ptr.reset(ptr);
        }

        /// Throw if the array cannot hold `items` elements
        void check_fits(size_t items, const char *what) const {
            if (_items >= items) return;
            throw std::invalid_argument(
                std::format(
                    "{} size mismatch: device array holds {} elements, requested {}",
                    what, _items, items
                )
            );
        }

        /// Owning pointer to device memory; its deleter carries the allocator
        std::unique_ptr<T, deleter> _ptr = nullptr;

        /// Number of elements
        size_t _items = 0;
    };

    /* @brief Device arrays whose contents outlive a single library call.
     *
     * These are pushed from Python and read back by the raytracing and
     * chemistry entry points without being re-uploaded, so they are kept in
     * plain device allocations rather than in the memory pool. Everything else
     * is per-call scratch and should be a local device_array obtained from
     * device::scratch.
     */
    struct resident_data {
        device_array<double> number_density;   ///< Matter number density
        device_array<double> photo_ion_thin;   ///< Optically thin photoion. table
        device_array<double> photo_ion_thick;  ///< Optically thick photoion. table
        device_array<double> source_flux;      ///< Source flux array
        device_array<int> source_position;     ///< Source position array
    };

    /* @brief Singleton managing only one GPU device, its memory pool and stream.
     *
     * The singleton pattern ensures thread-safe initialization of the class and
     * safe read-only access to its resident data, but modifications to that
     * data are not thread safe.
     */
    // TODO: make device thread safe!
    class device {
       public:
        /// Check if device has been initialized.
        static bool is_initialized() noexcept { return instance()._gpu_id >= 0; }

        /* @brief Throw exception if device is not initialized.
         *
         * @param[in] loc Source location for error reporting
         *
         * @throw std::runtime_error if device not initialized
         */
        static void check_initialized(
            const std::source_location &loc = std::source_location::current()
        );

        /* @brief Initialize the GPU device and its memory pool.
         *
         * @param[in] rank Device rank/ID to initialize
         *
         * @throw std::runtime_error if no device is available or if any CUDA call fails
         */
        static void initialize(unsigned int rank);

        /* @brief Release resident data, trim the memory pool and close the device.
         *
         * The device can be initialized again afterwards. This is also run when
         * the singleton is destroyed, so a process that forgets to close still
         * gives its device memory back before exiting.
         */
        static void close();

        /* @brief Access the arrays that outlive a single library call.
         *
         * @return Reference to the resident data
         *
         * @throw std::runtime_error if device not initialized
         */
        static resident_data &resident();

        /* @brief Allocate a per-call array from the stream-ordered memory pool.
         *
         * Falls back to a plain device allocation on platforms without memory
         * pool support, where initialize() leaves the pool null.
         *
         * @tparam T Element type
         * @param[in] items Number of elements
         *
         * @return Device array ordered on the device stream, or an unpooled
         * array if the platform has no memory pool
         * @throw std::runtime_error if device not initialized
         */
        template <typename T>
        static device_array<T> scratch(size_t items) {
            check_initialized();
            auto &self = instance();
            return self._pool ? device_array<T>(items, self._stream)
                              : device_array<T>(items);
        }

        /// Get the CUDA device ID, or -1 if not initialized
        static int device_id() noexcept { return instance()._gpu_id; }

        /* @brief Get the stream that orders all memory pool operations.
         *
         * This is currently the legacy default stream: kernel launches and
         * copies are still synchronous, and allocating from the pool on the
         * legacy stream keeps them correctly ordered without any explicit
         * synchronization. Switching to a dedicated stream is a change to
         * initialize() alone, provided the copies become asynchronous too.
         */
        static cudaStream_t stream() noexcept { return instance()._stream; }

        /* @brief Get the memory pool backing scratch allocations.
         * @return The pool, or null if the device is not initialized or the
         * platform does not support memory pools
         */
        static cudaMemPool_t pool() noexcept { return instance()._pool; }

       private:
        /// Private constructor to enforce singleton pattern
        device() {}
        device(const device &) = delete;
        device &operator=(const device &) = delete;

        /// Releases whatever close() would, as a backstop at process exit
        ~device() { release(); }

        /// Idempotent, non-throwing release of the device resources
        void release() noexcept;

        /// Get or create the singleton instance
        static device &instance() noexcept;

        /// Device ID (-1 means uninitialized)
        int _gpu_id = -1;

        /// Stream ordering pool allocations; 0 is the legacy default stream
        cudaStream_t _stream = 0;

        /// Stream-ordered pool for scratch allocations; null if unsupported
        cudaMemPool_t _pool = nullptr;

        /// Arrays that outlive a single library call
        resident_data _resident;
    };

}  // namespace asora
