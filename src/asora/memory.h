#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <source_location>
#include <span>
#include <stdexcept>
#include <unordered_map>

/* @file memory.h
 * @brief CUDA device and device memory management for ASORA raytracing library
 *
 * Provides:
 * - RAII wrapper for memory buffers with type-erased storage and typed views.
 * - Enum for identifying different types of buffers in the memory pool.
 * - Singleton class managing device initialization and a memory pool of buffers.
 *
 */

namespace asora {

    /* @brief Type-erased shared memory buffer on the GPU device.
     *
     * Provides RAII-style management of device memory with automatic cleanup.
     * Memory is shared across copies using reference counting (shared_ptr).
     *
     * @warning Do not access the underlying memory from host code!
     */
    class device_buffer {
       public:
        /// Default constructor creates an empty buffer
        device_buffer() = default;

        /// Allocates the given number of bytes and create a device buffer.
        explicit device_buffer(size_t nbytes);

        device_buffer(const device_buffer &other) = default;
        device_buffer(device_buffer &&other) = default;
        device_buffer &operator=(const device_buffer &other) = default;
        device_buffer &operator=(device_buffer &&other) = default;

        /// Swap two device buffers
        friend void swap(device_buffer &lhs, device_buffer &rhs) noexcept;

        /* @brief Get raw pointer to device memory.
         *
         * @return Pointer to device memory
         * @warning Do not dereference this from host code!
         */
        std::byte *data() { return _ptr.get(); }

        /// Const version of data()
        const std::byte *data() const { return _ptr.get(); }

        /* @brief Get a typed view of the device array.
         *
         * @tparam T Element type (default: std::byte)
         * @return Span view over the device memory
         * @warning Do not dereference this from host code!
         */
        template <typename T = std::byte>
        std::span<T> view() noexcept {
            return {reinterpret_cast<T *>(data()), _nbytes / sizeof(T)};
        }

        /// Const version of view()
        template <typename T = std::byte>
        std::span<const T> view() const noexcept {
            return {reinterpret_cast<const T *>(data()), _nbytes / sizeof(T)};
        }

        /* @brief Get a typed pointer to the device array.
         *
         * @tparam T Element type
         * @return Data pointer to the typed view
         * @warning Do not dereference this from host code!
         */
        template <typename T>
        T *data() {
            return view<T>().data();
        }

        /// Const version of data<T>()
        template <typename T>
        const T *data() const {
            return view<T>().data();
        }

        /// Get the size of the buffer in bytes
        size_t size() const { return _nbytes; }

        /* @brief Get the size of the typed view of the device array.
         *
         * @tparam T Element type
         * @return Size of the view
         */
        template <typename T>
        size_t size() const {
            return view<T>().size();
        }

        /* @brief Copy data from host to device.
         * @param[in] src Host memory source pointer
         * @param[in] nbytes Number of bytes to copy
         */
        void copyFromHost(const void *src, size_t nbytes);

        /// Like copyFromHost but all bytes of the buffer are copied
        void copyFromHost(const void *src);

        /*
         * @brief Copy data from device to host.
         * @param[out] dst Host memory destination pointer
         * @param[in] nbytes Number of bytes to copy
         */
        void copyToHost(void *dst, size_t nbytes) const;

        /// Like copyToHost but all bytes of the buffer are copied
        void copyToHost(void *dst) const;

       private:
        /// Shared pointer to device memory
        std::shared_ptr<std::byte> _ptr = nullptr;

        /// Size of the buffer in bytes
        size_t _nbytes = 0;
    };

    /* @brief Identifiers for different types of device buffers.
     *
     * Tags are used to manage named buffers in the device memory pool,
     * allowing type-safe access to simulation data structures.
     */
    enum class buffer_tag {
        number_density,          ///< Matter number density (0)
        fraction_HII,            ///< Hydrogen II fraction (1)
        fraction_HeII,           ///< Helium II fraction (2)
        fraction_HeIII,          ///< Helium III fraction (3)
        cross_section_HI,        ///< Ionization cross-section for HI (4)
        cross_section_HeI,       ///< Ionization cross-section for HeI (5)
        cross_section_HeII,      ///< Ionization cross-section for HeII (6)
        photo_ionization_HI,     ///< Photoionization rates for HI (7)
        photo_ionization_HeI,    ///< Photoionization rates for HeI (8)
        photo_ionization_HeII,   ///< Photoionization rates for HeII (9)
        photo_heating_HI,        ///< Photoheating rates for HI (10)
        photo_heating_HeI,       ///< Photoheating rates for HeI (11)
        photo_heating_HeII,      ///< Photoheating rates for HeII (12)
        column_density_HI,       ///< Column density along rays for HI (13)
        column_density_HeI,      ///< Column density along rays for HeI (14)
        column_density_HeII,     ///< Column density along rays for HeII (15)
        photo_ion_thin_table,    ///< Lookup table for optically thin photoion. (16)
        photo_ion_thick_table,   ///< Lookup table for optically thick photoion. (17)
        photo_heat_thin_table,   ///< Lookup table for optically thin photoheating (18)
        photo_heat_thick_table,  ///< Lookup table for optically thick photoheating (19)
        source_flux,             ///< Source flux array (20)
        source_position,         ///< Source position array (21)
        temperature,             ///< Gas temperature array (22)
        clumping_factor,         ///< Clumping factor array (23)
    };

    /* @brief Singleton managing only one GPU device and its memory pool.
     *
     * Provides static interface for device initialization and buffer de/allocation.
     * The singleton pattern ensures thread-safe initialization of the class and safe
     * read-only access of its memory pool, but modifications to the pool and its
     * buffers is not.
     */
    // TODO: make device thread safe!
    class device {
       public:
        /// Check if device has been initialized.
        static bool is_initialized() noexcept { return instance()._gpu_id >= 0; }

        /* @brief Throw exception if device is not initialized.
         * @param[in] loc Source location for error reporting
         * @throw std::runtime_error if device not initialized
         */
        static void check_initialized(
            const std::source_location &loc = std::source_location::current()
        );

        /* @brief Initialize the GPU device.
         * @param[in] rank Device rank/ID to initialize
         * @return Reference to the device instance
         * @throw std::runtime_error if cudaGetDeviceCount and cudaSetDevice fail
         */
        static device &initialize(unsigned int rank);

        /// Close device and release all resources.
        static void close();

        /* @brief Allocate buffer and copy data from host to device.
         *
         * If buffer identifier is not in use, a new buffer device is allocated and
         * added to the pool. Additionally if a source host pointer is provided, its
         * data is copied to the buffer.
         *
         * @tparam T Element type
         * @param[in] tag Buffer identifier
         * @param[in] src Host memory source pointer
         * @param[in] items Number of elements to copy
         */
        template <typename T>
        static void transfer(buffer_tag tag, const T *src, size_t items) {
            instance().allocate_or_copy(tag, items * sizeof(T), src);
        }

        /* @brief Allocate buffer and copy data from host to device.
         *
         * It behaves like 'transfer', except that a new buffer device of the right size
         * is allocated if the existing one is smaller than the required size.
         * If force_matching_size is true, the existing buffer must have exactly the
         * required size, otherwise reallocation is triggered.
         *
         * @tparam T Element type
         * @param[in] tag Buffer identifier
         * @param[in] src Host memory source pointer
         * @param[in] items Number of elements to copy
         * @param[in] force_matching_size If true, enforce exact size match; when false,
         * reuse existing buffers that are large enough and only reallocate when they
         * are too small.
         */
        template <typename T>
        static void ensure_transfer(
            buffer_tag tag, const T *src, size_t items, bool force_matching_size = false
        ) {
            instance().allocate_or_copy(
                tag, items * sizeof(T), src, true, force_matching_size
            );
        }

        /* @brief Allocate an empty buffer on device and add it to the pool.
         *
         * @tparam T Element type
         * @param[in] tag Buffer identifier
         * @param[in] items Number of elements to allocate
         * @throw std::runtime_error if buffer identifier already in use
         */
        template <typename T>
        static void add(buffer_tag tag, size_t items) {
            instance().allocate_or_copy(tag, items * sizeof(T));
        }

        /* @brief Ensure there is a buffer with the given tag and the required size.
         * If force_matching_size is true, the existing buffer must have exactly the
         * required size.
         *
         * @tparam T Element type
         * @param[in] tag Buffer identifier
         * @param[in] items Number of elements to allocate
         * @param[in] force_matching_size If true, the existing buffer must have exactly
         * the required size
         */
        template <typename T>
        static void ensure(
            buffer_tag tag, size_t items, bool force_matching_size = false
        ) {
            instance().allocate_or_copy(
                tag, items * sizeof(T), nullptr, true, force_matching_size
            );
        }

        /* @brief Retrieve a buffer from the device.
         * @param[in] tag Buffer identifier
         * @return Reference to the device buffer
         * @throw std::runtime_error if device not initialized
         * @throw std::out_of_range if buffer doesn't exist
         */
        static device_buffer &get(buffer_tag tag);

        /// Check if the buffer identifier is in use (= a buffer exists in the pool).
        static bool contains(buffer_tag tag);

        /// Get the CUDA device ID, or -1 if not initialized
        static int get_id() noexcept { return instance()._gpu_id; }

       private:
        /// Private constructor to enforce singleton pattern
        device() {}
        device(const device &) = delete;
        device &operator=(const device &) = delete;

        /// Get or create the singleton instance
        static device &instance() noexcept;

        /* @brief Allocate buffer or copy data to existing buffer.
         * @param[in] tag Buffer identifier
         * @param[in] nbytes Size in bytes
         * @param[in] src Optional host source pointer for copying
         * @param[in] ensure Ensure a buffer exists according to sizing policy.
         * @param[in] force_matching_size If ensure is true, enforce
         * exact size match; when false, reuse existing buffers that are large
         * enough and only reallocate when they are too small. Default is false.
         * @throw std::runtime_error if tag exists but no copy requested and
         * ensure is false
         */
        void allocate_or_copy(
            buffer_tag tag, size_t nbytes, const void *src = nullptr,
            bool ensure = false, bool force_matching_size = false
        );

        /// Device ID (-1 means uninitialized)
        int _gpu_id = -1;

        /// Memory pool for device buffers
        std::unordered_map<buffer_tag, device_buffer> _memory_pool;
    };

    /* @brief Set the persistent L2 window with given data.
     *
     * This function sets the persistent L2 window with the provided device memory.
     *
     * @param[in] ptr Pointer to the data to be copied to the persistent L2 window
     * @param[in] nbytes Size of the data to be copied (in bytes)
     * @param[in] stream CUDA stream for asynchronous operations (default: 0)
     * @param[in] hit_ratio Expected hit ratio for the persistent L2 window
     *            (default: 1.0f)
     */
    void enable_persistent_L2_memory(
        void *ptr, size_t nbytes, cudaStream_t stream = 0, float hit_ratio = 1.0f
    );

    /// Template overload for std::span input.
    template <typename T>
    void enable_persistent_L2_memory(
        const std::span<T> &data, cudaStream_t stream = 0, float hit_ratio = 1.0f
    ) {
        enable_persistent_L2_memory(data.data(), data.size_bytes(), stream, hit_ratio);
    }

    /* @brief Disable the persistent L2 window.
     *
     * This function disables the persistent L2 window, releasing any associated
     * resources. It can be called when the persistent L2 window is no longer needed.
     *
     * @param[in] stream CUDA stream for asynchronous operations (default: 0)
     */
    void disable_persistent_L2_memory(cudaStream_t stream = 0);

}  // namespace asora
