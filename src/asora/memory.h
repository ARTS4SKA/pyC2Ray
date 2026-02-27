#pragma once

#include <memory>
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
        number_density,          ///< Matter number density
        fraction_HII,            ///< Hydrogen II fraction
        fraction_HeII,           ///< Helium II fraction
        fraction_HeIII,          ///< Helium III fraction
        cross_section_HI,        ///< Ionization cross-section for HI
        cross_section_HeI,       ///< Ionization cross-section for HeI
        cross_section_HeII,      ///< Ionization cross-section for HeII
        photo_ionization_HI,     ///< Photoionization rates for HI
        photo_ionization_HeI,    ///< Photoionization rates for HeI
        photo_ionization_HeII,   ///< Photoionization rates for HeII
        photo_heating_HI,        ///< Photoheating rates for HI
        photo_heating_HeI,       ///< Photoheating rates for HeI
        photo_heating_HeII,      ///< Photoheating rates for HeII
        column_density_HI,       ///< Column density along rays for HI
        column_density_HeI,      ///< Column density along rays for HeI
        column_density_HeII,     ///< Column density along rays for HeII
        photo_ion_thin_table,    ///< Lookup table for optically thin photoionization
        photo_ion_thick_table,   ///< Lookup table for optically thick photoionization
        photo_heat_thin_table,   ///< Lookup table for optically thin photoheating
        photo_heat_thick_table,  ///< Lookup table for optically thick photoheating
        source_flux,             ///< Source flux array
        source_position,         ///< Source position array
        temperature,             ///< Gas temperature array
        clumping_factor,         ///< Clumping factor array
    };

    /* @brief Class managing memory allocations on a single GPU device.
     *
     * Provides interface to access and manage a device's memory pool.
     * The point of access should be through the device(id) free function which returns
     * a reference a unique memory_pool instance for the requested ID. This class is not
     * thread-safe.
     * @see get_device_pool(unsigned int)
     */
    class memory_pool {
       public:
        using id_t = unsigned int;

        memory_pool(const memory_pool &) = delete;
        memory_pool &operator=(const memory_pool &) = delete;
        memory_pool(memory_pool &&) = default;
        memory_pool &operator=(memory_pool &&) = default;

        /* @brief Set the associated device to be the current one.
         * @throw std::runtime_error if cudaGetDeviceCount and cudaSetDevice fail
         */
        void activate();

        /* @brief Return true if the device associated is not the current one.
         * @throw std::runtime_error if cudaGetCount fails
         */
        bool is_active() const;

        /// Just as is_active() but throws instead.
        void check_active() const;

        /// Clear the memory pool.
        void free();

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
         * @see allocate_or_copy
         */
        template <typename T>
        void transfer(buffer_tag tag, const T *src, size_t items) {
            allocate_or_copy(tag, items * sizeof(T), src);
        }

        /* @brief Allocate an empty buffer on device and add it to the pool.
         *
         * @tparam T Element type
         * @param[in] tag Buffer identifier
         * @param[in] items Number of elements to allocate
         * @see allocate_or_copy
         */
        template <typename T>
        void add(buffer_tag tag, size_t items) {
            allocate_or_copy(tag, items * sizeof(T));
        }

        /* @brief Retrieve a buffer from the device.
         * @param[in] tag Buffer identifier
         * @return Reference to the device buffer
         * @throw std::out_of_range if buffer doesn't exist
         */
        device_buffer &operator[](buffer_tag tag);
        const device_buffer &operator[](buffer_tag tag) const;

        /// Check if the buffer identifier exists in the pool.
        bool contains(buffer_tag tag) const;

        /// Get the CUDA device ID.
        id_t device_id() noexcept { return _gpu_id; }

       private:
        /// The free function get_device_pool(unsigned int) can create memory_pool
        /// instances and activate the associated device.
        friend memory_pool &get_device_pool(id_t id);

        memory_pool(id_t id) : _gpu_id(id) {}

        /* @brief Allocate buffer or copy data to existing buffer.
         * @param[in] tag Buffer identifier
         * @param[in] nbytes Size in bytes
         * @param[in] src Optional host source pointer for copying
         * @throw std::runtime_error if is_initialited() returns false
         * @throw std::runtime_error if tag exists but no copy requested
         */
        void allocate_or_copy(buffer_tag tag, size_t nbytes, const void *src = nullptr);

        /// Device ID.
        id_t _gpu_id;

        /// Underlying data structure.
        std::unordered_map<buffer_tag, device_buffer> _pool;
    };

    /* Create and return a unique memory_pool instance associated to the requested
     * device ID. Also call cudaSetDevice for the given device, meaning that following
     * CUDA calls or kernel launches will be executed on that device.
     */
    memory_pool &get_device_pool(memory_pool::id_t id);

}  // namespace asora
