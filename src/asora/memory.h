#pragma once

#include <memory>
#include <source_location>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace asora {

    // Type-erased, shared memory buffer living on the device.
    class device_buffer {
       public:
        device_buffer() = default;
        explicit device_buffer(size_t nbytes);

        device_buffer(const device_buffer &other) = default;
        device_buffer(device_buffer &&other) = default;
        device_buffer &operator=(const device_buffer &other) = default;
        device_buffer &operator=(device_buffer &&other) = default;

        friend void swap(device_buffer &lhs, device_buffer &rhs) noexcept;

        // Return a view on a device array.
        // NOTE: do not access this memory from the host!
        template <typename T = std::byte>
        std::span<T> view() noexcept {
            return {reinterpret_cast<T *>(data()), _nbytes / sizeof(T)};
        }
        template <typename T = std::byte>
        std::span<const T> view() const noexcept {
            return {reinterpret_cast<const T *>(data()), _nbytes / sizeof(T)};
        }

        // Return the array address on the device.
        // NOTE: do not access this memory from the host!
        std::byte *data() { return _ptr.get(); }
        const std::byte *data() const { return _ptr.get(); }

        template <typename T>
        T *data() {
            return view<T>().data();
        }
        template <typename T>
        const T *data() const {
            return view<T>().get();
        }

        // Return the size of the array.
        size_t size() const { return _nbytes; }

        template <typename T>
        size_t size() const {
            return _nbytes;
        }

        // Copy from or to memory on the host.
        void copyFromHost(const void *src);
        void copyToHost(void *dst) const;
        void copyFromHost(const void *src, size_t nbytes);
        void copyToHost(void *dst, size_t nbytes) const;

       private:
        std::shared_ptr<std::byte> _ptr = nullptr;
        size_t _nbytes = 0;
    };

    // Buffer identifiers.
    enum class buffer_tag {
        number_density,
        fraction_HII,
        fraction_HeII,
        fraction_HeIII,
        cross_section_HI,
        cross_section_HeI,
        cross_section_HeII,
        photo_ionization_HI,
        photo_ionization_HeI,
        photo_ionization_HeII,
        photo_heating_HI,
        photo_heating_HeI,
        photo_heating_HeII,
        column_density_HI,
        column_density_HeI,
        column_density_HeII,
        photo_ion_thin_table,
        photo_ion_thick_table,
        photo_heat_thin_table,
        photo_heat_thick_table,
        source_flux,
        source_position,
    };

    // Singleton with static interface that represents a device.
    class device {
       public:
        // Check if device is initialized.
        static bool is_initialized() noexcept { return instance()._gpu_id >= 0; }

        // Throw if device is not initialized.
        static void check_initialized(
            const std::source_location &loc = std::source_location::current()
        );

        // Initialize the device identified by the rank.
        static device &initialize(unsigned int rank);

        // Close the device and delete resources.
        static void close();

        // Add a buffer on the device if it does not exist and copy memory to it.
        template <typename T>
        static void transfer(buffer_tag tag, const T *src, size_t items) {
            instance().allocate_or_copy(tag, items * sizeof(T), src);
        }

        // Add an empty buffer on the device.
        template <typename T>
        static void add(buffer_tag tag, size_t items) {
            instance().allocate_or_copy(tag, items * sizeof(T));
        }

        // Get a buffer from the device.
        static device_buffer &get(buffer_tag tag);

        // Check if the buffer exists.
        static bool contains(buffer_tag tag);

        // Get device ID.
        static int get_device_id() { return instance()._gpu_id; }

       private:
        device() {}
        device(const device &) = delete;
        device &operator=(const device &) = delete;

        // Create the instance at the first call.
        static device &instance() noexcept;

        // Add new buffer with tag and copy, or just copy if tag alread exists.
        // Throw if tag exists but no copy was requested
        void allocate_or_copy(buffer_tag tag, size_t nbytes, const void *src = nullptr);

        // Identify the device.
        int _gpu_id = -1;

        // Memory cache for device buffers.
        std::unordered_map<buffer_tag, device_buffer> _memory_pool;
    };

}  // namespace asora
