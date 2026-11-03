#pragma once

#include <vector_types.h>

namespace asora {

    /// Typed-keys for resident storage. Construct resident arrays compile time using
    /// these tags, then access them by reference through device::resident<Tag>().
    namespace resident_tag {

        /// Base class for resident tags, carrying the element type.
        template <typename T>
        struct base {
            using type = T;
        };

        /// Common resident tags, used by both the raytracer and chemistry solver.

        /// Per-cell {number density, average HII fraction}. Both are read at the
        /// same scattered cell index, so pairing them into one double2 makes a cell
        /// cost one 32-byte sector instead of two.
        struct density_cell : base<double2> {};

        struct fraction_HII : base<double> {};  ///< HII fraction at timestep start

    }  // namespace resident_tag

}  // namespace asora
