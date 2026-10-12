#pragma once

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
        struct number_density : base<double> {};    ///< Matter number density
        struct fraction_HII_avg : base<double> {};  ///< Average HII fraction
        struct fraction_HII : base<double> {};      ///< HII fraction at timestep start

    }  // namespace resident_tag

}  // namespace asora
