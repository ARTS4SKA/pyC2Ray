#pragma once

#include <cuda_runtime.h>

#include <concepts>
#include <source_location>

namespace asora {

    // Throw exception for CUDA errors.
    void safe_cuda(
        cudaError_t err,
        const std::source_location &loc = std::source_location::current()
    );

    // Namespace for common constants.
    namespace c {

        template <std::floating_point F = double>
        constexpr F pi = F(3.1415926535897932385L);

        template <std::floating_point F = double>
        constexpr F sqrt3 = F(1.7320508075688772L);

        template <std::floating_point F = double>
        constexpr F sqrt2 = F(1.4142135623730951L);

    }  // namespace c

}  // namespace asora
