//
// IpplTypes
//   Typedefs for basic types used throughout IPPL
//

#ifndef IPPL_TYPES_H
#define IPPL_TYPES_H

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda/std/limits>
#endif

#include <cstddef>  // for std::size_t
#include <limits>

namespace ippl {
    namespace detail {
        typedef std::size_t size_type;

        // Returns +inf for `T`. Uses cuda::std::numeric_limits inside CUDA
        // device code (where std::numeric_limits is unavailable) and falls
        // back to std::numeric_limits on host. KOKKOS_IF_ON_DEVICE /
        // KOKKOS_IF_ON_HOST resolves the choice at codegen time per
        // execution context.
        template <typename T>
        KOKKOS_INLINE_FUNCTION T infinity() {
#if defined(KOKKOS_ENABLE_CUDA)
            KOKKOS_IF_ON_DEVICE(return cuda::std::numeric_limits<T>::infinity();)
            KOKKOS_IF_ON_HOST(return std::numeric_limits<T>::infinity();)
#else
            return std::numeric_limits<T>::infinity();
#endif
        }
    }  // namespace detail

}  // namespace ippl

#endif
