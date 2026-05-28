#ifndef IPPL_ALVINE_SINUSOIDAL_JITTER_HPP
#define IPPL_ALVINE_SINUSOIDAL_JITTER_HPP

#include <Kokkos_MathematicalFunctions.hpp>

namespace alvine {

KOKKOS_INLINE_FUNCTION
double sinusoidalJitter(double spacing, double xIndex, double yIndex, int component) {
    constexpr double amplitude = 0.1;
    constexpr double kx        = 12.9898;
    constexpr double ky        = 78.233;
    constexpr double phase     = 1.5707963267948966;

    return amplitude * spacing
           * Kokkos::sin(kx * (xIndex + 0.5) + ky * (yIndex + 0.5) + component * phase);
}

}  // namespace alvine

#endif
