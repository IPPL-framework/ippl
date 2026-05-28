#ifndef IPPL_ALVINE_SINUSOIDAL_JITTER_HPP
#define IPPL_ALVINE_SINUSOIDAL_JITTER_HPP

#include <Kokkos_MathematicalFunctions.hpp>

namespace alvine {

KOKKOS_INLINE_FUNCTION
double sinusoidalPerturbation(double xIndex, double yIndex, double phase = 0.0) {
    constexpr double kx        = 12.9898;
    constexpr double ky        = 78.233;

    return Kokkos::sin(kx * (xIndex + 0.5) + ky * (yIndex + 0.5) + phase);
}

KOKKOS_INLINE_FUNCTION
double sinusoidalJitter(double spacing, double xIndex, double yIndex, int component) {
    constexpr double amplitude      = 0.1;
    constexpr double componentPhase = 1.5707963267948966;

    return amplitude * spacing
           * sinusoidalPerturbation(xIndex, yIndex, component * componentPhase);
}

KOKKOS_INLINE_FUNCTION
double sinusoidalVorticityPerturbation(double xIndex, double yIndex) {
    double seed     = Kokkos::sin(12.9898 * (xIndex + 1.0) + 78.233 * (yIndex + 1.0)) * 43758.5453;
    double random01 = seed - Kokkos::floor(seed);

    return 0.1 * (random01 - 0.5);
}

}  // namespace alvine

#endif
