#ifndef IPPL_ALVINE_SINUSOIDAL_JITTER_HPP
#define IPPL_ALVINE_SINUSOIDAL_JITTER_HPP

#include <Kokkos_MathematicalFunctions.hpp>

namespace alvine {

KOKKOS_INLINE_FUNCTION
double sinusoidalPositionJitter(double spacing, double xIndex, double yIndex, int component) {
    double seed = Kokkos::sin(12.9898 * (xIndex + 1.0)
                              + 78.233 * (yIndex + 1.0)
                              + 37.719 * component)
                  * 43758.5453;
    double random01 = seed - Kokkos::floor(seed);

    return spacing * 0.2 * (random01 - 0.5);
}

KOKKOS_INLINE_FUNCTION
double sinusoidalVorticityPerturbation(double xIndex, double yIndex) {
    double seed     = Kokkos::sin(12.9898 * (xIndex + 1.0) + 78.233 * (yIndex + 1.0)) * 43758.5453;
    double random01 = seed - Kokkos::floor(seed);

    return 0.1 * (random01 - 0.5);
}

}  // namespace alvine

#endif
