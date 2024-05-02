#ifndef IPPL_VORTEX_IN_CELL_DITRIBUTIONS_H
#define IPPL_VORTEX_IN_CELL_DITRIBUTIONS_H

#include <memory>

#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"

using view_type   = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using host_type   = typename ippl::ParticleAttrib<T>::HostMirror;
using vector_type = ippl::Vector<double, Dim>;

class BaseDistribution {
public:
    view_type r;
    host_type omega;
    vector_type rmin, rmax, origin, center;

    BaseDistribution(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
                     vector_type origin)
        : r(r_)
        , omega(omega_)
        , rmin(r_min)
        , rmax(r_max)
        , origin(origin) {
        this->center = rmin + 0.5 * (rmax - rmin);
    }
    KOKKOS_INLINE_FUNCTION virtual void operator()(const size_t i) const = 0;
};

class AllOnes : BaseDistribution {
public:
    AllOnes(view_type r_, host_type omega_, vector_type r_min, vector_type r_max, vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const { this->omega(i) = 1; }
};

class UnitDisk : BaseDistribution {
public:
    UnitDisk(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
             vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        vector_type dist = this->r(i) - this->center;
        double norm      = 0.0;
        for (unsigned int d = 0; d < Dim; d++) {
            norm += std::pow(dist(d), 2);
        }
        norm         = std::sqrt(norm);
        float radius = 3.0;
        if (norm > radius) {
            this->omega(i) = 0;  // 15/radius_core seemed to be too strong
        } else {
            this->omega(i) = 1;
        }
    }
};

class HalfPlane : BaseDistribution {
public:
    HalfPlane(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
              vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        if (this->r(i)(1) > this->center(1)) {
            this->omega(i) = 1;
        } else {
            this->omega(i) = -1;
        }
    }
};

class Band : BaseDistribution {
public:
    Band(view_type r_, host_type omega_, vector_type r_min, vector_type r_max, vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        if (this->r(i)(1) > this->center(1) + 1 or this->r(i)(1) < this->center(1) - 1) {
            // Outside of the band
            this->omega(i) = 0;
        } else {
            this->omega(i) = 1;
        }
    }
};

class TwoBands : BaseDistribution {
public:
    TwoBands(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
             vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // On the y axis (index=1)
        float separation = this->center(1) / 2;
        float width      = 1;

        float axis_first_band  = this->center(1) + separation / 2;
        float axis_second_band = this->center(1) - separation / 2;

        if (this->r(i)(1) < axis_first_band + width and this->r(i)(1) > axis_first_band) {
            this->omega(i) = 1;
        } else if (this->r(i)(1) < axis_second_band and this->r(i)(1) > axis_second_band - width) {
            this->omega(i) = -1;
        } else {
            // Outside of the bands
            this->omega(i) = 0;
        }
    }
};

class Ring : BaseDistribution {
public:
    Ring(view_type r_, host_type omega_, vector_type r_min, vector_type r_max, vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        vector_type dist = this->r(i) - this->center;
        double norm      = 0.0;
        for (unsigned int d = 0; d < Dim; d++) {
            norm += std::pow(dist(d), 2);
        }
        norm = std::sqrt(norm);
        if (norm < 2 and norm > 1.5) {
            this->omega(i) = 1;
        } else {
            this->omega(i) = 0;
        }
    }
};

class ConcentricCircles : BaseDistribution {
public:
    ConcentricCircles(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
                      vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        vector_type dist = this->r(i) - this->center;
        double norm      = 0.0;
        for (unsigned int d = 0; d < Dim; d++) {
            norm += std::pow(dist(d), 2);
        }
        norm = std::sqrt(norm);
        if (norm < 2 and norm > 1.5) {
            this->omega(i) = 1;
        } else if (norm < 1 and norm > 0.5) {
            this->omega(i) = 1;
        } else {
            this->omega(i) = 0;
        }
    }
};

#endif