#ifndef IPPL_VORTEX_IN_CELL_DITRIBUTIONS_H
#define IPPL_VORTEX_IN_CELL_DITRIBUTIONS_H

#include <memory>

#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"

using view_type   = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using host_type   = typename ippl::ParticleAttrib<T>::host_mirror_type;;/*typename ippl::ParticleAttrib<T>::HostMirror;*/
using vector_type = ippl::Vector<double, Dim>;

class BaseDistribution {
public:
    view_type r;
    host_type omega;
    vector_type rmin, rmax, origin, center;
    unsigned np;

    BaseDistribution(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
                     vector_type origin, unsigned np)
        : r(r_)
        , omega(omega_)
        , rmin(r_min)
        , rmax(r_max)
        , origin(origin) 
	, np(np) {
        this->center = rmin + 0.5 * (rmax - rmin);
    }

    KOKKOS_INLINE_FUNCTION virtual void operator()(const size_t i) const = 0;
};

class UnitDisk : BaseDistribution {
public:
    UnitDisk(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
             vector_type origin, unsigned np)
        : BaseDistribution(r_, omega_, r_min, r_max, origin, np) {}

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
            //this->omega(i) = 1;
            this->omega(i) = 1 * ((rmax[1] - rmin[1]) * (rmax[0] - rmin[0])) / np;
        }
    }
};

class HalfPlane : BaseDistribution {
public:
    HalfPlane(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
              vector_type origin, unsigned np)
        : BaseDistribution(r_, omega_, r_min, r_max, origin, np) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        if (this->r(i)(1) > this->center(1)) {
            //this->omega(i) = 1;
            this->omega(i) = 1 * ((rmax[1] - rmin[1]) * (rmax[0] - rmin[0])) / np;
        } else {
            //this->omega(i) = -1;
            this->omega(i) = -1 * ((rmax[1] - rmin[1]) * (rmax[0] - rmin[0])) / np;
        }
    }
};

class Band : BaseDistribution {
public:
    Band(view_type r_, host_type omega_, vector_type r_min, vector_type r_max, vector_type origin, unsigned np)
        : BaseDistribution(r_, omega_, r_min, r_max, origin, np) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        //if (this->r(i)(1) > this->center(1) + 1 or this->r(i)(1) < this->center(1) - 1) {
        //    // Outside of the band
        //    this->omega(i) = 0;
        //} else {
            this->omega(i) = (2.0 * (rmax[0] - rmin[0])) / np;
        //}
    }
};

#endif
