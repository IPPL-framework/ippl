#ifndef IPPL_VORTEX_IN_CELL_DITRIBUTIONS_H
#define IPPL_VORTEX_IN_CELL_DITRIBUTIONS_H

#include <memory>
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"


using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using host_type = typename ippl::ParticleAttrib<T>::HostMirror;

template <unsigned dim>
    struct UnitDisk{
        using vector_type = ippl::Vector<double, dim>;

        view_type r;
        host_type omega;
        vector_type rmin, rmax, origin, center; 
        double radius_core;

        UnitDisk(view_type r_, host_type omega_, vector_type r_min, vector_type r_max, vector_type origin, double radius_core) : 
        r(r_),
        omega(omega_),
        rmin(r_min),
        rmax(r_max),
        origin(origin),
        radius_core(radius_core) {
            this->center = rmin + 0.5 * (rmax - rmin);
        }
        
        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
            vector_type dist = r(i) - this->center;
            double norm = 0.0;
            for (unsigned int d = 0; d < dim; d++) {
                norm += std::pow(dist(d), 2);
            }
            norm = std::sqrt(norm);

            if (norm > radius_core) {
                omega(i) = 1; // 15/radius_core seemed to be too strong
            } else {
                omega(i) = 0;
            }
        }

    };

#endif