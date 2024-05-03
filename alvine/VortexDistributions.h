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
    AllOnes(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
            vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const { this->omega(i) = 1; }
};

class Disk : BaseDistribution {
public:
    Disk(view_type r_, host_type omega_, vector_type r_min, vector_type r_max, vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        vector_type dist = this->r(i) - this->center;
        double norm      = 0.0;
        for (unsigned int d = 0; d < Dim; d++) {
            norm += std::pow(dist(d), 2);
        }
        norm         = std::sqrt(norm);
        float radius = 2.2;
        if (norm > radius) {
            this->omega(i) = 0;
        } else {
            this->omega(i) = 1;
        }
    }
};

class GaussianDisk : BaseDistribution {
public:
    GaussianDisk(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
                 vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        vector_type dist = this->r(i) - this->center;
        double dist_s    = 0.0;
        for (unsigned int d = 0; d < Dim; d++) {
            dist_s += std::pow(dist(d), 2);
        }
        float sigma  = 1.0;
        float A      = 1.0;
        float radius = 2.2;

        if (std::sqrt(dist_s) > radius) {
            this->omega(i) = 0;
        } else {
            this->omega(i) = A * std::exp(-dist_s / (2 * sigma * sigma));
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
        float width      = 0.8;

        float axis_first_band  = this->center(1) + separation / 2;
        float axis_second_band = this->center(1) - separation / 2;

        bool add_noise    = false;
        float random_toss = static_cast<float>(random()) / RAND_MAX;
        float increment   = static_cast<float>(random()) / RAND_MAX - 0.5;

        if (this->r(i)(1) < axis_first_band + width and this->r(i)(1) > axis_first_band) {
            this->omega(i) = 1;
            if (add_noise) {
                if (random_toss < 0.1) {
                    this->omega(i) += increment;
                }
            }
        } else if (this->r(i)(1) < axis_second_band and this->r(i)(1) > axis_second_band - width) {
            this->omega(i) = -1;
            if (add_noise) {
                if (random_toss < 0.1) {
                    this->omega(i) += increment;
                }
            }
        } else {
            // Outside of the bands
            this->omega(i) = 0;
        }
    }
};

class TwoBandsGaussian : BaseDistribution {
public:
    TwoBandsGaussian(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
                     vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // On the y axis (index=1)
        float separation = this->center(1) / 2;
        float width      = 1;

        float axis_first_band  = this->center(1) + separation / 2;
        float axis_second_band = this->center(1) - separation / 2;

        float A     = 2.0;
        float sigma = 0.5;

        float dist_s = 0.0;
        if (this->r(i)(1) < axis_first_band + width / 2
            and this->r(i)(1) > axis_first_band - width / 2) {
            dist_s         = r(i)(1) - axis_first_band;
            this->omega(i) = A * std::exp(-std::pow(dist_s, 2) / (2 * sigma * sigma));
        } else if (this->r(i)(1) < axis_second_band + width / 2
                   and this->r(i)(1) > axis_second_band - width / 2) {
            dist_s         = r(i)(1) - axis_second_band;
            this->omega(i) = -A * std::exp(-std::pow(dist_s, 2) / (2 * sigma * sigma));
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

class TwoDisks : BaseDistribution {
public:
    TwoDisks(view_type r_, host_type omega_, vector_type r_min, vector_type r_max,
             vector_type origin)
        : BaseDistribution(r_, omega_, r_min, r_max, origin) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        double proximity = 4.37;  // Distance between the center of the two disks

        // 1.5 -> first case
        // 4.37 -> near third case
        // 4.5 -> second case
        // 5 -> too far
        vector_type center_1 = this->center + vector_type(proximity / 2, 0);
        vector_type center_2 = this->center - vector_type(proximity / 2, 0);

        vector_type dist_1 = this->r(i) - center_1;
        vector_type dist_2 = this->r(i) - center_2;

        double norm_1 = 0.0;
        for (unsigned int d = 0; d < Dim; d++) {
            norm_1 += std::pow(dist_1(d), 2);
        }
        norm_1 = std::sqrt(norm_1);

        double norm_2 = 0.0;
        for (unsigned int d = 0; d < Dim; d++) {
            norm_2 += std::pow(dist_2(d), 2);
        }
        norm_2 = std::sqrt(norm_2);

        float radius = 1.0;
        if ((norm_1 < radius) or (norm_2 < radius)) {
            this->omega(i) = 1;
        } else {
            this->omega(i) = 0;
        }
    }
};

#endif