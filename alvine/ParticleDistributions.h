#ifndef IPPL_VORTEX_IN_CELL_PAR_DITRIBUTIONS_H
#define IPPL_VORTEX_IN_CELL_PAR_DITRIBUTIONS_H

#include <Kokkos_Random.hpp>
#include <memory>

#include "AlvineManager.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "ParticleDistributions.h"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randu.h"
#include "VortexDistributions.h"

using view_type     = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using vector_type   = ippl::Vector<double, Dim>;
using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

class BaseParticleDistribution {
public:
    view_type r;
    vector_type rmin, rmax, center;
    size_type np_m;

    BaseParticleDistribution(view_type r_, vector_type r_min, vector_type r_max, size_type np_m)
        : r(r_)
        , rmin(r_min)
        , rmax(r_max)
        , np_m(np_m) {
        this->center = rmin + 0.5 * (rmax - rmin);
    }

    KOKKOS_INLINE_FUNCTION virtual void operator()(const size_t i) const = 0;
};

class CircleDistribution : BaseParticleDistribution {
public:
    int n_circles_m = 24;
    int particles_in_circle[24 + 1];

    CircleDistribution(view_type r_, vector_type r_min, vector_type r_max, size_type np_m)
        : BaseParticleDistribution(r_, r_min, r_max, np_m) {
        // This implementationn follows the paper description
        int M = 10;

        particles_in_circle[0] = 2;
        for (int i = 1; i <= n_circles_m; i++) {
            particles_in_circle[i] = M * i + particles_in_circle[i - 1];
            std::cout << particles_in_circle[i] << std::endl;
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const size_t p) const {
        int p_ = p;

        // Find in which circle p is
        int circle = 0;
        for (int i = 1; i <= n_circles_m; ++i) {
            if (particles_in_circle[i - 1] < p_ + 1 and p_ + 1 <= particles_in_circle[i]) {
                circle = i;
                break;
            }
        }

        if (p_ < particles_in_circle[24]) {
            double d     = 0.3;
            double r_    = double(circle) * d;
            double theta = double(particles_in_circle[circle] - p_)
                           / (particles_in_circle[circle] - particles_in_circle[circle - 1]);

            this->r(p)(0) = std::cos(theta * 2 * pi) * r_ + this->center(0);
            this->r(p)(1) = std::sin(theta * 2 * pi) * r_ + this->center(1);
        } else {  // Place somewhere far to be ignored by the vortex distribution
            this->r(p)(0) = 23;
            this->r(p)(1) = 23;
        }
    }
};

class EquidistantDistribution : BaseParticleDistribution {
public:
    // Random number generator
    double increments[Dim];
    int nr_m;

    EquidistantDistribution(view_type r_, vector_type r_min, vector_type r_max, size_type np_m)
        : BaseParticleDistribution(r_, r_min, r_max, np_m) {
        nr_m = std::pow(np_m, 1.0 / Dim);
        std::cout << "EquidistantDistribution particles: " << nr_m << std::endl;
        for (unsigned int i = 0; i < Dim; i++) {
            // assume that the grid is equidistant
            increments[i] = (r_max[i] - r_min[i]) / nr_m;
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Create equidistant grid
        r(i)[0] = rmin[0] + increments[0] * (i % nr_m);
        r(i)[1] = rmin[1] + increments[1] * (i / nr_m);
    }
};

class RandDistribution : BaseParticleDistribution {
public:
    // Random number generator
    double rmin[Dim];
    double rmax[Dim];

    int seed                = 42;
    GeneratorPool rand_pool = GeneratorPool((size_type)(seed + 100 * ippl::Comm->rank()));

    RandDistribution(view_type r_, vector_type r_min, vector_type r_max, size_type np_m)
        : BaseParticleDistribution(r_, r_min, r_max, np_m) {
        for (unsigned int i = 0; i < Dim; i++) {
            rmin[i] = r_min[i];
            rmax[i] = r_max[i];
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            r(i)[d] = rand_gen.drand(rmin[d], rmax[d]);
        }
        rand_pool.free_state(rand_gen);
    }
};

#endif