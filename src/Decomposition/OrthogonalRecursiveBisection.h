//
// Class OrthogonalRecursiveBisection for Domain Decomposition
//
// Simple domain decomposition using an Orthogonal Recursive Bisection,
// domain is divided recursively so as to even weights on each side of the cut,
// works with 2^n processors only.
//
// Copyright (c) 2021, Michael Ligotino, ETH, Zurich;
// Paul Scherrer Institut, Villigen; Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#ifndef IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H
#define IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H

#include "FieldLayout/FieldLayout.h"
#include "Index/Index.h"
#include "Index/NDIndex.h"
#include "Particle/ParticleAttrib.h"
#include "Particle/ParticleSpatialLayout.h"
#include "Region/NDRegion.h"

namespace ippl {
    /*
     * @class OrthogonalRecursiveBisection
     * @tparam Tf type of field
     * @tparam Dim dimension
     * @tparam M mesh
     * @tparam Tp type of particle position. If not specified, it will be equal to the field's type
     */

    template <class Field, class Tp = typename Field::value_type>
    class OrthogonalRecursiveBisection {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using Tf                      = typename Field::value_type;

    public:
        // Weight for reduction
        Field bf_m;

        /*!
         * Initialize member field with mesh and field layout
         * @param fl
         * @param mesh Mesh
         * @param rho Density field
         */
        void initialize(FieldLayout<Dim>& fl, mesh_type& mesh, const Field& rho);

        /*!
         * Performs scatter operation of particle positions in field (weights) and
         * repartitions FieldLayout's global domain
         * @tparam Attrib the particle attribute type (memory space must be accessible to field
         * memory)
         * @param R Weights to scatter
         * @param fl FieldLayout
         * @param isFirstRepartition boolean which tells whether to scatter or not
         */
        template <typename Attrib>
        bool binaryRepartition(const Attrib& R, FieldLayout<Dim>& fl,
                               const bool& isFirstRepartition);

        /*!
         * Find cutting axis as the longest axis of the field layout.
         * @param dom Domain to reduce
         */
        int findCutAxis(NDIndex<Dim>& dom);

        /*!
         * Performs reduction on local field in all dimension except that determined
         * by cutAxis, stores result in res
         * @param res Array giving the result of reduction
         * @param dom Domain to reduce
         * @param cutAxis Index of cut axis
         */
        void perpendicularReduction(std::vector<Tf>& res, unsigned int cutAxis, NDIndex<Dim>& dom);

        /*!
         * Find median of array
         * @param w Array of real numbers
         */
        int findMedian(std::vector<Tf>& w);

        /*!
         * Splits the domain given by the iterator along the cut axis at the median,
         * the corresponding index will be cut between median and median+1
         * @param domains Set of subdomains which will be cut
         * @param procs Set of ranks count associated to each subdomain
         * @param it Iterator
         * @param cutAxis Index of cut axis
         * @param median Median
         */
        void cutDomain(std::vector<NDIndex<Dim>>& domains, std::vector<int>& procs, int it,
                       int cutAxis, int median);

        /*!
         * Scattering of particle positions in field using a CIC method
         * @tparam Attrib the particle attribute type (memory space must be accessible to field
         * memory)
         * @param r Weights
         */
        template <typename Attrib>
        void scatterR(const Attrib& r);

    };  // class

}  // namespace ippl

#include "Decomposition/OrthogonalRecursiveBisection.hpp"

#endif  // IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H
