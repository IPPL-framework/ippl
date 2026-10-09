//
// Class OrthogonalRecursiveBisection for Domain Decomposition
//
// Simple domain decomposition using an Orthogonal Recursive Bisection,
// domain is divided recursively so as to even weights on each side of the cut,
// works with 2^n processors only.
//
//

#ifndef IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H
#define IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "FieldLayout/FieldLayout.h"
#include "Index/Index.h"
#include "Index/NDIndex.h"
#include "Particle/ParticleAttrib.h"
#include "Particle/ParticleSpatialLayout.h"
#include "Region/NDRegion.h"

namespace ippl {
    /*
     * @class OrthogonalRecursiveBisection
     * @tparam Field the field type
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
         * repartitions FieldLayout's global domain, cutting only axes that are not forbidden.
         * The FieldLayout and the ORB weight field are updated only after the proposed
         * domains pass validation.
         * @tparam Attrib the particle attribute type (memory space must be accessible to field
         * memory)
         * @param R Weights to scatter
         * @param fl FieldLayout
         * @param isFirstRepartition boolean which tells whether to scatter or not
         * @param forbiddenAxes true for axes ORB must not cut. The default allows all axes,
         * which is the legacy ORB behavior.
         */
        template <typename Attrib>
        bool binaryRepartition(const Attrib& R, FieldLayout<Dim>& fl,
                               const bool& isFirstRepartition,
                               const std::array<bool, Dim>& forbiddenAxes = {});

        /*!
         * Find cutting axis as the longest axis of the field layout that is not forbidden.
         * @param dom Domain to reduce
         * @param forbiddenAxes true for axes ORB must not cut
         * @return the cut axis, or -1 if all axes are forbidden
         */
        int findCutAxis(const NDIndex<Dim>& dom, const std::array<bool, Dim>& forbiddenAxes = {});

        /*!
         * Check whether two domains overlap.
         */
        bool domainsOverlap(const NDIndex<Dim>& lhs, const NDIndex<Dim>& rhs) const;

        /*!
         * Check whether proposed domains tile the global domain without cutting forbidden axes.
         */
        bool domainsTileAllowedDecomposition(const std::vector<NDIndex<Dim>>& domains,
                                             const NDIndex<Dim>& globalDomain,
                                             const std::array<bool, Dim>& forbiddenAxes) const;

        /*!
         * Performs reduction on local field in all dimension except that determined
         * by cutAxis, stores result in res
         * @param rankWeights Array giving the result of reduction
         * @param cutAxis Index of cut axis
         * @param dom Domain to reduce
         */
        void perpendicularReduction(std::vector<Tf>& rankWeights, unsigned int cutAxis,
                                    NDIndex<Dim>& dom);

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
