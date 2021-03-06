//
// Class FieldLayout
//   FieldLayout describes how a given index space (represented by an NDIndex
//   object) is distributed among MPI ranks. It performs the initial
//   partitioning. The user may request that a particular dimension not be
//   partitioned by flagging that axis as 'SERIAL' (instead of 'PARALLEL').
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#include "FieldLayout/FieldLayout.h"
#include "Ippl.h"
#include "Utility/PAssert.h"

#include "Partition/Partitioner.h"

#include "Utility/IpplException.h"


#include <cstdlib>
#include <limits>

namespace ippl {

    template <unsigned Dim>
    FieldLayout<Dim>::FieldLayout()
    : dLocalDomains_m("local domains (device)", 0)
    , hLocalDomains_m(Kokkos::create_mirror_view(dLocalDomains_m))
    {
        for (unsigned int d = 0; d < Dim; ++d) {
            requestedLayout_m[d] = PARALLEL;
            minWidth_m[d] = 0;
        }
    }


    template <unsigned Dim>
    FieldLayout<Dim>::FieldLayout(const NDIndex<Dim>& domain, e_dim_tag* p)
    : FieldLayout()
    {
        initialize(domain, p);
    }


    template <unsigned Dim>
    FieldLayout<Dim>::~FieldLayout() { }


    template <unsigned Dim>
    void
    FieldLayout<Dim>::initialize(const NDIndex<Dim>& domain,
                                 e_dim_tag* userflags)
    {
        int nRanks = Ippl::Comm->size();

        gDomain_m = domain;

        if (nRanks < 2) {
            Kokkos::resize(dLocalDomains_m, nRanks);
            Kokkos::resize(hLocalDomains_m, nRanks);
            hLocalDomains_m(0) = domain;
            Kokkos::deep_copy(dLocalDomains_m, hLocalDomains_m);
            return;
        }


        // If the user did not specify parallel/serial flags then make all parallel.
        long totparelems = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            if (userflags == 0)
                requestedLayout_m[d] = PARALLEL;
            else
                requestedLayout_m[d] = userflags[d];

            if (requestedLayout_m[d] == PARALLEL) {
                totparelems *= domain[d].length();
            }
        }

        /* Check to see if we have too few elements to partition.  If so, reduce
         * the number of ranks (if necessary) to just the number of elements along
         * parallel dims.
         */
        if (totparelems < nRanks) {
            nRanks = totparelems;
        }

        Kokkos::resize(dLocalDomains_m, nRanks);
        Kokkos::resize(hLocalDomains_m, nRanks);

        detail::Partitioner<Dim> partition;

        partition.split(domain, hLocalDomains_m, requestedLayout_m, nRanks);

        findNeighbors();

        Kokkos::deep_copy(dLocalDomains_m, hLocalDomains_m);

        calcWidths();
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::NDIndex_t&
    FieldLayout<Dim>::getLocalNDIndex(int rank) const
    {
        return hLocalDomains_m(rank);
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::host_mirror_type&
    FieldLayout<Dim>::getHostLocalDomains() const
    {
        return hLocalDomains_m;
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::view_type&
    FieldLayout<Dim>::getDeviceLocalDomains() const
    {
        return dLocalDomains_m;
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::face_neighbor_type&
    FieldLayout<Dim>::getFaceNeighbors() const {
        return faceNeighbors_m;
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::edge_neighbor_type&
    FieldLayout<Dim>::getEdgeNeighbors() const {
        return edgeNeighbors_m;
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::vertex_neighbor_type&
    FieldLayout<Dim>::getVertexNeighbors() const {
        return vertexNeighbors_m;
    }


    template <unsigned Dim>
    void FieldLayout<Dim>::write(std::ostream& out) const
    {
        if (Ippl::Comm->rank() > 0) {
            return;
        }

        out << "Domain = " << gDomain_m << "\n"
            << "Total number of boxes = " << hLocalDomains_m.size() << "\n";

        using size_type = typename host_mirror_type::size_type;
        for (size_type i = 0; i < hLocalDomains_m.size(); ++i) {
            out << "    Box " << i << " " << hLocalDomains_m(i) << "\n";
        }
    }


    template <unsigned Dim>
    void FieldLayout<Dim>::calcWidths()
    {
        // initialize widths first
        for (unsigned int d = 0; d < Dim; ++d) {
            minWidth_m[d] = gDomain_m[d].length();
        }

        using size_type = typename host_mirror_type::size_type;
        for (size_type i = 0; i < hLocalDomains_m.size(); ++i) {
            const NDIndex_t &dom = hLocalDomains_m(i);
            for (unsigned int d = 0; d < Dim; ++d) {
                if ((unsigned int) dom[d].length() < minWidth_m[d])
                    minWidth_m[d] = dom[d].length();
            }
        }
    }


    template <unsigned Dim>
    void FieldLayout<Dim>::findNeighbors(int nghost) {

        /* just to be safe, we reset the neighbor list
         * (at the moment this is unnecessary, but as soon as
         * we have a repartitioner we need this call).
         */
        for (size_t i = 0; i < faceNeighbors_m.size(); ++i) {
            faceNeighbors_m[i].clear();
        }

        for (size_t i = 0; i < edgeNeighbors_m.size(); ++i) {
            edgeNeighbors_m[i].clear();
        }

        vertexNeighbors_m.fill(-1);


        int myRank = Ippl::Comm->rank();

        // get my local box
        auto& nd = hLocalDomains_m[myRank];

        // grow the box by nghost cells in each dimension
        auto gnd = nd.grow(nghost);

        for (int rank = 0; rank < Ippl::Comm->size(); ++rank) {
            if (rank == myRank) {
                // do not compare with my domain
                continue;
            }

            if (gnd.touches(hLocalDomains_m[rank])) {
                /* my grown domain touches another
                 * --> it is a neighbor
                 */
                auto intersect = gnd.intersect(hLocalDomains_m[rank]);

                /* check how many dimension have length > 1.
                 * Vertices are scalars --> all dimensions have length 1
                 * Edges are vectors --> 1 dimension has length > 1
                 * Faces are matrices --> 2 dimensions have length > 1
                 */
                int nDim = 0;
                for (unsigned int d = 0; d < Dim; ++d) {
                    const Index& index = intersect[d];
                    nDim += (index.length() > 1) ? 1 : 0;
                }


                switch (nDim) {

                case 0:
                    addVertex(gnd, intersect, rank);
                    break;
                case 1:
                    addEdge(gnd, intersect, rank);
                    break;
                case 2:
                    addFace(gnd, intersect, rank);
                    break;
                default:
                    throw IpplException(
                        "FieldLayout::findNeighbors()",
                        "Failed to identify grid point. Neither a face, edge or vertex grid point.");
                }
            }
        }
    }


    template <unsigned Dim>
    void FieldLayout<Dim>::addVertex(const NDIndex_t& grown,
                                     const NDIndex_t& intersect,
                                     int rank)
    {
        /* The following routine computes the correct index
         * of the vertex.
         *
         * Example vertex 5: x high, y low, z high:
         *
         * 1st iteration: add = 1 --> index = 1
         * 2nd iteration: add = 0 --> index += 0 --> index = 1
         * 3rd iteration: add = 1 --> index += 4 --> index = 5
         */
        size_t index = 0;
        for (size_t d = 0; d < Dim; ++d) {

            /* if lower --> 0
             * else upper --> 1
             */
            const bool isLower = (grown[d].first() == intersect[d].first());

            int add = (isLower) ? 0 : 1;

            index += (add << d);
        }

        PAssert(index < vertexNeighbors_m.size());

        vertexNeighbors_m[index] = rank;
    }


    template <unsigned Dim>
    void FieldLayout<Dim>::addEdge(const NDIndex_t& grown,
                                   const NDIndex_t& intersect,
                                   int rank)
    {
        int nEdgesPerDim = (1 << (Dim - 1));

        size_t index = 0;

        int num = 1;
        for (size_t d = 0; d < Dim; ++d) {

            if (intersect[d].length() == 1) {
                const bool isLower = (grown[d].first() == intersect[d].first());
                index += (isLower) ? 0 : num;
                ++num;
                continue;
            }


            int jump = d * nEdgesPerDim;
            index += jump;
        }

        PAssert(index < edgeNeighbors_m.size());

        edgeNeighbors_m[index].push_back(rank);
    }


    template <unsigned Dim>
    void FieldLayout<Dim>::addFace(const NDIndex_t& grown,
                                   const NDIndex_t& intersect,
                                   int rank)
    {
        for (unsigned int d = 0; d < Dim; ++d) {
            const Index& index = intersect[d];

            if (index.length() == 1) {
                /* We found the
                 * intersecting dimension.
                 * Now, we need to figure out which face
                 * (upper or lower)
                 */

                /* if lower --> 0
                 * else upper --> 1
                 */
                int inc = (grown[d].first() == index.first()) ? 0 : 1;

                /* x low  --> 0
                 * x high --> 1
                 * y low  --> 2
                 * y high --> 3
                 * z low  --> 4
                 * z high --> 5
                 */
                faceNeighbors_m[inc + 2 * d].push_back(rank);
                break;
            }
        }
    }
}
