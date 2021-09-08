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
#include "Utility/IpplTimings.h"


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
    FieldLayout<Dim>::FieldLayout(const NDIndex<Dim>& domain, e_dim_tag* p, bool isAllPeriodic)
    : FieldLayout()
    {
        initialize(domain, p, isAllPeriodic);
    }


    template <unsigned Dim>
    FieldLayout<Dim>::~FieldLayout() { }

    template <unsigned Dim>
    void
    FieldLayout<Dim>::updateLayout(const std::vector<NDIndex<Dim>>& domains) {
        if (domains.empty())
           return;
        
        for (unsigned int i = 0; i < domains.size(); i++)
           hLocalDomains_m(i) = domains[i];
        
        findNeighbors();

        Kokkos::deep_copy(dLocalDomains_m, hLocalDomains_m);

        calcWidths();
    }

    template <unsigned Dim>
    void
    FieldLayout<Dim>::initialize(const NDIndex<Dim>& domain,
                                 e_dim_tag* userflags, bool isAllPeriodic)
    {
        int nRanks = Ippl::Comm->size();

        gDomain_m = domain;

        isAllPeriodic_m = isAllPeriodic;

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
    const typename FieldLayout<Dim>::face_neighbor_range_type&
    FieldLayout<Dim>::getFaceNeighborsSendRange() const {
        return faceNeighborsSendRange_m;
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::edge_neighbor_range_type&
    FieldLayout<Dim>::getEdgeNeighborsSendRange() const {
        return edgeNeighborsSendRange_m;
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::vertex_neighbor_range_type&
    FieldLayout<Dim>::getVertexNeighborsSendRange() const {
        return vertexNeighborsSendRange_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::face_neighbor_range_type&
    FieldLayout<Dim>::getFaceNeighborsRecvRange() const {
        return faceNeighborsRecvRange_m;
    }


    template <unsigned Dim>
    const typename FieldLayout<Dim>::edge_neighbor_range_type&
    FieldLayout<Dim>::getEdgeNeighborsRecvRange() const {
        return edgeNeighborsRecvRange_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::vertex_neighbor_range_type&
    FieldLayout<Dim>::getVertexNeighborsRecvRange() const {
        return vertexNeighborsRecvRange_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::match_face_type&
    FieldLayout<Dim>::getMatchFace() const {
        return matchface_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::match_edge_type&
    FieldLayout<Dim>::getMatchEdge() const {
        return matchedge_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::match_vertex_type&
    FieldLayout<Dim>::getMatchVertex() const {
        return matchvertex_m;
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

        /* We need to reset the neighbor list
         * and its ranges because of the repartitioner.
         */
        for (size_t i = 0; i < faceNeighbors_m.size(); ++i) {
            faceNeighbors_m[i].clear();
            faceNeighborsSendRange_m[i].clear();
            faceNeighborsRecvRange_m[i].clear();
        }

        for (size_t i = 0; i < edgeNeighbors_m.size(); ++i) {
            edgeNeighbors_m[i].clear();
            edgeNeighborsSendRange_m[i].clear();
            edgeNeighborsRecvRange_m[i].clear();
        }

        vertexNeighbors_m.fill(-1);


        int myRank = Ippl::Comm->rank();

        // get my local box
        auto& nd = hLocalDomains_m[myRank];

        // grow the box by nghost cells in each dimension
        auto gnd = nd.grow(nghost);

        static IpplTimings::TimerRef findInternalNeighborsTimer = IpplTimings::getTimer("findInternal");
        static IpplTimings::TimerRef findPeriodicNeighborsTimer = IpplTimings::getTimer("findPeriodic");
        for (int rank = 0; rank < Ippl::Comm->size(); ++rank) {
            if (rank == myRank) {
                // do not compare with my domain
                continue;
            }

            auto& ndNeighbor = hLocalDomains_m[rank];
            IpplTimings::startTimer(findInternalNeighborsTimer);
            //For inter-processor neighbors
            if (gnd.touches(ndNeighbor)) {

                auto intersect = gnd.intersect(ndNeighbor);
                addNeighbors(gnd, nd, ndNeighbor, intersect, nghost, rank);

            }
            IpplTimings::stopTimer(findInternalNeighborsTimer);

            IpplTimings::startTimer(findPeriodicNeighborsTimer);
            if(isAllPeriodic_m) {

                int offsetd0, offsetd1, offsetd2;
                for (unsigned int d0 = 0; d0 < Dim; ++d0) {
                    //The k loop is for checking whether our local
                    //domain touches both min. and max. extents of the 
                    //global domain as this can happen in 1D, 2D decompositions
                    //and also in less no. of cores (like <=4)
                    for (int k0 = 0; k0 < 2; ++k0) {

                        offsetd0 = getPeriodicOffset(nd, d0, k0);
                        if(offsetd0 == 0)
                            continue;

                        gnd[d0] = gnd[d0] + offsetd0; 
                        if (gnd.touches(ndNeighbor)) {
                            auto intersect = gnd.intersect(ndNeighbor);
                            ndNeighbor[d0] = ndNeighbor[d0] - offsetd0;
                            addNeighbors(gnd, nd, ndNeighbor, intersect, 
                                         nghost, rank);
                            ndNeighbor[d0] = ndNeighbor[d0] + offsetd0;
                        }
                   
                        //The following loop is to find the periodic edge neighbors of
                        //the domain in the physical boundary
                        for (unsigned int d1 = d0 + 1; d1 < Dim; ++d1) {
                            for (int k1 = 0; k1 < 2; ++k1) {
                        
                                offsetd1 = getPeriodicOffset(nd, d1, k1);
                                if(offsetd1 == 0)
                                    continue;
                                
                                gnd[d1] = gnd[d1] + offsetd1; 
                                if (gnd.touches(ndNeighbor)) {
                                    auto intersect = gnd.intersect(ndNeighbor);
                                    ndNeighbor[d0] = ndNeighbor[d0] - offsetd0;
                                    ndNeighbor[d1] = ndNeighbor[d1] - offsetd1;
                                    addNeighbors(gnd, nd, ndNeighbor, intersect, 
                                                 nghost, rank);
                                    ndNeighbor[d0] = ndNeighbor[d0] + offsetd0;
                                    ndNeighbor[d1] = ndNeighbor[d1] + offsetd1;
                                }
                        
                                //The following loop is to find the vertex neighbors of
                                //the domain in the physical boundary
                                for (unsigned int d2 = d1 + 1; d2 < Dim; ++d2) {
                                    for (int k2 = 0; k2 < 2; ++k2) {
                            
                                        offsetd2 = getPeriodicOffset(nd, d2, k2);
                                        if(offsetd2 == 0)
                                            continue;
                                        
                                        gnd[d2] = gnd[d2] + offsetd2; 
                                        if (gnd.touches(ndNeighbor)) {
                                            auto intersect = gnd.intersect(ndNeighbor);
                                            ndNeighbor[d0] = ndNeighbor[d0] - offsetd0;
                                            ndNeighbor[d1] = ndNeighbor[d1] - offsetd1;
                                            ndNeighbor[d2] = ndNeighbor[d2] - offsetd2;
                                            addNeighbors(gnd, nd, ndNeighbor, intersect, 
                                                         nghost, rank);
                                            ndNeighbor[d0] = ndNeighbor[d0] + offsetd0;
                                            ndNeighbor[d1] = ndNeighbor[d1] + offsetd1;
                                            ndNeighbor[d2] = ndNeighbor[d2] + offsetd2;
                                        }
                                        gnd[d2] = gnd[d2] - offsetd2; 
                                    }
                                }
                                gnd[d1] = gnd[d1] - offsetd1;
                            }
                        }
                        gnd[d0] = gnd[d0] - offsetd0;
                    }
                }
            }
            IpplTimings::stopTimer(findPeriodicNeighborsTimer);
        }
    }
    
    template <unsigned Dim>
    void FieldLayout<Dim>::addNeighbors(NDIndex_t& gnd, 
                                        NDIndex_t& nd, 
                                        NDIndex_t& ndNeighbor,
                                        NDIndex_t& intersect,
                                        int nghost, 
                                        int rank) {
        
            bound_type rangeSend, rangeRecv;
            rangeSend = getBounds(nd, ndNeighbor, 
                                  nd, nghost);
                
            rangeRecv = getBounds(ndNeighbor, nd, 
                                  nd, nghost);
                
            int nDim = 0;
            for (unsigned int d = 0; d < Dim; ++d) {
                const Index& index = intersect[d];
                nDim += (index.length() > 1) ? 1 : 0;
            }

            switch (nDim) {

            case 0:
                addVertex(gnd, intersect, rank, rangeSend, rangeRecv);
                break;
            case 1:
                addEdge(gnd, intersect, rank, rangeSend, rangeRecv);
                break;
            case 2:
                addFace(gnd, intersect, rank, rangeSend, rangeRecv);
                break;
            default:
                throw IpplException(
                      "FieldLayout::addNeighbors()",
                      "Failed to identify grid point. Neither a face, edge or vertex grid point.");
            }

    }




    template <unsigned Dim>
    void FieldLayout<Dim>::addVertex(const NDIndex_t& grown,
                                     const NDIndex_t& intersect,
                                     int rank,
                                     const bound_type& rangeSend,
                                     const bound_type& rangeRecv)
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
        vertexNeighborsSendRange_m[index] = rangeSend;
        vertexNeighborsRecvRange_m[index] = rangeRecv;
        
    }


    template <unsigned Dim>
    void FieldLayout<Dim>::addEdge(const NDIndex_t& grown,
                                   const NDIndex_t& intersect,
                                   int rank,
                                   const bound_type& rangeSend,
                                   const bound_type& rangeRecv)
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
        edgeNeighborsSendRange_m[index].push_back(rangeSend);
        edgeNeighborsRecvRange_m[index].push_back(rangeRecv);
    }


    template <unsigned Dim>
    void FieldLayout<Dim>::addFace(const NDIndex_t& grown,
                                   const NDIndex_t& intersect,
                                   int rank,
                                   const bound_type& rangeSend,
                                   const bound_type& rangeRecv)
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
                faceNeighborsSendRange_m[inc + 2 * d].push_back(rangeSend);
                faceNeighborsRecvRange_m[inc + 2 * d].push_back(rangeRecv);
                break;
            }
        }
    }
    
    template <unsigned Dim>
    typename FieldLayout<Dim>::bound_type
    FieldLayout<Dim>::getBounds(const NDIndex_t& nd1,
                                const NDIndex_t& nd2,
                                const NDIndex_t& offset,
                                int nghost)
    {
        NDIndex<Dim> gnd = nd2.grow(nghost);

        NDIndex<Dim> overlap = gnd.intersect(nd1);

        bound_type intersect;

        /* Obtain the intersection bounds with local ranges of the view.
         * Add "+1" to the upper bound since Kokkos loops always to "< extent".
         */
        for (size_t i = 0; i < Dim; ++i) {
            intersect.lo[i] = overlap[i].first() - offset[i].first() /*offset*/ + nghost;
            intersect.hi[i] = overlap[i].last()  - offset[i].first() /*offset*/ + nghost + 1;
        }

        return intersect;
    }
    
    template <unsigned Dim>
    int FieldLayout<Dim>::getPeriodicOffset(const NDIndex_t& nd,
                                            const unsigned int d,
                                            const int k)
    {
        int offset=0;
        switch(k) {
            case 0:
                if(nd[d].max() == gDomain_m[d].max())
                    offset = -gDomain_m[d].length();

                break;
            case 1:
                if(nd[d].min() == gDomain_m[d].min())
                    offset = gDomain_m[d].length();

                break;
            default:
                throw IpplException("FieldLayout:getPeriodicOffset",
                                    "k  has to be either 0 or 1");
        }
        
        return offset;
    }

}
