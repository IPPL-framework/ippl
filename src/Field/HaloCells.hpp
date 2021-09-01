//
// Class HaloCells
//   The guard / ghost cells of BareField.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include <memory>
#include <vector>

#include "Communicate/Communicate.h"

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        HaloCells<T, Dim>::HaloCells()
        {
            static_assert(Dim == 3, "Dimension must be 3!");
        }

        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::accumulateHalo(view_type& view,
                                               const Layout_t* layout,
                                               int nghost)
        {
            exchangeFaces<plus_assign>(view, layout, nghost, HALO_TO_INTERNAL);

            exchangeEdges<plus_assign>(view, layout, nghost, HALO_TO_INTERNAL);

            exchangeVertices<plus_assign>(view, layout, nghost, HALO_TO_INTERNAL);
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::fillHalo(view_type& view,
                                         const Layout_t* layout,
                                         int nghost)
        {
            exchangeFaces<assign>(view, layout, nghost, INTERNAL_TO_HALO);

            exchangeEdges<assign>(view, layout, nghost, INTERNAL_TO_HALO);

            exchangeVertices<assign>(view, layout, nghost, INTERNAL_TO_HALO);
        }


        template <typename T, unsigned Dim>
        template <class Op>
        void HaloCells<T, Dim>::exchangeFaces(view_type& view,
                                              const Layout_t* layout,
                                              int nghost,
                                              SendOrder order)
        {
            /* The neighbor list has length 2 * Dim. Each index
             * denotes a face. The value tells which MPI rank
             * we need to send to.
             */
            using neighbor_type = typename Layout_t::face_neighbor_type;
            const neighbor_type& neighbors = layout->getFaceNeighbors();
            const auto& lDomains = layout->getHostLocalDomains();

            int myRank = Ippl::Comm->rank();

            size_t totalRequests = 0;
            for (auto& neighbor : neighbors) {
                totalRequests += neighbor.size();
            }

            using buffer_type = Communicate::buffer_type;
            std::vector<MPI_Request> requests(totalRequests);

            int tag = Ippl::Comm->next_tag(HALO_FACE_TAG, HALO_TAG_CYCLE);

            const size_t groupCount = neighbors.size();
            size_t requestIndex = 0;
            for (size_t face = 0; face < neighbors.size(); ++face) {
                for (size_t i = 0; i < neighbors[face].size(); ++i) {

                    int rank = neighbors[face][i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        // owned domain increased by nghost cells
                        range = getBounds(lDomains[myRank], lDomains[rank], 
                                          lDomains[myRank], nghost);
                    } else {
                        range = getBounds(lDomains[rank], lDomains[myRank], 
                                          lDomains[myRank], nghost);
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf = Ippl::Comm->getBuffer<T>(
                        IPPL_HALO_FACE_SEND + i * groupCount + face,
                        nsends);

                    Ippl::Comm->isend(rank, tag, haloData_m, *buf,
                        requests[requestIndex++], nsends);
                    buf->resetWritePos();
                }
            }

            // receive
            for (size_t face = 0; face < neighbors.size(); ++face) {
                for (size_t i = 0; i < neighbors[face].size(); ++i) {

                    int rank = neighbors[face][i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        // remote domain increased by nghost cells
                        range = getBounds(lDomains[rank], lDomains[myRank], 
                                          lDomains[myRank], nghost);
                    } else {
                        range = getBounds(lDomains[myRank], lDomains[rank], 
                                          lDomains[myRank], nghost);
                    }

                    size_type nrecvs = (int)((range.hi[0] - range.lo[0]) *
                                 (range.hi[1] - range.lo[1]) *
                                 (range.hi[2] - range.lo[2]));

                    buffer_type buf = Ippl::Comm->getBuffer<T>(
                        IPPL_HALO_FACE_RECV + i * groupCount + face,
                        nrecvs);

                    Ippl::Comm->recv(rank, tag, haloData_m, *buf,
                        nrecvs * sizeof(T), nrecvs);
                    buf->resetReadPos();

                    unpack<Op>(range, view, haloData_m);
                }
            }

            if (totalRequests > 0) {
                MPI_Waitall(totalRequests, requests.data(), MPI_STATUSES_IGNORE);
            }
        }


        template <typename T, unsigned Dim>
        template <class Op>
        void HaloCells<T, Dim>::exchangeEdges(view_type& view,
                                              const Layout_t* layout,
                                              int nghost,
                                              SendOrder order)
        {
            using neighbor_type = typename Layout_t::edge_neighbor_type;
            const neighbor_type& neighbors = layout->getEdgeNeighbors();
            const auto& lDomains = layout->getHostLocalDomains();

            int myRank = Ippl::Comm->rank();

            size_t totalRequests = 0;
            for (auto& neighbor : neighbors) {
                totalRequests += neighbor.size();
            }

            using buffer_type = Communicate::buffer_type;
            std::vector<MPI_Request> requests(totalRequests);

            int tag = Ippl::Comm->next_tag(HALO_EDGE_TAG, HALO_TAG_CYCLE);

            const size_t groupCount = neighbors.size();
            size_t requestIndex = 0;
            for (size_t edge = 0; edge < neighbors.size(); ++edge) {
                for (size_t i = 0; i < neighbors[edge].size(); ++i) {

                    int rank = neighbors[edge][i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        // owned domain increased by nghost cells
                        range = getBounds(lDomains[myRank], lDomains[rank], 
                                          lDomains[myRank], nghost);
                    } else {
                        range = getBounds(lDomains[rank], lDomains[myRank], 
                                          lDomains[myRank], nghost);
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf = Ippl::Comm->getBuffer<T>(
                        IPPL_HALO_EDGE_SEND + i * groupCount + edge,
                        nsends);

                    Ippl::Comm->isend(rank, tag, haloData_m, *buf,
                        requests[requestIndex++], nsends);
                    buf->resetWritePos();
                }
            }

            // receive
            for (size_t edge = 0; edge < neighbors.size(); ++edge) {
                for (size_t i = 0; i < neighbors[edge].size(); ++i) {

                    int rank = neighbors[edge][i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        // remote domain increased by nghost cells
                        range = getBounds(lDomains[rank], lDomains[myRank], 
                                          lDomains[myRank], nghost);
                    } else {
                        range = getBounds(lDomains[myRank], lDomains[rank], 
                                          lDomains[myRank], nghost);
                    }

                    size_type nrecvs = (int)((range.hi[0] - range.lo[0]) *
                                 (range.hi[1] - range.lo[1]) *
                                 (range.hi[2] - range.lo[2]));

                    buffer_type buf = Ippl::Comm->getBuffer<T>(
                        IPPL_HALO_EDGE_RECV + i * groupCount + edge,
                        nrecvs);

                    Ippl::Comm->recv(rank, tag, haloData_m, *buf,
                        nrecvs * sizeof(T), nrecvs);
                    buf->resetReadPos();

                    unpack<Op>(range, view, haloData_m);
                }
            }

            if (totalRequests > 0) {
                MPI_Waitall(totalRequests, requests.data(), MPI_STATUSES_IGNORE);
            }
        }


        template <typename T, unsigned Dim>
        template <class Op>
        void HaloCells<T, Dim>::exchangeVertices(view_type& view,
                                                 const Layout_t* layout,
                                                 int nghost,
                                                 SendOrder order)
        {
            using neighbor_type = typename Layout_t::vertex_neighbor_type;
            const neighbor_type& neighbors = layout->getVertexNeighbors();
            const auto& lDomains = layout->getHostLocalDomains();

            int myRank = Ippl::Comm->rank();

            using buffer_type = Communicate::buffer_type;
            std::vector<MPI_Request> requests(neighbors.size());

            int tag = Ippl::Comm->next_tag(HALO_VERTEX_TAG, HALO_TAG_CYCLE);

            size_t requestIndex = 0;
            for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
                if (neighbors[vertex] < 0) {
                    // we are on a mesh / physical boundary
                    continue;
                }

                int rank = neighbors[vertex];

                bound_type range;
                if (order == INTERNAL_TO_HALO) {
                    // owned domain increased by nghost cells
                    range = getBounds(lDomains[myRank], lDomains[rank], 
                                      lDomains[myRank], nghost);
                } else {
                    range = getBounds(lDomains[rank], lDomains[myRank], 
                                      lDomains[myRank], nghost);
                }

                size_type nsends;
                pack(range, view, haloData_m, nsends);

                buffer_type buf = Ippl::Comm->getBuffer<T>(
                    IPPL_HALO_VERTEX_SEND + vertex,
                    nsends);

                Ippl::Comm->isend(rank, tag, haloData_m, *buf,
                    requests[requestIndex++], nsends);
                buf->resetWritePos();
            }

            // receive
            for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
                if (neighbors[vertex] < 0) {
                    // we are on a mesh / physical boundary
                    continue;
                }

                int rank = neighbors[vertex];

                bound_type range;
                if (order == INTERNAL_TO_HALO) {
                    // remote domain increased by nghost cells
                    range = getBounds(lDomains[rank], lDomains[myRank], 
                                      lDomains[myRank], nghost);
                } else {
                    range = getBounds(lDomains[myRank], lDomains[rank], 
                                      lDomains[myRank], nghost);
                }

                size_type nrecvs = (int)((range.hi[0] - range.lo[0]) *
                             (range.hi[1] - range.lo[1]) *
                             (range.hi[2] - range.lo[2]));
                
                buffer_type buf = Ippl::Comm->getBuffer<T>(
                    IPPL_HALO_VERTEX_RECV + vertex,
                    nrecvs);

                Ippl::Comm->recv(rank, tag, haloData_m, *buf,
                    nrecvs * sizeof(T), nrecvs);
                buf->resetReadPos();

                unpack<Op>(range, view, haloData_m);
            }

            if (requestIndex > 0) {
                MPI_Waitall(requestIndex, requests.data(), MPI_STATUSES_IGNORE);
            }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::pack(const bound_type& range,
                                     const view_type& view,
                                     FieldBufferData<T>& fd,
                                     size_type& nsends)
        {
            auto subview = makeSubview(view, range);

            auto& buffer = fd.buffer;

            size_t size = subview.size();
            nsends = size;
            if (buffer.size() < size) {
                int overalloc = Ippl::Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "HaloCells::pack()",
                mdrange_type({0, 0, 0},
                             {subview.extent(0),
                              subview.extent(1),
                              subview.extent(2)}),
                KOKKOS_CLASS_LAMBDA(const size_t i,
                                    const size_t j,
                                    const size_t k)
                {
                    int l = i + j * subview.extent(0)
                            + k * subview.extent(0) * subview.extent(1);
                    buffer(l) = subview(i, j, k);
                }
            );
            Kokkos::fence();
        }


        template <typename T, unsigned Dim>
        template <typename Op>
        void HaloCells<T, Dim>::unpack(const bound_type& range,
                                       const view_type& view,
                                       FieldBufferData<T>& fd)
        {
            auto subview = makeSubview(view, range);
            auto buffer = fd.buffer;

            // 29. November 2020
            // https://stackoverflow.com/questions/3735398/operator-as-template-parameter
            Op op;

            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "HaloCells::unpack()",
                mdrange_type({0, 0, 0},
                             {subview.extent(0),
                              subview.extent(1),
                              subview.extent(2)}),
                KOKKOS_CLASS_LAMBDA(const size_t i,
                                    const size_t j,
                                    const size_t k)
                {
                    int l = i + j * subview.extent(0)
                            + k * subview.extent(0) * subview.extent(1);
                    op(subview(i, j, k), buffer(l));
                }
            );
            Kokkos::fence();
        }


        template <typename T, unsigned Dim>
        typename HaloCells<T, Dim>::bound_type
        HaloCells<T, Dim>::getBounds(const NDIndex<Dim>& nd1,
                                     const NDIndex<Dim>& nd2,
                                     const NDIndex<Dim>& offset,
                                     int nghost)
        {
            NDIndex<Dim> gnd = nd2.grow(nghost);

            NDIndex<Dim> overlap = gnd.intersect(nd1);

            bound_type intersect;

            /* Obtain the intersection bounds with local ranges of the view.
             * Add "+1" to the upper bound since Kokkos loops always to "< extent".
             */
            for (size_t i = 0; i < Dim; ++i) {
                intersect.lo[i] = overlap[i].first() - offset[i].first() /*offset*/
                                    + nghost;
                intersect.hi[i] = overlap[i].last()  - offset[i].first() /*offset*/
                                    + nghost + 1;
            }

            return intersect;
        }


        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::makeSubview(const view_type& view,
                                       const bound_type& intersect)
        {
            using Kokkos::make_pair;
            return Kokkos::subview(view,
                                   make_pair(intersect.lo[0], intersect.hi[0]),
                                   make_pair(intersect.lo[1], intersect.hi[1]),
                                   make_pair(intersect.lo[2], intersect.hi[2]));
        }
    }
}
