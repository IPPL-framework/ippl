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

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        HaloCells<T, Dim>::HaloCells()
        {
            static_assert(Dim == 3, "Dimension must be 3!");
            //Kokkos::resize(fd_m.buffer, 512*512*3);
        }

        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::initializeBuffers()
        {
            Kokkos::resize(fd_m.buffer, 512*512*3);
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

            // send
            std::vector<MPI_Request> requests(0);

            int tag = Ippl::Comm->next_tag(HALO_FACE_TAG, HALO_TAG_CYCLE);

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


                    requests.resize(requests.size() + 1);



                    int nsends;
                    //FieldBufferData<T> fd;
                    pack(range, view, fd_m, nsends);


                    Ippl::Comm->isend(rank, tag, fd_m, *(layout->sendFacear_m[face][i]), requests.back(), nsends);
                    layout->sendFacear_m[face][i]->resetWritePos();

                }
            }

            Ippl::Comm->barrier();
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

                    //FieldBufferData<T> fd;

                    //Kokkos::resize(fd.buffer,
                    //               (range.hi[0] - range.lo[0]) *
                    //               (range.hi[1] - range.lo[1]) *
                    //               (range.hi[2] - range.lo[2]));

                    int nrecvs = (int)((range.hi[0] - range.lo[0]) *
                                 (range.hi[1] - range.lo[1]) *
                                 (range.hi[2] - range.lo[2]));

                    
                    Ippl::Comm->recv(rank, tag, fd_m, *(layout->recvFacear_m[face][i]), nrecvs);
                    layout->recvFacear_m[face][i]->resetReadPos();


                    unpack<Op>(range, view, fd_m);
                }
            }

            if (requests.size() > 0) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            }
            Ippl::Comm->barrier();
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

            // send
            std::vector<MPI_Request> requests(0);

            int tag = Ippl::Comm->next_tag(HALO_EDGE_TAG, HALO_TAG_CYCLE);

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

                    requests.resize(requests.size() + 1);


                    //FieldBufferData<T> fd;
                    int nsends;
                    pack(range, view, fd_m, nsends);

                    Ippl::Comm->isend(rank, tag, fd_m, *(layout->sendEdgear_m[edge][i]), requests.back(), nsends);
                    layout->sendEdgear_m[edge][i]->resetWritePos();

                }
            }

            Ippl::Comm->barrier();
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

                    //FieldBufferData<T> fd;

                    //Kokkos::resize(fd.buffer,
                    //               (range.hi[0] - range.lo[0]) *
                    //               (range.hi[1] - range.lo[1]) *
                    //               (range.hi[2] - range.lo[2]));

                    int nrecvs = (int)((range.hi[0] - range.lo[0]) *
                                 (range.hi[1] - range.lo[1]) *
                                 (range.hi[2] - range.lo[2]));
                    
                    Ippl::Comm->recv(rank, tag, fd_m, *(layout->recvEdgear_m[edge][i]), nrecvs);
                    layout->recvEdgear_m[edge][i]->resetReadPos();


                    unpack<Op>(range, view, fd_m);
                }
            }

            if (requests.size() > 0) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            }
            Ippl::Comm->barrier();
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

            // send
            std::vector<MPI_Request> requests(0);

            int tag = Ippl::Comm->next_tag(HALO_VERTEX_TAG, HALO_TAG_CYCLE);

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

                requests.resize(requests.size() + 1);


                //FieldBufferData<T> fd;
                int nsends;
                pack(range, view, fd_m, nsends);

                Ippl::Comm->isend(rank, tag, fd_m, *(layout->sendVertexar_m[vertex]), requests.back(), nsends);
                layout->sendVertexar_m[vertex]->resetWritePos();
            }
            Ippl::Comm->barrier();

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

                //FieldBufferData<T> fd;

                //Kokkos::resize(fd.buffer,
                //               (range.hi[0] - range.lo[0]) *
                //               (range.hi[1] - range.lo[1]) *
                //               (range.hi[2] - range.lo[2]));

                int nrecvs = (int)((range.hi[0] - range.lo[0]) *
                             (range.hi[1] - range.lo[1]) *
                             (range.hi[2] - range.lo[2]));
                
                Ippl::Comm->recv(rank, tag, fd_m, *(layout->recvVertexar_m[vertex]), nrecvs);
                layout->recvVertexar_m[vertex]->resetReadPos();

                unpack<Op>(range, view, fd_m);
            }

            if (requests.size() > 0) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            }
            Ippl::Comm->barrier();
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::pack(const bound_type& range,
                                     const view_type& view,
                                     FieldBufferData<T>& fd,
                                     int& nsends)
        {
            auto subview = makeSubview(view, range);

            auto& buffer = fd.buffer;

            //Kokkos::resize(buffer, subview.extent(0) * subview.extent(1) * subview.extent(2));
            nsends = (int)(subview.extent(0) * subview.extent(1) * subview.extent(2));

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
                    int l = i + j * subview.extent(0) + k * subview.extent(0) * subview.extent(1);
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
                    int l = i + j * subview.extent(0) + k * subview.extent(0) * subview.extent(1);
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
                intersect.lo[i] = overlap[i].first() - offset[i].first() /*offset*/ + nghost;
                intersect.hi[i] = overlap[i].last()  - offset[i].first() /*offset*/ + nghost + 1;
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
