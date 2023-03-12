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

#include "Utility/IpplException.h"

#include "Communicate/Communicate.h"

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        HaloCells<T, Dim>::HaloCells() {
            static_assert(Dim == 3, "Dimension must be 3!");
        }

        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::accumulateHalo(view_type& view, const Layout_t* layout) {
            exchangeFaces<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL);

            exchangeEdges<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL);

            exchangeVertices<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL);
        }

        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::fillHalo(view_type& view, const Layout_t* layout) {
            exchangeFaces<assign>(view, layout, INTERNAL_TO_HALO);

            exchangeEdges<assign>(view, layout, INTERNAL_TO_HALO);

            exchangeVertices<assign>(view, layout, INTERNAL_TO_HALO);
        }

        template <typename T, unsigned Dim>
        template <class Op>
        void HaloCells<T, Dim>::exchangeFaces(view_type& view, const Layout_t* layout,
                                              SendOrder order) {
            /* The neighbor list has length 2 * Dim. Each index
             * denotes a face. The value tells which MPI rank
             * we need to send to.
             */
            using neighbor_type            = typename Layout_t::face_neighbor_type;
            const neighbor_type& neighbors = layout->getFaceNeighbors();
            using neighbor_range_type      = typename Layout_t::face_neighbor_range_type;
            const neighbor_range_type& neighborsSendRange = layout->getFaceNeighborsSendRange();
            const neighbor_range_type& neighborsRecvRange = layout->getFaceNeighborsRecvRange();
            using match_face_type                         = typename Layout_t::match_face_type;
            const match_face_type& matchface              = layout->getMatchFace();

            size_t totalRequests = 0;
            for (auto& neighbor : neighbors) {
                totalRequests += neighbor.size();
            }

            using buffer_type = Communicate::buffer_type;
            std::vector<MPI_Request> requests(totalRequests);

            std::array<int, 2 * Dim> face_tag;
            const size_t groupCount = neighbors.size();
            size_t requestIndex     = 0;
            for (size_t face = 0; face < neighbors.size(); ++face) {
                face_tag[face] = Ippl::Comm->next_tag(HALO_FACE_TAG, HALO_TAG_CYCLE);
                for (size_t i = 0; i < neighbors[face].size(); ++i) {
                    int rank = neighbors[face][i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        /*We store only the sending and receiving ranges
                         * of INTERNAL_TO_HALO and use the fact that the
                         * sending range of HALO_TO_INTERNAL is the receiving
                         * range of INTERNAL_TO_HALO and vice versa
                         */
                        range = neighborsSendRange[face][i];
                    } else {
                        range = neighborsRecvRange[face][i];
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf = Ippl::Comm->getBuffer<T>(
                        IPPL_HALO_FACE_SEND + i * groupCount + face, nsends);

                    Ippl::Comm->isend(rank, face_tag[face], haloData_m, *buf,
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
                        range = neighborsRecvRange[face][i];
                    } else {
                        range = neighborsSendRange[face][i];
                    }

                    size_type nrecvs =
                        (int)((range.hi[0] - range.lo[0]) * (range.hi[1] - range.lo[1])
                              * (range.hi[2] - range.lo[2]));

                    buffer_type buf = Ippl::Comm->getBuffer<T>(
                        IPPL_HALO_FACE_RECV + i * groupCount + face, nrecvs);

                    Ippl::Comm->recv(rank, face_tag[matchface[face]], haloData_m, *buf,
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
        void HaloCells<T, Dim>::exchangeEdges(view_type& view, const Layout_t* layout,
                                              SendOrder order) {
            using neighbor_type            = typename Layout_t::edge_neighbor_type;
            const neighbor_type& neighbors = layout->getEdgeNeighbors();
            using neighbor_range_type      = typename Layout_t::edge_neighbor_range_type;
            const neighbor_range_type& neighborsSendRange = layout->getEdgeNeighborsSendRange();
            const neighbor_range_type& neighborsRecvRange = layout->getEdgeNeighborsRecvRange();
            using match_edge_type                         = typename Layout_t::match_edge_type;
            const match_edge_type& matchedge              = layout->getMatchEdge();

            size_t totalRequests = 0;
            for (auto& neighbor : neighbors) {
                totalRequests += neighbor.size();
            }

            using buffer_type = Communicate::buffer_type;
            std::vector<MPI_Request> requests(totalRequests);

            std::array<int, Dim * (1 << (Dim - 1))> edge_tag;
            const size_t groupCount = neighbors.size();
            size_t requestIndex     = 0;
            for (size_t edge = 0; edge < neighbors.size(); ++edge) {
                edge_tag[edge] = Ippl::Comm->next_tag(HALO_EDGE_TAG, HALO_TAG_CYCLE);
                for (size_t i = 0; i < neighbors[edge].size(); ++i) {
                    int rank = neighbors[edge][i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        range = neighborsSendRange[edge][i];
                    } else {
                        range = neighborsRecvRange[edge][i];
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf = Ippl::Comm->getBuffer<T>(
                        IPPL_HALO_EDGE_SEND + i * groupCount + edge, nsends);

                    Ippl::Comm->isend(rank, edge_tag[edge], haloData_m, *buf,
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
                        range = neighborsRecvRange[edge][i];
                    } else {
                        range = neighborsSendRange[edge][i];
                    }

                    size_type nrecvs =
                        (int)((range.hi[0] - range.lo[0]) * (range.hi[1] - range.lo[1])
                              * (range.hi[2] - range.lo[2]));

                    buffer_type buf = Ippl::Comm->getBuffer<T>(
                        IPPL_HALO_EDGE_RECV + i * groupCount + edge, nrecvs);

                    Ippl::Comm->recv(rank, edge_tag[matchedge[edge]], haloData_m, *buf,
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
        void HaloCells<T, Dim>::exchangeVertices(view_type& view, const Layout_t* layout,
                                                 SendOrder order) {
            using neighbor_type            = typename Layout_t::vertex_neighbor_type;
            const neighbor_type& neighbors = layout->getVertexNeighbors();
            using neighbor_range_type      = typename Layout_t::vertex_neighbor_range_type;
            const neighbor_range_type& neighborsSendRange = layout->getVertexNeighborsSendRange();
            const neighbor_range_type& neighborsRecvRange = layout->getVertexNeighborsRecvRange();
            using match_vertex_type                       = typename Layout_t::match_vertex_type;
            const match_vertex_type& matchvertex          = layout->getMatchVertex();

            using buffer_type = Communicate::buffer_type;
            std::vector<MPI_Request> requests(neighbors.size());

            std::array<int, 2 << (Dim - 1)> vertex_tag;
            size_t requestIndex = 0;
            for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
                vertex_tag[vertex] = Ippl::Comm->next_tag(HALO_VERTEX_TAG, HALO_TAG_CYCLE);
                if (neighbors[vertex] < 0) {
                    // we are on a non-periodic mesh / physical boundary
                    continue;
                }

                int rank = neighbors[vertex];

                bound_type range;
                if (order == INTERNAL_TO_HALO) {
                    range = neighborsSendRange[vertex];
                } else {
                    range = neighborsRecvRange[vertex];
                }

                size_type nsends;
                pack(range, view, haloData_m, nsends);

                buffer_type buf = Ippl::Comm->getBuffer<T>(IPPL_HALO_VERTEX_SEND + vertex, nsends);

                Ippl::Comm->isend(rank, vertex_tag[vertex], haloData_m, *buf,
                                  requests[requestIndex++], nsends);
                buf->resetWritePos();
            }

            // receive
            for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
                if (neighbors[vertex] < 0) {
                    // we are on a non-periodic mesh / physical boundary
                    continue;
                }

                int rank = neighbors[vertex];

                bound_type range;
                if (order == INTERNAL_TO_HALO) {
                    range = neighborsRecvRange[vertex];
                } else {
                    range = neighborsSendRange[vertex];
                }

                size_type nrecvs = (int)((range.hi[0] - range.lo[0]) * (range.hi[1] - range.lo[1])
                                         * (range.hi[2] - range.lo[2]));

                buffer_type buf = Ippl::Comm->getBuffer<T>(IPPL_HALO_VERTEX_RECV + vertex, nrecvs);

                Ippl::Comm->recv(rank, vertex_tag[matchvertex[vertex]], haloData_m, *buf,
                                 nrecvs * sizeof(T), nrecvs);
                buf->resetReadPos();

                unpack<Op>(range, view, haloData_m);
            }

            if (requestIndex > 0) {
                MPI_Waitall(requestIndex, requests.data(), MPI_STATUSES_IGNORE);
            }
        }

        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::pack(const bound_type& range, const view_type& view,
                                     FieldBufferData<T>& fd, size_type& nsends) {
            auto subview = makeSubview(view, range);

            auto& buffer = fd.buffer;

            size_t size = subview.size();
            nsends      = size;
            if (buffer.size() < size) {
                int overalloc = Ippl::Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            Kokkos::parallel_for(
                "HaloCells::pack()", detail::getRangePolicy<Dim>(subview),
                KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    int l = i + j * subview.extent(0) + k * subview.extent(0) * subview.extent(1);
                    buffer(l) = subview(i, j, k);
                });
            Kokkos::fence();
        }

        template <typename T, unsigned Dim>
        template <typename Op>
        void HaloCells<T, Dim>::unpack(const bound_type& range, const view_type& view,
                                       FieldBufferData<T>& fd) {
            auto subview = makeSubview(view, range);
            auto buffer  = fd.buffer;

            // 29. November 2020
            // https://stackoverflow.com/questions/3735398/operator-as-template-parameter
            Op op;

            Kokkos::parallel_for(
                "HaloCells::unpack()", detail::getRangePolicy<Dim>(subview),
                KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    int l = i + j * subview.extent(0) + k * subview.extent(0) * subview.extent(1);
                    op(subview(i, j, k), buffer(l));
                });
            Kokkos::fence();
        }

        template <typename T, unsigned Dim>
        auto HaloCells<T, Dim>::makeSubview(const view_type& view, const bound_type& intersect) {
            using Kokkos::make_pair;
            return Kokkos::subview(view, make_pair(intersect.lo[0], intersect.hi[0]),
                                   make_pair(intersect.lo[1], intersect.hi[1]),
                                   make_pair(intersect.lo[2], intersect.hi[2]));
        }

        template <typename T, unsigned Dim>
        template <typename Op>
        void HaloCells<T, Dim>::applyPeriodicSerialDim(view_type& view, const Layout_t* layout,
                                                       const int nghost) {
            int myRank           = Ippl::Comm->rank();
            const auto& lDomains = layout->getHostLocalDomains();
            const auto& domain   = layout->getDomain();
            using mdrange_type   = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
            Kokkos::Array<long, Dim> ext, begin, end;

            for (size_t i = 0; i < Dim; ++i) {
                ext[i]   = view.extent(i);
                begin[i] = 0;
            }

            Op op;

            for (unsigned d = 0; d < Dim; ++d) {
                end    = ext;
                end[d] = nghost;

                if (lDomains[myRank][d].length() == domain[d].length()) {
                    int N = view.extent(d) - 1;

                    Kokkos::parallel_for(
                        "applyPeriodicSerialDim", mdrange_type(begin, end),
                        KOKKOS_LAMBDA<typename... Idx>(const Idx... args) {
                            // The ghosts are filled starting from the inside of
                            // the domain proceeding outwards for both lower and
                            // upper faces. The extra brackets and explicit mention

                            using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                            index_type coords[Dim] = {args...};

                            // nghost + i
                            coords[d] += nghost;
                            auto&& left = apply<Dim>(view, coords);

                            // N - nghost - i
                            coords[d]    = N - coords[d];
                            auto&& right = apply<Dim>(view, coords);

                            // nghost - 1 - i
                            coords[d] += 2 * nghost - 1 - N;
                            op(apply<Dim>(view, coords), right);

                            // N - (nghost - 1 - i) = N - (nghost - 1) + i
                            coords[d] = N - coords[d];
                            op(apply<Dim>(view, coords), left);
                        });

                    Kokkos::fence();
                }
            }
        }
    }  // namespace detail
}  // namespace ippl
