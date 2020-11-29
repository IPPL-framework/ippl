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
            static_assert(Dim < 4, "Dimension must be less than 4!");
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::fillLocalHalo(view_type& view,
                                              const T& value,
                                              int nghost)
        {
//             using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
//
//             intersect_type domain;
//
//             for (unsigned int d = 0; d < Dim; ++d) {
//
//                 domain.lo.fill(0);
//                 domain.hi.fill(nghost);
//
//                 for (unsigned int i = 0; i < 2; ++i) {
//                     domain.hi[d] = view.extent(d);
//
//                     auto halo = makeSubview(view, domain);
//
//                     std::cout << halo.extent(0) << " " << halo.extent(1) << " " << halo.extent(2) << std::endl;
//
//                     Kokkos::parallel_for(
//                         "HaloCells::fillLocalHalo()",
//                         mdrange_type({0, 0, 0},
//                                      {halo.extent(0),
//                                       halo.extent(1),
//                                       halo.extent(2)}),
//                         KOKKOS_CLASS_LAMBDA(const size_t i,
//                                             const size_t j,
//                                             const size_t k)
//                         {
//                             halo(i, j, k) = value;
//                         }
//                     );
//                 }
//             }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::accumulateHalo(view_type& /*view*/,
                                               const Layout_t* /*layout*/,
                                               int /*nghost*/)
        {

        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::exchangeHalo(view_type& view,
                                             const Layout_t* layout,
                                             int nghost)
        {
            exchangeFaces(view, layout, nghost);

            exchangeEdges(view, layout, nghost);

            exchangeVertices(view, layout, nghost);
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::exchangeFaces(view_type& view,
                                              const Layout_t* layout,
                                              int nghost)
        {
            /* The neighbor list has length 2 * Dim. Each index
             * denotes a face. The value tells which MPI rank
             * we need to send to.
             */
            using neighbor_type = typename Layout_t::face_neighbor_type;
            const neighbor_type& neighbors = layout->getFaceNeighbors();
            const auto& lDomains = layout->getHostLocalDomains();

            int myRank = Ippl::Comm->rank();


            std::cout << "myRank = " << myRank << " " << lDomains[myRank] << std::endl;

            // send
            std::vector<MPI_Request> requests(0);
            using archive_type = Communicate::archive_type;
            std::vector<std::unique_ptr<archive_type>> archives(0);

            int tag = Ippl::Comm->next_tag(HALO_FACE_TAG, HALO_TAG_CYCLE);

            for (size_t face = 0; face < neighbors.size(); ++face) {
                for (size_t i = 0; i < neighbors[face].size(); ++i) {

                    int rank = neighbors[face][i];

                    intersect_type range = getInternalBounds(lDomains[myRank], lDomains[rank], nghost);


                    archives.push_back(std::make_unique<archive_type>());
                    requests.resize(requests.size() + 1);


                    FieldData<T> fd;
                    pack(range, view, fd);

                    Ippl::Comm->isend(rank, tag, fd, *(archives.back()),
                                      requests.back());

                }
            }

            // receive
            for (size_t face = 0; face < neighbors.size(); ++face) {
                for (size_t i = 0; i < neighbors[face].size(); ++i) {

                    int rank = neighbors[face][i];

                    intersect_type range = getHaloBounds(lDomains[myRank], lDomains[rank], nghost);

                    FieldData<T> fd;

                    Kokkos::resize(fd.buffer,
                                   (range.hi[0] - range.lo[0]) *
                                   (range.hi[1] - range.lo[1]) *
                                   (range.hi[2] - range.lo[2]));

                    Ippl::Comm->recv(rank, tag, fd);

                    unpack(range, view, fd);
                }
            }

            if (requests.size() > 0) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                archives.clear();
            }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::exchangeEdges(view_type& view,
                                              const Layout_t* layout,
                                              int nghost)
        {
            using neighbor_type = typename Layout_t::edge_neighbor_type;
            const neighbor_type& neighbors = layout->getEdgeNeighbors();
            const auto& lDomains = layout->getHostLocalDomains();

            int myRank = Ippl::Comm->rank();

            // send
            std::vector<MPI_Request> requests(0);
            using archive_type = Communicate::archive_type;
            std::vector<std::unique_ptr<archive_type>> archives(0);

            int tag = Ippl::Comm->next_tag(HALO_EDGE_TAG, HALO_TAG_CYCLE);

            for (size_t edge = 0; edge < neighbors.size(); ++edge) {
                for (size_t i = 0; i < neighbors[edge].size(); ++i) {

                    int rank = neighbors[edge][i];

                    intersect_type range = getInternalBounds(lDomains[myRank], lDomains[rank], nghost);


                    archives.push_back(std::make_unique<archive_type>());
                    requests.resize(requests.size() + 1);


                    FieldData<T> fd;
                    pack(range, view, fd);

                    Ippl::Comm->isend(rank, tag, fd, *(archives.back()),
                                      requests.back());

                }
            }

            // receive
            for (size_t edge = 0; edge < neighbors.size(); ++edge) {
                for (size_t i = 0; i < neighbors[edge].size(); ++i) {

                    int rank = neighbors[edge][i];

                    intersect_type range = getHaloBounds(lDomains[myRank], lDomains[rank], nghost);


                    FieldData<T> fd;

                    Kokkos::resize(fd.buffer,
                                   (range.hi[0] - range.lo[0]) *
                                   (range.hi[1] - range.lo[1]) *
                                   (range.hi[2] - range.lo[2]));

                    Ippl::Comm->recv(rank, tag, fd);

                    unpack(range, view, fd);
                }
            }

            if (requests.size() > 0) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                archives.clear();
            }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::exchangeVertices(view_type& view,
                                                 const Layout_t* layout,
                                                 int nghost)
        {
            using neighbor_type = typename Layout_t::vertex_neighbor_type;
            const neighbor_type& neighbors = layout->getVertexNeighbors();
            const auto& lDomains = layout->getHostLocalDomains();

            int myRank = Ippl::Comm->rank();

            // send
            std::vector<MPI_Request> requests(0);
            using archive_type = Communicate::archive_type;
            std::vector<std::unique_ptr<archive_type>> archives(0);

            int tag = Ippl::Comm->next_tag(HALO_VERTEX_TAG, HALO_TAG_CYCLE);

            for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
                if (neighbors[vertex] < 0) {
                    // we are on a mesh / physical boundary
                    continue;
                }

                int rank = neighbors[vertex];

                intersect_type range = getInternalBounds(lDomains[myRank], lDomains[rank], nghost);


                archives.push_back(std::make_unique<archive_type>());
                requests.resize(requests.size() + 1);


                FieldData<T> fd;
                pack(range, view, fd);

                Ippl::Comm->isend(rank, tag, fd, *(archives.back()),
                                    requests.back());
            }

            // receive
            for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
                if (neighbors[vertex] < 0) {
                    // we are on a mesh / physical boundary
                    continue;
                }

                int rank = neighbors[vertex];

                intersect_type range = getHaloBounds(lDomains[myRank], lDomains[rank], nghost);


                FieldData<T> fd;

                Kokkos::resize(fd.buffer,
                               (range.hi[0] - range.lo[0]) *
                               (range.hi[1] - range.lo[1]) *
                               (range.hi[2] - range.lo[2]));

                Ippl::Comm->recv(rank, tag, fd);

                unpack(range, view, fd);
            }

            if (requests.size() > 0) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                archives.clear();
            }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::pack(const intersect_type& range,
                                     const view_type& view,
                                     FieldData<T>& fd)
        {
            auto subview = makeSubview(view, range);

            auto& buffer = fd.buffer;

            Kokkos::resize(buffer, subview.extent(0) * subview.extent(1) * subview.extent(2));

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
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::unpack(const intersect_type& range,
                                       const view_type& view,
                                       FieldData<T>& fd)
        {
            auto subview = makeSubview(view, range);
            auto buffer = fd.buffer;

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
                    subview(i, j, k) = buffer(l);
                }
            );
        }


        template <typename T, unsigned Dim>
        typename
        HaloCells<T, Dim>::intersect_type HaloCells<T, Dim>::getInternalBounds(const NDIndex<Dim>& owned,
                                                                               const NDIndex<Dim>& remote,
                                                                               int nghost)
        {
            // remote domain increased by nghost cells
            NDIndex<Dim> gnd = remote.grow(nghost);

            NDIndex<Dim> overlap = gnd.intersect(owned);

            intersect_type intersect;

            /* Obtain the intersection bounds with local ranges of the view.
             * Add "+1" to the upper bound since Kokkos loops always to "< extent".
             */
            for (size_t i = 0; i < Dim; ++i) {
                intersect.lo[i] = overlap[i].first() - owned[i].first() /*offset*/ + nghost;
                intersect.hi[i] = overlap[i].last()  - owned[i].first() /*offset*/ + nghost + 1;
            }

            return intersect;
        }


        template <typename T, unsigned Dim>
        typename
        HaloCells<T, Dim>::intersect_type HaloCells<T, Dim>::getHaloBounds(const NDIndex<Dim>& owned,
                                                                               const NDIndex<Dim>& remote,
                                                                               int nghost)
        {
            // remote domain increased by nghost cells
            NDIndex<Dim> gnd = owned.grow(nghost);

            NDIndex<Dim> overlap = gnd.intersect(remote);

            intersect_type intersect;

            /* Obtain the intersection bounds with local ranges of the view.
             * Add "+1" to the upper bound since Kokkos loops always to "< extent".
             */
            for (size_t i = 0; i < Dim; ++i) {
                intersect.lo[i] = overlap[i].first() - owned[i].first() /*offset*/ + nghost;
                intersect.hi[i] = overlap[i].last()  - owned[i].first() /*offset*/ + nghost + 1;
            }

            return intersect;
        }

        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::makeSubview(const view_type& view,
                                       const intersect_type& intersect)
        {
            using Kokkos::make_pair;
            return Kokkos::subview(view,
                                   make_pair(intersect.lo[0], intersect.hi[0]),
                                   make_pair(intersect.lo[1], intersect.hi[1]),
                                   make_pair(intersect.lo[2], intersect.hi[2]));
        }
    }
}