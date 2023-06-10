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

#include "Communicate/Communicator.h"

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        HaloCells<T, Dim>::HaloCells() {}

        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::accumulateHalo(view_type& view, const Layout_t* layout) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL);
        }

        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::fillHalo(view_type& view, const Layout_t* layout) {
            exchangeBoundaries<assign>(view, layout, INTERNAL_TO_HALO);
        }

        template <typename T, unsigned Dim>
        template <class Op>
        void HaloCells<T, Dim>::exchangeBoundaries(view_type& view, const Layout_t* layout,
                                                   SendOrder order) {
            using neighbor_list = typename Layout_t::neighbor_list;
            using range_list    = typename Layout_t::neighbor_range_list;

            const neighbor_list& neighbors = layout->getNeighbors();
            const range_list &sendRanges   = layout->getNeighborsSendRange(),
                             &recvRanges   = layout->getNeighborsRecvRange();

            size_t totalRequests = 0;
            for (const auto& componentNeighbors : neighbors) {
                totalRequests += componentNeighbors.size();
            }

            using buffer_type = mpi::Communicator::buffer_type;
            std::vector<MPI_Request> requests(totalRequests);

            // sending loop
            constexpr size_t cubeCount = detail::countHypercubes(Dim) - 1;
            size_t requestIndex        = 0;
            for (size_t index = 0; index < cubeCount; index++) {
                int tag                        = mpi::tag::HALO + index;
                const auto& componentNeighbors = neighbors[index];
                for (size_t i = 0; i < componentNeighbors.size(); i++) {
                    int targetRank = componentNeighbors[i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        /*We store only the sending and receiving ranges
                         * of INTERNAL_TO_HALO and use the fact that the
                         * sending range of HALO_TO_INTERNAL is the receiving
                         * range of INTERNAL_TO_HALO and vice versa
                         */
                        range = sendRanges[index][i];
                    } else {
                        range = recvRanges[index][i];
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf =
                        Comm->getBuffer<T>(mpi::tag::HALO_SEND + i * cubeCount + index, nsends);

                    Comm->isend(targetRank, tag, haloData_m, *buf, requests[requestIndex++],
                                nsends);
                    buf->resetWritePos();
                }
            }

            // receiving loop
            for (size_t index = 0; index < cubeCount; index++) {
                int tag                        = mpi::tag::HALO + Layout_t::getMatchingIndex(index);
                const auto& componentNeighbors = neighbors[index];
                for (size_t i = 0; i < componentNeighbors.size(); i++) {
                    int sourceRank = componentNeighbors[i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        range = recvRanges[index][i];
                    } else {
                        range = sendRanges[index][i];
                    }

                    size_type nrecvs = range.size();

                    buffer_type buf =
                        Comm->getBuffer<T>(mpi::tag::HALO_RECV + i * cubeCount + index, nrecvs);

                    Comm->recv(sourceRank, tag, haloData_m, *buf, nrecvs * sizeof(T), nrecvs);
                    buf->resetReadPos();

                    unpack<Op>(range, view, haloData_m);
                }
            }

            if (totalRequests > 0) {
                MPI_Waitall(totalRequests, requests.data(), MPI_STATUSES_IGNORE);
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
                int overalloc = Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            using index_array_type = typename RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "HaloCells::pack()", getRangePolicy(subview),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    int l = 0;

                    for (unsigned d1 = 0; d1 < Dim; d1++) {
                        int next = args[d1];
                        for (unsigned d2 = 0; d2 < d1; d2++) {
                            next *= subview.extent(d2);
                        }
                        l += next;
                    }

                    buffer(l) = apply(subview, args);
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

            using index_array_type = typename RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "HaloCells::unpack()", getRangePolicy(subview),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    int l = 0;

                    for (unsigned d1 = 0; d1 < Dim; d1++) {
                        int next = args[d1];
                        for (unsigned d2 = 0; d2 < d1; d2++) {
                            next *= subview.extent(d2);
                        }
                        l += next;
                    }

                    op(apply(subview, args), buffer(l));
                });
            Kokkos::fence();
        }

#if __cplusplus < 202002L
        template <typename View, typename Bounds, size_t... Idx>
        auto makeSubview_impl(const View& view, const Bounds& intersect,
                              const std::index_sequence<Idx...>&) {
            return Kokkos::subview(view,
                                   Kokkos::make_pair(intersect.lo[Idx], intersect.hi[Idx])...);
        };
#endif

        template <typename T, unsigned Dim>
        auto HaloCells<T, Dim>::makeSubview(const view_type& view, const bound_type& intersect) {
#if __cplusplus < 202002L
            return makeSubview_impl(view, intersect, std::make_index_sequence<Dim>{});
#else
            auto makeSub = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
                return Kokkos::subview(view,
                                       Kokkos::make_pair(intersect.lo[Idx], intersect.hi[Idx])...);
            };
            return makeSub(std::make_index_sequence<Dim>{});
#endif
        }

        template <typename T, unsigned Dim>
        template <typename Op>
        void HaloCells<T, Dim>::applyPeriodicSerialDim(view_type& view, const Layout_t* layout,
                                                       const int nghost) {
            int myRank           = Comm->rank();
            const auto& lDomains = layout->getHostLocalDomains();
            const auto& domain   = layout->getDomain();
            using index_type     = typename RangePolicy<Dim>::index_type;
            Kokkos::Array<index_type, Dim> ext, begin, end;

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

                    using index_array_type = typename RangePolicy<Dim>::index_array_type;
                    ippl::parallel_for(
                        "applyPeriodicSerialDim", createRangePolicy<Dim>(begin, end),
                        KOKKOS_LAMBDA(index_array_type & coords) {
                            // The ghosts are filled starting from the inside
                            // of the domain proceeding outwards for both lower
                            // and upper faces. The extra brackets and explicit
                            // mention

                            // nghost + i
                            coords[d] += nghost;
                            auto&& left = apply(view, coords);

                            // N - nghost - i
                            coords[d]    = N - coords[d];
                            auto&& right = apply(view, coords);

                            // nghost - 1 - i
                            coords[d] += 2 * nghost - 1 - N;
                            op(apply(view, coords), right);

                            // N - (nghost - 1 - i) = N - (nghost - 1) + i
                            coords[d] = N - coords[d];
                            op(apply(view, coords), left);
                        });

                    Kokkos::fence();
                }
            }
        }
    }  // namespace detail
}  // namespace ippl
