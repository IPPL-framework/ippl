//
// Class HaloCells
//   The guard / ghost cells of BareField.
//

#include <memory>
#include <vector>

#include "Utility/IpplException.h"

#include "Communicate/Communicator.h"

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, class... ViewArgs>
        HaloCells<T, Dim, ViewArgs...>::HaloCells() {}

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::accumulateHalo(view_type& view, Layout_t* layout) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL);
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::accumulateHalo_noghost(view_type& view,
                                                                    Layout_t* layout, int nghost) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL_NOGHOST, nghost);
        }
        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::fillHalo(view_type& view, Layout_t* layout) {
            exchangeBoundaries<assign>(view, layout, INTERNAL_TO_HALO);
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <class Op>
        void HaloCells<T, Dim, ViewArgs...>::exchangeBoundaries(view_type& view, Layout_t* layout,
                                                                SendOrder order, int nghost) {
            using neighbor_list = typename Layout_t::neighbor_list;
            using range_list    = typename Layout_t::neighbor_range_list;

            auto& comm = layout->comm;

            const neighbor_list& neighbors = layout->getNeighbors();
            const range_list &sendRanges   = layout->getNeighborsSendRange(),
                             &recvRanges   = layout->getNeighborsRecvRange();

            auto ldom = layout->getLocalNDIndex();
            for (const auto& axis : ldom) {
                if ((axis.length() == 1) && (Dim != 1)) {
                    throw std::runtime_error(
                        "HaloCells: Cannot do neighbour exchange when domain decomposition "
                        "contains planes!");
                }
            }

            // needed for the NOGHOST approach - we want to remove the ghost
            // cells on the boundaries of the global domain from the halo
            // exchange when we set HALO_TO_INTERNAL_NOGHOST
            const auto domain    = layout->getDomain();
            const auto& ldomains = layout->getHostLocalDomains();

            size_t totalRequests = 0;
            for (const auto& componentNeighbors : neighbors) {
                totalRequests += componentNeighbors.size();
            }

            int me = Comm->rank();

            using memory_space = typename view_type::memory_space;
            using buffer_type  = mpi::Communicator::buffer_type<memory_space>;
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
                    } else if (order == HALO_TO_INTERNAL_NOGHOST) {
                        range = recvRanges[index][i];

                        for (size_t j = 0; j < Dim; ++j) {
                            const int stride = ldomains[me][j].stride();

                            const int globalLo =
                                ldomains[me][j].first() + (range.lo[j] - nghost) * stride;
                            const int globalHi =
                                ldomains[me][j].first() + (range.hi[j] - 1 - nghost) * stride;

                            bool isLower = (globalLo == domain[j].min());
                            bool isUpper = (globalHi == domain[j].max());

                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = recvRanges[index][i];
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf = comm.template getBuffer<memory_space, T>(nsends);

                    comm.isend(targetRank, tag, haloData_m, *buf, requests[requestIndex++], nsends);
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
                    } else if (order == HALO_TO_INTERNAL_NOGHOST) {
                        range = sendRanges[index][i];

                        for (size_t j = 0; j < Dim; ++j) {
                            const int stride = ldomains[me][j].stride();

                            const int globalLo =
                                ldomains[me][j].first() + (range.lo[j] - nghost) * stride;
                            const int globalHi =
                                ldomains[me][j].first() + (range.hi[j] - 1 - nghost) * stride;

                            bool isLower = (globalLo == domain[j].min());
                            bool isUpper = (globalHi == domain[j].max());

                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = sendRanges[index][i];
                    }

                    size_type nrecvs = rangeSize(range);

                    buffer_type buf = comm.template getBuffer<memory_space, T>(nrecvs);

                    comm.recv(sourceRank, tag, haloData_m, *buf, nrecvs * sizeof(T), nrecvs);
                    buf->resetReadPos();

                    unpack<Op>(range, view, haloData_m);
                }
            }

            if (totalRequests > 0) {
                MPI_Waitall(totalRequests, requests.data(), MPI_STATUSES_IGNORE);
            }

            comm.freeAllBuffers();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        size_type HaloCells<T, Dim, ViewArgs...>::rangeSize(const bound_type& range) {
            size_type total = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                const long extent = range.hi[d] - range.lo[d];
                if (extent <= 0) {
                    return 0;
                }
                total *= static_cast<size_type>(extent);
            }
            return total;
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::pack(const bound_type& range, const view_type& view,
                                                  databuffer_type& fd, size_type& nsends) {
            auto& fieldBuffer = fd.buffer;

            size_type size = rangeSize(range);
            nsends      = size;
            if (size == 0) {
                return;
            }
            if (fieldBuffer.size() < size) {
                int overalloc = Comm->getDefaultOverallocation();
                Kokkos::realloc(fieldBuffer, size * overalloc);
            }
            auto buffer   = fieldBuffer;
            auto fullView = view;

            using exec_space = typename view_type::execution_space;
            using index_type = typename RangePolicy<Dim, exec_space>::index_type;
            Kokkos::Array<index_type, Dim> begin, end, extent;
            for (unsigned d = 0; d < Dim; ++d) {
                begin[d]  = range.lo[d];
                end[d]    = range.hi[d];
                extent[d] = end[d] - begin[d];
            }

            using index_array_type =
                typename RangePolicy<Dim, exec_space>::index_array_type;
            ippl::parallel_for(
                "HaloCells::pack()", createRangePolicy<Dim, exec_space>(begin, end),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    size_type l      = 0;
                    size_type stride = 1;

                    for (unsigned d = 0; d < Dim; ++d) {
                        l += static_cast<size_type>(args[d] - begin[d]) * stride;
                        stride *= static_cast<size_type>(extent[d]);
                    }

                    buffer(l) = apply(fullView, args);
                });
            Kokkos::fence();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <typename Op>
        void HaloCells<T, Dim, ViewArgs...>::unpack(const bound_type& range, const view_type& view,
                                                    databuffer_type& fd) {
            if (rangeSize(range) == 0) {
                return;
            }

            auto buffer   = fd.buffer;
            auto fullView = view;

            // 29. November 2020
            // https://stackoverflow.com/questions/3735398/operator-as-template-parameter
            Op op;

            using exec_space = typename view_type::execution_space;
            using index_type = typename RangePolicy<Dim, exec_space>::index_type;
            Kokkos::Array<index_type, Dim> begin, end, extent;
            for (unsigned d = 0; d < Dim; ++d) {
                begin[d]  = range.lo[d];
                end[d]    = range.hi[d];
                extent[d] = end[d] - begin[d];
            }

            using index_array_type =
                typename RangePolicy<Dim, exec_space>::index_array_type;
            ippl::parallel_for(
                "HaloCells::unpack()", createRangePolicy<Dim, exec_space>(begin, end),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    size_type l      = 0;
                    size_type stride = 1;

                    for (unsigned d = 0; d < Dim; ++d) {
                        l += static_cast<size_type>(args[d] - begin[d]) * stride;
                        stride *= static_cast<size_type>(extent[d]);
                    }

                    op(apply(fullView, args), buffer(l));
                });
            Kokkos::fence();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <typename Op>
        void HaloCells<T, Dim, ViewArgs...>::applyPeriodicSerialDim(view_type& view,
                                                                    const Layout_t* layout,
                                                                    const int nghost) {
            int myRank           = layout->comm.rank();
            const auto& lDomains = layout->getHostLocalDomains();
            const auto& domain   = layout->getDomain();

            using exec_space = typename view_type::execution_space;
            using index_type = typename RangePolicy<Dim, exec_space>::index_type;

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

                    using index_array_type =
                        typename RangePolicy<Dim,
                                             typename view_type::execution_space>::index_array_type;
                    ippl::parallel_for(
                        "applyPeriodicSerialDim", createRangePolicy<Dim, exec_space>(begin, end),
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
