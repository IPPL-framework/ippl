//
// Class HaloCells
//   The guard / ghost cells of BareField.
//

#include <memory>
#include <vector>

#include "Utility/IpplException.h"
#include "Utility/Logging.h"

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
            SPDLOG_SCOPE("Start exchangeBoundaries, {}", comm.rank());

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

            size_t sendRequests = 0;
            for (const auto& componentNeighbors : neighbors) {
                sendRequests += componentNeighbors.size();
            }

            int me = comm.rank();

            // ------------------------------------
            // async MPI buffers and management
            // ------------------------------------
            using execution_space = Kokkos::DefaultExecutionSpace;
            using policy_type     = Kokkos::RangePolicy<execution_space>;
            using memory_space    = typename view_type::memory_space;
            using buffer_type     = mpi::Communicator::buffer_type<memory_space>;
            //
            struct async_recv_data {
                buffer_type async_buffer;
                bound_type range;
                int tag;
                MPI_Request request;
            };

            std::vector<async_recv_data> recv_requests;
            std::vector<mpi::Communicator::async_send_data<memory_space>> send_requests;
            recv_requests.reserve(sendRequests);
            send_requests.reserve(sendRequests);

            // ------------------------------------
            // pre-post receives loop
            // ------------------------------------
            constexpr size_t cubeCount = detail::countHypercubes(Dim) - 1;
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
                            bool isLower = ((range.lo[j] + ldomains[me][j].first() - nghost)
                                            == domain[j].min());
                            bool isUpper = ((range.hi[j] - 1 + ldomains[me][j].first() - nghost)
                                            == domain[j].max());
                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = sendRanges[index][i];
                    }

                    size_type nrecvs = range.size();

                    // std::cout << haloData_m.get_buffer();
                    buffer_type buf = comm.template getBuffer<memory_space, T>(nrecvs);
                    buf->resetReadPos();
                    buf->resetWritePos();

                    MPI_Request request = MPI_REQUEST_NULL;
                    comm.irecv(sourceRank, tag, *buf, request, nrecvs * sizeof(T));
                    recv_requests.push_back({buf, range, tag, request});
                }
            }

            // sending loop
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
                            bool isLower = ((range.lo[j] + ldomains[me][j].first() - nghost)
                                            == domain[j].min());
                            bool isUpper = ((range.hi[j] - 1 + ldomains[me][j].first() - nghost)
                                            == domain[j].max());
                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = recvRanges[index][i];
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf = comm.template getBuffer<memory_space, T>(nsends);
                    buf->resetReadPos();
                    buf->resetWritePos();
                    MPI_Request request = MPI_REQUEST_NULL;

                    comm.isend(targetRank, tag, haloData_m, *buf, request, nsends);
                    SPDLOG_TRACE("halo serialized, {}", static_cast<uintptr_t>(request));
                    send_requests.push_back({buf, tag, request});
                }
            }

            // ------------------------------------
            // receive, then deserialize and unpack pre-posted receives
            // ------------------------------------
            if ((sendRequests > 0) || (recv_requests.size() > 0)) {
                bool redo = true;
                while (redo) {
                    redo = false;
                    for (auto it = recv_requests.begin(); it != recv_requests.end(); ++it) {
                        int flag = 0;
                        MPI_Status status;
                        SPDLOG_TRACE("iRecv MPI_Test, {} {}", comm.rank(),
                                     static_cast<uintptr_t>(it->request));
                        if (it->request != MPI_REQUEST_NULL) {
                            SPDLOG_TRACE("MPI_Test recv tag {:04}, req {}", it->tag,
                                         static_cast<uintptr_t>(it->request));
                            auto old_request = it->request;
                            MPI_Test(&it->request, &flag, &status);
                            if (flag) {
                                SPDLOG_DEBUG("SUCCESS iRecv MPI_Test, {} {}", comm.rank(),
                                             static_cast<uintptr_t>(old_request));
                                it->request = MPI_REQUEST_NULL;

                                auto buf = it->async_buffer;
                                int N    = it->range.size();
                                haloData_m.deserialize(*(buf), N);
                                unpack<Op>(it->range, view, haloData_m);
                                comm.template freeBuffer(buf);
                            } else {
                                SPDLOG_TRACE("FAIL iRecv MPI_Test, {} {}", comm.rank(),
                                             static_cast<uintptr_t>(it->request));
                                redo = true;
                            }
                        }
                    }
                    for (auto it = send_requests.begin(); it != send_requests.end(); ++it) {
                        int flag = 0;
                        MPI_Status status;
                        SPDLOG_TRACE("MPI_Test send tag {:04}, req {}", it->tag,
                                     static_cast<uintptr_t>(it->request));
                        if (it->request != MPI_REQUEST_NULL) {
                            auto old_request = it->request;
                            MPI_Test(&it->request, &flag, &status);
                            if (flag) {
                                SPDLOG_DEBUG("SUCCESS iSend MPI_Test, {} {}", comm.rank(),
                                             static_cast<uintptr_t>(old_request));
                                it->request = MPI_REQUEST_NULL;
                                comm.template freeBuffer(it->async_buffer);
                            } else {
                                SPDLOG_TRACE("FAIL iSend MPI_Test, {} {}", comm.rank(),
                                             static_cast<uintptr_t>(it->request));
                                redo = true;
                            }
                        }
                    }
                }
            }
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::pack(const bound_type& range, const view_type& view,
                                                  databuffer_type& fd, size_type& nsends) {
            auto subview = makeSubview(view, range);

            auto& buffer = fd.buffer;

            size_t size = subview.size();
            nsends      = size;
            if (buffer.size() < size) {
                int overalloc = Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            using index_array_type =
                typename RangePolicy<Dim, typename view_type::execution_space>::index_array_type;
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

        template <typename T, unsigned Dim, class... ViewArgs>
        template <typename Op>
        void HaloCells<T, Dim, ViewArgs...>::unpack(const bound_type& range, const view_type& view,
                                                    databuffer_type& fd) {
            auto subview = makeSubview(view, range);
            auto buffer  = fd.buffer;

            // 29. November 2020
            // https://stackoverflow.com/questions/3735398/operator-as-template-parameter
            Op op;

            using index_array_type =
                typename RangePolicy<Dim, typename view_type::execution_space>::index_array_type;
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

        template <typename T, unsigned Dim, class... ViewArgs>
        auto HaloCells<T, Dim, ViewArgs...>::makeSubview(const view_type& view,
                                                         const bound_type& intersect) {
            auto makeSub = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
                return Kokkos::subview(view,
                                       Kokkos::make_pair(intersect.lo[Idx], intersect.hi[Idx])...);
            };
            return makeSub(std::make_index_sequence<Dim>{});
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
