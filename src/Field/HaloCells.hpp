//
// Class HaloCells
//   The guard / ghost cells of BareField.
//

#include <memory>
#include <vector>

#include "Utility/IpplException.h"
#include "Utility/ParallelDispatch.h"

#include "Communicate/Communicator.h"

namespace ippl {
    namespace detail {
        // These kernels are instantiated in multiple translation units. Named functors avoid
        // NVCC's translation-unit-local extended-lambda copy/delete helper state.
        template <typename Subview, typename Buffer, typename IndexArray, unsigned Dim>
        struct HaloPackFunctor {
            Subview subview;
            Buffer buffer;

            KOKKOS_INLINE_FUNCTION void operator()(const IndexArray& args) const {
                int l = 0;

                for (unsigned d1 = 0; d1 < Dim; ++d1) {
                    int next = args[d1];
                    for (unsigned d2 = 0; d2 < d1; ++d2) {
                        next *= subview.extent(d2);
                    }
                    l += next;
                }

                buffer(l) = apply(subview, args);
            }
        };

        template <typename Subview, typename Buffer, typename Op, typename IndexArray, unsigned Dim>
        struct HaloUnpackFunctor {
            Subview subview;
            Buffer buffer;
            Op op;

            KOKKOS_INLINE_FUNCTION void operator()(const IndexArray& args) const {
                int l = 0;

                for (unsigned d1 = 0; d1 < Dim; ++d1) {
                    int next = args[d1];
                    for (unsigned d2 = 0; d2 < d1; ++d2) {
                        next *= subview.extent(d2);
                    }
                    l += next;
                }

                op(apply(subview, args), buffer(l));
            }
        };

        // Match Archive's field-value representation without a second copy kernel.
        // Byte access avoids alignment and object-lifetime assumptions about pooled storage.
        template <typename T>
        struct HaloWireValue {
            static constexpr size_type BYTE_COUNT = sizeof(T);

            KOKKOS_INLINE_FUNCTION static void pack(char* destination, const T& value) {
                copyBytes(destination, reinterpret_cast<const char*>(&value), BYTE_COUNT);
            }

            KOKKOS_INLINE_FUNCTION static T unpack(const char* source) {
                T value{};
                copyBytes(reinterpret_cast<char*>(&value), source, BYTE_COUNT);
                return value;
            }
        };

        // Archive serializes Vector components, excluding any object padding.
        template <typename T, unsigned N>
        struct HaloWireValue<Vector<T, N>> {
            static constexpr size_type BYTE_COUNT = N * sizeof(T);

            KOKKOS_INLINE_FUNCTION static void pack(char* destination, const Vector<T, N>& value) {
                for (unsigned component = 0; component < N; ++component) {
                    const T entry = value[component];
                    copyBytes(destination + component * sizeof(T),
                              reinterpret_cast<const char*>(&entry), sizeof(T));
                }
            }

            KOKKOS_INLINE_FUNCTION static Vector<T, N> unpack(const char* source) {
                Vector<T, N> value;
                for (unsigned component = 0; component < N; ++component) {
                    T entry{};
                    copyBytes(reinterpret_cast<char*>(&entry),
                              source + component * sizeof(T), sizeof(T));
                    value[component] = entry;
                }
                return value;
            }
        };

        // The communicator's existing archive overload sends these bytes as MPI_BYTE.
        // Ownership stays with its buffer pool until exchangeBoundaries completes.
        struct HaloPackedBuffer {
            char* data;
            size_type byteCount;

            char* getBuffer() const { return data; }
            size_type getSize() const { return byteCount; }
        };

        template <typename T>
        size_type haloMessageBytes(const mpi::Communicator& comm, size_type count) {
            constexpr size_type MAX_BYTES = static_cast<size_type>(INT_MAX);
            if (count > MAX_BYTES / HaloWireValue<T>::BYTE_COUNT) {
                // Check before multiplying to avoid overflow as well as MPI truncation.
                comm.assertMessageSize(MAX_BYTES + 1);
            }
            return count * HaloWireValue<T>::BYTE_COUNT;
        }

        template <typename T, typename Subview, typename IndexArray, unsigned Dim>
        struct HaloBytePackFunctor {
            Subview subview;
            char* buffer;

            KOKKOS_INLINE_FUNCTION void operator()(const IndexArray& args) const {
                size_type offset = 0;
                size_type stride = 1;
                for (unsigned dimension = 0; dimension < Dim; ++dimension) {
                    offset += args[dimension] * stride;
                    stride *= subview.extent(dimension);
                }
                HaloWireValue<T>::pack(buffer + offset * HaloWireValue<T>::BYTE_COUNT,
                                       apply(subview, args));
            }
        };

        template <typename T, typename Subview, typename Op, typename IndexArray, unsigned Dim>
        struct HaloByteUnpackFunctor {
            Subview subview;
            const char* buffer;
            Op op;

            KOKKOS_INLINE_FUNCTION void operator()(const IndexArray& args) const {
                size_type offset = 0;
                size_type stride = 1;
                for (unsigned dimension = 0; dimension < Dim; ++dimension) {
                    offset += args[dimension] * stride;
                    stride *= subview.extent(dimension);
                }
                T value = HaloWireValue<T>::unpack(
                    buffer + offset * HaloWireValue<T>::BYTE_COUNT);
                op(apply(subview, args), value);
            }
        };

        template <typename T, unsigned Dim, typename Exec, typename Subview>
        void packHaloBytes(const Subview& subview, char* buffer) {
            using index_array_type = typename RangePolicy<Dim, Exec>::index_array_type;
            using functor_type = HaloBytePackFunctor<T, Subview, index_array_type, Dim>;
            Kokkos::Array<typename RangePolicy<Dim, Exec>::index_type, Dim> begin{}, end;
            for (unsigned d = 0; d < Dim; ++d) end[d] = subview.extent(d);
            ippl::parallel_for("HaloCells::pack()", createRangePolicy<Dim, Exec>(begin, end),
                               functor_type{subview, buffer});
            // exchangeBoundaries fences the complete pack batch before MPI reads it.
        }

        template <typename T, unsigned Dim, typename Op, typename Exec, typename Subview>
        void unpackHaloBytes(const Subview& subview, const char* buffer) {
            using index_array_type = typename RangePolicy<Dim, Exec>::index_array_type;
            using functor_type = HaloByteUnpackFunctor<T, Subview, Op, index_array_type, Dim>;
            Kokkos::Array<typename RangePolicy<Dim, Exec>::index_type, Dim> begin{}, end;
            for (unsigned d = 0; d < Dim; ++d) end[d] = subview.extent(d);
            ippl::parallel_for("HaloCells::unpack()", createRangePolicy<Dim, Exec>(begin, end),
                               functor_type{subview, buffer, Op{}});
            // Default policies share one execution instance. Ordered launches preserve
            // additions to overlapping ranges; the caller fences before buffer reuse.
        }

        template <typename View, typename Op, typename IndexArray>
        struct HaloPeriodicFunctor {
            View view;
            Op op;
            unsigned dimension;
            int extent;
            int nghost;

            KOKKOS_INLINE_FUNCTION void operator()(IndexArray coords) const {
                // The ghosts are filled starting from the inside of the domain proceeding outwards
                // for both lower and upper faces. The extra brackets and explicit mention

                // nghost + i
                coords[dimension] += nghost;
                auto&& left = apply(view, coords);

                // N - nghost - i
                coords[dimension] = extent - coords[dimension];
                auto&& right      = apply(view, coords);

                // nghost - 1 - i
                coords[dimension] += 2 * nghost - 1 - extent;
                op(apply(view, coords), right);

                // N - (nghost - 1 - i) = N - (nghost - 1) + i
                coords[dimension] = extent - coords[dimension];
                op(apply(view, coords), left);
            }
        };

        template <typename T, unsigned Dim, class... ViewArgs>
        HaloCells<T, Dim, ViewArgs...>::HaloCells() {}

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::accumulateHalo(view_type& view, Layout_t* layout,
                                                            int nghost) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL, nghost);
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::accumulateHalo_noghost(view_type& view,
                                                                    Layout_t* layout, int nghost) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL_NOGHOST, nghost);
        }
        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::fillHalo(view_type& view, Layout_t* layout,
                                                      int nghost) {
            exchangeBoundaries<assign>(view, layout, INTERNAL_TO_HALO, nghost);
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

            // Thin local domains are valid. FieldLayout classifies neighbor
            // directions from domain positions, including one-entry parallel axes.

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
            struct SendMessage {
                int rank;
                int tag;
                buffer_type buffer;
                size_type byteCount;
            };
            struct ReceiveMessage {
                bound_type range;
                buffer_type buffer;
            };
            std::vector<SendMessage> sends;
            std::vector<ReceiveMessage> receives;
            sends.reserve(totalRequests);
            receives.reserve(totalRequests);
            std::vector<MPI_Request> requests(2 * totalRequests);
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

                    const size_type nsends = range.size();
                    const size_type byteCount = haloMessageBytes<T>(comm, nsends);
                    buffer_type buf = comm.template getBuffer<memory_space>(byteCount);
                    packHaloBytes<T, Dim, typename view_type::execution_space>(
                        makeSubview(view, range), buf->getBuffer());
                    sends.push_back({targetRank, tag, buf, byteCount});
                }
            }

            // MPI, including CUDA-aware MPI, may read send storage only after all
            // packs finish. Keep one readiness fence for the entire batch.
            if (!sends.empty()) {
                Kokkos::fence();
            }

            // Post every receive before sending; retain storage through ordered unpack.
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
                            const bool isLower = globalLo == domain[j].min();
                            const bool isUpper = globalHi == domain[j].max();
                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = sendRanges[index][i];
                    }

                    const size_type nrecvs = range.size();
                    const size_type byteCount = haloMessageBytes<T>(comm, nrecvs);
                    buffer_type buf = comm.template getBuffer<memory_space>(byteCount);
                    receives.push_back({range, buf});
                    comm.irecv(sourceRank, tag, *buf, requests[requestIndex++], byteCount);
                }
            }

            for (const auto& message : sends) {
                HaloPackedBuffer packed{message.buffer->getBuffer(), message.byteCount};
                comm.isend(message.rank, message.tag, packed, requests[requestIndex++]);
            }
            if (requestIndex > 0) {
                MPI_Waitall(requestIndex, requests.data(), MPI_STATUSES_IGNORE);
            }

            // Keep the original direction/neighbor order for overlapping additions.
            // All kernels use the view execution space's default instance.
            for (const auto& message : receives) {
                unpackHaloBytes<T, Dim, Op, typename view_type::execution_space>(
                    makeSubview(view, message.range), message.buffer->getBuffer());
            }
            if (!receives.empty()) {
                Kokkos::fence();
            }
            comm.freeAllBuffers();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::pack(const bound_type& range, const view_type& view,
                                                  databuffer_type& fd, size_type& nsends) {
            auto subview = makeSubview(view, range);

            auto& buffer = fd.buffer;
            size_t size  = subview.size();
            nsends       = size;
            if (buffer.size() < size) {
                int overalloc = Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            using exec_space       = typename view_type::execution_space;
            using range_type       = RangePolicy<Dim, exec_space>;
            using index_type       = typename range_type::index_type;
            using index_array_type = typename range_type::index_array_type;
            using buffer_view_type = typename databuffer_type::view_type;
            using functor_type =
                HaloPackFunctor<decltype(subview), buffer_view_type, index_array_type, Dim>;

            Kokkos::Array<index_type, Dim> begin{};
            Kokkos::Array<index_type, Dim> end{};
            for (unsigned d = 0; d < Dim; ++d) {
                end[d] = static_cast<index_type>(subview.extent(d));
            }
            auto policy = createRangePolicy<Dim, exec_space>(begin, end);
            ippl::parallel_for("HaloCells::pack()", policy, functor_type{subview, buffer});
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

            using exec_space       = typename view_type::execution_space;
            using range_type       = RangePolicy<Dim, exec_space>;
            using index_type       = typename range_type::index_type;
            using index_array_type = typename range_type::index_array_type;
            using functor_type =
                HaloUnpackFunctor<decltype(subview), decltype(buffer), Op, index_array_type, Dim>;

            Kokkos::Array<index_type, Dim> begin{};
            Kokkos::Array<index_type, Dim> end{};
            for (unsigned d = 0; d < Dim; ++d) {
                end[d] = static_cast<index_type>(subview.extent(d));
            }
            auto policy = createRangePolicy<Dim, exec_space>(begin, end);
            ippl::parallel_for("HaloCells::unpack()", policy,
                               functor_type{subview, buffer, op});
            Kokkos::fence();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        auto HaloCells<T, Dim, ViewArgs...>::makeSubview(const view_type& view,
                                                         const bound_type& intersect) {
            auto makeSub = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
                return Kokkos::submdspan(
                    view.to_mdspan(),
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
                    using functor_type = HaloPeriodicFunctor<view_type, Op, index_array_type>;
                    ippl::parallel_for("applyPeriodicSerialDim",
                                       createRangePolicy<Dim, exec_space>(begin, end),
                                       functor_type{view, op, d, N, nghost});

                    Kokkos::fence();
                }
            }
        }
    }  // namespace detail
}  // namespace ippl
