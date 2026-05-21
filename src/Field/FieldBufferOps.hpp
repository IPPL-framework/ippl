//
// Shared pack / unpack / solver_send / solver_recv helpers for Field-like
// communication patterns. Originally inlined in FFTOpenPoissonSolver.hpp and
// promoted here so other utilities (e.g. mirrorField) can reuse the exact same
// device-parallel, CUDA-aware-MPI-compatible primitives.
//
// All functions operate on Kokkos views and ippl::detail::FieldBufferData
// buffers whose memory space is inferred from the view, so data stays on the
// device when the underlying MPI is GPU-aware.
//
#ifndef IPPL_FIELD_BUFFER_OPS_HPP
#define IPPL_FIELD_BUFFER_OPS_HPP

#include <vector>

#include "Types/Vector.h"

#include "Communicate/Archive.h"
#include "Field/HaloCells.h"
#include "Index/NDIndex.h"

namespace ippl {
    namespace detail {
        // Access a view that either contains a scalar, vector, or matrix field
        // so that the correct element is selected at compile time. Used by
        // unpack_impl to support rank-0 (scalar), rank-1 (vector), and rank-2
        // (matrix) field types with a single kernel.
        template <int tensorRank, typename View, unsigned Dim>
        struct ViewAccess;

        template <typename View, unsigned Dim>
        struct ViewAccess<2, View, Dim> {
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            KOKKOS_INLINE_FUNCTION constexpr static auto& get(View&& view, unsigned dim1,
                                                              unsigned dim2,
                                                              const index_array_type& args) {
                return apply(view, args)[dim1][dim2];
            }
        };

        template <typename View, unsigned Dim>
        struct ViewAccess<1, View, Dim> {
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            KOKKOS_INLINE_FUNCTION constexpr static auto& get(View&& view, unsigned dim1,
                                                              [[maybe_unused]] unsigned dim2,
                                                              const index_array_type& args) {
                return apply(view, args)[dim1];
            }
        };

        template <typename View, unsigned Dim>
        struct ViewAccess<0, View, Dim> {
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            KOKKOS_INLINE_FUNCTION constexpr static auto& get(View&& view,
                                                              [[maybe_unused]] unsigned dim1,
                                                              [[maybe_unused]] unsigned dim2,
                                                              const index_array_type& args) {
                return apply(view, args);
            }
        };

        // Pack a 3D view region into a linear buffer via the real-part of each
        // cell (FFT-solver semantics: tolerates real or Kokkos::complex
        // views; discards imaginary parts of complex views).
        template <typename Tb, typename View, unsigned Dim>
        inline void pack(const ippl::NDIndex<Dim> intersect, View& view,
                         ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                         const ippl::NDIndex<Dim> ldom,
                         ippl::mpi::Communicator::size_type& nsends) {
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

            Kokkos::View<Tb*>& buffer = fd.buffer;

            size_t size = intersect.size();
            nsends      = size;
            if (buffer.size() < size) {
                const int overalloc = ippl::Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            using index_type = typename ippl::RangePolicy<Dim>::index_type;
            Kokkos::Array<index_type, Dim> first, last;
            for (unsigned d = 0; d < Dim; ++d) {
                first[d] = intersect[d].first() - ldom[d].first() + nghost;
                last[d]  = intersect[d].last() - ldom[d].first() + nghost + 1;
            }

            ippl::parallel_for("ippl::detail::pack()", ippl::createRangePolicy(first, last),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    Vector<int, Dim> igVec = args;
                    for (unsigned d = 0; d < Dim; ++d) {
                        igVec[d] -= first[d];
                    }

                    int l = igVec[0];
                    for (unsigned d = 1; d < Dim; ++d) {
                        int factor = 1;
                        for (unsigned d1 = 0; d1 < d; ++d1) {
                            factor *= intersect[d1].length();
                        }
                        l += igVec[d] * factor; 
                    }

                    Kokkos::complex<Tb> val = apply(view, args);
                    buffer(l)               = Kokkos::real(val);
                });
            Kokkos::fence();
        }

        // Pack a 3D view region into a linear buffer by direct element copy.
        // Works for any value type (scalar real, ippl::Vector, ...). Used by
        // mirrorField and any caller that needs value-preserving communication.
        template <typename Tb, typename View, unsigned Dim>
        inline void pack_field(const ippl::NDIndex<Dim> intersect, View& view,
                               ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                               const ippl::NDIndex<Dim> ldom,
                               ippl::mpi::Communicator::size_type& nsends) {
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

            Kokkos::View<Tb*>& buffer = fd.buffer;

            size_t size = intersect.size();
            nsends      = size;
            if (buffer.size() < size) {
                const int overalloc = ippl::Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }
            
            using index_type = typename ippl::RangePolicy<Dim>::index_type;
            Kokkos::Array<index_type, Dim> first, last;
            for (unsigned d = 0; d < Dim; ++d) {
                first[d] = intersect[d].first() - ldom[d].first() + nghost;
                last[d]  = intersect[d].last() - ldom[d].first() + nghost + 1;
            }

            ippl::parallel_for("ippl::detail::pack_field()", ippl::createRangePolicy(first, last),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    Vector<int, Dim> igVec = args;
                    for (unsigned d = 0; d < Dim; ++d) {
                        igVec[d] -= first[d];
                    }

                    int l = igVec[0];
                    for (unsigned d = 1; d < Dim; ++d) {
                        int factor = 1;
                        for (unsigned d1 = 0; d1 < d; ++d1) {
                            factor *= intersect[d1].length();
                        }
                        l += igVec[d] * factor; 
                    }
                    buffer(l) = apply(view, args);
                });
            Kokkos::fence();
        }

        // Unpack a linear buffer into a view region, with per-axis conditional
        // reflection of the buffer index. Setting a dimension to true via the
        // coordBool list reverses the buffer ordering along that axis as it is 
        // placed into the view — the primitive operation behind `mirrorField`
        // and behind the Vico solver's reflected-quadrant assembly.
        template <int tensorRank, typename Tb, typename View, unsigned Dim>
        inline void unpack_impl(const ippl::NDIndex<Dim> intersect,
                                const View& view,
                                ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                                const ippl::NDIndex<Dim> ldom,
                                ippl::Vector<bool, Dim> coordBool,
                                size_t dim1 = 0, size_t dim2 = 0) {
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

            Kokkos::View<Tb*>& buffer = fd.buffer;

            using index_type = typename ippl::RangePolicy<Dim>::index_type;
            Kokkos::Array<index_type, Dim> first, last;
            for (unsigned d = 0; d < Dim; ++d) {
                first[d] = intersect[d].first() - ldom[d].first() + nghost;
                last[d]  = intersect[d].last() - ldom[d].first() + nghost + 1;
            }

            ippl::parallel_for("ippl::detail::unpack_impl()",
                ippl::createRangePolicy(first, last),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    Vector<int, Dim> igVec = args;
                    for (unsigned d = 0; d < Dim; ++d) {
                        igVec[d] -= first[d];
                    }


                    for (unsigned d = 0; d < Dim; ++d) {
                        igVec[d] = coordBool[d] 
                                   * (intersect[d].length() - 2 * igVec[d] - 1)
                                   + igVec[d];
                    }

                    int l = igVec[0];
                    for (unsigned d = 1; d < Dim; ++d) {
                        int factor = 1;
                        for (unsigned d1 = 0; d1 < d; ++d1) {
                            factor *= intersect[d1].length();
                        }
                        l += igVec[d] * factor; 
                    }

                    ippl::detail::ViewAccess<tensorRank, decltype(view), Dim>::get(view,
                        dim1, dim2, args) = buffer(l);
                });
            Kokkos::fence();
        }

        template <typename Tb, typename View, unsigned Dim>
        inline void unpack(const ippl::NDIndex<Dim> intersect, const View& view,
                           ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                           const ippl::NDIndex<Dim> ldom,
                           ippl::Vector<bool, Dim> coordBool = false) {
            unpack_impl<0, Tb, View, Dim>(intersect, view, fd, nghost, ldom, coordBool);
        }

        template <typename Tb, typename View, unsigned Dim>
        inline void unpack(const ippl::NDIndex<Dim> intersect,
                           const View& view, size_t dim1,
                           ippl::detail::FieldBufferData<Tb>& fd, 
                           int nghost, const ippl::NDIndex<Dim> ldom,
                           ippl::Vector<bool, Dim> coordBool = false) {
            unpack_impl<1, Tb, View, Dim>(intersect, view, fd, nghost, ldom, coordBool,
                                          dim1);
        }

        template <typename Tb, typename View, unsigned Dim>
        inline void unpack(const ippl::NDIndex<Dim> intersect,
                           View& view, ippl::detail::FieldBufferData<Tb>& fd,
                           int nghost, const ippl::NDIndex<Dim> ldom,
                           size_t dim1, size_t dim2, 
                           ippl::Vector<bool, Dim> coordBool = false) {
            unpack_impl<2, Tb, View, Dim>(intersect, view, fd, nghost, ldom, coordBool,
                                          dim1, dim2);
        }

        // Async MPI_Isend wrapper: pack the intersection into a device-resident
        // buffer (picked up from the view's memory space via Comm->getBuffer),
        // issue Comm->isend, and append the resulting request to `requests`.
        template <typename Tb, typename View, unsigned Dim>
        inline void solver_send(int TAG, int id, int i, const ippl::NDIndex<Dim> intersection,
                                const ippl::NDIndex<Dim> ldom, int nghost,
                                View& view, ippl::detail::FieldBufferData<Tb>& fd,
                                std::vector<MPI_Request>& requests) {
            using memory_space = typename View::memory_space;

            requests.resize(requests.size() + 1);

            ippl::mpi::Communicator::size_type nsends;
            pack(intersection, view, fd, nghost, ldom, nsends);

            ippl::mpi::Communicator::buffer_type<memory_space> buf =
                ippl::Comm->getBuffer<memory_space, Tb>(nsends);

            int tag = TAG + id;

            ippl::Comm->isend(i, tag, fd, *buf, requests.back(), nsends);
            buf->resetWritePos();
        }

        template <typename Tb, typename View, unsigned Dim>
        inline void solver_recv(int TAG, int id, int i, const ippl::NDIndex<Dim> intersection,
                                const ippl::NDIndex<Dim> ldom, int nghost,
                                View& view, ippl::detail::FieldBufferData<Tb>& fd,
                                ippl::Vector<bool, Dim> coordBool = false) {
            using memory_space = typename View::memory_space;

            ippl::mpi::Communicator::size_type nrecvs;
            nrecvs = intersection.size();

            ippl::mpi::Communicator::buffer_type<memory_space> buf =
                ippl::Comm->getBuffer<memory_space, Tb>(nrecvs);

            int tag = TAG + id;

            ippl::Comm->recv(i, tag, fd, *buf, nrecvs * sizeof(Tb), nrecvs);
            buf->resetReadPos();

            unpack(intersection, view, fd, nghost, ldom, coordBool);
        }

        // Variant of solver_send using pack_field (direct element copy). Needed
        // when the view's element type is not a real scalar (e.g. ippl::Vector).
        template <typename Tb, typename View, unsigned Dim>
        inline void solver_send_field(int TAG, int id, int i,
                                      const ippl::NDIndex<Dim> intersection,
                                      const ippl::NDIndex<Dim> ldom, int nghost,
                                      View& view,
                                      ippl::detail::FieldBufferData<Tb>& fd,
                                      std::vector<MPI_Request>& requests) {
            using memory_space = typename View::memory_space;

            requests.resize(requests.size() + 1);

            ippl::mpi::Communicator::size_type nsends;
            pack_field(intersection, view, fd, nghost, ldom, nsends);

            ippl::mpi::Communicator::buffer_type<memory_space> buf =
                ippl::Comm->getBuffer<memory_space, Tb>(nsends);

            int tag = TAG + id;

            ippl::Comm->isend(i, tag, fd, *buf, requests.back(), nsends);
            buf->resetWritePos();
        }

    }  // namespace detail
}  // namespace ippl

#endif  // IPPL_FIELD_BUFFER_OPS_HPP
