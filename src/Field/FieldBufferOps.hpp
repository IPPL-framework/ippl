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
        template <int tensorRank, typename>
        struct ViewAccess;

        template <typename View>
        struct ViewAccess<2, View> {
            KOKKOS_INLINE_FUNCTION constexpr static auto& get(View&& view, unsigned dim1,
                                                              unsigned dim2, size_t i, size_t j,
                                                              size_t k) {
                return view(i, j, k)[dim1][dim2];
            }
        };

        template <typename View>
        struct ViewAccess<1, View> {
            KOKKOS_INLINE_FUNCTION constexpr static auto& get(View&& view, unsigned dim1,
                                                              [[maybe_unused]] unsigned dim2,
                                                              size_t i, size_t j, size_t k) {
                return view(i, j, k)[dim1];
            }
        };

        template <typename View>
        struct ViewAccess<0, View> {
            KOKKOS_INLINE_FUNCTION constexpr static auto& get(View&& view,
                                                              [[maybe_unused]] unsigned dim1,
                                                              [[maybe_unused]] unsigned dim2,
                                                              size_t i, size_t j, size_t k) {
                return view(i, j, k);
            }
        };

        // Pack a 3D view region into a linear buffer via the real-part of each
        // cell (FFT-solver semantics: tolerates real or Kokkos::complex
        // views; discards imaginary parts of complex views).
        template <typename Tb, typename Tf>
        inline void pack(const ippl::NDIndex<3> intersect, Kokkos::View<Tf***>& view,
                         ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                         const ippl::NDIndex<3> ldom,
                         ippl::mpi::Communicator::size_type& nsends) {
            Kokkos::View<Tb*>& buffer = fd.buffer;

            size_t size = intersect.size();
            nsends      = size;
            if (buffer.size() < size) {
                const int overalloc = ippl::Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            const int first0 = intersect[0].first() + nghost - ldom[0].first();
            const int first1 = intersect[1].first() + nghost - ldom[1].first();
            const int first2 = intersect[2].first() + nghost - ldom[2].first();

            const int last0 = intersect[0].last() + nghost - ldom[0].first() + 1;
            const int last1 = intersect[1].last() + nghost - ldom[1].first() + 1;
            const int last2 = intersect[2].last() + nghost - ldom[2].first() + 1;

            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "ippl::detail::pack()",
                mdrange_type({first0, first1, first2},
                             {(long int)last0, (long int)last1, (long int)last2}),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    const int ig = i - first0;
                    const int jg = j - first1;
                    const int kg = k - first2;

                    int l = ig + jg * intersect[0].length()
                            + kg * intersect[1].length() * intersect[0].length();

                    Kokkos::complex<Tb> val = view(i, j, k);
                    buffer(l)               = Kokkos::real(val);
                });
            Kokkos::fence();
        }

        // Pack a 3D view region into a linear buffer by direct element copy.
        // Works for any value type (scalar real, ippl::Vector, ...). Used by
        // mirrorField and any caller that needs value-preserving communication.
        template <typename Tb, typename Tf>
        inline void pack_field(const ippl::NDIndex<3> intersect, Kokkos::View<Tf***>& view,
                               ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                               const ippl::NDIndex<3> ldom,
                               ippl::mpi::Communicator::size_type& nsends) {
            Kokkos::View<Tb*>& buffer = fd.buffer;

            size_t size = intersect.size();
            nsends      = size;
            if (buffer.size() < size) {
                const int overalloc = ippl::Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            const int first0 = intersect[0].first() + nghost - ldom[0].first();
            const int first1 = intersect[1].first() + nghost - ldom[1].first();
            const int first2 = intersect[2].first() + nghost - ldom[2].first();

            const int last0 = intersect[0].last() + nghost - ldom[0].first() + 1;
            const int last1 = intersect[1].last() + nghost - ldom[1].first() + 1;
            const int last2 = intersect[2].last() + nghost - ldom[2].first() + 1;

            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "ippl::detail::pack_field()",
                mdrange_type({first0, first1, first2},
                             {(long int)last0, (long int)last1, (long int)last2}),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    const int ig = i - first0;
                    const int jg = j - first1;
                    const int kg = k - first2;

                    int l = ig + jg * intersect[0].length()
                            + kg * intersect[1].length() * intersect[0].length();

                    buffer(l) = view(i, j, k);
                });
            Kokkos::fence();
        }

        // Unpack a linear buffer into a 3D view region, with per-axis conditional
        // reflection of the buffer index. Setting x / y / z to true reverses the
        // buffer ordering along that axis as it is placed into the view — the
        // primitive operation behind `mirrorField` and behind the Vico solver's
        // reflected-quadrant assembly.
        template <int tensorRank, typename Tb, typename Tf>
        inline void unpack_impl(const ippl::NDIndex<3> intersect,
                                const Kokkos::View<Tf***>& view,
                                ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                                const ippl::NDIndex<3> ldom, size_t dim1 = 0, size_t dim2 = 0,
                                bool x = false, bool y = false, bool z = false) {
            Kokkos::View<Tb*>& buffer = fd.buffer;

            const int first0 = intersect[0].first() + nghost - ldom[0].first();
            const int first1 = intersect[1].first() + nghost - ldom[1].first();
            const int first2 = intersect[2].first() + nghost - ldom[2].first();

            const int last0 = intersect[0].last() + nghost - ldom[0].first() + 1;
            const int last1 = intersect[1].last() + nghost - ldom[1].first() + 1;
            const int last2 = intersect[2].last() + nghost - ldom[2].first() + 1;

            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "ippl::detail::unpack_impl()",
                mdrange_type({first0, first1, first2}, {last0, last1, last2}),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    int ig = i - first0;
                    int jg = j - first1;
                    int kg = k - first2;

                    ig = x * (intersect[0].length() - 2 * ig - 1) + ig;
                    jg = y * (intersect[1].length() - 2 * jg - 1) + jg;
                    kg = z * (intersect[2].length() - 2 * kg - 1) + kg;

                    int l = ig + jg * intersect[0].length()
                            + kg * intersect[1].length() * intersect[0].length();

                    ippl::detail::ViewAccess<tensorRank, decltype(view)>::get(view, dim1, dim2, i,
                                                                              j, k) = buffer(l);
                });
            Kokkos::fence();
        }

        template <typename Tb, typename Tf>
        inline void unpack(const ippl::NDIndex<3> intersect, const Kokkos::View<Tf***>& view,
                           ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                           const ippl::NDIndex<3> ldom, bool x = false, bool y = false,
                           bool z = false) {
            unpack_impl<0, Tb, Tf>(intersect, view, fd, nghost, ldom, 0, 0, x, y, z);
        }

        template <typename Tb, typename Tf>
        inline void unpack(const ippl::NDIndex<3> intersect,
                           const Kokkos::View<ippl::Vector<Tf, 3>***>& view, size_t dim1,
                           ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                           const ippl::NDIndex<3> ldom) {
            unpack_impl<1, Tb, ippl::Vector<Tf, 3>>(intersect, view, fd, nghost, ldom, dim1);
        }

        template <typename Tb, typename Tf>
        inline void unpack(const ippl::NDIndex<3> intersect,
                           const Kokkos::View<ippl::Vector<ippl::Vector<Tf, 3>, 3>***>& view,
                           ippl::detail::FieldBufferData<Tb>& fd, int nghost,
                           const ippl::NDIndex<3> ldom, size_t dim1, size_t dim2) {
            unpack_impl<2, Tb, ippl::Vector<ippl::Vector<Tf, 3>, 3>>(intersect, view, fd, nghost,
                                                                    ldom, dim1, dim2);
        }

        // Async MPI_Isend wrapper: pack the intersection into a device-resident
        // buffer (picked up from the view's memory space via Comm->getBuffer),
        // issue Comm->isend, and append the resulting request to `requests`.
        template <typename Tb, typename Tf, unsigned Dim>
        inline void solver_send(int TAG, int id, int i, const ippl::NDIndex<Dim> intersection,
                                const ippl::NDIndex<Dim> ldom, int nghost,
                                Kokkos::View<Tf***>& view,
                                ippl::detail::FieldBufferData<Tb>& fd,
                                std::vector<MPI_Request>& requests) {
            using memory_space = typename Kokkos::View<Tf***>::memory_space;

            requests.resize(requests.size() + 1);

            ippl::mpi::Communicator::size_type nsends;
            pack(intersection, view, fd, nghost, ldom, nsends);

            ippl::mpi::Communicator::buffer_type<memory_space> buf =
                ippl::Comm->getBuffer<memory_space, Tf>(nsends);

            int tag = TAG + id;

            ippl::Comm->isend(i, tag, fd, *buf, requests.back(), nsends);
            buf->resetWritePos();
        }

        template <typename Tb, typename Tf, unsigned Dim>
        inline void solver_recv(int TAG, int id, int i, const ippl::NDIndex<Dim> intersection,
                                const ippl::NDIndex<Dim> ldom, int nghost,
                                Kokkos::View<Tf***>& view,
                                ippl::detail::FieldBufferData<Tb>& fd, bool x = false,
                                bool y = false, bool z = false) {
            using memory_space = typename Kokkos::View<Tf***>::memory_space;

            ippl::mpi::Communicator::size_type nrecvs;
            nrecvs = intersection.size();

            ippl::mpi::Communicator::buffer_type<memory_space> buf =
                ippl::Comm->getBuffer<memory_space, Tf>(nrecvs);

            int tag = TAG + id;

            ippl::Comm->recv(i, tag, fd, *buf, nrecvs * sizeof(Tf), nrecvs);
            buf->resetReadPos();

            unpack(intersection, view, fd, nghost, ldom, x, y, z);
        }

        // Variant of solver_send using pack_field (direct element copy). Needed
        // when the view's element type is not a real scalar (e.g. ippl::Vector).
        template <typename Tb, typename Tf, unsigned Dim>
        inline void solver_send_field(int TAG, int id, int i,
                                      const ippl::NDIndex<Dim> intersection,
                                      const ippl::NDIndex<Dim> ldom, int nghost,
                                      Kokkos::View<Tf***>& view,
                                      ippl::detail::FieldBufferData<Tb>& fd,
                                      std::vector<MPI_Request>& requests) {
            using memory_space = typename Kokkos::View<Tf***>::memory_space;

            requests.resize(requests.size() + 1);

            ippl::mpi::Communicator::size_type nsends;
            pack_field(intersection, view, fd, nghost, ldom, nsends);

            ippl::mpi::Communicator::buffer_type<memory_space> buf =
                ippl::Comm->getBuffer<memory_space, Tf>(nsends);

            int tag = TAG + id;

            ippl::Comm->isend(i, tag, fd, *buf, requests.back(), nsends);
            buf->resetWritePos();
        }

    }  // namespace detail
}  // namespace ippl

#endif  // IPPL_FIELD_BUFFER_OPS_HPP
