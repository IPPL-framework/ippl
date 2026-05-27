//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
#include "Archive.h"

namespace ippl {
    namespace detail {
        KOKKOS_INLINE_FUNCTION void copyBytes(char* dst, const char* src, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                dst[i] = src[i];
            }
        }

        template <class... Properties>
        Archive<Properties...>::Archive(size_type size)
            : writepos_m(0)
            , readpos_m(0)
            , buffer_m("buffer", size) {}

        // -----------------------------------
        // Scalar serialize
        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::serialize(const Kokkos::View<T*, ViewArgs...>& view,
                                               size_type nsends) {
            constexpr size_t size = sizeof(T);
            char* dst_ptr         = (char*)(buffer_m.data()) + writepos_m;
            char* src_ptr         = (char*)(view.data());
            assert(writepos_m + (nsends * size) <= buffer_m.size());
            // construct temp views of the src/dst buffers of the correct size (bytes)
            using src_view_type =
                Kokkos::View<char*, typename Kokkos::View<T*, ViewArgs...>::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            using dst_view_type =
                Kokkos::View<char*, typename buffer_type::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            src_view_type src_view(src_ptr, size * nsends);
            dst_view_type dst_view(dst_ptr, size * nsends);
            Kokkos::deep_copy(dst_view, src_view);
            Kokkos::fence();
            writepos_m += (nsends * size);
        }

        // -----------------------------------
        // Vector serialize
        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::serialize(
            const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view, size_type nsends) {
            constexpr size_t size         = sizeof(T);
            char* dst_ptr                 = (char*)(buffer_m.data());
            ippl::Vector<T, Dim>* src_ptr = view.data();
            auto wp                       = writepos_m;
            // The Kokkos range policies expect int64
            // so we have to explicitly specify size_type (uint64)
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            Kokkos::parallel_for(
                "Archive::serialize()", mdrange_t({0, 0}, {(long int)nsends, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    const char* src = reinterpret_cast<const char*>(&src_ptr[i][d]);
                    char* dst       = dst_ptr + (Dim * i + d) * size + wp;
                    copyBytes(dst, src, size);
                });

            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        // -----------------------------------
        // Scalar Deserialize
        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<T*, ViewArgs...>& view,
                                                 size_type nrecvs) {
            // if we have to enlarge the destination view
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            //
            constexpr size_t size = sizeof(T);
            char* src_ptr         = (char*)(buffer_m.data()) + readpos_m;
            char* dst_ptr         = (char*)(view.data());
            assert(readpos_m + (nrecvs * size) <= buffer_m.size());
            // construct temp views of the src/dst buffers of the correct size (bytes)
            using src_view_type =
                Kokkos::View<char*, typename buffer_type::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            using dst_view_type =
                Kokkos::View<char*, typename Kokkos::View<T*, ViewArgs...>::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            src_view_type src_view(src_ptr, size * nrecvs);
            dst_view_type dst_view(dst_ptr, size * nrecvs);
            Kokkos::deep_copy(dst_view, src_view);
            Kokkos::fence();
            readpos_m += (nrecvs * size);
        }

        // -----------------------------------
        // Vector Deserialize
        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                                 size_type nrecvs) 
        {
            // if we have to enlarge the destination view
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            //
            constexpr size_t size         = sizeof(T);
            char* src_ptr                 = (char*)(buffer_m.data());
            ippl::Vector<T, Dim>* dst_ptr = view.data();
            auto rp                       = readpos_m;
            using exec_space              = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            Kokkos::parallel_for(
                "Archive::deserialize()", mdrange_t({0, 0}, {(long int)nrecvs, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    const char* src = src_ptr + (Dim * i + d) * size + rp;
                    char* dst       = reinterpret_cast<char*>(&dst_ptr[i][d]);
                    copyBytes(dst, src, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }
    }  // namespace detail
}  // namespace ippl
