//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
#include <cstring>

#include "Utility/Logging.h"

#include "Archive.h"

namespace ippl {
    namespace detail {

        template <typename BufferType>
        Archive<BufferType>::Archive(size_type size)
            : writepos_m(0)
            , readpos_m(0)
            , buffer_m("buffer", size) {}

        // -----------------------------------
        // Scalar serialize
        template <typename BufferType>
        template <typename T, class... ViewArgs>
        void Archive<BufferType>::serialize(const Kokkos::View<T*, ViewArgs...>& view,
                                            size_type nsends) {
            constexpr size_t size = sizeof(T);
            char* dst_ptr         = (char*)(buffer_m.data()) + writepos_m;
            char* src_ptr         = (char*)(view.data());
            assert(writepos_m + (nsends * size) <= buffer_m.size());
            // construct temp views of the src/dst buffers of the correct size (bytes)
            Kokkos::View<char*, Kokkos::MemoryUnmanaged> src_view(src_ptr, size * nsends);
            Kokkos::View<char*, Kokkos::MemoryUnmanaged> dst_view(dst_ptr, size * nsends);
            Kokkos::deep_copy(dst_view, src_view);
            Kokkos::fence();
            SPDLOG_TRACE("Incrementing writepos: {}, from {}, to {}", (void*)dst_view.data(),
                         writepos_m, writepos_m + (nsends * size));
            writepos_m += (nsends * size);
        }

        // -----------------------------------
        // Vector serialize
        template <typename BufferType>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<BufferType>::serialize(const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                            size_type nsends) {
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
                    std::memcpy(dst_ptr + (Dim * i + d) * size + wp, &(*(src_ptr + i))[d], size);
                });

            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        // -----------------------------------
        // Scalar Deserialize
        template <typename BufferType>
        template <typename T, class... ViewArgs>
        void Archive<BufferType>::deserialize(Kokkos::View<T*, ViewArgs...>& view,
                                              size_type nrecvs) {
            // if we have to enlarge the destination view
            if (nrecvs > view.extent(0)) {
                SPDLOG_WARN("DeSerialization realloc: {}, from {}, to {}", (void*)view.data(),
                            view.extent(0), nrecvs);
                Kokkos::realloc(view, nrecvs);
            }
            //
            constexpr size_t size = sizeof(T);
            char* src_ptr         = (char*)(buffer_m.data()) + readpos_m;
            char* dst_ptr         = (char*)(view.data());
            assert(readpos_m + (nrecvs * size) <= buffer_m.size());
            // construct temp views of the src/dst buffers of the correct size (bytes)
            Kokkos::View<char*, Kokkos::MemoryUnmanaged> src_view(src_ptr, size * nrecvs);
            Kokkos::View<char*, Kokkos::MemoryUnmanaged> dst_view(dst_ptr, size * nrecvs);
            Kokkos::deep_copy(dst_view, src_view);
            Kokkos::fence();
            SPDLOG_TRACE("Incrementing readpos: {}, from {}, to {}", (void*)buffer_m.data(),
                         readpos_m, readpos_m + (nrecvs * size));
            readpos_m += (nrecvs * size);
        }

        // -----------------------------------
        // Vecto Deserialize
        template <typename BufferType>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<BufferType>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                              size_type nrecvs)  //
        {
            // if we have to enlarge the destination view
            if (nrecvs > view.extent(0)) {
                SPDLOG_WARN("DeSerialization realloc: {}, from {}, to {}", (void*)view.data(),
                            view.extent(0), nrecvs);
                Kokkos::realloc(view, nrecvs);
            }
            //
            constexpr size_t size         = sizeof(T);
            char* src_ptr                 = (char*)(buffer_m.data()) + readpos_m;
            ippl::Vector<T, Dim>* dst_ptr = view.data();
            auto rp                       = readpos_m;
            using exec_space              = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            Kokkos::parallel_for(
                "Archive::deserialize()", mdrange_t({0, 0}, {(long int)nrecvs, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    std::memcpy(&(*(dst_ptr + i))[d], src_ptr + (Dim * i + d) * size + rp, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }
    }  // namespace detail
}  // namespace ippl
