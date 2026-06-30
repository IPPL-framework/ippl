//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
//   When data is exchanged between MPI ranks, it is stored in one dimensional
//   arrays. These have the type detail::Archive, which are wrappers around
//   one dimensional Kokkos views of type char. The data is then transferred using
//   MPI send/recv calls. Note that the archive type differs from other buffers in
//   that they have type char and thus contain raw bytes, unlike other typed buffers
//   such as detail::FieldBufferData used by HaloCells.
//
//   On CUDA/HIP device-memory archives, the internal buffer is allocated directly
//   via cudaMalloc/hipMalloc so that the pointer is compatible with MPI IPC.
//   HIP device allocations are rounded to the 64 KiB HSA IPC granularity.
//
#ifndef IPPL_ARCHIVE_H
#define IPPL_ARCHIVE_H

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

#include <type_traits>

#include "Types/Vector.h"

namespace ippl {
    namespace detail {
        /*!
         * @file Archive.h
         * Serialize and desesrialize particle attributes.
         * @tparam Properties variadic template for Kokkos::View
         */

        template <class... Properties>
        class Archive {
        public:
            using buffer_type  = typename ViewType<char, 1, Properties...>::view_type;
            using memory_space = typename buffer_type::memory_space;
            using pointer_type = typename buffer_type::pointer_type;

            Archive(size_type size = 0);
            ~Archive();

            /*!
             * Serialize.
             * @param view to take data from.
             */
            template <typename T, class... ViewArgs>
            void serialize(const Kokkos::View<T*, ViewArgs...>& view, size_type nsends);

            template <typename T, class... ViewArgs, typename HashView>
            void serialize(const Kokkos::View<T*, ViewArgs...>& view, const HashView& hash,
                           size_type nsends);

            /*!
             * Serialize vector attributes
             *
             *\remark We need a specialized function for vectors since the vector is
             * not a trivially copyable class. Hence, we cannot use std::memcpy directly.
             *
             * @param view to take data from.
             */
            template <typename T, unsigned Dim, class... ViewArgs>
            void serialize(const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                           size_type nsends);

            template <typename T, unsigned Dim, class... ViewArgs, typename HashView>
            void serialize(const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                           const HashView& hash, size_type nsends);

            /*!
             * Deserialize.
             * @param view to put data to
             */
            template <typename T, class... ViewArgs>
            void deserialize(Kokkos::View<T*, ViewArgs...>& view, size_type nrecvs);

            template <typename T, class... ViewArgs>
            void deserialize(Kokkos::View<T*, ViewArgs...>& view, size_type offset,
                             size_type nrecvs);

            template <typename T, unsigned Dim, class... ViewArgs>
            void deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view, size_type offset,
                             size_type nrecvs);

            /*!
             * Deserialize vector attributes
             *
             * \remark We need a specialized function for vectors since the vector is
             * not a trivially copyable class. Hence, we cannot use std::memcpy directly.
             *
             * @param view to put data to
             */
            template <typename T, unsigned Dim, class... ViewArgs>
            void deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view, size_type nrecvs);

            /*!
             * @returns a pointer to the data of the buffer.
             *          On GPU this is a page-aligned device pointer from cudaMalloc/hipMalloc.
             */
            pointer_type getBuffer() { return bufferData(); }

            /*!
             * @returns the number of bytes written so far
             */
            size_type getSize() const { return writepos_m; }

            /*!
             * @returns the total capacity of the buffer in bytes
             */
            size_type getBufferSize() const { return bufferSize(); }

            void resizeBuffer(size_type size);
            void reallocBuffer(size_type size);

            void resetWritePos() { writepos_m = 0; }
            void resetReadPos() { readpos_m = 0; }

        private:
            //! write position for serialization
            size_type writepos_m;
            //! read position for deserialization
            size_type readpos_m;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            //! Serialized data for host-accessible memory spaces.
            buffer_type buffer_m;

            //! Raw device pointer from cudaMalloc/hipMalloc (page-aligned, IPC-safe)
            pointer_type buffer_ptr_m = nullptr;
            size_type buffer_size_m   = 0;

            static constexpr bool useRawGpuBuffer() {
#if defined(KOKKOS_ENABLE_CUDA)
                if constexpr (std::is_same_v<memory_space, Kokkos::CudaSpace>) {
                    return true;
                }
#endif
#if defined(KOKKOS_ENABLE_HIP)
                if constexpr (std::is_same_v<memory_space, Kokkos::HIPSpace>) {
                    return true;
                }
#endif
                return false;
            }

            pointer_type bufferData() const {
                if constexpr (useRawGpuBuffer()) {
                    return buffer_ptr_m;
                } else {
                    return buffer_m.data();
                }
            }
            size_type bufferSize() const {
                if constexpr (useRawGpuBuffer()) {
                    return buffer_size_m;
                } else {
                    return buffer_m.size();
                }
            }

            void gpuAlloc(size_type size);
            void gpuFree();
#else
            //! serialized data (standard Kokkos view on CPU)
            buffer_type buffer_m;

            pointer_type bufferData() const { return buffer_m.data(); }
            size_type bufferSize() const { return buffer_m.size(); }
#endif
        };
    }  // namespace detail
}  // namespace ippl

#include "Archive.hpp"

#endif
