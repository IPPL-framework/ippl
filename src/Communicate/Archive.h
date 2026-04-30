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
//   On CUDA/HIP the internal buffer is allocated directly via cudaMalloc/hipMalloc
//   so that the pointer is page-aligned (4K) and compatible with MPI IPC.
//
#ifndef IPPL_ARCHIVE_H
#define IPPL_ARCHIVE_H

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

#include "Types/Vector.h"

namespace ippl {
    namespace detail {
        /*!
         * Serialize and deserialize particle attributes.
         * @tparam Properties variadic template for Kokkos::View
         */

        template <class... Properties>
        class Archive {
        public:
            using buffer_type  = typename ViewType<char, 1, Properties...>::view_type;
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

            using memory_space = typename buffer_type::memory_space;

            //! True iff this Archive's memory space is host-inaccessible
            //! (CUDA device or HIP device). UVM is excluded — it works with
            //! the regular Kokkos::View path because the host can address
            //! the memory directly. For a HostSpace archive the host-side
            //! memcpy in serialize() requires a host-accessible buffer, so
            //! the raw device allocation path must NOT be used there.
            static constexpr bool uses_raw_device_alloc =
#if defined(KOKKOS_ENABLE_CUDA)
                std::is_same_v<memory_space, Kokkos::CudaSpace>
#elif defined(KOKKOS_ENABLE_HIP)
                std::is_same_v<memory_space, Kokkos::HIPSpace>
#else
                false
#endif
                ;

        private:
            //! write position for serialization
            size_type writepos_m;
            //! read position for deserialization
            size_type readpos_m;

            //! Raw device pointer (only valid when uses_raw_device_alloc).
            pointer_type buffer_ptr_m = nullptr;
            size_type buffer_size_m   = 0;
            //! Standard Kokkos view buffer (used for host-accessible spaces).
            buffer_type buffer_m;

            pointer_type bufferData() const {
                if constexpr (uses_raw_device_alloc) {
                    return buffer_ptr_m;
                } else {
                    return buffer_m.data();
                }
            }
            size_type bufferSize() const {
                if constexpr (uses_raw_device_alloc) {
                    return buffer_size_m;
                } else {
                    return buffer_m.size();
                }
            }

            void gpuAlloc(size_type size);
            void gpuFree();
        };
    }  // namespace detail
}  // namespace ippl

#include "Archive.hpp"

#endif
