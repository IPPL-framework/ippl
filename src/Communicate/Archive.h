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
#ifndef IPPL_ARCHIVE_H
#define IPPL_ARCHIVE_H

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

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
            using pointer_type = typename buffer_type::pointer_type;

            Archive(size_type size = 0);

            /*!
             * Serialize.
             * @param view to take data from.
             */
            template <typename T, class... ViewArgs>
            void serialize(const Kokkos::View<T*, ViewArgs...>& view, size_type nsends);

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

            /*!
             * Deserialize.
             * @param view to put data to
             */
            template <typename T, class... ViewArgs>
            void deserialize(Kokkos::View<T*, ViewArgs...>& view, size_type nrecvs);

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
             * @returns a pointer to the data of the buffer
             */
            pointer_type getBuffer() { return buffer_m.data(); }

            /*!
             * @returns the size of the buffer
             */
            size_type getSize() const { return writepos_m; }

            size_type getBufferSize() const { return buffer_m.size(); }

            void resizeBuffer(size_type size) { Kokkos::resize(buffer_m, size); }

            void reallocBuffer(size_type size) { Kokkos::realloc(buffer_m, size); }

            void resetWritePos() { writepos_m = 0; }
            void resetReadPos() { readpos_m = 0; }

            ~Archive() = default;

        private:
            //! write position for serialization
            size_type writepos_m;
            //! read position for deserialization
            size_type readpos_m;
            //! serialized data
            buffer_type buffer_m;
        };
    }  // namespace detail
}  // namespace ippl

#include "Archive.hpp"

#endif
