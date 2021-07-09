//
// Class Ippl
//   Class to (de-)serialize in MPI communication.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
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
            using buffer_type = typename ViewType<char, 1, Properties...>::view_type;
            using pointer_type = typename buffer_type::pointer_type;

            Archive(size_type size = 0);

            /*!
             * Serialize.
             * @param view to take data from.
             */
            template <typename T>
            void operator<<(const Kokkos::View<T*>& view);

            template <typename T>
            void serialize(const Kokkos::View<T*>& view, count_type nsends);

            /*!
             * Serialize vector attributes
             *
             *\remark We need a specialized function for vectors since the vector is
             * not a trivially copyable class. Hence, we cannot use std::memcpy directly.
             *
             * @param view to take data from.
             */
            template <typename T, unsigned Dim>
            void operator<<(const Kokkos::View<Vector<T, Dim>*>& view);

            template <typename T, unsigned Dim>
            void serialize(const Kokkos::View<Vector<T, Dim>*>& view, count_type nsends);

            /*!
             * Deserialize.
             * @param view to put data to
             */
            template <typename T>
            void operator>>(Kokkos::View<T*>& view);

            template <typename T>
            void deserialize(Kokkos::View<T*>& view, count_type nrecvs);

            /*!
             * Deserialize vector attributes
             *
             * \remark We need a specialized function for vectors since the vector is
             * not a trivially copyable class. Hence, we cannot use std::memcpy directly.
             *
             * @param view to put data to
             */
            template <typename T, unsigned Dim>
            void operator>>(Kokkos::View<Vector<T, Dim>*>& view);

            template <typename T, unsigned Dim>
            void deserialize(Kokkos::View<Vector<T, Dim>*>& view, count_type nrecvs);

            /*!
             * @returns a pointer to the data of the buffer
             */
            pointer_type getBuffer() {
                return buffer_m.data();
            }


            /*!
             * @returns the size of the buffer
             */
            size_type getSize() const {
                return writepos_m;
            }

            size_type getBufferSize() const {
                return buffer_m.size();
            }

            void resizeBuffer(size_type size) {
                Kokkos::resize(buffer_m, size);
            }
            
            void resetWritePos() {
                writepos_m = 0;
            }
            void resetReadPos() {
                readpos_m = 0;
            }

            ~Archive() = default;

        private:
            //! write position for serialization
            size_type writepos_m;
            //! read position for deserialization
            size_type readpos_m;
            //! serialized data
            buffer_type buffer_m;
        };
    }
}

#include "Archive.hpp"

#endif
