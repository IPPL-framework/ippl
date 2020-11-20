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

#include "Types/ViewTypes.h"
#include "Types/Vector.h"

namespace ippl {
    namespace detail {
        /*!
         * @tparam Properties variadic template for Kokkos::View
         */
        template <class... Properties>
        class Archive {

        public:
            using buffer_type = typename ViewType<char, 1, Properties...>::view_type;
            using pointer_type = typename buffer_type::pointer_type;
            using size_type = typename buffer_type::size_type;

            Archive(int size = 0);

            /*!
             * Serialize.
             * @param view to take data from.
             */
            template <typename T>
            void operator<<(const /*typename ViewType<T, 1, Properties...>::view_type*/Kokkos::View<T*>& view);


            /*!
             * Serialize vector attributes
             * @param view to take data from.
             */
            template <typename T, unsigned Dim>
            void operator<<(const Kokkos::View<Vector<T, Dim>*>& view);


            /*!
             * Deserialize.
             * @param view to put data to
             */
            template <typename T>
            void operator>>(typename /*ViewType<T, 1, Properties...>::view_type*/Kokkos::View<T*>& view);


            /*!
             * Deserialize vector attributes
             * @param view to put data to
             */
            template <typename T, unsigned Dim>
            void operator>>(Kokkos::View<Vector<T, Dim>*>& view);


            /*!
             * @returns a pointer to the data of the buffer
             */
            /*pointer_type*/void* getBuffer() /*const*/ {
                return buffer_m.data();
            }


            /*!
             * @returns the size of the buffer
             */
            size_type getSize() const {
                return buffer_m.size();
            }

            ~Archive() { }

        private:
            //! write position for serialization
            size_t writepos_m;
            //! read position for deserialization
            size_t readpos_m;
            //! serialized data
            buffer_type buffer_m;
        };
    }
}

#include "Archive.hpp"

#endif