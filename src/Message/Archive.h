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

namespace ippl {
    namespace detail {
        /*!
         * @tparam Properties variadic template for Kokkos::View
         */
        template <class... Properties>
        class Archive {

        public:
            using buffer_type = typename ViewType<char*, 1, Properties...>;

            Archive(size_t size = 0);

            /*!
             * Serialize.
             * @param view to take data from.
             */
            template <typename T>
            void operator<<(const ViewType<T, 1, Properties...>& view);


            /*!
             * Deserialize.
             * @param view to put data to
             */
            template <typename T>
            void operator>>(const ViewType<T, 1, Properties...>& view);


            /*!
             * @returns a pointer to the data of the buffer
             */
            void* getBuffer() const {
                return buffer_m.data();
            }


            /*!
             * @returns the size of the buffer
             */
            size_t getSize() const {
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