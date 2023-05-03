//
// Struct ViewType
//   Kokkos::Views of different dimensions.
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
#ifndef IPPL_VIEW_TYPES_H
#define IPPL_VIEW_TYPES_H

#include <Kokkos_Core.hpp>

namespace ippl {
    /**
     * @file ViewTypes.h
     * This file defines multi-dimensional arrays to store mesh and particle attributes.
     * It provides specialized versions for 1, 2 and 3 dimensions. The file further
     * provides write functions for the different view types.
     */
    namespace detail {
        /*!
         * Recursively templated struct for defining pointers with arbitrary
         * indirection depth.
         * @tparam T data type
         * @tparam N indirection level
         */
        template <typename T, int N>
        struct NPtr {
            typedef typename NPtr<T, N - 1>::type* type;
        };

        /*!
         * Base case template specialization for a simple pointer.
         */
        template <typename T>
        struct NPtr<T, 1> {
            typedef T* type;
        };

        /*!
         * View type for an arbitrary number of dimensions.
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         */
        template <typename T, unsigned Dim, class... Properties>
        struct ViewType {
            typedef Kokkos::View<typename NPtr<T, Dim>::type, Properties...> view_type;
        };
    }  // namespace detail
}  // namespace ippl

#endif
