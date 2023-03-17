//
// Class NDRegion
//   NDRegion is a simple container of N PRegion objects. It is templated
//   on the type of data (T) and the number of PRegions (Dim).
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef IPPL_NDREGION_H
#define IPPL_NDREGION_H

#include <initializer_list>

#include "Region/PRegion.h"

namespace ippl {
    template <typename T, unsigned Dim>
    /*!
     * @file NDRegion.h
     * @tparam T data type
     * @tparam Dim number of PRegions
     */
    class NDRegion {
    public:
        /*!
         * Create an empty NDregion
         */
        KOKKOS_FUNCTION
        NDRegion() {}

        KOKKOS_FUNCTION
        ~NDRegion() {}

        /*!
         * Create a NDregion from PRegions
         * @param ...args list of PRegions
         *
         * \remark See also (November 21, 2020)
         * https://stackoverflow.com/questions/16478089/converting-variadic-template-pack-into-stdinitializer-list
         */
        template <class... Args>
        KOKKOS_FUNCTION NDRegion(const Args&... args);

        KOKKOS_INLINE_FUNCTION NDRegion(const NDRegion<T, Dim>& nr);

        KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& operator=(const NDRegion<T, Dim>& nr);

        KOKKOS_INLINE_FUNCTION const PRegion<T>& operator[](unsigned d) const;

        KOKKOS_INLINE_FUNCTION PRegion<T>& operator[](unsigned d);

        KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& operator+=(const T t);

        KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& operator-=(const T t);

        KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& operator*=(const T t);

        KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& operator/=(const T t);

        KOKKOS_INLINE_FUNCTION bool empty() const;

    private:
        KOKKOS_FUNCTION
        NDRegion(std::initializer_list<PRegion<T>> regions);

        //! Array of PRegions
        PRegion<T> regions_m[Dim];
    };
}  // namespace ippl

#include "Region/NDRegion.hpp"

#endif