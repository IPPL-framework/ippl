//
// Class NDIndex
//   This is a simple wrapper around Index that just keeps track of
//   N of them and passes along requests for intersect, etc.
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
#ifndef IPPL_NDINDEX_H
#define IPPL_NDINDEX_H

#include <initializer_list>

#include "Index/Index.h"

namespace ippl {
    /*!
     * @file NDIndex.h
     * @tparam Dim the number of index dimensions
     */
    template <unsigned Dim>
    class NDIndex
    {
    public:
        KOKKOS_FUNCTION
        NDIndex() {}

        template <class... Args>
        KOKKOS_FUNCTION
        NDIndex(const Args&... args);

        /*!
         * @returns a reference to any of the Indexes.
         */
        KOKKOS_INLINE_FUNCTION
        const ippl::Index& operator[](unsigned d) const noexcept;

        KOKKOS_INLINE_FUNCTION
        Index& operator[](unsigned d) noexcept;

        /*!
         * @returns the total size.
         */
        KOKKOS_INLINE_FUNCTION
        unsigned size() const noexcept;

        /*!
         * @returns true if empty.
         */
        KOKKOS_INLINE_FUNCTION
        bool empty() const noexcept;

        /*!
         * Intersect with another NDIndex.
         */
        KOKKOS_INLINE_FUNCTION
        NDIndex<Dim> intersect(const NDIndex<Dim>&) const;

        /*!
         * Intersect with another NDIndex.
         */
        KOKKOS_INLINE_FUNCTION
        NDIndex<Dim> grow(int ncells) const;

        KOKKOS_INLINE_FUNCTION
        NDIndex<Dim> grow(int ncells, unsigned int dim) const;


        KOKKOS_INLINE_FUNCTION
        bool touches(const NDIndex<Dim>&) const;

        KOKKOS_INLINE_FUNCTION
        bool contains(const NDIndex<Dim>& a) const;

        // Split on dimension d with the given ratio 0<a<1.
        KOKKOS_INLINE_FUNCTION
        bool split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d, double a) const;

        // Split on dimension d, or the longest dimension.
        KOKKOS_INLINE_FUNCTION
        bool split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d) const;

        KOKKOS_INLINE_FUNCTION
        bool split(NDIndex<Dim>& l, NDIndex<Dim>& r) const;

    private:
        KOKKOS_FUNCTION
        NDIndex(std::initializer_list<Index> indices);

        //! Array of indices
        Index indices_m[Dim];
    };
}

#include "Index/NDIndex.hpp"

#endif