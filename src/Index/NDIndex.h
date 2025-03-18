//
// Class NDIndex
//   This is a simple wrapper around Index that just keeps track of
//   N of them and passes along requests for intersect, etc.
//
#ifndef IPPL_NDINDEX_H
#define IPPL_NDINDEX_H

#include <initializer_list>

#include "Types/Vector.h"

#include "Index/Index.h"

namespace ippl {
    /*!
     * @file NDIndex.h
     * @tparam Dim the number of index dimensions
     */
    template <unsigned Dim>
    class NDIndex {
    public:
        KOKKOS_FUNCTION
        NDIndex() {}

        template <class... Args>
        KOKKOS_FUNCTION NDIndex(const Args&... args);

        KOKKOS_FUNCTION NDIndex(const Vector<unsigned, Dim>& sizes);

        /*!
         * @returns a reference to any of the Indexes.
         */
        KOKKOS_INLINE_FUNCTION const ippl::Index& operator[](unsigned d) const noexcept;

        KOKKOS_INLINE_FUNCTION Index& operator[](unsigned d) noexcept;

        /*!
         * @returns the total size.
         */
        KOKKOS_INLINE_FUNCTION unsigned size() const noexcept;

        /*!
         * @returns true if empty.
         */
        KOKKOS_INLINE_FUNCTION bool empty() const noexcept;

        /*!
         * Intersect with another NDIndex.
         */
        KOKKOS_INLINE_FUNCTION NDIndex<Dim> intersect(const NDIndex<Dim>&) const;

        /*!
         * Intersect with another NDIndex.
         */
        KOKKOS_INLINE_FUNCTION NDIndex<Dim> grow(int ncells) const;

        KOKKOS_INLINE_FUNCTION NDIndex<Dim> grow(int ncells, unsigned int dim) const;

        KOKKOS_INLINE_FUNCTION bool touches(const NDIndex<Dim>&) const;

        KOKKOS_INLINE_FUNCTION bool contains(const NDIndex<Dim>& a) const;

        // Split on dimension d with at position i
        KOKKOS_INLINE_FUNCTION bool split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d,
                                          int i) const;

        // Split on dimension d with the given ratio 0<a<1.
        KOKKOS_INLINE_FUNCTION bool split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d,
                                          double a) const;

        // Split on dimension d, or the longest dimension.
        KOKKOS_INLINE_FUNCTION bool split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d) const;

        KOKKOS_INLINE_FUNCTION bool split(NDIndex<Dim>& l, NDIndex<Dim>& r) const;

        KOKKOS_INLINE_FUNCTION Vector<size_t, Dim> length() const;
        KOKKOS_INLINE_FUNCTION Vector<int, Dim> first() const;
        KOKKOS_INLINE_FUNCTION Vector<int, Dim> last() const;

        using iterator       = Index*;
        using const_iterator = const Index*;
        KOKKOS_INLINE_FUNCTION constexpr iterator begin();
        KOKKOS_INLINE_FUNCTION constexpr iterator end();
        KOKKOS_INLINE_FUNCTION constexpr const_iterator begin() const;
        KOKKOS_INLINE_FUNCTION constexpr const_iterator end() const;

    private:
        KOKKOS_FUNCTION
        NDIndex(std::initializer_list<Index> indices);

        //! Array of indices
        Index indices_m[Dim];
    };
}  // namespace ippl

#include "Index/NDIndex.hpp"

#endif
