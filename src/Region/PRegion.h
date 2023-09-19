//
// Class PRegion
//   PRegion represents a (possibly continuous) numeric interval.  It is
//   similar to Index, with the following differences:
//      1. It is templated on the data type; Index always uses integers
//      2. A PRegion is defined between two endpoints A and B; the PRegion
//         includes values X where A <= X < B (i.e., X in [A,B) )
//      3. PRegion does not keep track of a base Index, and does not
//         supply the plugBase operation.  It is not designed for use in
//         Field operations like Index is, it is meant instead for use in
//         Particle construction and usage.
//
//   PRegion<T>()      --> make a PRegion on [0,1)
//   PRegion<T>(B)     --> make a PRegion on [0,B)
//   PRegion<T>(A,B)   --> make a PRegion on [A,B)
//
#ifndef IPPL_PREGION_H
#define IPPL_PREGION_H

namespace ippl {
    /*!
     * @file PRegion.h
     * @tparam T type of interval
     */
    template <typename T>
    class PRegion {
    public:
        /*!
         * Default region [0, 1[
         */
        KOKKOS_FUNCTION
        PRegion();

        /*!
         * Region [0, b[
         */
        KOKKOS_FUNCTION
        PRegion(T b);

        /*!
         * Region [a, b[
         */
        KOKKOS_FUNCTION
        PRegion(T a, T b);

        KOKKOS_DEFAULTED_FUNCTION
        ~PRegion() = default;

        KOKKOS_FUNCTION
        PRegion(const PRegion<T>&);

        KOKKOS_INLINE_FUNCTION PRegion<T>& operator=(const PRegion<T>& rhs);

        /*!
         * @returns the lower bound
         */
        KOKKOS_INLINE_FUNCTION T min() const noexcept;

        /*!
         * @returns the upper bound
         */
        KOKKOS_INLINE_FUNCTION T max() const noexcept;

        /*!
         * @returns the length of the region
         */
        KOKKOS_INLINE_FUNCTION T length() const noexcept;

        /*!
         * @returns true if empty
         */
        KOKKOS_INLINE_FUNCTION bool empty() const noexcept;

        KOKKOS_INLINE_FUNCTION PRegion<T>& operator+=(T t) noexcept;

        KOKKOS_INLINE_FUNCTION PRegion<T>& operator-=(T t) noexcept;

        KOKKOS_INLINE_FUNCTION PRegion<T>& operator*=(T t) noexcept;

        KOKKOS_INLINE_FUNCTION PRegion<T>& operator/=(T t) noexcept;

    private:
        //! Interval start point
        T a_m;

        //! Interval end point
        T b_m;
    };
}  // namespace ippl

#include "PRegion.hpp"

#endif
