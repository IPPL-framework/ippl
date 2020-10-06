//
// Class Kokkos_LField
//   Local Field class
//
// Copyright (c) 2003 - 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//

// #include <type_traits>

namespace ippl {

    template<class T, unsigned Dim>
    Kokkos_LField<T,Dim>::Kokkos_LField(const Domain_t& owned, int vnode)
        : vnode_m(vnode)
        , owned_m(owned)
    {
        static_assert(Dim == 3, "Only 3-dimensional fields supported at the momment!");

        if constexpr(Dim == 1) {
            this->resize(owned[0].length());
        } else if constexpr(Dim == 2) {
            this->resize(owned[0].length(),
                        owned[1].length());
        } else if constexpr(Dim == 3) {
            this->resize(owned[0].length(),
                        owned[1].length(),
                        owned[2].length());
        }
    }


    template<class T, unsigned Dim>
    void Kokkos_LField<T,Dim>::write(std::ostream& out) const {
        Kokkos::fence();
        write_<T>(dview_m, out);
    }


    template<class T, unsigned Dim>
    template<typename ...Args>
    void Kokkos_LField<T,Dim>::resize(Args... args) {
        Kokkos::resize(dview_m, args...);
    }


    template<class T, unsigned Dim>
    Kokkos_LField<T, Dim>& Kokkos_LField<T, Dim>::operator=(T x) {
    //     if constexpr(Dim == 1) {
    //         Kokkos::parallel_for("Kokkos_LField::operator=()",
    //                              dview_m.extent(0),
    //                              KOKKOS_CLASS_LAMBDA(const int i) {
    //                                  dview_m(i) = x;
    //                             });
    //     } else if constexpr(Dim == 2) {
    //         Kokkos::parallel_for("Kokkos_LField::operator=()",
    //                              Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
    //                                  {0, 0},
    //                                  {dview_m.extent(0), dview_m.extent(1)}),
    //                              KOKKOS_CLASS_LAMBDA(const int i, const int j) {
    //                                  dview_m(i, j) = x;
    //                             });
    //     } else if constexpr(Dim == 3) {
            Kokkos::parallel_for("Kokkos_LField::operator=()",
                                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                    {0, 0, 0},
                                    {dview_m.extent(0), dview_m.extent(1), dview_m.extent(2)}),
                                KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                                    dview_m(i, j, k) = x;
                                });
    //     }
        return *this;
    }


    template <class T, unsigned Dim>
    template <typename E, size_t N>
    //           std::enable_if_t<
    //     //               // essentially equivalent to:
    //     //               //   requires std::derived_from<E, VecExpr<E>>
    //                 std::is_base_of_v<Expression<E>, E>,
    //     //               // -------------------------------------------
    //                 int> >
    Kokkos_LField<T,Dim>& Kokkos_LField<T,Dim>::operator=(Expression<E, N> const& expr) {
        CapturedExpression<E, N> expr_ = reinterpret_cast<const CapturedExpression<E, N>&>(expr);
    //     if constexpr(Dim == 1) {
    //         Kokkos::parallel_for("Kokkos_LField<T,Dim>::operator=",
    //                              dview_m.extent(0), KOKKOS_CLASS_LAMBDA(const int i) {
    //                                  dview_m(i) = expr_(i);
    //                             });
    //     } else if constexpr(Dim == 2) {
    //         Kokkos::parallel_for("Kokkos_LField::operator=()",
    //                              Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
    //                                                                     {dview_m.extent(0), dview_m.extent(1)}),
    //                              KOKKOS_CLASS_LAMBDA(const int i, const int j) {
    //                                  dview_m(i, j) = expr_(i, j);
    //                             });
    //     } else if constexpr(Dim == 3) {
            Kokkos::parallel_for("Kokkos_LField::operator=()",
                                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                        {dview_m.extent(0), dview_m.extent(1), dview_m.extent(2)}),
                                KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                                    dview_m(i, j, k) = expr_(i, j, k);
                                });
    //     }
        return *this;
    }
}