//
// Class LField
//   Local fields. A BareField consists of mulitple LFields.
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
namespace ippl {

    template<class T, unsigned Dim>
    LField<T,Dim>::LField(const Domain_t& owned, int vnode, int nghost)
        : vnode_m(vnode)
        , nghost_m(nghost)
        , owned_m(owned)
    {
        static_assert(Dim == 3, "Only 3-dimensional fields supported at the momment!");

        if constexpr(Dim == 1) {
            this->resize(owned[0].length() + 2 * nghost);
        } else if constexpr(Dim == 2) {
            this->resize(owned[0].length() + 2 * nghost,
                         owned[1].length() + 2 * nghost);
        } else if constexpr(Dim == 3) {
            this->resize(owned[0].length() + 2 * nghost,
                         owned[1].length() + 2 * nghost,
                         owned[2].length() + 2 * nghost);
        }
    }


    template<class T, unsigned Dim>
    void LField<T,Dim>::write(std::ostream& out) const {
        Kokkos::fence();
        detail::write<T>(dview_m, out);
    }


    template<class T, unsigned Dim>
    template<typename ...Args>
    void LField<T,Dim>::resize(Args... args) {
        Kokkos::resize(dview_m, args...);
    }


    template<class T, unsigned Dim>
    LField<T, Dim>& LField<T, Dim>::operator=(T x) {
    //     if constexpr(Dim == 1) {
    //         Kokkos::parallel_for("LField::operator=()",
    //                              dview_m.extent(0),
    //                              KOKKOS_CLASS_LAMBDA(const int i) {
    //                                  dview_m(i) = x;
    //                             });
    //     } else if constexpr(Dim == 2) {
    //         Kokkos::parallel_for("LField::operator=()",
    //                              Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
    //                                  {0, 0},
    //                                  {dview_m.extent(0), dview_m.extent(1)}),
    //                              KOKKOS_CLASS_LAMBDA(const int i, const int j) {
    //                                  dview_m(i, j) = x;
    //                             });
    //     } else if constexpr(Dim == 3) {
            Kokkos::parallel_for("LField::operator=()",
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
    LField<T,Dim>& LField<T,Dim>::operator=(Expression<E, N> const& expr) {
        detail::CapturedExpression<E, N> expr_ = reinterpret_cast<const detail::CapturedExpression<E, N>&>(expr);
    //     if constexpr(Dim == 1) {
    //         Kokkos::parallel_for("LField<T,Dim>::operator=",
    //                              dview_m.extent(0), KOKKOS_CLASS_LAMBDA(const int i) {
    //                                  dview_m(i) = expr_(i);
    //                             });
    //     } else if constexpr(Dim == 2) {
    //         Kokkos::parallel_for("LField::operator=()",
    //                              Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
    //                                                                     {dview_m.extent(0), dview_m.extent(1)}),
    //                              KOKKOS_CLASS_LAMBDA(const int i, const int j) {
    //                                  dview_m(i, j) = expr_(i, j);
    //                             });
    //     } else if constexpr(Dim == 3) {
            Kokkos::parallel_for("LField::operator=()",
                                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost_m, nghost_m, nghost_m},
                                                                       {dview_m.extent(0) - nghost_m,
                                                                        dview_m.extent(1) - nghost_m,
                                                                        dview_m.extent(2) - nghost_m}),
                                KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                                    dview_m(i, j, k) = expr_(i, j, k);
                                });
    //     }
        return *this;
    }

    #define DefineReduction(fun, name, op)                                                                   \
    template<class T, unsigned Dim>                                                                          \
    T LField<T, Dim>::name() {                                                                               \
        T temp = 0.0;                                                                                        \
        Kokkos::parallel_reduce("fun",                                                                       \
                               Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost_m, nghost_m, nghost_m},        \
                                                                      {dview_m.extent(0) - nghost_m,         \
                                                                       dview_m.extent(1) - nghost_m,         \
                                                                       dview_m.extent(2) - nghost_m}),       \
                               KOKKOS_CLASS_LAMBDA(const int i, const int j,                                 \
                                                   const int k, T& valL) {                                   \
                                    T myVal = dview_m(i, j, k);                                              \
                                    op;                                                                      \
                               }, Kokkos::fun<T>(temp));                                                     \
        return temp;                                                                                         \
    }                                                                                                        \

    DefineReduction(Sum,  sum,  valL += myVal)
    DefineReduction(Max,  max,  if(myVal > valL) valL = myVal)
    DefineReduction(Min,  min,  if(myVal < valL) valL = myVal)
    DefineReduction(Prod, prod, valL *= myVal)
}
