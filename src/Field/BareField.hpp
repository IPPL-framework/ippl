//
// Class BareField
//   A BareField consists of multple LFields and represents a field.
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
#include "Field/BareField.h"
#include "FieldLayout/FieldLayout.h"
#include "Utility/Inform.h"
#include "Utility/Unique.h"
#include "Utility/IpplInfo.h"

#include <map>
#include <utility>
#include <cstdlib>

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        struct isExpression<BareField<T, Dim>> : std::true_type {};
    }

    template< typename T, unsigned Dim>
    BareField<T, Dim>::BareField()
    : nghost_m(1)
    , layout_m(nullptr)
    { }


    template< typename T, unsigned Dim>
    BareField<T, Dim>::BareField(Layout_t& l, int nghost)
    : nghost_m(nghost)
//     , owned_m(0)
    , layout_m(&l)
    {
        setup();
    }

    template<typename T, unsigned Dim>
    void BareField<T, Dim>::initialize(Layout_t& l, int nghost) {
        if (layout_m == 0) {
            layout_m = &l;
            nghost_m = nghost;
            setup();
        }
    }


    template<typename T, unsigned Dim>
    void BareField<T, Dim>::setup() {
        static_assert(Dim == 3, "Only 3-dimensional fields supported at the momment!");

        owned_m = layout_m->getLocalNDIndex();

        if constexpr(Dim == 1) {
            this->resize(owned_m[0].length() + 2 * nghost_m);
        } else if constexpr(Dim == 2) {
            this->resize(owned_m[0].length() + 2 * nghost_m,
                         owned_m[1].length() + 2 * nghost_m);
        } else if constexpr(Dim == 3) {
            this->resize(owned_m[0].length() + 2 * nghost_m,
                         owned_m[1].length() + 2 * nghost_m,
                         owned_m[2].length() + 2 * nghost_m);
        }
    }


    template<class T, unsigned Dim>
    template<typename ...Args>
    void BareField<T,Dim>::resize(Args... args) {
        Kokkos::resize(dview_m, args...);
    }


    template<typename T, unsigned Dim>
    BareField<T, Dim>& BareField<T, Dim>::operator=(T x) {
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for("BareField::operator=(T)",
                             mdrange_type({0, 0, 0},
                                          {dview_m.extent(0),
                                           dview_m.extent(1),
                                           dview_m.extent(2)
                                    }),
                             KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                                 dview_m(i, j, k) = x;
                             });
        return *this;
    }


    template<typename T, unsigned Dim>
    template <typename E, size_t N>
    BareField<T, Dim>& BareField<T, Dim>::operator=(const detail::Expression<E, N>& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for("BareField::operator=(const Expression&)",
                             mdrange_type({nghost_m, nghost_m, nghost_m},
                                          {dview_m.extent(0) - nghost_m,
                                           dview_m.extent(1) - nghost_m,
                                           dview_m.extent(2) - nghost_m}),
                             KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                             dview_m(i, j, k) = expr_(i, j, k);
                             });
        return *this;
    }


    template<class T, unsigned Dim>
    void BareField<T,Dim>::write(std::ostream& out) const {
        Kokkos::fence();
        detail::write<T>(dview_m, out);
    }


    #define DefineReduction(fun, name, op)                                                                   \
    template <typename T, unsigned Dim>                                                                      \
    T BareField<T, Dim>::name(int nghost) {                                                                  \
        PAssert_LE(nghost, nghost_m);                                                                        \
        T temp = 0.0;                                                                                        \
        const int shift = nghost_m - nghost;                                                                 \
        Kokkos::parallel_reduce("fun",                                                                       \
                               Kokkos::MDRangePolicy<Kokkos::Rank<3>>({shift, shift, shift},                 \
                                                                      {dview_m.extent(0) - shift,            \
                                                                       dview_m.extent(1) - shift,            \
                                                                       dview_m.extent(2) - shift}),          \
                               KOKKOS_CLASS_LAMBDA(const int i, const int j,                                 \
                                                   const int k, T& valL) {                                   \
                                    T myVal = dview_m(i, j, k);                                              \
                                    op;                                                                      \
                               }, Kokkos::fun<T>(temp));                                                     \
        return temp;                                                                                         \
    }

    DefineReduction(Sum,  sum,  valL += myVal)
    DefineReduction(Max,  max,  if(myVal > valL) valL = myVal)
    DefineReduction(Min,  min,  if(myVal < valL) valL = myVal)
    DefineReduction(Prod, prod, valL *= myVal)


}
