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
#include "Field/Kokkos_LField.h"

#include "Field/FieldExpr.h"

#include <type_traits>

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
//                 std::is_base_of_v<FieldExpr<T, E>, E>,
//     //               // -------------------------------------------
//                 int> >
Kokkos_LField<T,Dim>& Kokkos_LField<T,Dim>::operator=(FieldExpr<T, E, N> const& expr) {
    LFieldCaptureExpr<T, E,N> expr_ = reinterpret_cast<const LFieldCaptureExpr<T, E, N>&>(expr);
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


/*
 *
 * Expression Template Operations
 *
 */
template <typename T, typename E1, typename E2>
class LFieldAdd : public FieldExpr<T, LFieldAdd<T, E1, E2>, sizeof(E1) + sizeof(E2)> {
public:
    LFieldAdd(E1 const& u, E2 const& v) : _u(u), _v(v) { }

    template<typename ...Args>
    KOKKOS_INLINE_FUNCTION
    T operator()(Args... args) const {
        return _u(args...) + _v(args...);
    }

private:
    E1 const _u;
    E2 const _v;
};


template <typename T, typename E1, typename E2, size_t N1, size_t N2>
LFieldAdd<T, E1, E2>
operator+(FieldExpr<T, E1, N1> const& u, FieldExpr<T, E2, N2> const& v) {
  return LFieldAdd<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}


template <typename T, typename E1, typename E2>
class LFieldSubtract : public FieldExpr<T, LFieldSubtract<T, E1, E2>, sizeof(E1) + sizeof(E2)> {
public:
    LFieldSubtract(E1 const& u, E2 const& v) : _u(u), _v(v) { }

    template<typename ...Args>
    KOKKOS_INLINE_FUNCTION
    T operator()(Args... args) const {
        return _u(args...) - _v(args...);
    }

private:
    E1 const _u;
    E2 const _v;
};


template <typename T, typename E1, typename E2, size_t N1, size_t N2>
LFieldSubtract<T, E1, E2>
operator-(FieldExpr<T, E1, N1> const& u, FieldExpr<T, E2, N2> const& v) {
  return LFieldSubtract<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}


template <typename T, typename E1, typename E2>
class LFieldMultiply : public FieldExpr<T, LFieldMultiply<T, E1, E2>, sizeof(E1) + sizeof(E2) >{
public:
    LFieldMultiply(E1 const& u, E2 const& v) : _u(u), _v(v) { }

    template<typename ...Args>
    KOKKOS_INLINE_FUNCTION
    T operator()(Args... args) const {
        return _u(args...) * _v(args...);
    }

private:
    E1 const _u;
    E2 const _v;
};


template <typename T, typename E1, typename E2, size_t N1, size_t N2>
LFieldMultiply<T, E1, E2>
operator*(FieldExpr<T, E1, N1> const& u, FieldExpr<T, E2, N2> const& v) {
  return LFieldMultiply<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}


template <typename T, typename E1, typename E2>
class LFieldDivide : public FieldExpr<T, LFieldDivide<T, E1, E2>, sizeof(E1) + sizeof(E2) >{
public:
    LFieldDivide(E1 const& u, E2 const& v) : _u(u), _v(v) { }

    template<typename ...Args>
    KOKKOS_INLINE_FUNCTION
    T operator()(Args... args) const {
        return _u(args...) / _v(args...);
    }

private:
    E1 const _u;
    E2 const _v;
};


template <typename T, typename E1, typename E2, size_t N1, size_t N2>
LFieldDivide<T, E1, E2>
operator/(FieldExpr<T, E1, N1> const& u, FieldExpr<T, E2, N2> const& v) {
  return LFieldDivide<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}


/*
 *
 *
 *
 */
template <typename T, typename E1>
class LFieldAddScalarRight : public FieldExpr<T, LFieldAddScalarRight<T, E1>, sizeof(E1) + sizeof(T)> {
public:
    LFieldAddScalarRight(E1 const& u, T const& v) : _u(u), _v(v) { }

    template<typename ...Args>
    KOKKOS_INLINE_FUNCTION
    T operator()(Args... args) const {
        return _u(args...) + _v;
    }

private:
    E1 const _u;
    T const _v;
};


template <typename T, typename E1, size_t N1>
LFieldAddScalarRight<T, E1>
operator+(FieldExpr<T, E1, N1> const& u, T const& v) {
  return LFieldAddScalarRight<T, E1>(*static_cast<const E1*>(&u), *static_cast<const T*>(&v));

}

template <typename T, typename E1>
class LFieldAddScalarLeft : public FieldExpr<T, LFieldAddScalarLeft<T, E1>, sizeof(E1) + sizeof(T)> {
public:
    LFieldAddScalarLeft(T const& u, E1 const& v) : _u(u), _v(v) { }

    template<typename ...Args>
    KOKKOS_INLINE_FUNCTION
    T operator()(Args... args) const {
        return _u + _v(args...);
    }

private:
    T const _u;
    E1 const _v;
};


template <typename T, typename E1, size_t N1>
LFieldAddScalarLeft<T, E1>
operator+(T const& u, FieldExpr<T, E1, N1> const& v) {
  return LFieldAddScalarLeft<T, E1>(*static_cast<const T*>(&u), *static_cast<const E1*>(&v));

}