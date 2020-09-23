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

// #include "Utility/PAssert.h"


template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::Kokkos_LField(const Domain_t& owned,
                      int vnode)
: vnode_m(vnode),
  owned_m(owned)
{
    //FIXME
    this->resize(owned.size());
}


template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::Kokkos_LField(const Kokkos_LField<T,Dim>& lf)
  : vnode_m(lf.vnode_m),
    owned_m(lf.owned_m)
{
    Kokkos::resize(dview_m, lf.dview_m.size());
    Kokkos::deep_copy(dview_m, lf.dview_m);
}


template<class T, unsigned Dim>
void Kokkos_LField<T,Dim>::write(std::ostream& out) const
{

    write_<T>(dview_m, out);
}


template<class T, unsigned Dim>
template<typename ...Args>
void Kokkos_LField<T,Dim>::resize(Args... args)
{
    Kokkos::resize(dview_m, args...);
}



template <typename T, typename E1, typename E2>
class LFieldAdd : public FieldExpr<T, LFieldAdd<T, E1, E2> >{
public:
  LFieldAdd(E1 const& u, E2 const& v) : _u(u), _v(v) { }

  KOKKOS_INLINE_FUNCTION
  T operator()(size_t i) const { return _u(i) + _v(i); }

private:
  E1 const _u;
  E2 const _v;
};




template <typename T, typename E1, typename E2>
LFieldAdd<T, E1, E2>
operator+(FieldExpr<T, E1> const& u, FieldExpr<T, E2> const& v) {
  return LFieldAdd<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}



template <typename T, typename E1, typename E2>
class LFieldSubtract : public FieldExpr<T, LFieldSubtract<T, E1, E2> >{
public:
  LFieldSubtract(E1 const& u, E2 const& v) : _u(u), _v(v) { }

  KOKKOS_INLINE_FUNCTION
  T operator()(size_t i) const { return _u(i) - _v(i); }

private:
  E1 const _u;
  E2 const _v;
};


template <typename T, typename E1, typename E2>
LFieldSubtract<T, E1, E2>
operator-(FieldExpr<T, E1> const& u, FieldExpr<T, E2> const& v) {
  return LFieldSubtract<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}



template <typename T, typename E1, typename E2>
class LFieldMultiply : public FieldExpr<T, LFieldMultiply<T, E1, E2> >{
public:
  LFieldMultiply(E1 const& u, E2 const& v) : _u(u), _v(v) { }

  KOKKOS_INLINE_FUNCTION
  T operator()(size_t i) const { return _u(i) * _v(i); }

private:
  E1 const _u;
  E2 const _v;
};



template <typename T, typename E1, typename E2>
LFieldMultiply<T, E1, E2>
operator*(FieldExpr<T, E1> const& u, FieldExpr<T, E2> const& v) {
  return LFieldMultiply<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}



template <typename T, typename E1, typename E2>
class LFieldDivide : public FieldExpr<T, LFieldDivide<T, E1, E2> >{
public:
  LFieldDivide(E1 const& u, E2 const& v) : _u(u), _v(v) { }

  KOKKOS_INLINE_FUNCTION
  T operator()(size_t i) const { return _u(i) / _v(i); }

private:
  E1 const _u;
  E2 const _v;
};


template <typename T, typename E1, typename E2>
LFieldDivide<T, E1, E2>
operator/(FieldExpr<T, E1> const& u, FieldExpr<T, E2> const& v) {
  return LFieldDivide<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}

