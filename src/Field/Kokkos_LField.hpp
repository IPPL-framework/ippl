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

// #include "Utility/PAssert.h"


template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::Kokkos_LField(const NDIndex<Dim>& owned,
                      const NDIndex<Dim>& allocated,
                      int vnode)
: vnode_m(vnode),
  owned_m(owned),
  allocated_m(allocated)
{
    //FIXME
    this->resize(owned.size());
}


template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::Kokkos_LField(const Kokkos_LField<T,Dim>& lf)
  : vnode_m(lf.vnode_m),
    owned_m(lf.owned_m),
    allocated_m(lf.allocated_m)
{
    Kokkos::resize(dview_m, lf.getDeviceView().size());
    Kokkos::deep_copy(dview_m, lf.getDeviceView());
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
