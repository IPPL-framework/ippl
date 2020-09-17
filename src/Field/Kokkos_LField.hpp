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
  Owned(owned),
  Allocated(allocated)
{}


template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::Kokkos_LField(const Kokkos_LField<T,Dim>& lf)
  : vnode_m(lf.vnode_m),
    Owned(lf.Owned),
    Allocated(lf.Allocated)
{
    Kokkos::deep_copy(dview_m, lf.getDeviceView());
}


template<class T, unsigned Dim>
void Kokkos_LField<T,Dim>::write(std::ostream& out) const
{
    typename view_type::HostView hview = Kokkos::create_mirror_view(dview_m);

//     for (iterator p = begin(); p!=end(); ++p)
//     out << *p << " ";
}