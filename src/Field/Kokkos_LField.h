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
#ifndef KOKKOS_LField_H
#define KOKKOS_LField_H

// include files
#include "Field/CompressedBrickIterator.h"

#include <Kokkos_Core.hpp>

#include "Field/ViewTypes.h"

#include <iostream>

// forward declarations
template <class T, unsigned Dim> class Kokkos_LField;
template <class T, unsigned Dim>
std::ostream& operator<<(std::ostream&, const Kokkos_LField<T,Dim>&);


//////////////////////////////////////////////////////////////////////

// This stores the local data for a Field.
template<class T, unsigned Dim>
class Kokkos_LField
{

public:
  // An iterator for the contents of this Kokkos_LField.
  typedef CompressedBrickIterator<T,Dim> iterator;

  // The type of domain stored here
  typedef NDIndex<Dim> Domain_t;

  typedef typename ViewType<T, Dim>::view_type view_type;


  //
  // Constructors and destructor
  //

  // Ctors for an Kokkos_LField.  Arguments:
  //     owned = domain of "owned" region of Kokkos_LField (without guards)
  //     allocated = domain of "allocated" region, which includes guards
  //     vnode = global vnode ID number (see below)
  Kokkos_LField(const NDIndex<Dim>& owned,
         const NDIndex<Dim>& allocated,
         int vnode = -1);

  // Copy constructor.
  Kokkos_LField(const Kokkos_LField<T,Dim>&);

  // Destructor: just free the memory, if it's there.

  ~Kokkos_LField() {};

  //
  // General information accessors
  //

  // Return information about the Kokkos_LField.
  int size(unsigned d) const { return Owned[d].length(); }
  const NDIndex<Dim>& getAllocated()   const { return Allocated; }
  const NDIndex<Dim>& getOwned()       const { return Owned; }

  view_type&    getDeviceView() { return dview_m; }

  // Return global vnode ID number (between 0 and nvnodes - 1)
  int getVnode() const { return vnode_m; }

  //
  // I/O
  //

  // print an Kokkos_LField out
  void write(std::ostream&) const;

private:
  // Global vnode ID number for the associated Vnode (useful with more recent
  // FieldLayouts which store a logical "array" of vnodes; user specifies
  // numbers of vnodes along each direction). Classes or user codes that use
  // Kokkos_LField are responsible for setting and managing the values of this index;
  // if unset, it has the value -1. Generally, this parameter value is set on
  // construction of the vnode:

  int vnode_m;

  // actual field data
  view_type dview_m;

  // What domain in the data is owned by this Kokkos_LField.

  NDIndex<Dim>   Owned;

  // How total domain is actually allocated for thie Kokkos_LField (including guards)

  NDIndex<Dim>   Allocated;

  Kokkos_LField();
  const Kokkos_LField<T,Dim> &operator=(const Kokkos_LField<T,Dim> &);
};


template<class T, unsigned Dim>
inline
std::ostream& operator<<(std::ostream& out, const Kokkos_LField<T,Dim>& a)
{


  a.write(out);
  return out;
}

//////////////////////////////////////////////////////////////////////

#include "Field/Kokkos_LField.hpp"

#endif // Kokkos_LField_H