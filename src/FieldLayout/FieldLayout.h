//
// Class FieldLayout
//   FieldLayout describes how a given index space (represented by an NDIndex
//   object) is distributed among MPI ranks (vnodes). It performs the initial
//   partitioning, and stores a list of local and remote vnodes. The user may
//   request that a particular dimension not be partitioned by flagging that
//   axis as 'SERIAL' (instead of 'PARALLEL').
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef IPPL_FIELD_LAYOUT_H
#define IPPL_FIELD_LAYOUT_H

#include "FieldLayout/FieldLayoutUser.h"
#include "FieldLayout/Vnode.h"
#include "DomainMap/DomainMap.h"
#include "Index/NDIndex.h"
#include "Field/GuardCellSizes.h"
#include "Utility/IpplInfo.h"
#include "Utility/UserList.h"
#include "Utility/vmap.h"
#include "Utility/Unique.h"
#include "Utility/my_auto_ptr.h"
#include "Utility/RefCounted.h"

#include <iostream>

// forward declarations
template <unsigned Dim> class FieldLayout;
template <unsigned Dim>
std::ostream& operator<<(std::ostream&, const FieldLayout<Dim>&);

// enumeration used to select serial or parallel axes
enum e_dim_tag { SERIAL=0, PARALLEL=1 } ;


// class definition ... inheritance is private, so that we hide the
// UserList checkin routines and instead replace them with our own
template<unsigned Dim>
class FieldLayout : private UserList
{

public:
  using Index = ippl::Index;
  // Typedefs for containers.
  typedef vmap<Unique::type,my_auto_ptr<Vnode<Dim> > > ac_id_vnodes;
  typedef DomainMap<NDIndex<Dim>,RefCountedP< Vnode<Dim> >,
                    Touches<Dim>,Contains<Dim>,Split<Dim> > ac_domain_vnodes;
  typedef vmap<GuardCellSizes<Dim>,my_auto_ptr<ac_domain_vnodes> >
          ac_gc_domain_vnodes;

  // Typedefs for iterators.
  typedef typename ac_id_vnodes::iterator           iterator_iv;
  typedef typename ac_id_vnodes::const_iterator     const_iterator_iv;
  typedef typename ac_domain_vnodes::iterator       iterator_dv;
  typedef typename ac_domain_vnodes::touch_iterator touch_iterator_dv;
  typedef std::pair<touch_iterator_dv,touch_iterator_dv> touch_range_dv;
  typedef typename ac_gc_domain_vnodes::iterator    iterator_gdv;
  typedef iterator_user                             iterator_if;
  typedef size_type_user                            size_type_if;

private:
  // utility to return a zero-guard cell structure.
  static GuardCellSizes<Dim> gc0() { return GuardCellSizes<Dim>(0U); }

public:
  //
  // constructors and destructors
  //

  // Default constructor, which should only be used if you are going to
  // call 'initialize' soon after (before using in any context)
  FieldLayout();

  FieldLayout(const NDIndex<Dim>& domain, e_dim_tag *p=0) {
    initialize(domain,p);
  }

  // Destructor: Everything deletes itself automatically ... the base
  // class destructors inform all the FieldLayoutUser's we're going away.
  virtual ~FieldLayout();

  // Initialization functions, only to be called by the user of FieldLayout
  // objects when the FieldLayout was created using the default constructor;
  // otherwise these are only called internally by the various non-default
  // FieldLayout constructors:

  void initialize(const NDIndex<Dim>& domain, e_dim_tag *p=0);

  //
  // FieldLayout operations and information
  //

  // this does everything necessary to realign all the fields
  // associated with this FieldLayout!
  void Repartition(const NDIndex<Dim>*, const NDIndex<Dim>*);
  void Repartition(const NDIndex<Dim>& domain) { Repartition(&domain,(&domain)+1); }

  // This differs from the previous prototype in that it allows preservation of
  // global Vnode integer ID numbers associated with the input Vnodes. --tjw
  void Repartition(const Vnode<Dim>*, const Vnode<Dim>*);

  // Return the domain.
  const NDIndex<Dim>& getDomain() const { return Domain; }

  // Compare FieldLayouts to see if they represent the same domain; if
  // dimensionalities are different, the NDIndex operator==() will return
  // false:
  template <unsigned Dim2>
  bool operator==(const FieldLayout<Dim2>& x) const {
    return Domain == x.getDomain();
  }

  //
  // local vnode, remote vnode, touch range, and FieldLayoutUser iterators
  //
  int numVnodes(void) const {
    return (size_iv() + size_rdv());
  }

  // Accessors for the locals by Id.
  typename ac_id_vnodes::size_type size_iv() const;
  iterator_iv                      begin_iv();
  iterator_iv                      end_iv();
  const_iterator_iv                begin_iv() const;
  const_iterator_iv                end_iv() const;

  // Accessors for the remote vnode containers.
  typename ac_gc_domain_vnodes::size_type size_rgdv() const;
  iterator_gdv                            begin_rgdv();
  iterator_gdv                            end_rgdv();

  // Accessors for the remote vnodes themselves.
  typename ac_domain_vnodes::size_type 
    size_rdv(const GuardCellSizes<Dim>& gc = gc0()) const;
  iterator_dv begin_rdv(const GuardCellSizes<Dim>& gc = gc0());
  iterator_dv end_rdv(const GuardCellSizes<Dim>& gc = gc0());
  touch_range_dv touch_range_rdv(const NDIndex<Dim>& domain,
				 const GuardCellSizes<Dim>& gc = gc0()) const; 

  // Accessors for the users accessing this FieldLayout
  size_type_if      size_if() const { return getNumUsers(); }
  iterator_if       begin_if() { return begin_user(); }
  iterator_if       end_if() { return end_user(); }

  //
  // Query for information about the vnode sizes
  //

  // check if the vnode sizes are OK, to match the given GuardCellSizes
  bool fitsGuardCells(const GuardCellSizes<Dim>& gc) const {
    for (unsigned int d=0; d < Dim; ++d)
      if (MinWidth[d] < gc.left(d) || MinWidth[d] < gc.right(d))
        return false;
    return true;
  }

  // for the requested dimension, report if the distribution is
  // SERIAL or PARALLEL
  e_dim_tag getDistribution(unsigned int d) const {
    e_dim_tag retval = PARALLEL;
    if (MinWidth[d] == (unsigned int) Domain[d].length())
      retval = SERIAL;
    return retval;
  }

  // for the requested dimension, report if the distribution was requested to
  // be SERIAL or PARALLEL
  e_dim_tag getRequestedDistribution(unsigned int d) const {
    return RequestedLayout[d];
  }

  //
  // UserList operations
  //

  // Return our ID, as generated by UserList.
  UserList::ID_t get_Id() const { return getUserListID(); }

  NDIndex<Dim> getLocalNDIndex();

  //
  // I/O
  //

  // Print it out.
  void write(std::ostream&) const;

private:
  // Container definitions.
  ac_id_vnodes        Local_ac;
  ac_gc_domain_vnodes Remotes_ac;

  // Record the domain.
  NDIndex<Dim> Domain;

  // The minimum width of vnodes in each dimension, and the type of
  // layout that the user requested (this might not be the case anymore).
  unsigned int MinWidth[Dim];
  e_dim_tag RequestedLayout[Dim];

  // calculate the minimum vnode sizes in each dimension
  void calcWidths();

  // utility to return a zero-guard cell structure.
  void new_gc_layout(const GuardCellSizes<Dim>&);

  // The routine which actually sets things up.
  void setup(const NDIndex<Dim>&, e_dim_tag *);
};


//////////////////////////////////////////////////////////////////////

// Definitions for the specialized constructors.
// Just turn it into a call to the general ctor.


// Accessor definitions.

template<unsigned Dim>
inline typename FieldLayout<Dim>::ac_id_vnodes::size_type 
FieldLayout<Dim>::size_iv() const
{
  return Local_ac.size();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_iv
FieldLayout<Dim>::begin_iv()
{
  return Local_ac.begin();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_iv
FieldLayout<Dim>::end_iv()
{
  return Local_ac.end();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::const_iterator_iv
FieldLayout<Dim>::begin_iv() const
{
  return Local_ac.begin();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::const_iterator_iv
FieldLayout<Dim>::end_iv() const
{
  return Local_ac.end();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::ac_gc_domain_vnodes::size_type
FieldLayout<Dim>::size_rgdv() const
{
  return Remotes_ac.size();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_gdv
FieldLayout<Dim>::begin_rgdv()
{
  return Remotes_ac.begin();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_gdv
FieldLayout<Dim>::end_rgdv()
{
  return Remotes_ac.end();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::ac_domain_vnodes::size_type
FieldLayout<Dim>::size_rdv(const GuardCellSizes<Dim>& gc) const
{
  return Remotes_ac[gc]->size();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_dv
FieldLayout<Dim>::begin_rdv(const GuardCellSizes<Dim>& gc)
{
  return Remotes_ac[gc]->begin();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_dv
FieldLayout<Dim>::end_rdv(const GuardCellSizes<Dim>& gc)
{
  return Remotes_ac[gc]->end();
}

template<unsigned Dim>
inline 
typename FieldLayout<Dim>::touch_range_dv
FieldLayout<Dim>::touch_range_rdv(const NDIndex<Dim>& domain,
				  const GuardCellSizes<Dim>& gc) const
{
  return Remotes_ac[gc]->touch_range(domain);
}


// I/O

template<unsigned Dim>
inline
std::ostream& operator<<(std::ostream& out, const FieldLayout<Dim>& f) {
  
  

  f.write(out);
  return out;
}


#include "FieldLayout/FieldLayout.hpp"

#endif