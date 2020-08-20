// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "Index/SIndex.h"
#include "Index/IndexedSIndex.h"
#include "FieldLayout/FieldLayout.h"
#include "Utility/PAssert.h"




////////////////////////////////////////////////////////////////////////////
// default constructor: this requires the user to call 'initialize'
// before any other actions are carried out with this SIndex
template<unsigned int Dim>
SIndex<Dim>::SIndex() : Layout(0) {
  
}


////////////////////////////////////////////////////////////////////////////
// constructor: requires a FieldLayout, and optionally an offset amount
template<unsigned int Dim>
SIndex<Dim>::SIndex(FieldLayout<Dim>& fl) : Layout(&fl) {
  
  

  setup();
}


////////////////////////////////////////////////////////////////////////////
// copy constructor
template<unsigned int Dim>
SIndex<Dim>::SIndex(const SIndex<Dim>& si)
  : Layout(si.Layout), Offset(si.Offset), IndexList(si.IndexList),
    BoundingBox(si.BoundingBox)
{
  
  

  // check in as a user of this Layout
  Layout->checkin(*this);
}


////////////////////////////////////////////////////////////////////////////
// a special constructor, taking another SIndex and an Offset.  This
// version is almost like a copy constructor, except that the given Offset
// is added to our own offset.
template<unsigned int Dim>
SIndex<Dim>::SIndex(const SIndex<Dim>& si, const SOffset<Dim>& so)
  : Layout(si.Layout), Offset(si.Offset), IndexList(si.IndexList),
    BoundingBox(si.BoundingBox)
{
  
  

  // add in the extra offset amount
  Offset += so;

  // check in as a user of this Layout
  Layout->checkin(*this);
}


////////////////////////////////////////////////////////////////////////////
// a special constructor, taking another SIndex and an int * offset.  This
// version is almost like a copy constructor, except that the given Offset
// is added to our own offset.
template<unsigned int Dim>
SIndex<Dim>::SIndex(const SIndex<Dim>& si, const int *so)
  : Layout(si.Layout), Offset(si.Offset), IndexList(si.IndexList),
    BoundingBox(si.BoundingBox)
{
  
  

  // add in the extra offset amount
  Offset += so;

  // check in as a user of this Layout
  Layout->checkin(*this);
}


////////////////////////////////////////////////////////////////////////////
// destructor: frees memory used to store indices, and check out from Layout
template<unsigned int Dim>
SIndex<Dim>::~SIndex() {
  
  if (Layout != 0)
    Layout->checkout(*this);
}


////////////////////////////////////////////////////////////////////////////
// initialize the object, if we were constructed with the default
// constructor
template<unsigned int Dim>
void SIndex<Dim>::initialize(FieldLayout<Dim>& fl) {
  
  

  Layout = &fl;
  setup();
}


////////////////////////////////////////////////////////////////////////////
// set up our internal data structures from the constructor.  Assumes
// the Layout and Offset have been set.
template<unsigned int Dim>
void SIndex<Dim>::setup() {
  

  // check in as a user of the FieldLayout
  Layout->checkin(*this);

  // get the number of vnodes, and make sure we have storage for them
  IndexList.reserve(Layout->size_iv());

  // create new LSIndex objects for the local vnodes
  typename FieldLayout<Dim>::iterator_iv locvn = Layout->begin_iv();
  for ( ; locvn != Layout->end_iv(); ++locvn)
    IndexList.push_back(std::make_shared<LSIndex<Dim>>((*locvn).second.get()));

  // save our bounding box information
  BoundingBox = Layout->getDomain();
}


////////////////////////////////////////////////////////////////////////////
// add a new index point, specified as an Offset or as a single-point NDIndex.
// return success (this can fail if the point is outsize the field's domain)
template<unsigned int Dim>
bool SIndex<Dim>::addIndex(const SOffset<Dim>& so) {
  
  

  // right now, this is straightforward: try to add to each LSIndex in
  // succession until one actually works
  for (iterator_iv curr = begin_iv(); curr != end_iv(); ++curr) {
    if (addIndex(curr, so)) {
      return true;
    }
  }

  // if we're here, we could not find a vnode to take this point
  return false;
}


////////////////////////////////////////////////////////////////////////////
// add a new index point, specified as an Offset or as a single-point NDIndex.
// This version specifically tells which LField to use.
// return success (this can fail if the point is outsize the field's domain)
template<unsigned int Dim>
bool SIndex<Dim>::addIndex(SIndex<Dim>::iterator_iv& curr,
			   const SOffset<Dim>& so) {
  
  

  // if the point is in the LField's region, add it
  if ((*curr)->contains(so) && ! (*curr)->hasIndex(so)) {
    (*curr)->addIndex(so);
    return true;
  }

  // if we're here, the point was outside the given LField's domain
  return false;
}


////////////////////////////////////////////////////////////////////////////
// same as above, but taking an NDIndex.  One point is added to our lists
// for each point in the NDIndex region (e.g., if a 2x3 NDIndex is given,
// 6 points in total will be added to this SIndex).
template<unsigned int Dim>
void SIndex<Dim>::addIndex(const NDIndex<Dim>& constndi) {
  
  

  // cast away const; we are not modifying the NDIndex, so this should be ok
  NDIndex<Dim>& ndi((NDIndex<Dim> &)constndi);

  // calculate the total number of points, and counters for looping
  unsigned int d, totalnum = ndi.size();
  Index::iterator counter[Dim];
  for (d=0; d < Dim; ++d)
    counter[d] = ndi[d].begin();

  while (totalnum-- > 0) {
    // create an SOffset obj and add it to our lists
    SOffset<Dim> newpoint;
    for (d=0; d < Dim; ++d)
      newpoint[d] = *(counter[d]);
    addIndex(newpoint);

    // increment the index iterators
    unsigned int chkdim = 0;
    while(chkdim < Dim) {
      ++(counter[chkdim]);
      if (counter[chkdim] == ndi[chkdim].end()) {
	counter[chkdim] = ndi[chkdim].begin();
	chkdim++;
      } else {
	break;
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////
// add a new index point, specified as an Offset or as a single-point NDIndex.
// return success (this can fail if the point is outsize the field's domain)
template<unsigned int Dim>
bool SIndex<Dim>::removeIndex(const SOffset<Dim>& so) {
  
  

  // right now, this is straightforward: try to add to each LSIndex in
  // succession until one actually works
  for (iterator_iv curr = begin_iv(); curr != end_iv(); ++curr)
    if (removeIndex(curr, so))
      return true;

  // if we're here, we could not find a vnode to take this point
  return false;
}


////////////////////////////////////////////////////////////////////////////
// add a new index point, specified as an Offset or as a single-point NDIndex.
// This version specifically tells which LField to use.
// return success (this can fail if the point is outsize the field's domain)
template<unsigned int Dim>
bool SIndex<Dim>::removeIndex(SIndex<Dim>::iterator_iv& curr,
			   const SOffset<Dim>& so) {
  
  
  if ((*curr)->hasIndex(so)) {
    (*curr)->removeIndex(so);
    return true;
  }
  return false;
}


////////////////////////////////////////////////////////////////////////////
// same as above, but taking an NDIndex.  One point is added to our lists
// for each point in the NDIndex region (e.g., if a 2x3 NDIndex is given,
// 6 points in total will be added to this SIndex).
template<unsigned int Dim>
void SIndex<Dim>::removeIndex(const NDIndex<Dim>& constndi) {
  
  

  // cast away const; we are not modifying the NDIndex, so this should be ok
  NDIndex<Dim>& ndi((NDIndex<Dim> &)constndi);

  // calculate the total number of points, and counters for looping
  unsigned int d, totalnum = ndi.size();
  Index::iterator counter[Dim];
  for (d=0; d < Dim; ++d)
    counter[d] = ndi[d].begin();

  while (totalnum-- > 0) {
    // create an SOffset obj and add it to our lists
    SOffset<Dim> newpoint;
    for (d=0; d < Dim; ++d)
      newpoint[d] = *(counter[d]);
    removeIndex(newpoint);

    // increment the index iterators
    unsigned int chkdim = 0;
    while(chkdim < Dim) {
      ++(counter[chkdim]);
      if (counter[chkdim] == ndi[chkdim].end()) {
	counter[chkdim] = ndi[chkdim].begin();
	chkdim++;
      } else {
	break;
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////
// reserve storage space equal to the given fraction of the size of
// each vnode.  if fraction=1.0, reserve storage for the entire vnode.
template<unsigned int Dim>
void SIndex<Dim>::reserve(double fraction) {
  for (iterator_iv a = begin_iv(); a != end_iv(); ++a) {
    typename LSIndex<Dim>::size_type newcapacity = (*a)->getDomain().size();
    if (fraction < 0.9999 && fraction > 0.0)
      newcapacity = static_cast<typename LSIndex<Dim>::size_type>(fraction *
				static_cast<double>(newcapacity));
    (*a)->reserve(newcapacity);
  }
}


////////////////////////////////////////////////////////////////////////////
// clear out the existing indices
template<unsigned int Dim>
void SIndex<Dim>::clear() {
  

  // tell all LSIndex objects to remove their points
  for (iterator_iv a = begin_iv(); a != end_iv(); ++a) {
    (*a)->clear();
  }
}


////////////////////////////////////////////////////////////////////////////
// return whether the given point is contained here
template<unsigned int Dim>
bool SIndex<Dim>::hasIndex(const SOffset<Dim>& so) const {
  
  

  for (const_iterator_iv a = begin_iv(); a != end_iv(); ++a)
    if ((*a)->hasIndex(so))
      return true;
  return false;
}


////////////////////////////////////////////////////////////////////////////
// assignment operator
// NOTE: this right now only works properly when the layout's match
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator=(const SIndex<Dim>& si) {
  
  

  if (&si != this) {
    // copy the offset and layout, checking ourselves in if necessary
    Offset = si.Offset;
    if (Layout == 0 || Layout != si.Layout) {
      if (Layout != 0)
	Layout->checkout(*this);
      Layout = si.Layout;
      Layout->checkin(*this);
    }

    // copy the list of index objects, replacing our current contents
    IndexList = si.IndexList;

    // copy the bounding box
    BoundingBox = si.BoundingBox;
  }

  return *this;
}


////////////////////////////////////////////////////////////////////////////
// assignment operator from a single SOffset.  This will leave this SIndex
// with just the one point.
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator=(const SOffset<Dim>& so) {
  
  

  // put in the single point
  clear();
  addIndex(so);

  // set our bounding box to just this point
  toNDIndex(so, BoundingBox);

  return *this;
}


////////////////////////////////////////////////////////////////////////////
// assignment operator from an NDIndex.  All the points in the index space
// will be added.
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator=(const NDIndex<Dim>& ndi) {
  
  

  // put in all the points from the NDIndex
  clear();
  addIndex(ndi);

  // set our bounding box to this NDIndex domain
  BoundingBox = ndi;

  return *this;
}


////////////////////////////////////////////////////////////////////////////
// intersection operator, with another SIndex object.
// NOTE: this right now only works properly when the layout's match
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator&=(const SIndex<Dim>& si) {
  
  

  if (&si != this) {
    // for all our own points, only keep those which are in the other one
    SIndex<Dim> newval(*Layout);
    const_iterator_iv a  = begin_iv();
    const_iterator_iv ea = end_iv();
    iterator_iv na = newval.begin_iv();
    for ( ; a != ea; ++a, ++na) {
      typename LSIndex<Dim>::const_iterator ls_i = (*a)->begin();
      typename LSIndex<Dim>::const_iterator ls_e = (*a)->end();
      for ( ; ls_i != ls_e ; ++ls_i) {
	if (si.hasIndex(*ls_i))
	  newval.addIndex(na, *ls_i);
      }
    }

    // copy over the points to here
    *this = newval;

    // the bounding box of the intersection will be the intersection
    // of the two original bounding boxes
    BoundingBox = BoundingBox.intersect(si.BoundingBox);
  }

  return *this;
}


////////////////////////////////////////////////////////////////////////////
// intersection operator, with another SOffset object.
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator&=(const SOffset<Dim>& so) {
  
  

  bool found = hasIndex(so);
  clear();
  if (found)
    addIndex(so);
  toNDIndex(so, BoundingBox);

  return *this;
}


////////////////////////////////////////////////////////////////////////////
// intersection operator, with the points in an NDIndex
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator&=(const NDIndex<Dim>& ndi) {
  
  

  // for all our own points, only keep those which are in the other one
  SIndex<Dim> newval(*Layout);
  const_iterator_iv a  = begin_iv();
  const_iterator_iv ea = end_iv();
  iterator_iv na = newval.begin_iv();
  for ( ; a != ea; ++a, ++na) {
    typename LSIndex<Dim>::const_iterator ls_i = (*a)->begin();
    typename LSIndex<Dim>::const_iterator ls_e = (*a)->end();
    for ( ; ls_i != ls_e ; ++ls_i) {
      if ((*ls_i).inside(ndi))
	newval.addIndex(na, *ls_i);
    }
  }

  // the bounding box of the intersection will be the intersection
  // of the two original bounding boxes
  BoundingBox = BoundingBox.intersect(ndi);

  // copy over the points to here
  *this = newval;
  return *this;
}


////////////////////////////////////////////////////////////////////////////
// union operator, with another SIndex or SOffset object.  This will
// append the point if it is not already present.
// NOTE: this right now only works properly when the layout's match.
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator|=(const SIndex<Dim>& si) {
  
  

  if (&si != this) {
    const_iterator_iv a  = si.begin_iv();
    const_iterator_iv ea = si.end_iv();
    iterator_iv na = begin_iv();
    for ( ; a != ea; ++a, ++na) {
      typename LSIndex<Dim>::const_iterator ls_i = (*a)->begin();
      typename LSIndex<Dim>::const_iterator ls_e = (*a)->end();
      for ( ; ls_i != ls_e ; ++ls_i)
	addIndex(na, *ls_i);
    }

    // just reset the bounding box to the original full domain
    BoundingBox = Layout->getDomain();
  }

  return *this;
}


////////////////////////////////////////////////////////////////////////////
// union operator, with another SIndex or SOffset object.  This will
// append the point if it is not already present.
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator|=(const SOffset<Dim>& so) {
  
  

  addIndex(so);

  // just reset the bounding box to the original full domain
  BoundingBox = Layout->getDomain();

  return *this;
}


////////////////////////////////////////////////////////////////////////////
// union operator, with an NDIndex object.  This just needs to add the
// points in the NDIndex object
template<unsigned int Dim>
SIndex<Dim>& SIndex<Dim>::operator|=(const NDIndex<Dim>& ndi) {
  
  

  addIndex(ndi);

  // just reset the bounding box to the original full domain
  BoundingBox = Layout->getDomain();

  return *this;
}


////////////////////////////////////////////////////////////////////////////
// () operators which make a copy of this SIndex with an extra offset.
// These are functionally identical to the operator+, but provide a
// nicer syntax.  That is, si(1,1) means  si + SOffset<Dim>(1,1)
template<unsigned int Dim>
SIndex<Dim> SIndex<Dim>::operator()(int i0) {
  
  

  CTAssert(Dim==1);
  return SIndex(*this, SOffset<Dim>(i0));
}

template<unsigned int Dim>
SIndex<Dim> SIndex<Dim>::operator()(int i0, int i1) {
  
  
  CTAssert(Dim==2);
  return SIndex(*this, SOffset<Dim>(i0,i1));
}

template<unsigned int Dim>
SIndex<Dim> SIndex<Dim>::operator()(int i0, int i1, int i2) {
  
  

  CTAssert(Dim==3);
  return SIndex(*this, SOffset<Dim>(i0,i1,i2));
}

template<unsigned int Dim>
SIndex<Dim> SIndex<Dim>::operator()(int i0, int i1, int i2, int i3) {
  
  

  CTAssert(Dim==4);
  return SIndex(*this, SOffset<Dim>(i0,i1,i2,i3));
}

template<unsigned int Dim>
SIndex<Dim> SIndex<Dim>::operator()(int i0, int i1, int i2, int i3, int i4) {
  
  

  CTAssert(Dim==5);
  return SIndex(*this, SOffset<Dim>(i0,i1,i2,i3,i4));
}

template<unsigned int Dim>
SIndex<Dim> SIndex<Dim>::operator()(int i0, int i1, int i2, int i3, int i4,
				    int i5) {
  
  

  CTAssert(Dim==6);
  return SIndex(*this, SOffset<Dim>(i0,i1,i2,i3,i4,i5));
}

template<unsigned int Dim>
SIndex<Dim> SIndex<Dim>::operator()(const SOffset<Dim>& so) {
  
  

  return SIndex(*this, so);
}

template<unsigned int Dim>
SIndex<Dim> SIndex<Dim>::operator()(const int *so) {
  
  

  return SIndex(*this, SOffset<Dim>(so));
}


////////////////////////////////////////////////////////////////////////////
// operator[], which is used with Index or NDIndex objects to further
// subset the data.  This will only work if the dimension of the Index
// arguments + Brackets is <= Dim.  Otherwise, too many dimensions worth
// of Index objects are being applied
template<unsigned int Dim>
IndexedSIndex<Dim,1> SIndex<Dim>::operator[](const Index &i) {
  
  

  CTAssert(Dim >= 1);
  NDIndex<Dim> dom;
  dom[0] = i;
  return IndexedSIndex<Dim,1>(*this, dom);
}



////////////////////////////////////////////////////////////////////////////
// convert from the given SOffset value to an NDIndex, with offset added
template<unsigned int Dim>
void SIndex<Dim>::toNDIndex(const SOffset<Dim>& val, NDIndex<Dim>& NDI) {
  
  

  for (unsigned int d=0; d < Dim; ++d) {
    int m = val[d] + Offset[d];
    NDI[d] = Index(m, m);
  }
}


////////////////////////////////////////////////////////////////////////////
// return the total size, which is the sum of the individual sizes
template<unsigned int Dim>
typename SIndex<Dim>::size_type_iv SIndex<Dim>::size() const {
  
  

  size_type_iv retval = 0;
  for (const_iterator_iv a = begin_iv(); a != end_iv(); ++a)
    retval += (*a)->size();

  return retval;
}


////////////////////////////////////////////////////////////////////////////
// change to using a new layout.
// NOTE: this needs to be fixed to properly redistribute points to
// their proper nodes.
template<unsigned int Dim>
void SIndex<Dim>::setFieldLayout(FieldLayout<Dim>& fl) {
  
  

  // create a new, empty SIndex
  SIndex<Dim> newindx(fl, Offset);

  // NOTE: PUT CODE TO REDISTRIBUTE POINTS PROPERLY HERE


  // copy over this new SIndex to ourselves
  *this = newindx;
}


////////////////////////////////////////////////////////////////////////////
// Repartition onto a new layout
template<unsigned int Dim>
void SIndex<Dim>::Repartition(UserList *userlist) {
  
  

  if (Layout != 0 && userlist->getUserListID() == Layout->get_Id()) {
    // it is indeed our layout which is being repartitioned.  The easiest
    // way to do this is to just use the setFieldLayout with our current
    // layout, but which we now know is partitioned differently.
    setFieldLayout(*Layout);
  }
}


////////////////////////////////////////////////////////////////////////////
// Tell this object that an object is being deleted
template<unsigned int Dim>
void SIndex<Dim>::notifyUserOfDelete(UserList *userlist) {
  
  

  if (Layout != 0 && userlist->getUserListID() == Layout->get_Id())
    Layout = 0;
}


////////////////////////////////////////////////////////////////////////////
// write contents to given ostream
template<unsigned int Dim>
std::ostream& operator<<(std::ostream& o, const SIndex<Dim>& si) {
  
  

  o << "vnodes = " << si.size_iv();
  o << ", offset = " << si.getOffset();
  o << ", bounding box = " << si.getDomain();
  o << ", points in each LField (w/offset):" << std::endl;
  for (typename SIndex<Dim>::const_iterator_iv a=si.begin_iv(); a!=si.end_iv(); ++a) {
    o << "  In LField w/domain=" << (*a)->getDomain() << ":" << std::endl;
    o <<   "    compressed = " << (*a)->IsCompressed() << std::endl;
    unsigned int lsize = (*a)->size();
    for (unsigned int i=0; i < lsize; ++i)
      o << "    " << (*a)->getIndex(i) + si.getOffset() << std::endl;
  }

  return o;
}


////////////////////////////////////////////////////////////////////////////
// print out debugging info
template<unsigned int Dim>
void SIndex<Dim>::printDebug(Inform& o) const {
  

  o << *this << endl;
}


/***************************************************************************
 * $RCSfile: SIndex.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: SIndex.cpp,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $
 ***************************************************************************/
