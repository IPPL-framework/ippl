// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef NDINDEX_INLINES_H
#define NDINDEX_INLINES_H

// include files
#include "Utility/PAssert.h"


//////////////////////////////////////////////////////////////////////
// Construct from a simple array of Indexes
template<unsigned Dim>
inline
NDIndex<Dim>::NDIndex(const Index* idx)
{
  for (unsigned i=0; i<Dim; ++i)
    p[i]=idx[i];
}

// Intersect with another NDIndex.
template<unsigned Dim>
inline NDIndex<Dim> 
NDIndex<Dim>::intersect(const NDIndex<Dim>& i) const 
{
  NDIndex<Dim> r;
  for (unsigned d=0;d<Dim;++d)
    r[d] = p[d].intersect( i[d] );
  return r;
}

//////////////////////////////////////////////////////////////////////

// Forward substitute the base of one index into this one.

template<unsigned D1, unsigned D2>
inline NDIndex<D1> plugBase(const NDIndex<D1>& i1, const NDIndex<D2>& i2)
{
  // Construct and return ret.
  NDIndex<D1> ret(i1);
  // Loop over each of the Indexes in ret.
  for (unsigned d1=0; d1<D1; ++d1)
    // Try to find the corresponding Index in i2.
    for (unsigned d2=0; d2<D2; ++d2)
      if ( i1[d1].sameBase( i2[d2] ) )
	{
	  // Found it.  Substitute for this one.
	  ret[d1] = ret[d1].plugBase( i2[d2] );
	  break;
	}
  return ret;
}



//////////////////////////////////////////////////////////////////////


template<unsigned Dim>
inline unsigned 
NDIndex<Dim>::size() const
{
  unsigned s = p[0].length();
  for (unsigned int d=1; d<Dim; ++d)
    s *= p[d].length();
  return s;
}

//----------------------------------------------------------------------

template <unsigned Dim>
inline bool
NDIndex<Dim>::touches(const NDIndex<Dim>& a) const
{
  bool touch = true;
  for (unsigned int d=0; (d<Dim)&&touch ; ++d)
    touch = touch && p[d].touches(a.p[d]);
  return touch;
}

//----------------------------------------------------------------------

template <unsigned Dim>
inline bool
NDIndex<Dim>::contains(const NDIndex<Dim>& a) const
{
  bool cont = true;
  for (unsigned int d=0; (d<Dim)&&cont ; ++d)
    cont = cont && p[d].contains(a.p[d]);
  return cont;
}

//----------------------------------------------------------------------

template <unsigned Dim>
inline bool
NDIndex<Dim>::containsAllPoints(const NDIndex<Dim>& a) const
{
  bool cont = true;
  for (unsigned int d=0; (d<Dim)&&cont ; ++d)
    cont = cont && p[d].containsAllPoints(a.p[d]);
  return cont;
}

//----------------------------------------------------------------------

template<unsigned Dim>
inline bool
NDIndex<Dim>::empty() const
{
  bool r = false;
  for (unsigned d=0; d<Dim; ++d)
    r = r || p[d].empty();
  return r;
}

//----------------------------------------------------------------------

template<unsigned Dim>
inline bool
NDIndex<Dim>::split(NDIndex<Dim>& l, NDIndex<Dim>& r,
		    unsigned d, double a) const
{
  if ( &l != this )
    l = *this;
  if ( &r != this )
    r = *this;
  return p[d].split(l[d],r[d],a);
}

template<unsigned Dim>
inline bool
NDIndex<Dim>::split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d) const
{
  if ( &l != this )
    l = *this;
  if ( &r != this )
    r = *this;
  return p[d].split(l[d],r[d]);
}

//----------------------------------------------------------------------

template<unsigned Dim>
inline bool
NDIndex<Dim>::split(NDIndex<Dim>& l, NDIndex<Dim>& r) const
{
  unsigned int max_dim = 0;
  unsigned int max_length = 0;
  for (unsigned int d=0; d<Dim; ++d)
    if ( p[d].length() > max_length ) {
      max_dim = d;
      max_length = p[d].length();
    }
  return split(l,r,max_dim);
}

//////////////////////////////////////////////////////////////////////

template<unsigned Dim>
inline
NDIndex<Dim>::NDIndex(const Index& i0)
{
  PInsist(Dim==1, "Number of arguments does not match NDIndex dimension!!");
  p[0] = i0;
}

template<unsigned Dim>
inline
NDIndex<Dim>::NDIndex(const Index& i0,const Index& i1)
{
  PInsist(Dim==2, "Number of arguments does not match NDIndex dimension!!");
  p[0] = i0;
  p[1] = i1;
}

template<unsigned Dim>
inline
NDIndex<Dim>::NDIndex(const Index& i0,const Index& i1,const Index& i2)
{
  PInsist(Dim==3, "Number of arguments does not match NDIndex dimension!!");
  p[0] = i0;
  p[1] = i1;
  p[2] = i2;
}

template<unsigned Dim>
inline
NDIndex<Dim>::NDIndex(const Index& i0,const Index& i1,const Index& i2,
                      const Index& i3)
{
  PInsist(Dim==4, "Number of arguments does not match NDIndex dimension!!");
  p[0] = i0;
  p[1] = i1;
  p[2] = i2;
  p[3] = i3;
}

template<unsigned Dim>
inline
NDIndex<Dim>::NDIndex(const Index& i0,const Index& i1,const Index& i2,
                      const Index& i3,const Index& i4)
{
  PInsist(Dim==5, "Number of arguments does not match NDIndex dimension!!");
  p[0] = i0;
  p[1] = i1;
  p[2] = i2;
  p[3] = i3;
  p[4] = i4;
}

template<unsigned Dim>
inline
NDIndex<Dim>::NDIndex(const Index& i0,const Index& i1,const Index& i2,
                      const Index& i3,const Index& i4,const Index& i5)
{
  PInsist(Dim==6, "Number of arguments does not match NDIndex dimension!!");
  p[0] = i0;
  p[1] = i1;
  p[2] = i2;
  p[3] = i3;
  p[4] = i4;
  p[5] = i5;
}

template<> inline
NDIndex<2>::NDIndex(const NDIndex<1>& ndi, const Index& i)
{
  p[0] = ndi[0];
  p[1] = i;
}
template<> inline
NDIndex<3>::NDIndex(const NDIndex<2>& ndi, const Index& i)
{
  p[0] = ndi[0];
  p[1] = ndi[1];
  p[2] = i;
}
template<> inline
NDIndex<4>::NDIndex(const NDIndex<3>& ndi, const Index& i)
{
  p[0] = ndi[0];
  p[1] = ndi[1];
  p[2] = ndi[2];
  p[3] = i;
}
template<> inline
NDIndex<5>::NDIndex(const NDIndex<4>& ndi, const Index& i)
{
  p[0] = ndi[0];
  p[1] = ndi[1];
  p[2] = ndi[2];
  p[3] = ndi[3];
  p[4] = i;
}
template<> inline
NDIndex<6>::NDIndex(const NDIndex<5>& ndi, const Index& i)
{
  p[0] = ndi[0];
  p[1] = ndi[1];
  p[2] = ndi[2];
  p[3] = ndi[3];
  p[4] = ndi[4];
  p[5] = i;
}

//////////////////////////////////////////////////////////////////////
//
// Attempts to determine whether i2 represents a stencil relative
// to i1. For each index in i1, it checks through all of the indices
// in i2 to see if the id's match. If a match is obtained, but the 
// indexes are not equivalent, this must be a stencil. If i1's index
// is not found in i2, we make the safe choice of returning
// true (is-a-stencil). We only conclude this is not a stencil if
// all indices in i1 appear in i2 and are equivalent.
//
//////////////////////////////////////////////////////////////////////

template<unsigned D1, unsigned D2>
inline bool isStencil(const NDIndex<D1> &i1, const NDIndex<D2> &i2)
{
  for (unsigned i = 0; i < D1; i++)
    {
      bool found = false;
      for (unsigned j = 0; j < D2; j++)
	{
	  if (i1[i].id() == i2[j].id())
	    {
	      found = true;
	      if (!(i1[i] == i2[j]))
		return true;
	    }
	}
      if (!found)
	return true;
    }

  return false;
}

#endif // NDINDEX_INLINES_H

/***************************************************************************
 * $RCSfile: NDIndexInlines.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: NDIndexInlines.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
