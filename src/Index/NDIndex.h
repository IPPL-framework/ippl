// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef NDINDEX_H
#define NDINDEX_H

// include files
#include "Index/Index.h"

#include <iostream>

// forward declarations
template <unsigned Dim> class NDIndex;

template <unsigned Dim>
NDIndex<Dim> operator+(const NDIndex<Dim>&, const int *);
template <unsigned Dim>
NDIndex<Dim> operator+(const int *,const NDIndex<Dim>&);

template <unsigned Dim>
NDIndex<Dim> operator-(const NDIndex<Dim>&, const int *);
template <unsigned Dim>
NDIndex<Dim> operator-(const int *,const NDIndex<Dim>&);

template <unsigned Dim>
NDIndex<Dim> operator-(const NDIndex<Dim>&);

template <unsigned Dim>
NDIndex<Dim> operator*(const NDIndex<Dim>&, const int *);
template <unsigned Dim>
NDIndex<Dim> operator*(const int *,const NDIndex<Dim>&);

template <unsigned Dim>
NDIndex<Dim> operator/(const NDIndex<Dim>&, const int *);

template <unsigned Dim>
bool operator<(const NDIndex<Dim>&, const NDIndex<Dim>&);

template <unsigned Dim, unsigned Dim2>
bool operator==(const NDIndex<Dim>&, const NDIndex<Dim2>&);

template <unsigned Dim>
std::ostream& operator<<(std::ostream&, const NDIndex<Dim>&);

//////////////////////////////////////////////////////////////////////

//
// Implementation of plugbase that the member template and enumerated
// plugbase member functions use.
//

template<unsigned D1, unsigned D2>
NDIndex<D1> plugBase(const NDIndex<D1>&, const NDIndex<D2>&);



/***********************************************************************

This is a simple wrapper around Index that just keeps track of
N of them and passes along requests for intersect, plugBase and
so on.

***********************************************************************/

template<unsigned Dim>
class NDIndex
{
public: 

  // Null ctor does nothing.
  NDIndex() {}

  // Construct from a simple array of Indexes
  NDIndex(const Index *idx);

  // Construct from individual indexes.
  // Only instantiate the ones that make sense.
  NDIndex(const Index&);
  NDIndex(const Index&,const Index&);
  NDIndex(const Index&,const Index&,const Index&);
  NDIndex(const Index&,const Index&,const Index&,
          const Index&);
  NDIndex(const Index&,const Index&,const Index&,
          const Index&,const Index&);
  NDIndex(const Index&,const Index&,const Index&,
          const Index&,const Index&,const Index&);
  NDIndex(const NDIndex<Dim-1>&, const Index&);

  // Return a reference to any of the Indexes.
  const Index& operator[](unsigned d) const
    {
      return p[d];
    }
  Index& operator[](unsigned d)
    {
      return p[d];
    }

  // Get the total size.
  unsigned size() const;

  // Stuff for doing index mapping calculations.
  bool empty() const;
  NDIndex<Dim> intersect(const NDIndex<Dim>&) const;

  template<unsigned D>
  NDIndex<Dim> plugBase(const NDIndex<D>& i)const { 
      return ::plugBase(*this,i);
  }

  // useful functions with DomainMap.
  bool touches(const NDIndex<Dim>&) const;
  bool contains(const NDIndex<Dim>& a) const;
  bool containsAllPoints(const NDIndex<Dim> &b) const;
  
  // Split on dimension d with the given ratio 0<a<1.
  bool split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d, double a) const;
  // Split on dimension d, or the longest dimension.
  bool split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d) const;
  bool split(NDIndex<Dim>& l, NDIndex<Dim>& r) const;

  // put data into a message to send to another node
  Message& putMessage(Message& m) const {
    unsigned d;
    for ( d = 0 ; d < Dim ; ++d )
      p[d].putMessage(m);
    return m;
  }

  // get data out from a message
  Message& getMessage(Message& m) {
    unsigned d;
    for ( d = 0 ; d < Dim ; ++d )
      p[d].getMessage(m);
    return m;
  }
private:
  Index p[Dim==0?1:Dim];			// Pointer to the indexes.

};


// Additive operations.
template <unsigned Dim>
inline
NDIndex<Dim> operator+(const NDIndex<Dim>& ndi, const int * off)
{
  NDIndex<Dim> newNdi;
  for (unsigned d=0; d<Dim; d++) newNdi[d] = ndi[d] + off[d];
  return newNdi;
}
template <unsigned Dim>
inline
NDIndex<Dim> operator+(const int * off, const NDIndex<Dim>& ndi)
{
  NDIndex<Dim> newNdi;
  for (unsigned d=0; d<Dim; d++) newNdi[d] = off[d] + ndi[d];
  return newNdi;
}
template <unsigned Dim>
inline
NDIndex<Dim> operator-(const NDIndex<Dim>& ndi, const int * off)
{
  NDIndex<Dim> newNdi;
  for (unsigned d=0; d<Dim; d++) newNdi[d] = ndi[d] - off[d];
  return newNdi;
} 
template <unsigned Dim>
inline
NDIndex<Dim> operator-(const int * off, const NDIndex<Dim>& ndi)
{
  NDIndex<Dim> newNdi;
  for (unsigned d=0; d<Dim; d++) newNdi[d] = off[d] - ndi[d];
  return newNdi;
}

// Multipplicative operations.
template <unsigned Dim>
inline
NDIndex<Dim> operator-(const NDIndex<Dim>& ndi)
{
  NDIndex<Dim> newNdi;
  for (unsigned d=0; d<Dim; d++) newNdi[d] = -ndi[d];
  return newNdi;
}
template <unsigned Dim>
inline
NDIndex<Dim> operator*(const NDIndex<Dim>& ndi, const int * mult)
{
  NDIndex<Dim> newNdi;
  for (unsigned d=0; d<Dim; d++) newNdi[d] = ndi[d] * mult[d];
  return newNdi;
}
template <unsigned Dim>
inline
NDIndex<Dim> operator*(const int * mult, const NDIndex<Dim>& ndi)
{
  NDIndex<Dim> newNdi;
  for (unsigned d=0; d<Dim; d++) newNdi[d] = mult[d] * ndi[d];
  return newNdi;
}
template <unsigned Dim>
inline
NDIndex<Dim> operator/(const NDIndex<Dim>& ndi, const int *denom)
{
  NDIndex<Dim> newNdi;
  for (unsigned d=0; d<Dim; d++) newNdi[d] = ndi[d]/denom[d];
  return newNdi;
}

// Comparison operators so we can use a map container
// Just compare the Indexes in turn.
template <unsigned Dim>
inline
bool operator<(const NDIndex<Dim>& lhs, const NDIndex<Dim>& rhs) {
  for (unsigned d=0; d<Dim; ++d) {
    if (lhs[d] < rhs[d]) return true;
    if ( !(lhs[d]==rhs[d]) ) return false;
  }
  return false;
}

template <unsigned Dim, unsigned Dim2>
inline
bool operator==(const NDIndex<Dim>& lhs, const NDIndex<Dim2>& rhs) {
  if (Dim != Dim2) {
    return false;
  } else {
    for (unsigned d=0; d<Dim; ++d)
      if ( !(lhs[d]==rhs[d]) ) return false;
    return true;
  }
}

// write NDIndex out to the given stream
template <unsigned Dim>
inline std::ostream&
operator<<(std::ostream& out, const NDIndex<Dim>& idx) {
  unsigned d;
  out << '{';
  for (d = 0; d < Dim; ++d) 
    out << idx[d] << ((d==Dim-1) ? '}' : ',');
  return out;
}




//////////////////////////////////////////////////////////////////////

// Build some helper objects for use in maps.

template<unsigned Dim>
class Touches 
{
public:
  Touches() {}
  static bool test(const NDIndex<Dim>& a, const NDIndex<Dim>& b) 
  {
    return a.touches(b);
  }
};

template<unsigned Dim>
class Contains 
{
public:
  Contains() {}
  static bool test(const NDIndex<Dim>& a, const NDIndex<Dim>& b) 
  {
    return a.contains(b);
  }
};

template<unsigned Dim>
class Split 
{
public:
  Split() {}
  static bool test(NDIndex<Dim>& l,
		   NDIndex<Dim>& r,
		   const NDIndex<Dim>& a) 
  {
    return a.split(l,r);
  }
};

//////////////////////////////////////////////////////////////////////

#ifndef NDINDEX_INLINES_H
#include "Index/NDIndexInlines.h"
#endif

//////////////////////////////////////////////////////////////////////

#endif // NDINDEX_H

/***************************************************************************
 * $RCSfile: NDIndex.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: NDIndex.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
