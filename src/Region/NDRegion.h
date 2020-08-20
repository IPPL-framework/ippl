// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef NDREGION_H
#define NDREGION_H

/***********************************************************************
 * NDRegion is a simple container of N PRegion objects.  It is templated
 * on the type of data (T) and the number of PRegions (Dim).  It can also
 * be templated on the Mesh type, in order to provide a mesh for converting
 * an NDIndex into an NDRegion.
 ***********************************************************************/

// include files
#include "Region/PRegion.h"
#include "Utility/PAssert.h"

#include <iostream>

// forward declarations
class Message;
template < class T, unsigned Dim > class NDRegion;
template <class T, unsigned Dim>
NDRegion<T,Dim> operator+(const NDRegion<T,Dim>&, T);
template <class T, unsigned Dim>
NDRegion<T,Dim> operator+(T, const NDRegion<T,Dim>&);
template <class T, unsigned Dim>
NDRegion<T,Dim> operator-(const NDRegion<T,Dim>&, T);
template <class T, unsigned Dim>
NDRegion<T,Dim> operator-(T, const NDRegion<T,Dim>&);
template <class T, unsigned Dim>
NDRegion<T,Dim> operator-(const NDRegion<T,Dim>&);
template <class T, unsigned Dim>
NDRegion<T,Dim> operator*(const NDRegion<T,Dim>&, T);
template <class T, unsigned Dim>
NDRegion<T,Dim> operator*(T, const NDRegion<T,Dim>&);
template <class T, unsigned Dim>
NDRegion<T,Dim> operator/(const NDRegion<T,Dim>&, T);
template <class T, unsigned Dim>
bool operator<(const NDRegion<T,Dim>&, const NDRegion<T,Dim>&);
template <class T, unsigned Dim>
bool operator==(const NDRegion<T,Dim>&, const NDRegion<T,Dim>&);
template <class T, unsigned Dim>
bool operator!=(const NDRegion<T,Dim>&, const NDRegion<T,Dim>&);
template < class T, unsigned Dim >
std::ostream& operator<<(std::ostream&, const NDRegion<T,Dim>&);

template < class T, unsigned Dim >
class NDRegion {

public: 
  // Null ctor does nothing.
  NDRegion() {}

  // Construct from a simple array of PRegions
  NDRegion(PRegion<T>* idx) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] = idx[i];
  }

  // Construct from individual PRegions, for 1D thru 6D cases.
  NDRegion(const PRegion<T>& r1) {
    PInsist(Dim==1, "Number of arguments does not match NDRegion dimension!!");
    p[0] = r1;
  }
  NDRegion(const PRegion<T>& r1, const PRegion<T>& r2) {
    PInsist(Dim==2, "Number of arguments does not match NDRegion dimension!!");
    p[0] = r1;
    p[1] = r2;
  }
  NDRegion(const PRegion<T>& r1, const PRegion<T>& r2, const PRegion<T>& r3) {
    PInsist(Dim==3, "Number of arguments does not match NDRegion dimension!!");
    p[0] = r1;
    p[1] = r2;
    p[2] = r3;
  }
  NDRegion(const PRegion<T>& r1, const PRegion<T>& r2, const PRegion<T>& r3,
           const PRegion<T>& r4) {
    PInsist(Dim==4, "Number of arguments does not match NDRegion dimension!!");
    p[0] = r1;
    p[1] = r2;
    p[2] = r3;
    p[3] = r4;
  }
  NDRegion(const PRegion<T>& r1, const PRegion<T>& r2, const PRegion<T>& r3,
           const PRegion<T>& r4, const PRegion<T>& r5) {
    PInsist(Dim==5, "Number of arguments does not match NDRegion dimension!!");
    p[0] = r1;
    p[1] = r2;
    p[2] = r3;
    p[3] = r4;
    p[4] = r5;
  }
  NDRegion(const PRegion<T>& r1, const PRegion<T>& r2, const PRegion<T>& r3,
           const PRegion<T>& r4, const PRegion<T>& r5, const PRegion<T>& r6) {
    PInsist(Dim==6, "Number of arguments does not match NDRegion dimension!!");
    p[0] = r1;
    p[1] = r2;
    p[2] = r3;
    p[3] = r4;
    p[4] = r5;
    p[5] = r6;
  }

  // copy constructor
  NDRegion(const NDRegion<T,Dim>& nr) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] = nr.p[i];
  }

  // operator= definitions
  NDRegion<T,Dim>& operator=(const NDRegion<T,Dim>& nr) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] = nr.p[i];
    return *this;
  }

  // Return a reference to any of the PRegion<T> objects.
  const PRegion<T>& operator[](unsigned d) const { return p[d]; }
  PRegion<T>& operator[](unsigned d) { return p[d]; }

  // return the volume of this region
  T volume() const {
    T v = p[0].length();
    for (unsigned int i=1; i < Dim; i++)
      v *= p[i].length();
    return v;
  }

  // compute-assign operators
  NDRegion<T,Dim>& operator+=(const T t) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] += t;
    return *this;
  }
  NDRegion<T,Dim>& operator-=(const T t) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] -= t;
    return *this;
  }
  NDRegion<T,Dim>& operator*=(const T t) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] *= t;
    return *this;
  }
  NDRegion<T,Dim>& operator/=(const T t) {
    if (t != 0) {
      for (unsigned int i=0; i < Dim; i++) p[i] /= t;
    }
    return *this;
  }

  bool empty() const {
    for (unsigned int i=0; i < Dim; i++)
      if ( ! p[i].empty() )
	return false;
    return true;
  }

  // useful functions with DomainMap.
  NDRegion<T,Dim> intersect(const NDRegion<T,Dim>& nr) const {
    NDRegion<T,Dim> retval;
    for (unsigned int i=0; i < Dim; i++)
      retval.p[i] = p[i].intersect(nr.p[i]);
    return retval;
  }

  bool touches(const NDRegion<T,Dim>& nr) const {
    for (unsigned int i=0; i < Dim; i++)
      if ( ! (p[i].touches(nr.p[i])) )
	return false;
    return true;
  }

  bool contains(const NDRegion<T,Dim>& nr) const {
    for (unsigned int i=0; i < Dim; i++)
      if ( ! (p[i].contains(nr.p[i])) )
	return false;
    return true;
  }

  // Split on dimension d, or the longest dimension.
  bool split(NDRegion<T,Dim>& l, NDRegion<T,Dim>& r,
	     unsigned d) const {
    for (unsigned int i=0; i < Dim; i++) {
      if (i == d) {
	p[i].split(l.p[i], r.p[i]);
      }
      else {
	l.p[i] = p[i];
	r.p[i] = p[i];
      }
    }
    return true;
  }

  bool split(NDRegion<T,Dim>& l, NDRegion<T,Dim>& r) const {
    // find longest dimension
    unsigned d = 0;
    T maxlen = p[0].length();
    for (unsigned i=1; i < Dim; i++) {
      if (p[i].length() > maxlen) {
	maxlen = p[i].length();
	d = i;
      }
    }
    // split on this dimension
    return split(l, r, d);
  }

  // put data into a message to send to another node
  Message& putMessage(Message& m) {
    for ( unsigned d = 0 ; d < Dim ; ++d )
      p[d].putMessage(m);
    return m;
  }

  // get data out from a message
  Message& getMessage(Message& m) {
    for ( unsigned d = 0 ; d < Dim ; ++d )
      p[d].getMessage(m);
    return m;
  }

private:
  PRegion<T> p[Dim];			// Array of PRegions

};


// Additive operations.
template <class T, unsigned Dim>
inline
NDRegion<T,Dim> operator+(const NDRegion<T,Dim>& nr, T t) {
  NDRegion<T,Dim> retval(nr);
  retval += t;
  return retval;
}
template <class T, unsigned Dim>
inline
NDRegion<T,Dim> operator+(T t, const NDRegion<T,Dim>& nr) {
  return (nr + t);
}
template <class T, unsigned Dim>
inline
NDRegion<T,Dim> operator-(const NDRegion<T,Dim>& nr, T t) {
  return (nr + (-t));
}
template <class T, unsigned Dim>
inline
NDRegion<T,Dim> operator-(T t, const NDRegion<T,Dim>& nr) {
  return (-nr + t);
}

// Multipplicative operations.
template <class T, unsigned Dim>
inline
NDRegion<T,Dim> operator-(const NDRegion<T,Dim>& nr) {
  NDRegion<T,Dim> retval;
  for (unsigned int i=0; i < Dim; i++)
    retval[i] = -nr[i];
  return retval;
}
template <class T, unsigned Dim>
inline
NDRegion<T,Dim> operator*(const NDRegion<T,Dim>& nr, T t) {
  NDRegion<T,Dim> retval(nr);
  retval *= t;
  return retval;
}
template <class T, unsigned Dim>
inline
NDRegion<T,Dim> operator*(T t, const NDRegion<T,Dim>& nr) {
  return (nr * t);
}
template <class T, unsigned Dim>
inline
NDRegion<T,Dim> operator/(const NDRegion<T,Dim>& nr, T t) {
  return (t != 0 ? (nr * (1/t)) : nr);
}

// Comparison operators so we can use a map container
// Just compare the PRegions in turn.
template <class T, unsigned Dim>
inline
bool operator<(const NDRegion<T,Dim>& A, const NDRegion<T,Dim>& B) {
  for (unsigned int i=0; i < Dim; i++)
    if ( !(A[i] < B[i]) ) return false;
  return true;
}
template <class T, unsigned Dim>
inline
bool operator==(const NDRegion<T,Dim>& A, const NDRegion<T,Dim>& B) {
  for (unsigned int i=0; i < Dim; i++)
    if ( !(A[i] == B[i]) ) return false;
  return true;
}
template <class T, unsigned Dim>
inline
bool operator!=(const NDRegion<T,Dim>& A, const NDRegion<T,Dim>& B) {
  return !(A == B);
}

// write NDRegion out to the given stream
template <class T, unsigned Dim>
inline
std::ostream& operator<<(std::ostream& out, const NDRegion<T,Dim>& idx) {
  out << '{';
  for (unsigned d = 0; d < Dim; ++d) 
    out << idx[d] << ((d==Dim-1) ? '}' : ',');
  return out;
}


//////////////////////////////////////////////////////////////////////

// Build some helper objects for use in DomainMap
template < class T, unsigned Dim >
class TouchesRegion {
public:
  TouchesRegion() {}
  static bool test(const NDRegion<T,Dim>& a,
		   const NDRegion<T,Dim>& b) {
    return a.touches(b);
  }
};

template < class T, unsigned Dim >
class ContainsRegion {
public:
  ContainsRegion() {}
  static bool test(const NDRegion<T,Dim>& a,
		   const NDRegion<T,Dim>& b) {
    return a.contains(b);
  }
};

template < class T, unsigned Dim >
class SplitRegion {
public:
  SplitRegion() {}
  static bool test(NDRegion<T,Dim>& l, NDRegion<T,Dim>& r,
		   const NDRegion<T,Dim>& a) {
    return a.split(l,r);
  }
};


#endif // NDREGION_H

/***************************************************************************
 * $RCSfile: NDRegion.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:32 $
 * IPPL_VERSION_ID: $Id: NDRegion.h,v 1.1.1.1 2003/01/23 07:40:32 adelmann Exp $ 
 ***************************************************************************/
