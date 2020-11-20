//
// Class NDRegion
//   NDRegion is a simple container of N PRegion objects. It is templated
//   on the type of data (T) and the number of PRegions (Dim). It can also
//   be templated on the Mesh type, in order to provide a mesh for converting
//   an NDIndex into an NDRegion.
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
#ifndef IPPL_NDREGION_H
#define IPPL_NDREGION_H

#include "Region/PRegion.h"
//#include "Utility/PAssert.h"

//#include <iostream>
/*
// forward declarations
class Message;
*/
namespace ippl {
template < class T, unsigned Dim > class NDRegion;
/*
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
*/
template < class T, unsigned Dim >
std::ostream& operator<<(std::ostream&, const NDRegion<T,Dim>&);
}

namespace ippl {
    template <typename T, unsigned Dim>
    class NDRegion
    {
    public:
        KOKKOS_FUNCTION
        NDRegion() {}

        KOKKOS_FUNCTION
        ~NDRegion() { };

    /*
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
    }*/

  KOKKOS_FUNCTION
  NDRegion(const PRegion<T>& r1, const PRegion<T>& r2, const PRegion<T>& r3) {
      //    PInsist(Dim==3, "Number of arguments does not match NDRegion dimension!!");
    p[0] = r1;
    p[1] = r2;
    p[2] = r3;
  }

  // copy constructor
  KOKKOS_INLINE_FUNCTION
  NDRegion(const NDRegion<T,Dim>& nr) {
      for (unsigned int i=0; i < Dim; i++)
	  p[i] = nr.p[i];
  }

  // operator= definitions
  KOKKOS_INLINE_FUNCTION
  NDRegion<T,Dim>& operator=(const NDRegion<T,Dim>& nr) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] = nr.p[i];
    return *this;
    }

  // Return a reference to any of the PRegion<T> objects.
  KOKKOS_INLINE_FUNCTION
  const PRegion<T>& operator[](unsigned d) const { return p[d]; }

  KOKKOS_INLINE_FUNCTION
  PRegion<T>& operator[](unsigned d) { return p[d]; }

    /*
  // return the volume of this region
  T volume() const {
    T v = p[0].length();
    for (unsigned int i=1; i < Dim; i++)
      v *= p[i].length();
    return v;
  }
    */
  // compute-assign operators
  KOKKOS_INLINE_FUNCTION
  NDRegion<T,Dim>& operator+=(const T t) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] += t;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  NDRegion<T,Dim>& operator-=(const T t) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] -= t;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  NDRegion<T,Dim>& operator*=(const T t) {
    for (unsigned int i=0; i < Dim; i++)
      p[i] *= t;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  NDRegion<T,Dim>& operator/=(const T t) {
    if (t != 0) {
      for (unsigned int i=0; i < Dim; i++) p[i] /= t;
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  bool empty() const {
    for (unsigned int i=0; i < Dim; i++)
      if ( ! p[i].empty() )
	return false;
    return true;
  }
    /*

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
  }*/

    private:
        //! Array of PRegions
        PRegion<T> p[Dim];
    };
}

/*
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
*/
// write NDRegion out to the given stream


namespace ippl {

template <class T, unsigned Dim>
inline
std::ostream& operator<<(std::ostream& out, const NDRegion<T,Dim>& idx) {
  out << '{';
  for (unsigned d = 0; d < Dim; ++d) 
    out << idx[d] << ((d==Dim-1) ? '}' : ',');
  return out;
}

}

/*
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
*/

#endif