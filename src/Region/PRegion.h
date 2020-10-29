#ifndef IPPL_PREGION_H
#define IPPL_PREGION_H

/***********************************************************************
 * PRegion represents a (possibly continuous) numeric interval.  It is 
 * similar to Index, with the following differences:
 *	1. It is templated on the data type; Index always uses integers
 *	2. A PRegion is defined between two endpoints A and B; the PRegion
 *	   includes values X where A <= X < B (i.e., X in [A,B) )
 *	3. PRegion does not keep track of a base Index, and does not
 *	   supply the plugBase operation.  It is not designed for use in
 *	   Field operations like Index is, it is meant instead for use in
 *	   Particle construction and usage.
 *
 ***********************************************************************
 *
 * PRegion<T>()      --> make a PRegion on [0,1)
 * PRegion<T>(B)     --> make a PRegion on [0,B)
 * PRegion<T>(A,B)   --> make a PRegion on [A,B)
 *
 ***********************************************************************/

// #include <Kokkos_Core.hpp>

// include files
//#include "Message/Message.h"

//#include <iostream>

// forward declarations
namespace ippl {
template<class T> class PRegion;
/*template <class T>
PRegion<T> operator+(const PRegion<T>&, T);
template <class T>
PRegion<T> operator+(T, const PRegion<T>&);
template <class T>
PRegion<T> operator-(const PRegion<T>&, T);
template <class T>
PRegion<T> operator-(T, const PRegion<T>&);
template <class T>
PRegion<T> operator-(const PRegion<T>&);
template <class T>
PRegion<T> operator*(const PRegion<T>&, T);
template <class T>
PRegion<T> operator*(T, const PRegion<T>&);
template <class T>
PRegion<T> operator/(const PRegion<T>&, T);*/
template<class T>
std::ostream& operator<<(std::ostream&, const PRegion<T>&);

}

namespace ippl {

template<typename T>
class PRegion {

public:
  // 
  // Member functions.  Make these inline for efficiency.
  //

  // Constructors
  KOKKOS_FUNCTION
  PRegion() : First(0), Last(1) {}                       // [0,1)

  KOKKOS_FUNCTION
  PRegion(T B) : First(0), Last(B) {}                    // [0,B)
   
  KOKKOS_FUNCTION
  PRegion(T A, T B) : First(A), Last(B) {}               // [A,B)

   // Destructor ... nothing to do
  KOKKOS_FUNCTION
  ~PRegion() { }
   
    //KOKKOS_FUNCTION
  PRegion(const PRegion<T>&) = default;
   
    //KOKKOS_INLINE_FUNCTION
  PRegion<T>& operator=(const PRegion<T>& rhs) = default;
    
  // General query functions
  KOKKOS_INLINE_FUNCTION
  T min() const { return (First < Last ? First : Last); }  // smallest elem
  
  KOKKOS_INLINE_FUNCTION  
  T max() const { return (First > Last ? First : Last); }  // largest elem

  KOKKOS_INLINE_FUNCTION
  T length() const { return (max() - min()); }	           // length of region

  KOKKOS_INLINE_FUNCTION
  T first() const { return First; }		           // first element.

  KOKKOS_INLINE_FUNCTION
  T last() const { return Last; }		           // last element.

  KOKKOS_INLINE_FUNCTION
  bool empty() const { return (First == Last);}	           // is it empty?
    
  // compute-assign operators
  KOKKOS_INLINE_FUNCTION
  PRegion<T>& operator+=(T t) {
    First += t;
    Last += t;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  PRegion<T>& operator-=(T t) {
    First -= t;
    Last -= t;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  PRegion<T>& operator*=(T t) {
    First *= t;
    Last *= t;
    return *this;
  }
  
  KOKKOS_INLINE_FUNCTION
  PRegion<T>& operator/=(T t) {
    if (t != 0) {
      First /= t;
      Last /= t;
    }
    return *this;
  }
  /*
  // Intersect with another PRegion.  Since we have possibly continuous
  // variables, we do not consider the stride here, just where the two
  // intervals overlap (if at all)
  PRegion<T> intersect(const PRegion<T>& r) const {
    T A = 0;
    T B = 0;

    // find min and max of both ranges
    T Amin = min();
    T Amax = max();
    T Bmin = r.min();
    T Bmax = r.max();

    // make sure these regions overlap
    // must check special case of single points
    if (Amin == Amax) {
      if ((Bmin == Bmax && Amin == Bmin) ||
	  (Bmin != Bmax && Amin >= Bmin && Amin < Bmax))
	A = B = Amin;
    }
    else if (Bmin == Bmax) {
      if (Bmin >= Amin && Bmin < Amax)
	A = B = Bmin;
    }
    else {
      if (Amax > Bmin && Bmax > Amin) {
	A = (Amin > Bmin ? Amin : Bmin);
	B = (Amax < Bmax ? Amax : Bmax);
      }
    }

    // now return the intersecting region
    return PRegion<T>(A, B);
  }

  // Test to see if there is any overlap between two PRegions
  bool touches(const PRegion<T>& r) const {
	  bool retval = false;
	  // find min and max of both ranges
	  T Amin = min();
	  T Amax = max();
	  T Bmin = r.min();
	  T Bmax = r.max();

	  // check for overlap ... must check special case of single points
	  if (Amin == Amax) {
		  if ((Bmin == Bmax && Amin == Bmin) ||
				  (Bmin != Bmax && Amin >= Bmin && Amin < Bmax))
			  retval = true;
	  }
	  else if (Bmin == Bmax) {
		  if (Bmin >= Amin && Bmin < Amax)
			  retval = true;
	  }
	  else {
		  if (Amax > Bmin && Bmax > Amin)
			  retval = true;
	  }
	  return retval;
  }

  // Test to see if the given PRegion is contained within this one
  bool contains(const PRegion<T>& r) const {
    return ( min() <= r.min() && max() >= r.max() );
  }

  // Split one into two.
  bool split(PRegion<T>& l, PRegion<T>& r) const {
    T mid = First + (Last - First) / T(2);
    l = PRegion<T>(First, mid);
    r = PRegion<T>(mid, Last);
    return true;
  }

  // An operator< so we can impose some sort of ordering.
  bool operator<(const PRegion<T>& r) const {
    T L1 = length();
    T L2 = r.length();
    T Amin = min();
    T Bmin = r.min();
    return ( (L1 < L2) || ( (L1 == L2) && ( (Amin < Bmin) ||
                   ( (Amin == Bmin) && (L1 > 0) ) ) ) );
  }

  // Test for equality.
  bool operator==(const PRegion<T>& r) const {
    return ( (Last==r.Last) && (First==r.First) );
  }

  // put data into a message to send to another node
  Message& putMessage(Message& m) {
    T d[2];
    d[0] = First;
    d[1] = Last;
    m.put(d, d + 2);
    return m;
  }

  // get data out from a message
  Message& getMessage(Message& m) {
    T d[2];
    m.get_iter(d);
    *this = PRegion<T>(d[0], d[1]);
    return m;
    }*/

private: 
  // The interval endpoints
  T First, Last;
};

}

/*
// Additive operations.
template <class T>
inline 
PRegion<T> operator+(const PRegion<T>& r, T t) {
  return PRegion<T>(r.first() + t, r.last() + t);
}
template <class T>
inline 
PRegion<T> operator+(T t, const PRegion<T>& r) {
  return PRegion<T>(r.first() + t, r.last() + t);
}
template <class T>
inline 
PRegion<T> operator-(const PRegion<T>& r, T t) {
  return PRegion<T>(r.first() - t, r.last() - t);
}
template <class T>
inline 
PRegion<T> operator-(T t, const PRegion<T>& r) {
  return PRegion<T>(t - r.first(), t - r.last());
}

// Multipplicative operations.
template <class T>
inline 
PRegion<T> operator-(const PRegion<T>& r) {
  return PRegion<T>(-r.first(), -r.last());
}
template <class T>
inline 
PRegion<T> operator*(const PRegion<T>& r, T t) {
  return PRegion<T>(r.first() * t, r.last() * t);
}
template <class T>
inline 
PRegion<T> operator*(T t, const PRegion<T>& r) {
  return PRegion<T>(r.first() * t, r.last() * t);
}
template <class T>
inline 
PRegion<T> operator/(const PRegion<T>& r, T t) {
  if (t != 0)
    return PRegion<T>(r.first() / t, r.last() / t);
  else        // This is an error!!
    return r;
}
*/


namespace ippl {
// Print out PRegion.
template <class T>
inline 
std::ostream& operator<<(std::ostream& out, const PRegion<T>& r) {
  out << '[' << r.min();
  out << ',' << r.max();
  out << ')';
  return out;
}
}

#endif // PREGION_H

/***************************************************************************
 * $RCSfile: PRegion.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:32 $
 * IPPL_VERSION_ID: $Id: PRegion.h,v 1.1.1.1 2003/01/23 07:40:32 adelmann Exp $ 
 ***************************************************************************/
