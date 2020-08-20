// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef VEC_H
#define VEC_H

// include files
#include "Message/Message.h"
#include "Utility/PAssert.h"

//////////////////////////////////////////////////////////////////////

template<class T, unsigned Length>
class vec
{
public:
  vec() {}
  vec(T v0);
  vec(T v0, T v1);
  vec(T v0, T v1, T v2);
  T& operator[](unsigned d) { return Ptr[d]; }
  const T& operator[](unsigned d) const { return Ptr[d]; }
  Message& putMessage(Message &m) {
    m.put(Ptr, Ptr + Length);
    return m;
  }
  Message& getMessage(Message &m) {
    m.get_iter(Ptr);
    return m;
  }

  static T dot(const T*,const T*);
private:
  T Ptr[Length];
};

//////////////////////////////////////////////////////////////////////

template<class T, unsigned Length>
inline 
vec<T,Length>::vec(T v0)
{
  CTAssert(Length==1);
  Ptr[0] = v0;
}

template<class T, unsigned Length>
inline 
vec<T,Length>::vec(T v0, T v1)
{
  CTAssert(Length==2);
  Ptr[0] = v0;
  Ptr[1] = v1;
}

template<class T, unsigned Length>
inline 
vec<T,Length>::vec(T v0, T v1, T v2)
{
  CTAssert(Length==3);
  Ptr[0] = v0;
  Ptr[1] = v1;
  Ptr[2] = v2;
}

//////////////////////////////////////////////////////////////////////

//
// Define a global function for taking the dot product between two 
// short arrays of objects of type T.
//
template<class T, unsigned Length>
inline T
vec<T,Length>::dot(const T* l, const T* r)
{
  T ret = l[0]*r[0];
  for (unsigned int i=1; i<Length; ++i)
    ret += l[i]*r[i];
  return ret;
}

//////////////////////////////////////////////////////////////////////

#endif // VEC_H

/***************************************************************************
 * $RCSfile: Vec.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:34 $
 * IPPL_VERSION_ID: $Id: Vec.h,v 1.1.1.1 2003/01/23 07:40:34 adelmann Exp $ 
 ***************************************************************************/
