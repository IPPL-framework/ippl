// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef APPTYPE_TRAITS_H
#define APPTYPE_TRAITS_H



// templated traits struct for general IPPL AppType
template <class T>
struct AppTypeTraits {
  typedef typename T::Element_t Element_t;
  enum { ElemDim = T::ElemDim };
};

// specialized versions for built-in types
#define BUILTIN_APPTYPE_TRAITS(T)                                \
template <>                                                      \
struct AppTypeTraits< T > {                                      \
  typedef T Element_t;                                           \
  enum { ElemDim = 0 };                                          \
};

BUILTIN_APPTYPE_TRAITS(bool)
BUILTIN_APPTYPE_TRAITS(char)
BUILTIN_APPTYPE_TRAITS(unsigned char)
BUILTIN_APPTYPE_TRAITS(short)
BUILTIN_APPTYPE_TRAITS(unsigned short)
BUILTIN_APPTYPE_TRAITS(int)
BUILTIN_APPTYPE_TRAITS(unsigned int)
BUILTIN_APPTYPE_TRAITS(long)
BUILTIN_APPTYPE_TRAITS(unsigned long)
BUILTIN_APPTYPE_TRAITS(long long)
BUILTIN_APPTYPE_TRAITS(float)
BUILTIN_APPTYPE_TRAITS(double)
BUILTIN_APPTYPE_TRAITS(long double)
BUILTIN_APPTYPE_TRAITS(std::complex<float>)
BUILTIN_APPTYPE_TRAITS(std::complex<double>)

#undef BUILTIN_APPTYPE_TRAITS

#endif // APPTYPE_TRAITS_H

/***************************************************************************
 * $RCSfile: AppTypeTraits.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: AppTypeTraits.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/

