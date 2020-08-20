// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PASSIGN_DEFS_H
#define PASSIGN_DEFS_H

// include files
#include "PETE/IpplExpressions.h"
#include "AppTypes/AppTypeTraits.h"

// forward declarations
template<class T> class ParticleAttrib;
template<class T, unsigned Dim> class ParticleAttribElem;


//////////////////////////////////////////////////////////////////////
//
// Evaluation functor.
//
//////////////////////////////////////////////////////////////////////

template<class T>
inline T&
for_each(ParticleAttribIterator<T>& p, EvalFunctor_0)
{
  return *p;
}

template<class T>
inline const T&
for_each(ParticleAttribConstIterator<T>& p, EvalFunctor_0)
{
  return *p;
}

#define DEFINE_EVALFUNCTOR_PAE(D)                                          \
                                                                           \
template<class T>                                                          \
inline typename AppTypeTraits<T>::Element_t&                               \
for_each(ParticleAttribElemIterator<T,D>& p, EvalFunctor_0)                \
{                                                                          \
  return *p;                                                               \
}

DEFINE_EVALFUNCTOR_PAE(1)
DEFINE_EVALFUNCTOR_PAE(2)
DEFINE_EVALFUNCTOR_PAE(3)
DEFINE_EVALFUNCTOR_PAE(4)
DEFINE_EVALFUNCTOR_PAE(5)
DEFINE_EVALFUNCTOR_PAE(6)


//////////////////////////////////////////////////////////////////////
//
// Count the elements in an expression.
//
//////////////////////////////////////////////////////////////////////

template<class T, class C>
inline int
for_each(const ParticleAttribIterator<T>& p, PETE_CountElems, C)
{
  return p.size();
}

template<class T, class C>
inline int
for_each(const ParticleAttribConstIterator<T>& p, PETE_CountElems, C)
{
  return p.size();
}

#define DEFINE_COUNTFUNCTOR_PAE(D)                                          \
                                                                            \
template<class T, class C>                                                  \
inline int                                                                  \
for_each(const ParticleAttribElemIterator<T,D>& p, PETE_CountElems, C)      \
{                                                                           \
  return p.getParticleAttribElem().size();                                  \
}

DEFINE_COUNTFUNCTOR_PAE(1)
DEFINE_COUNTFUNCTOR_PAE(2)
DEFINE_COUNTFUNCTOR_PAE(3)
DEFINE_COUNTFUNCTOR_PAE(4)
DEFINE_COUNTFUNCTOR_PAE(5)
DEFINE_COUNTFUNCTOR_PAE(6)


//////////////////////////////////////////////////////////////////////
//
// Increment the pointers in an expression.
//
//////////////////////////////////////////////////////////////////////

template<class T, class C>
inline int
for_each(ParticleAttribIterator<T>& p, PETE_Increment, C)
{
  ++p;
  return 0;
}

template<class T, class C>
inline int
for_each(ParticleAttribConstIterator<T>& p, PETE_Increment, C)
{
  ++p;
  return 0;
}

#define DEFINE_INCFUNCTOR_PAE(D)                                            \
                                                                            \
template<class T, class C>                                                  \
inline int                                                                  \
for_each(ParticleAttribElemIterator<T,D>& p, PETE_Increment, C)             \
{                                                                           \
  ++p;                                                                      \
  return 0;                                                                 \
}

DEFINE_INCFUNCTOR_PAE(1)
DEFINE_INCFUNCTOR_PAE(2)
DEFINE_INCFUNCTOR_PAE(3)
DEFINE_INCFUNCTOR_PAE(4)
DEFINE_INCFUNCTOR_PAE(5)
DEFINE_INCFUNCTOR_PAE(6)



#endif // PASSIGN_DEFS_H

/***************************************************************************
 * $RCSfile: PAssignDefs.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: PAssignDefs.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/

