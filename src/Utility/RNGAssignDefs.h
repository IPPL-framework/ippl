// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef RNG_ASSIGN_DEFS_H
#define RNG_ASSIGN_DEFS_H

// include files
#include "PETE/IpplExpressions.h"
#include "Field/AssignTags.h"


// interactions for basic tags

//////////////////////////////////////////////////////////////////////
//
// Evaluation functor.
//
//////////////////////////////////////////////////////////////////////

template<class GT>
inline typename SequenceGen<GT>::Return_t
for_each(const SequenceGen<GT>& p, EvalFunctor_0)
{
  return p();
}

//////////////////////////////////////////////////////////////////////
//
// Count the elements in an expression.
//
//////////////////////////////////////////////////////////////////////

template<class GT, class C>
inline int
for_each(const SequenceGen<GT>&, PETE_CountElems, C)
{
  // just return a code value
  return -1;
}

//////////////////////////////////////////////////////////////////////
//
// Increment the pointers in an expression.
//
//////////////////////////////////////////////////////////////////////

template<class GT, class C>
inline int
for_each(const SequenceGen<GT>&, PETE_Increment, C)
{
  // just return (increment happens automatically after evaluation!)
  return 0;
}


// Field-specific tag interactions


//////////////////////////////////////////////////////////////////////
//
// Go to beginning of LField
//
//////////////////////////////////////////////////////////////////////

template<class GT, class C>
inline int
for_each(const SequenceGen<GT>&, BeginLField, C)
{
  // just return
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Go on to the next LField
//
//////////////////////////////////////////////////////////////////////

template<class GT, class C>
inline int
for_each(const SequenceGen<GT>&, NextLField, C)
{
  // just return
  return 0;
}

//
// If there is an RNG in the expr, it can't have the same ID.
//
template<class GT, class C>
inline bool
for_each(const SequenceGen<GT>&, SameFieldID, C)
{
  return false;
}

//
// for plugbase, just return true for RNG
//
template <unsigned D> struct PlugBase;

template<class GT, class C, unsigned D>
inline bool
for_each(const SequenceGen<GT>&, const PlugBase<D>&, C)
{
  return true;
}

//
// RNG objects are not compressed!
//
template<class GT, class C>
inline bool
for_each(const SequenceGen<GT>&, IsCompressed, C)
{
  return false;
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors with arguments
//
//////////////////////////////////////////////////////////////////////

template<class GT>
inline typename SequenceGen<GT>::Return_t
for_each(const SequenceGen<GT>& p, const EvalFunctor_1&)
{
  return p();
}

template<class GT>
inline typename SequenceGen<GT>::Return_t
for_each(const SequenceGen<GT>& p, const EvalFunctor_2&)
{
  return p();
}

template<class GT>
inline typename SequenceGen<GT>::Return_t
for_each(const SequenceGen<GT>& p, const EvalFunctor_3&)
{
  return p();
}

//
// Does it have unit stride?
// Not really a sensible question, but it is safe to say it does.
//

template<class GT, class C>
inline bool
for_each(const SequenceGen<GT>& /*p*/, HasUnitStride, C)
{
  return true;
}

//
// RNG ignores step functor
//
template<class GT, class C>
inline int
for_each(const SequenceGen<GT>&, StepFunctor, C)
{
  return 0;
}

//
// RNG ignores rewind functor
//
template<class GT, class C>
inline int
for_each(const SequenceGen<GT>&, RewindFunctor, C)
{
  return 0;
}

// RNG ignores filling guard cells question
template<class GT, class C, unsigned int D, class T1>
inline int
for_each(const SequenceGen<GT>&,
         const FillGCIfNecessaryTag<D,T1> &, C)
{
  return 0;
}


#endif // RNG_ASSIGN_DEFS_H

/***************************************************************************
 * $RCSfile: RNGAssignDefs.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: RNGAssignDefs.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/

