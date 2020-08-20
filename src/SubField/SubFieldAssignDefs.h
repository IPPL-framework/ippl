// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SUB_FIELD_ASSIGN_DEFS_H
#define SUB_FIELD_ASSIGN_DEFS_H

// include files
#include "Field/AssignDefs.h"

// forward references
template<class T, unsigned int D, class S> class SubFieldIter;


//////////////////////////////////////////////////////////////////////
//
// Is the domain specification object compressed?
//
//////////////////////////////////////////////////////////////////////

struct DomainCompressed {
  typedef bool PETE_Return_t;
};

template<class T, class S, class C, unsigned int D>
inline bool
for_each(SubFieldIter<T,D,S> &p, DomainCompressed, C)
{
  return p.DomainCompressed();
}

template<class T, class C, unsigned int D>
inline bool
for_each(typename BareField<T,D>::iterator&, DomainCompressed, C)
{
  return true;
}

template<class C>
inline bool
for_each(Index::cursor&, DomainCompressed, C)
{
  return false;
}

template<class T, class C>
inline bool
for_each(PETE_Scalar<T>&, DomainCompressed, C)
{
  return true;
}


//////////////////////////////////////////////////////////////////////
//
// Do the terms all use the same kind of subset object?
//
//////////////////////////////////////////////////////////////////////

struct SameSubsetType {
  typedef bool PETE_Return_t;
  int fID;
  SameSubsetType(int id) : fID(id) {}
};

template<class T, class S, class C, unsigned int D>
inline bool
for_each(SubFieldIter<T,D,S> &p, SameSubsetType s, C)
{
  return p.matchType(s.fID);
}

template<class T, class C, unsigned int D>
inline bool
for_each(typename BareField<T,D>::iterator&, SameSubsetType /*s*/, C)
{
  return false;
}

template<class C>
inline bool
for_each(Index::cursor&, SameSubsetType, C)
{
  return true;
}

template<class T, class C>
inline bool
for_each(PETE_Scalar<T>&, SameSubsetType, C)
{
  return true;
}


//////////////////////////////////////////////////////////////////////
//
// Initialize all subset objects in an expression before the loop starts
//
//////////////////////////////////////////////////////////////////////

struct SubsetInit {
  typedef int PETE_Return_t;
};

template<class T, class S, class C, unsigned int D>
inline int
for_each(SubFieldIter<T,D,S> &p, SubsetInit, C) 
{
  p.initialize();
  return 0;
}

template<class T, class C, unsigned int D>
inline int
for_each(typename BareField<T,D>::iterator &/*p*/, SubsetInit, C)
{
  return 0;
}

template<class C>
inline int
for_each(Index::cursor&, SubsetInit, C)
{
  return 0;
}

template<class T, class C>
inline int
for_each(PETE_Scalar<T>&, SubsetInit, C)
{
  return 0;
}


//////////////////////////////////////////////////////////////////////
//
// Set a subfield iterator to point to the next lfield
//
//////////////////////////////////////////////////////////////////////

struct SubsetNextLField {
  typedef int PETE_Return_t;
};

template<class T, class S, class C, unsigned int D>
inline int
for_each(SubFieldIter<T,D,S> &p, SubsetNextLField, C)
{
  p.nextLField();
  return 0;
}

template<class T, class C, unsigned int D>
inline int
for_each(typename BareField<T,D>::iterator&, SubsetNextLField, C)
{
  return 0;
}

template<class C>
inline int
for_each(Index::cursor&, SubsetNextLField, C)
{
  return 0;
}

template<class T, class C>
inline int
for_each(PETE_Scalar<T>&, SubsetNextLField, C)
{
  return 0;
}


//////////////////////////////////////////////////////////////////////
//
// Do any of the terms in an expression have an ID equal to a given one?
//
//////////////////////////////////////////////////////////////////////

template<class T, class S, class C, unsigned int D>
inline bool
for_each(SubFieldIter<T,D,S> &p, SameFieldID s, C)
{
  return (int)p.getBareField().get_Id() == s.fID;
}

//////////////////////////////////////////////////////////////////////
//
// Plugbase.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class S, class C>
inline bool
for_each(SubFieldIter<T,D,S> &p, const PlugBase<D>& f, C)
{
  return p.plugBase(f.Domain);
}

//////////////////////////////////////////////////////////////////////
//
// Check for compression.
//
//////////////////////////////////////////////////////////////////////

template<class T, class S, class C, unsigned int D>
inline bool
for_each(SubFieldIter<T,D,S> &p, IsCompressed, C)
{
  return p.IsCompressed();
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// First, no arguments.
//
//////////////////////////////////////////////////////////////////////

template<class T, class S, unsigned int D>
inline T&
for_each(SubFieldIter<T,D,S> &p, EvalFunctor_0)
{
  return *p;
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// One argument.
//
//////////////////////////////////////////////////////////////////////

template<class T, class S, unsigned int D>
inline T&
for_each(SubFieldIter<T,D,S> &p, const EvalFunctor_1& e)
{
  return p.offset(e.I);
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// Two arguments.
//
//////////////////////////////////////////////////////////////////////

template<class T, class S, unsigned int D>
inline T&
for_each(SubFieldIter<T,D,S> &p, const EvalFunctor_2& e)
{
  return p.offset(e.I,e.J);
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// Three arguments.
//
//////////////////////////////////////////////////////////////////////

template<class T, class S, unsigned int D>
inline T&
for_each(SubFieldIter<T,D,S> &p, const EvalFunctor_3& e)
{
  return p.offset(e.I,e.J,e.K);
}

//////////////////////////////////////////////////////////////////////
//
// Step in some dimension.
//
//////////////////////////////////////////////////////////////////////

template<class T, class S, class C, unsigned int D>
inline int
for_each(SubFieldIter<T,D,S> &p, StepFunctor s, C)
{
  p.step(s.D);
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Rewind in some dimension.
//
//////////////////////////////////////////////////////////////////////

template<class T, class S, class C, unsigned int D>
inline int
for_each(SubFieldIter<T,D,S> &p, RewindFunctor s, C)
{
  p.rewind(s.D);
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Does an iterator reference something with unit stride?
// Don't worry about it for now.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned int D, class S, class C>
inline bool
for_each(SubFieldIter<T,D,S> &/*p*/, HasUnitStride, C)
{
  return false;
}

//////////////////////////////////////////////////////////////////////
//
// Ask each term to fill guard cells and compress itself
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned int D, class S, class C, class T1>
inline int
for_each(SubFieldIter<T,D,S> &p, const FillGCIfNecessaryTag<D,T1> &/*f*/, C)
{
  //tjw3/3/99  p.FillGCIfNecessary(f.I, f.I);
  p.FillGCIfNecessary();
  return 0;
}

#endif // SUB_FIELD_ASSIGN_DEFS_H

/***************************************************************************
 * $RCSfile: SubFieldAssignDefs.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubFieldAssignDefs.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
