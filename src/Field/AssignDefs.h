// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef ASSIGN_DEFS_H
#define ASSIGN_DEFS_H

// include files
#include "Field/AssignTags.h"

// forward declarations
template<class T, unsigned int D> class BareField;
template<class T, unsigned int D> class BareFieldIterator;
template<class T, unsigned int D> class IndexedBareFieldIterator;

//////////////////////////////////////////////////////////////////////
//
// Do any of the terms in an expression have an ID equal to a given one?
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D>
inline bool
for_each(const BareFieldIterator<T,D>& p, SameFieldID s, C)
{
  return p.GetBareField().get_Id() == (unsigned int) s.fID;
}

template<class T, class C, unsigned int D>
inline bool
for_each(const IndexedBareFieldIterator<T,D>& p, SameFieldID s, C)
{
  return p.GetBareField().get_Id() == (unsigned int) s.fID;
}

//
// If there is an index in the expr, it can't have the same ID.
//
template<class C>
inline bool
for_each(const Index::cursor&, SameFieldID, C)
{
  return false;
}

//
// If there is a scalar in the expr, it can't have the same ID.
//
template<class T, class C>
inline bool
for_each(const PETE_Scalar<T>&, SameFieldID, C)
{
  return false;
}

//////////////////////////////////////////////////////////////////////
//
// Plugbase.
//
//////////////////////////////////////////////////////////////////////

template<unsigned D>
struct PlugBase
{
  NDIndex<D> Domain;
  PlugBase( const NDIndex<D>& domain ) : Domain(domain) {}
  typedef bool PETE_Return_t;
};

//
// Plug into an IndexedBareField.
//
template<class T, unsigned int D1, unsigned D2, class C>
inline bool
for_each(IndexedBareFieldIterator<T,D1>& p, const PlugBase<D2>& f, C)
{
  return p.plugBase(f.Domain);
}

//
// plugbase
//
template<class C, unsigned D>
inline bool
for_each(Index::cursor& p, const PlugBase<D>& f, C)
{
  //cout << "Plugging " << f.Domain << endl;
  //cout << "  p.id() = " << p.id() << endl;
  for (unsigned d=0; d<D; ++d) 
    {
      //cout << "  d=" << d << ", id=" << f.Domain[d].id() << endl;
      if ( p.id() == f.Domain[d].id() )
	{
	  //cout << "  matched." << endl;
	  return p.plugBase( f.Domain[d], d );
	}
    }
  return false;
}

//
// just return true for scalar
//
template<class T, class C, unsigned D>
inline bool
for_each(const PETE_Scalar<T>&, const PlugBase<D>&, C)
{
  return true;
}

//////////////////////////////////////////////////////////////////////
//
// Ask each term if it is compressed.
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D>
inline bool
for_each(const IndexedBareFieldIterator<T,D>& p, IsCompressed, C)
{
  return p.IsCompressed();
}

template<class T, class C, unsigned int D>
inline bool
for_each(const BareFieldIterator<T,D>& p, IsCompressed, C)
{
  return p.IsCompressed();
}

//
// Indexes are never compressed.
//
template<class C>
inline bool
for_each(const Index::cursor&, IsCompressed, C)
{
  return false;
}

//
// Scalars are always compressed.
//
template<class T, class C>
inline bool
for_each(const PETE_Scalar<T>&, IsCompressed, C)
{
  return true;
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// First, no arguments.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned int D>
inline T&
for_each(const BrickIterator<T,D>& p, EvalFunctor_0)
{
  return *p;
}

//
// Evaluate an Index.
//
inline int
for_each(const Index::cursor& p, EvalFunctor_0)
{
  return *p;
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// One argument.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned int D>
inline T&
for_each(const BrickIterator<T,D>& p, const EvalFunctor_1& e)
{
  return p.offset(e.I);
}

template<class T, unsigned int D>
inline T&
for_each(const BrickIterator<T,D>& p, const UnitEvalFunctor_1& e)
{
  return p.unit_offset(e.I);
}

//
// Evaluate an Index.
//
inline int
for_each(const Index::cursor& p, const EvalFunctor_1& e)
{
  return p.offset(e.I);
}

//
// Evaluate a scalar.
//
template<class T>
inline T
for_each(const PETE_Scalar<T>& p, const EvalFunctor_1&)
{
  return T(p);
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// Two arguments.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned int D>
inline T&
for_each(const BrickIterator<T,D>& p, const EvalFunctor_2& e)
{
  return p.offset(e.I, e.J);
}

template<class T, unsigned int D>
inline T&
for_each(const BrickIterator<T,D>& p, const UnitEvalFunctor_2& e)
{
  return p.unit_offset(e.I, e.J);
}

//
// Evaluate an Index.
//
inline int
for_each(const Index::cursor& p, const EvalFunctor_2& e)
{
  return p.offset(e.I, e.J);
}

//
// Evaluate a scalar.
//
template<class T>
inline T
for_each(const PETE_Scalar<T>& p, const EvalFunctor_2&)
{
  return T(p);
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// Three arguments.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned int D>
inline T&
for_each(const BrickIterator<T,D>& p, const EvalFunctor_3& e)
{
  return p.offset(e.I, e.J, e.K);
}

template<class T, unsigned int D>
inline T&
for_each(const BrickIterator<T,D>& p, const UnitEvalFunctor_3& e)
{
  return p.offset(e.I, e.J, e.K);
}

//
// Evaluate an Index.
//
inline int
for_each(const Index::cursor& p, const EvalFunctor_3& e)
{
  return p.offset(e.I, e.J, e.K);
}

//
// Evaluate a scalar.
//
template<class T>
inline T
for_each(const PETE_Scalar<T>& p, const EvalFunctor_3&)
{
  return T(p);
}

//////////////////////////////////////////////////////////////////////
//
// Get started in a given LField 
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D>
inline int
for_each(BareFieldIterator<T,D>& p, BeginLField, C)
{
  p.beginLField();
  return 0;
}

//
// ignore this signal
//
template<class T, class C>
inline int
for_each(PETE_Scalar<T>&, BeginLField, C)
{
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Go on to the next LField.
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D>
inline int
for_each(BareFieldIterator<T,D>& p, NextLField, C)
{
  p.nextLField();
  return 0;
}

//
// If there is a scalar in the expr, it ignores this signal.
//
template<class T, class C>
inline int
for_each(PETE_Scalar<T>&, NextLField, C)
{
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Step in some dimension.
//
//////////////////////////////////////////////////////////////////////

template<class T,  class C, unsigned int D>
inline int
for_each(IndexedBareFieldIterator<T,D>& p, StepFunctor s, C)
{
  p.step(s.D);
  return 0;
}

template<class T,  class C, unsigned int D>
inline int
for_each(BareFieldIterator<T,D>& p, StepFunctor s, C)
{
  p.step(s.D);
  return 0;
}

template<class T,  class C, unsigned int D>
inline int
for_each(BrickIterator<T,D>& p, StepFunctor s, C)
{
  p.step(s.D);
  return 0;
}

//
// Step along an Index.
//
template<class C>
inline int
for_each(Index::cursor& p, StepFunctor s, C)
{
  p.step(s.D);
  return 0;
}

//
// scalar ignores step functor
//
template<class T, class C>
inline int
for_each(PETE_Scalar<T>&, StepFunctor, C)
{
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Rewind in some dimension.
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D>
inline int
for_each(IndexedBareFieldIterator<T,D>& p, RewindFunctor s, C)
{
  p.rewind(s.D);
  return 0;
}


template<class T, class C, unsigned int D>
inline int
for_each(BareFieldIterator<T,D>& p, RewindFunctor s, C)
{
  p.rewind(s.D);
  return 0;
}


template<class T, class C, unsigned int D>
inline int
for_each(BrickIterator<T,D>& p, RewindFunctor s, C)
{
  p.rewind(s.D);
  return 0;
}

//
// Rewind an Index.
//
template<class C>
inline int
for_each(Index::cursor& p, RewindFunctor s, C)
{
  p.rewind(s.D);
  return 0;
}

//
// scalar ignores rewind functor
//
template<class T, class C>
inline int
for_each(PETE_Scalar<T>&, RewindFunctor, C)
{
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Count the number of elements in an expression.
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D>
inline int
for_each(BrickIterator<T,D>& p, PETE_CountElems, C)
{
  int size = p.size(0);
  for (unsigned int i=1; i<D; ++i)
    size *= p.size(i);
  return size;
}

template<class T, class C, unsigned int D>
inline int
for_each(BareFieldIterator<T,D>& p, PETE_CountElems, C)
{
  BareField<T,D>& f = p.GetBareField();
  int n = 0;
  for (typename BareField<T,D>::iterator_if lf=f.begin_if(); lf!=f.end_if(); ++lf)
    n += (*lf).second->getOwned().size();
  return n;
}

template<class T, class C, unsigned int D>
inline int
for_each(IndexedBareFieldIterator<T,D>& p, PETE_CountElems, C)
{
  BareField<T,D>& f = p.GetBareField();
  int n = 0;
  for (typename BareField<T,D>::iterator_if lf=f.begin_if(); lf!=f.end_if(); ++lf) {
    const NDIndex<D>& domain = (*lf).second->getOwned();
    if ( p.getDomain().touches(domain) )
      n += p.getDomain().intersect(domain).size();
  }
  return n;
}

//////////////////////////////////////////////////////////////////////
//
// Increment the pointers in an expression.
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D>
inline int
for_each(IndexedBareFieldIterator<T,D>& p, PETE_Increment, C)
{
  ++p;
  return 0;
}


template<class T, class C, unsigned int D>
inline int
for_each(BareFieldIterator<T,D>& p, PETE_Increment, C)
{
  ++p;
  return 0;
}


template<class T, class C, unsigned int D>
inline int
for_each(BrickIterator<T,D>& p, PETE_Increment , C)
{
  ++p;
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Find out if all the BrickIterators in an expression 
// have unit stride in the first dimension.
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D>
inline bool
for_each(const IndexedBareFieldIterator<T,D>& iter, HasUnitStride, C)
{
  return iter.Stride(0) == 1;
}

template<class T, class C, unsigned int D>
inline bool
for_each(const BareFieldIterator<T,D>& iter, HasUnitStride, C)
{
  return iter.Stride(0) == 1;
}

template<class T, class C, unsigned int D>
inline bool
for_each(const BrickIterator<T,D>& iter, HasUnitStride, C)
{
  return iter.Stride(0) == 1;
}

template<class C>
inline bool
for_each(const Index::cursor&, HasUnitStride, C)
{
  return true;
}

template<class T, class C>
inline bool
for_each(const PETE_Scalar<T>&, HasUnitStride, C)
{
  return true;
}

//////////////////////////////////////////////////////////////////////
//
// Ask each term to fill guard cells and compress itself
//
//////////////////////////////////////////////////////////////////////

template<class T, class C, unsigned int D, unsigned int D1, class T1>
inline int
for_each(const IndexedBareFieldIterator<T,D>& p, 
  const FillGCIfNecessaryTag<D1,T1> &f, C)
{
  p.FillGCIfNecessary(*(f.bf_m));
  return 0;
}

template<class T, class C, unsigned int D, unsigned int D1, class T1>
inline int
for_each(const BareFieldIterator<T,D>& /*p*/, 
  const FillGCIfNecessaryTag<D1,T1> &, C)
{
  return 0;
}

template<class T, class C, unsigned int D, class T1>
inline int
for_each(const PETE_Scalar<T>&, const FillGCIfNecessaryTag<D,T1> &, C)
{
  return 0;
}

template<class C, unsigned int D, class T1>
inline int
for_each(const Index::cursor&, const FillGCIfNecessaryTag<D,T1> &, C)
{
  return 0;
}

#endif // ASSIGN_DEFS_H

/***************************************************************************
 * $RCSfile: AssignDefs.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: AssignDefs.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
