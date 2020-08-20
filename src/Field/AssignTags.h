// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef ASSIGN_TAGS_H
#define ASSIGN_TAGS_H

// include files
#include "Index/NDIndex.h"

// Forward declarations:
template<class T, unsigned int D> class BareField;

//
// A functor that knows an id.
//
struct SameFieldID
{
  int fID;
  SameFieldID(int id) : fID(id) {}
  typedef bool PETE_Return_t;
};


struct IsCompressed
{
  typedef bool PETE_Return_t;
};


struct EvalFunctor_1
{
  int I;
  EvalFunctor_1(int i) : I(i) {}
};

struct UnitEvalFunctor_1 : public EvalFunctor_1
{
  UnitEvalFunctor_1(int i) : EvalFunctor_1(i) {}
};

struct EvalFunctor_2
{
  int I, J;
  EvalFunctor_2(int i, int j) : I(i), J(j) {}
};

struct UnitEvalFunctor_2 : public EvalFunctor_2
{
  UnitEvalFunctor_2(int i, int j) : EvalFunctor_2(i,j) {}
};

struct EvalFunctor_3
{
  int I, J, K;
  EvalFunctor_3(int i, int j, int k) : I(i), J(j), K(k) {}
};

struct UnitEvalFunctor_3 : public EvalFunctor_3
{
  UnitEvalFunctor_3(int i, int j, int k) : EvalFunctor_3(i,j,k) {}
};

//
// A tag for beginning an LField
//
struct BeginLField
{
  typedef int PETE_Return_t;
};

//
// A tag for going on to next LField
//
struct NextLField
{
  typedef int PETE_Return_t;
};


struct StepFunctor
{
  unsigned D;
  StepFunctor(unsigned d) : D(d) {}
  typedef int PETE_Return_t;
};

struct RewindFunctor
{
  unsigned D;
  RewindFunctor(unsigned d) : D(d) {}
  typedef int PETE_Return_t;
};


//
// Do the iterators have unit stride in the inner loop?
//

struct HasUnitStride
{
  typedef bool PETE_Return_t;
};

// Do we need to fill the guard cells?

template<unsigned D, class T1>
struct FillGCIfNecessaryTag
{
  FillGCIfNecessaryTag(const BareField<T1,D> &bf) : bf_m(&bf) { }
//tjw added 3/3/99:
  FillGCIfNecessaryTag() : bf_m(0) { }
//tjw added 3/3/99.
  typedef int PETE_Return_t;
  const BareField<T1,D> *bf_m;
};

template<unsigned D, class T1>
inline FillGCIfNecessaryTag<D,T1> 
FillGCIfNecessary(const BareField<T1,D> &bf)
{
  return FillGCIfNecessaryTag<D,T1>(bf);
}

//tjw added 3/3/99: 

// This weird tag is needed, because writing a no-argument FillGCIFNEcessary()
// function below didn't work for some reason, at least with some compilers
// like the pre-7.3 SGI compiler and CodeWarrior Pro4. Once the global
// function invocation syntax FillGCIfNecessary<T,D>() is supported by all our
// compilers (it's not there yet in the non-beta SGI compiler), should be able
// to eliminate this FGCINTag business. --tjw 3/3/1999
template<unsigned D, class T1>
struct FGCINTag
{
};

template<unsigned D, class T1>
inline FillGCIfNecessaryTag<D,T1> 
FillGCIfNecessary(FGCINTag<D,T1>)
{
  return FillGCIfNecessaryTag<D,T1>();
}

//tjw added 3/3/99.


#endif // ASSIGN_TAGS_H

/***************************************************************************
 * $RCSfile: AssignTags.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: AssignTags.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
