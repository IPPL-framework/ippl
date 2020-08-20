// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 * This program was prepared by PSI. 
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "Field/ReductionLoc.h"
#include "Index/NDIndex.h"
#include "Field/AssignDefs.h"
#include "Message/Message.h"
#include "Message/GlobalComm.h"


//////////////////////////////////////////////////////////////////////
//
// Reduce and find the location.
//
//////////////////////////////////////////////////////////////////////

template<class T, class LOC>
struct ReduceLoc
{
  T Val;
  LOC Loc;
  ReduceLoc() {}
  ReduceLoc(const T& t, const LOC& l) : Val(t), Loc(l) {}
  Message& putMessage(Message& mess)
    {
      ::putMessage(mess,Val);      
      ::putMessage(mess,Loc);      
      return mess;
    }
  Message& getMessage(Message& mess)
    {
      ::getMessage(mess,Val);      
      ::getMessage(mess,Loc);
      return mess;
    }
  bool operator<(const ReduceLoc<T,LOC>& rhs) const
    {
      return Val < rhs.Val;
    }
};

template <class T1, class T2, class LOC>
inline
ReduceLoc<typename PETEBinaryReturn<T1,T2,FnMin>::type, LOC>
Min(const ReduceLoc<T1,LOC>& lhs, const ReduceLoc<T2,LOC>& rhs)
{
  typedef typename PETEBinaryReturn<T1,T2,FnMin>::type T0;
  if (lhs < rhs)
    return ReduceLoc<T0,LOC>(lhs.Val,lhs.Loc);
  else
    return ReduceLoc<T0,LOC>(rhs.Val,rhs.Loc);
}

template <class T1, class T2, class LOC>
inline
ReduceLoc<typename PETEBinaryReturn<T1,T2,FnMax>::type, LOC>
Max(const ReduceLoc<T1,LOC>& lhs, const ReduceLoc<T2,LOC>& rhs)
{
  typedef typename PETEBinaryReturn<T1,T2,FnMax>::type T0;
  if (lhs < rhs)
    return ReduceLoc<T0,LOC>(rhs.Val,rhs.Loc);
  else
    return ReduceLoc<T0,LOC>(lhs.Val,lhs.Loc);
}




template<class T, class CompOp, class AccOp, unsigned D>
typename T::PETE_Return_t
Reduction(const PETE_Expr<T>& const_expr,
	  CompOp comp_op,
	  AccOp  acc_op,
	  NDIndex<D>& loc)
{
  
  // Extract the expression. 
  typename T::PETE_Expr_t expr ( const_expr.PETE_unwrap().MakeExpression() );

  // The type and value that we will return. 
  typedef typename T::PETE_Return_t R;
  ReduceLoc< R, NDIndex<D> > global_rloc;

  // Get the number of elements we will be looping over. 
  // The combiner makes sure each term has 
  // the same number of elements. 
  int n = for_each(expr,PETE_CountElems(),AssertEquals());
  if (n > 0) {
    // Get the first value. 
    R ret = for_each(expr,EvalFunctor_0());

    // Make an operator that will store the location in loc.
    FindLocation<D> find_loc(loc);

    // Get the first location. 
    for_each(expr,find_loc,PETE_NullCombiner());

    // Loop over all the local elements. 
    for (int i=1; i<n; ++i) {
      // Increment the pointers. 
      for_each(expr,PETE_Increment(),PETE_NullCombiner());
      // check and see if this is the new min. 
      R val = for_each(expr,EvalFunctor_0());
      if ( PETE_apply(comp_op,val,ret) ) {
	// Record the value.
	ret = val;
	// Record the current location.
	for_each(expr,find_loc,PETE_NullCombiner());
      }
    }

    // Do the cross processor reduction.
    ReduceLoc< R, NDIndex<D> > rloc( ret , loc );
    reduce_masked(rloc, global_rloc, acc_op, true);
  } else {
    // do reduction, but indicate we have no local contribution
    reduce_masked(global_rloc, global_rloc, acc_op, false);
  }

  // Return the calculated values
  loc = global_rloc.Loc;
  return global_rloc.Val;
}

/***************************************************************************
 * $RCSfile: ReductionLoc.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: ReductionLoc.cpp,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
