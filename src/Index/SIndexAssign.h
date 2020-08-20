// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SINDEX_ASSIGN_H
#define SINDEX_ASSIGN_H

/***************************************************************************
 Special versions of assign which take SIndex objects on the left-hand side
 ***************************************************************************/

// include files
#include "PETE/IpplExpressions.h"

// forward declarations
template<unsigned Dim> class SIndex;

// helper class used to determine if things are expressions or not
template<bool IsExpr>  class SIExprTag { };


//////////////////////////////////////////////////////////////////////

// SIndex = Expression 
template<unsigned Dim, class RHS, class OP, bool IsExpr>
void
assign(SIndex<Dim>&, RHS, OP, const NDIndex<Dim> &, SIExprTag<IsExpr>);


//////////////////////////////////////////////////////////////////////

// a macro for creating different assignment operators for SIndex objects
#define SI_ASSIGNMENT_FUNCTIONS(FUNC,OP)			\
								\
template<unsigned Dim, class RHS>				\
inline void							\
FUNC(SIndex<Dim>& lhs, const PETE_Expr<RHS>& rhs)		\
{								\
  assign(lhs, rhs.PETE_unwrap().MakeExpression(), OP(),		\
         lhs.getFieldLayout().getDomain(), SIExprTag<false>());	\
}								\
								\
template<unsigned Dim, class RHS>				\
inline void							\
FUNC(SIndex<Dim>& lhs, const PETE_Expr<RHS>& rhs,		\
     const NDIndex<Dim>& domain)				\
{								\
  assign(lhs, rhs.PETE_unwrap().MakeExpression(), OP(),		\
         domain, SIExprTag<true>());				\
}

#define SI_ASSIGNMENT_OPERATORS(FUNC,OP)			\
								\
template<unsigned Dim, class RHS>				\
inline void							\
FUNC(SIndex<Dim>& lhs, const PETE_Expr<RHS>& rhs)		\
{								\
  assign(lhs, rhs.PETE_unwrap().MakeExpression(), OP(),		\
         lhs.getFieldLayout().getDomain(), SIExprTag<false>());	\
}



// use the macro to create a number of different assign functions which
// will in turn call the general  SIndex = expression  version of assign
SI_ASSIGNMENT_FUNCTIONS(assign,OpAssign)
SI_ASSIGNMENT_OPERATORS(operator<<,OpAssign)
SI_ASSIGNMENT_OPERATORS(operator|=,OpBitwiseOrAssign)
SI_ASSIGNMENT_OPERATORS(operator&=,OpBitwiseAndAssign)

#include "Index/SIndexAssign.hpp"

#endif // SINDEX_ASSIGN_H

/***************************************************************************
 * $RCSfile: SIndexAssign.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: SIndexAssign.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
