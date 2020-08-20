// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef ASSIGN_H
#define ASSIGN_H

// include files
#include "PETE/IpplExpressions.h"
#include <complex>

// forward declarations
template<class T, unsigned Dim> class BareField;
template<class T, unsigned Dim> class LField;
template<class T, unsigned Dim, unsigned Brackets> class IndexedBareField;

// a debugging output message macro for assign functions.  This is
// only enabled if you specificall add -DDEBUG_ASSIGN somewhere.
#ifdef DEBUG_ASSIGN
#define ASSIGNMSG(x) x
#else
#define ASSIGNMSG(x)
#endif


//////////////////////////////////////////////////////////////////////
//
// Currently, some of the assignment algorithms depend
// on knowing that there is just one Field on the rhs.
// We detect these here by building a simple tag class ExprTag
// that has a bool template argument.  The arg is true
// if the the assign should deal with general expressions,
// and false if it can use the special purpose single field version.
//
// This wouldn't be needed if we had partial specialization,
// but this will disambiguate expressions even without it.
//
//////////////////////////////////////////////////////////////////////

template <bool IsExpr>
class ExprTag
{
};

//////////////////////////////////////////////////////////////////////

// BareField = Expression 
template<class T, unsigned Dim, class RHS, class OP>
void
assign(const BareField<T,Dim>& a, RHS b, OP op, ExprTag<true>);

// IndexedBareField = Expression 
template<class T1, unsigned Dim, class RHS, class OP>
void
assign(const IndexedBareField<T1,Dim,Dim> &a, RHS b, OP op, ExprTag<true>,
       bool fillGC); // tjw added fillGC 12/16/1997 new BC hack

// IndexedBareField = Expression 
template<class T1, unsigned Dim, class RHS, class OP>
inline void
assign(const IndexedBareField<T1,Dim,Dim> &a, RHS b, OP op, ExprTag<true> et) {
  assign(a, b, op, et, true);
}

// Component = Expression
template<class A, class RHS, class OP, class Tag, class TP>
void
assign(PETE_TUTree<OpParens<TP>,A> lhs, RHS rhs, OP op, Tag,
       bool fillGC=true); // tjw added fillGC 12/16/1997 new BC hack

// BareField = BareField, with communication.
template<class T1, unsigned D1, class RHS, class Op>
void
assign(const BareField<T1,D1>& lhs, RHS rhs, Op op, ExprTag<false>);

// IndexedBareField = IndexedBareField, different dimensions.
template<class T1, unsigned D1, class RHS, class Op>
void
assign(IndexedBareField<T1,D1,D1> lhs, RHS rhs, Op, ExprTag<false>);

//////////////////////////////////////////////////////////////////////

// Expression = constant; 
template<class T, unsigned D, class OP>
inline void
assign(BareField<T,D>& a,const T& b, OP op, ExprTag<true>)
{
  assign(a, PETE_Scalar<T>(b), op, ExprTag<true>());
}

template<class T, unsigned D, class OP>
inline void
assign(IndexedBareField<T,D,D> a, const T& b, OP op, ExprTag<true>)
{
  assign(a, PETE_Scalar<T>(b), op, ExprTag<true>());
}

//////////////////////////////////////////////////////////////////////

template<class T> struct IsExprTrait { enum { IsExpr = T::IsExpr } ; };
template<> struct IsExprTrait<double>   { enum { IsExpr = 1 }; };
template<> struct IsExprTrait<std::complex<double>> { enum { IsExpr = 1 }; };
template<> struct IsExprTrait<float>    { enum { IsExpr = 1 }; };
template<> struct IsExprTrait<short>    { enum { IsExpr = 1 }; };
template<> struct IsExprTrait<int>      { enum { IsExpr = 1 }; };
template<> struct IsExprTrait<long>     { enum { IsExpr = 1 }; };
template<> struct IsExprTrait<Index>    { enum { IsExpr = 1 }; };

#define ASSIGNMENT_OPERATORS(FUNC,OP)				\
								\
template<class LHS, class RHS>					\
inline void							\
FUNC(const PETE_Expr<LHS>& lhs, const PETE_Expr<RHS>& rhs)	\
{								\
  assign(lhs.PETE_unwrap(), rhs.PETE_unwrap().MakeExpression(),	\
	 OP(),ExprTag< IsExprTrait<RHS>::IsExpr >());		\
}								\
								\
template<class T, unsigned D>					\
inline void							\
FUNC(const IndexedBareField<T,D,D>& lhs, const T& rhs)		\
{								\
  assign(lhs,PETE_Scalar<T>(rhs), OP(),ExprTag<true>(),true);	\
}								\
								\
template<class T, unsigned D>					\
inline void							\
FUNC(const BareField<T,D>& lhs, const T& rhs)			\
{								\
  assign(lhs,PETE_Scalar<T>(rhs),OP(),ExprTag<true>());		\
}								\
								\
template<class A, class TP>					\
inline void							\
FUNC(const PETE_TUTree<OpParens<TP>,A>& lhs, const bool& rhs)	\
{								\
  assign(lhs,PETE_Scalar<bool>(rhs),OP(),ExprTag<true>());	\
}								\
template<class A, class TP>					\
inline void							\
FUNC(const PETE_TUTree<OpParens<TP>,A>& lhs, const char& rhs)	\
{								\
  assign(lhs,PETE_Scalar<char>(rhs),OP(),ExprTag<true>());	\
}								\
template<class A, class TP>					\
inline void							\
FUNC(const PETE_TUTree<OpParens<TP>,A>& lhs, const int& rhs)	\
{								\
  assign(lhs,PETE_Scalar<int>(rhs),OP(),ExprTag<true>());	\
}								\
template<class A,class TP>					\
inline void							\
FUNC(const PETE_TUTree<OpParens<TP>,A>& lhs, const float& rhs)	\
{								\
  assign(lhs,PETE_Scalar<float>(rhs),OP(),ExprTag<true>());	\
}								\
template<class A, class TP>					\
inline void							\
FUNC(const PETE_TUTree<OpParens<TP>,A>& lhs, const double& rhs)	\
{								\
  assign(lhs,PETE_Scalar<double>(rhs),OP(),ExprTag<true>());	\
}								\
template<class A, class TP>					\
inline void							\
FUNC(const PETE_TUTree<OpParens<TP>,A>& lhs, const std::complex<double>& rhs)\
{								\
  assign(lhs,PETE_Scalar<std::complex<double>>(rhs),OP(),ExprTag<true>());	\
}

ASSIGNMENT_OPERATORS(assign,OpAssign)
ASSIGNMENT_OPERATORS(operator<<,OpAssign)
ASSIGNMENT_OPERATORS(operator+=,OpAddAssign)				      
ASSIGNMENT_OPERATORS(operator-=,OpSubtractAssign)			      
ASSIGNMENT_OPERATORS(operator*=,OpMultipplyAssign)
ASSIGNMENT_OPERATORS(operator/=,OpDivideAssign)
ASSIGNMENT_OPERATORS(mineq,OpMinAssign)
ASSIGNMENT_OPERATORS(maxeq,OpMaxAssign)


// Determine whether to compress or uncompress the
// left hand side given information about the expression.
// This prototype is defined here because the SubField assignment
// files need this functionality as well.
template<class T, unsigned Dim, class A, class Op>
bool TryCompressLHS(LField<T,Dim>&, A&, Op, const NDIndex<Dim>&);


// Include the .cpp function definitions.
#include "Field/Assign.hpp"
#include "Field/AssignGeneralBF.hpp"
#include "Field/AssignGeneralIBF.hpp"

#endif // ASSIGN_H

/***************************************************************************
 * $RCSfile: Assign.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: Assign.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
