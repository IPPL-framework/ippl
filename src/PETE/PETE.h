// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/


///////////////////////////////////////////////////////////////////////////
//
// FILE NAME
//    PETE.h
//
// CREATED
//    July 11, 1997
//
// DESCRIPTION
//    PETE: Portable Expression Template Engine.
//
//    This header file defines the objects and operators necessary to use
//    expression templates for a fairly large class of problems.
//
//    It defines template functions for the standard unary, binary and 
//    trinary operators.  Each of them is defined for only certain types to 
//    prevent clashes with global operator+.
//
///////////////////////////////////////////////////////////////////////////

#ifndef PETE_H
#define PETE_H


// include files
#include "PETE/TypeComputations.h"
#include "Utility/PAssert.h"

#include <cstdlib>

//=========================================================================
//
// PETE BASE CLASS DEFINITIONS
//
//=========================================================================

///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETE_Expr<W>
//
// DESCRIPTION
//    The base class for all objects that will participate in expressions
//    All this wrapper class needs to do is let you convert back to 
//    the type W.
//
//    The intended use for PETE_Expr is as a base class for
//    a class you want to be able to enter into PETE expressions.
//    You construct a PETE aware class by saying:
//
//    class W : public PETE_Expr<W>
//    {
//       ...
//    };
//
//    Then a function that will recognize PETE expressions can do:
//
//    template<class W>
//    void foo(const PETE_Expr<W>& wrapped_expr)
//    {
//      const W& expr = wrapped_expr.PETE_unwrap();
//      ... do stuff with expr ...
//    }
// 
///////////////////////////////////////////////////////////////////////////

template<class WrappedExpr>
class PETE_Expr 
{
public:
  typedef WrappedExpr Wrapped;

  WrappedExpr& PETE_unwrap() 
  { 
    return static_cast<WrappedExpr&>(*this);
  }
  const WrappedExpr& PETE_unwrap() const 
  { 
    return static_cast<const WrappedExpr&>(*this);
  }

};


///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETE_Scalar<T>
//
// DESCRIPTION
//    A wrapper around a scalar to be used in PETE expressions.
//
//    This is a simple illustration of how one makes a class PETE-aware.
//    First, notice that PETE_Scalar<T> inherits publically from 
//    PETE_Expr< PETE_Scalar<T> >. In addition, a PETE-aware class must
//    provide two typedefs:
//
//      PETE_Expr_t: this is the actual class that will be used to
//        compute an expression. Many simple classes act as both
//        containers and cursor-like classes, so, often, this
//        will simply be the class itself. However, classes with
//        an STL-like iterator should use the iterator as PETE_Expr_t.
//      PETE_Return_t: this is the type returned by evaluating the
//        expression. Typically, this is the element-type of a container.
//        For PETE_Scalar, this is the type of scalar.
//
//   Classes need to supply a member function of the form:
//
//      PETE_Expr_t MakeExpression() const { ... }
//
//   This function should construct and return an appropriate object of
//   type PETE_Expr_t.
//
//   Finally, PETE really does its work by applying Functors recursively
//   through the tree that represents an expression. At the leaves of the
//   tree, these Functors encounter user classes like PETE_Scalar. There
//   are three Functors that all user classes should support:
//
//      EvalFunctor_0: returns a value for the leaf part of the expression.
//        This Functor has no arguments so most containers will need to
//        supply a cursor-like class that returns the current container value.
//        The cursor is bumped using another Functor, PETE_Increment. 
//      PETE_Increment: moves the cursor to the next element.
//      PETE_CountElems: returns the number of elements in the container.
//        Used for testing conformance, for example.
//
//   The actual code for evaluating these Functors for a particular user class
//   is contained in a global function named 'for_each'. Users classes need
//   to define a different 'for_each' for each Functor.
// 
///////////////////////////////////////////////////////////////////////////

template<class T>
class PETE_Scalar : public PETE_Expr< PETE_Scalar<T> >
{
public:

  // Required PETE typedefs and expression creation function.
  
  typedef PETE_Scalar<T> PETE_Expr_t;
  typedef T PETE_Return_t;

  PETE_Expr_t MakeExpression() const { return *this; }

  // Default constructor takes no action.

  PETE_Scalar() { }

  // Constructor from a single value.

  PETE_Scalar(const T& t) : scalar(t) { }

  // Conversion to a type T.
  operator T() const { return scalar; }

private:

  T scalar;

};

//=========================================================================
//
// BASIC FUNCTORS AND for_each FUNCTIONS FOR PETE_Scalar
//
//=========================================================================

// Evaluation functor, no offsets.

struct EvalFunctor_0
{
};

// Increment functor

struct PETE_Increment
{
  typedef int PETE_Return_t;
};

// Element count functor.

struct PETE_CountElems
{
  typedef int PETE_Return_t;
};

// Behavior of basic Functors for PETE_Scalar:

// Evaluate a scalar.

template<class T>
inline T
for_each(const PETE_Scalar<T>& p, EvalFunctor_0)
{
  return T(p);
}

// Cursor increment, ignored by scalar.

template<class T, class C>
inline int
for_each(PETE_Scalar<T>&, PETE_Increment, C)
{
  return 0;
}

// Count elements, scalar returns code value -1.

template<class T, class C>
inline int
for_each(const PETE_Scalar<T>&, PETE_CountElems, C)
{
  return -1;
}


//=========================================================================
//
// PETE TREE CLASSES
//
//=========================================================================

///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETE_TUTree<Value_t, Child_t>
//
// DESCRIPTION
//    A tree node for representing unary expressions. The node holds a
//    Child (of type Child_t), which is the type of the expression sub tree,
//    a Value (of type Value_t), which is typically the operation applied to
//    the sub tree.
// 
///////////////////////////////////////////////////////////////////////////

template<class Value_t, class Child_t>
struct PETE_TUTree : public PETE_Expr< PETE_TUTree<Value_t, Child_t> >
{
  enum { IsExpr = 1 };
  typedef PETE_TUTree<Value_t, Child_t> PETE_Expr_t;
  typedef typename PETEUnaryReturn<typename Child_t::PETE_Return_t, 
    Value_t>::type PETE_Return_t;
  const PETE_Expr_t& MakeExpression() const { return *this; };

  // The value and child expression stored in this node of the tree.
  
  Value_t Value;
  Child_t Child;

  // Default constructor: takes no action.

  PETE_TUTree() { }

  // Constructor using both a value and the child.

  PETE_TUTree(const Value_t& v, const Child_t& c)
  : Value(v), Child(c) { }

  // Constructor using just the child.
  
  PETE_TUTree(const Child_t& c)
  : Child(c) { }
};


///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETE_TBTree<Value_t, Left_t, Right_t>
//
// DESCRIPTION
//    A tree node for representing binary expressions. The node holds a
//    Left child (of type Left_t), which is the type of the LHS expression 
//    sub tree, a Right child (of type Right_t), which is the type of the RHS 
//    expression sub tree, and a Value (of type Value_t), which is typically 
//    the operation applied to the two sub trees.
// 
///////////////////////////////////////////////////////////////////////////

template<class Value_t, class Left_t, class Right_t>
struct PETE_TBTree :
  public PETE_Expr< PETE_TBTree<Value_t, Left_t, Right_t> >
{
  enum { IsExpr = 1 };
  typedef PETE_TBTree<Value_t,Left_t,Right_t> PETE_Expr_t;
  typedef typename PETEBinaryReturn<typename Left_t::PETE_Return_t,
    typename Right_t::PETE_Return_t, Value_t>::type  PETE_Return_t;
  const PETE_Expr_t& MakeExpression() const { return *this; };

  // The value and left/right sub expressions stored in this node of the tree.
  
  Value_t Value;
  Left_t  Left;
  Right_t Right;

  // Default ctor: takes no action.
  
  PETE_TBTree() { }

  // Constructor using both the value and the two children.
  
  PETE_TBTree(const Value_t& v, const Left_t& l, const Right_t& r)
  : Value(v), Left(l), Right(r) { }

  // Constructor using just the two children.
  
  PETE_TBTree(const Left_t& l, const Right_t& r)
  : Left(l), Right(r) { }
};


///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETE_TTTree<Value_t, Left_t, Middle_t, Right_t>
//
// DESCRIPTION
//    A tree node for representing trinary expressions. The node holds a
//    Left child (of type Left_t), which is the type of the LHS expression 
//    sub tree (typically a comparison operation); a Middle child (of type
//    Middle_t), which is the type of the middle (true branch) expression
//    sub tree; a Right child (of type Right_t), which is the type of 
//    the expression (false branch) sub tree; and a Value (of type Value_t), 
//    which is typically the operation applied to the three sub trees.
// 
///////////////////////////////////////////////////////////////////////////

template< class Value_t, class Left_t, class Middle_t, class Right_t >
class PETE_TTTree
  : public PETE_Expr< PETE_TTTree< Value_t, Left_t, Middle_t, Right_t > >
{
public:
  enum { IsExpr = 1 };
  typedef PETE_TTTree<Value_t,Left_t,Middle_t,Right_t> PETE_Expr_t;
  typedef typename PETETrinaryReturn<typename Left_t::PETE_Return_t, 
    typename Middle_t::PETE_Return_t,
    typename Right_t::PETE_Return_t, Value_t>::type PETE_Return_t;
  const PETE_Expr_t& MakeExpression() const { return *this; };

  // The value and left, right, and middle sub trees stored at this node.

  Value_t  Value;
  Left_t   Left;
  Middle_t Middle;
  Right_t  Right;

  // Default ctor: takes no action.

  PETE_TTTree() { }

  // Constructor using the value and three children.

  PETE_TTTree(const Value_t& v, const Left_t& l, 
    const Middle_t& m, const Right_t& r)
    : Value(v), Left(l), Middle(m), Right(r) { }

  // Constructor with just the three children.

  PETE_TTTree(const Left_t& l, const Middle_t& m,const Right_t& r)
    : Left(l), Middle(m), Right(r) {}
};


//=========================================================================
//
// DEFAULT for_each FUNCTIONS INVOLVING PETE TREES
// 
//=========================================================================

template<class Op, class T1, class Functor, class Combiner>
inline
typename Functor::PETE_Return_t
for_each(PETE_TUTree<Op,T1>& node, Functor f, Combiner c)
{
  return c(for_each(node.Child, f, c));
}

template<class Op, class T1, class T2, class Functor, class Combiner>
inline
typename Functor::PETE_Return_t
for_each(PETE_TBTree<Op,T1,T2>& node, Functor f, Combiner c)
{
  return c(for_each(node.Left, f, c), for_each(node.Right, f, c));
}

template<class Op, class T1, class T2, class T3, class Functor, class Combiner>
inline
typename Functor::PETE_Return_t
for_each(PETE_TTTree<Op,T1,T2,T3>& node, Functor f, Combiner c)
{
  return c(for_each(node.Left, f, c), for_each(node.Middle, f, c) ,
    for_each(node.Right, f, c));
}


//
// A useful combiner.
//
template<class T, class Op>
struct PETE_Combiner
{
  T operator()(T x) { return x; }
  T operator()(T x, T y) { return PETE_apply(Op(),x,y); }
  T operator()(T x, T y, T z) {return PETE_apply(Op(),x,PETE_apply(Op(),y,z));}
};

struct AssertEquals
{
  int operator()(int l)
  {
    return l;
  }
  int operator()(int l, int r)
  {
    int ret = l;
    if ( (l>=0) && (r>=0) ) {
      PInsist(l==r,"Arguments not equal in AssertEquals()!!");
    }
    else {
      if ( r>=0 ) return ret = r;
    }
    return ret;
  }
  int operator()(int l, int m, int r)
  {
    int ret = l;
    if ( (l>=0) && (m>=0) && (r>=0) ) {
      PInsist(m==l && m==r,"Arguments not equal in AssertEquals()!!");
    }
    else if ( l>=0 ) {
      return l;
    }
    else if ( m>=0) {
      return m;
    }
    else if ( r>=0) {
      return r;
    }
    return ret;
  }
};

//
// A combiner for when you don't want to return a value.
//

struct PETE_NullCombiner
{
  int operator()(int) { return 0; }
  int operator()(int, int) { return 0; }
  int operator()(int, int, int) { return 0; }
};

//
// Some shorthand for common combiners.
// 

typedef PETE_Combiner<bool,OpAnd> PETE_AndCombiner;
typedef PETE_Combiner<bool,OpOr> PETE_OrCombiner;
typedef PETE_Combiner<int,OpAdd> PETE_SumCombiner;


//////////////////////////////////////////////////////////////////////

//
// for_each functions that recurse through the tree doing evaluation.
//

template<class Op, class T1, class Functor>
inline
typename PETEUnaryReturn<typename T1::PETE_Return_t,Op>::type
for_each(PETE_TUTree<Op,T1>& node, Functor f)
{
  return PETE_apply(node.Value,
		    for_each(node.Child,f));
}


template<class Op, class T1, class T2, class Functor>
struct struct_for_each
{
  typedef typename PETEBinaryReturn<typename T1::PETE_Return_t,
    typename T2::PETE_Return_t,Op>::type
  Return_t;

  static inline Return_t
  apply(PETE_TBTree<Op,T1,T2>& node, Functor f)
    {
      return PETE_apply(node.Value, 
			for_each(node.Left,f),
			for_each(node.Right,f) );
    }
};

template<class T>
struct ConditionalAssign
{
  ConditionalAssign(bool q, const T& v) : cond(q), value(v) {}
  bool cond;
  T value;
};

template<class T1, class T2, class Functor>
struct struct_for_each<OpWhere,T1,T2,Functor>
{
  typedef typename T2::PETE_Return_t T3;
  typedef ConditionalAssign<T3> Return_t;

  static inline Return_t
  apply(PETE_TBTree<OpWhere,T1,T2>& node, Functor f)
    {
      if ( for_each(node.Left,f) )
	return Return_t(true,for_each(node.Right,f));
      else
	return Return_t(false,T3());
    }
};

template<class Op, class T1, class T2, class Functor>
inline
typename struct_for_each<Op,T1,T2,Functor>::Return_t
for_each(PETE_TBTree<Op,T1,T2>& node, Functor f)
{
  return struct_for_each<Op,T1,T2,Functor>::apply(node,f);
}

/***********************************************************************

  There are two ways to evaluate the trinary where operator:

      if (a) { return b; } else { return c; }

      return a ? b : c;

  The first is safer since b or c could have a divide by zero
  or something when it is not supposed to be evaluated, but the
  second can be much faster since there are more optimization
  opportunities, particularly strength reduction.

  Below we have some trait magic for looking at the right hand sides
  and deciding if the expression is "dangerous" and needs to be done
  the first way.  Otherwise it does it the second way.

  This uses partial specialization so it is user extensible for
  special cases.

***********************************************************************/

template<int I1, int I2> struct SafeCombine {};
template<> struct SafeCombine<0,0> { enum { safe=0 }; };
template<> struct SafeCombine<1,0> { enum { safe=0 }; };
template<> struct SafeCombine<0,1> { enum { safe=0 }; };
template<> struct SafeCombine<1,1> { enum { safe=1 }; };

// Expressions are safe unless proven otherwise.
template<class Expr> struct SafeExpression { enum { safe=1 }; };

// Unary expressions are safe if the sub-tree is safe.
template<class Op, class Sub>
struct SafeExpression< PETE_TUTree<Op,Sub> >
{
  enum { safe = SafeExpression<Sub>::safe };
};

// Binary expressions are safe if both sub-trees are safe.
template<class Op, class Left, class Right>
struct SafeExpression< PETE_TBTree<Op,Left,Right> >
{
  enum { safe = SafeCombine<SafeExpression<Left>::safe,SafeExpression<Right>::safe>::safe };
};

// Trinary expressions are safe if the true and false sub-trees are safe.
template<class Op, class Left, class Middle, class Right>
struct SafeExpression< PETE_TTTree<Op,Left,Middle,Right> >
{
  enum { safe = SafeCombine<SafeExpression<Middle>::safe,SafeExpression<Right>::safe>::safe };
};

// OpDivide is unsafe.
template<class Left, class Right>
struct SafeExpression< PETE_TBTree<OpDivide,Left,Right> >
{
  enum { safe = 0 };
};

// Functor class to evaluate the for_each.
// If no specialization matches, use PETE_apply.
// For OpWhere, this will use ?: through PETE_apply
template<int Safe, class T1, class T2, class T3, class Op, class Functor>
struct TrinaryForEach
{
  static inline 
    typename PETETrinaryReturn<
      typename T1::PETE_Return_t, 
      typename T2::PETE_Return_t,
      typename T3::PETE_Return_t,
      Op>::type
  apply(PETE_TTTree<Op,T1,T2,T3>& node, Functor f) {
    return PETE_apply(node.Value,
		      for_each(node.Left,f),
		      for_each(node.Middle,f),
		      for_each(node.Right,f));
  }
};

// For an unsafe OpWhere, don't evaluate both args.
template<class T1, class T2, class T3, class Functor>
struct
TrinaryForEach<0,T1,T2,T3,OpWhere,Functor>
{
  static inline
    typename PETETrinaryReturn<
      typename T1::PETE_Return_t, 
      typename T2::PETE_Return_t,
      typename T3::PETE_Return_t,
      OpWhere>::type
  apply(PETE_TTTree<OpWhere,T1,T2,T3>& node, Functor f) {
    return for_each(node.Left,f) ? 
           for_each(node.Middle,f) : for_each(node.Right,f);
  }
};

// The general definition of for_each for trinary.
// This just turns around and calls the class function.
template<class T1, class T2, class T3, class Functor,class Op>
inline
typename PETETrinaryReturn<typename T1::PETE_Return_t, 
  typename T2::PETE_Return_t,
  typename T3::PETE_Return_t,Op>::type
for_each(PETE_TTTree<Op,T1,T2,T3>& node, Functor f)
{
  return TrinaryForEach<
    SafeExpression< PETE_TTTree<Op,T1,T2,T3> >::safe
    ,T1,T2,T3,Op,Functor>::apply(node,f);
}

//=========================================================================
//
// GENERAL PETE REDUCTION CODE
// 
//=========================================================================

#if !defined(PETE_USER_REDUCTION_CODE)

#define PETE_USER_REDUCTION_CODE

#endif

template< class R, class T, class InitOp, class AccOp>
inline void 
Reduction(R& ret, const PETE_Expr<T>& const_expr,
  InitOp init_op, AccOp acc_op )
{
  //  Extract the expression.

  typename T::PETE_Expr_t expr(const_expr.PETE_unwrap().MakeExpression());

  // Get the number of elements we will be looping over.
  
  int n = for_each(expr, PETE_CountElems(), AssertEquals());

  // Make sure there is something to do.
  
  if (n > 0) {

    // Get the first value.
  
    PETE_apply(init_op, ret, for_each(expr, EvalFunctor_0()));

    // Loop over all the elements.
    for (int i = 1; i < n; ++i)
      {
	// Increment the cursors.
	
	for_each(expr, PETE_Increment(), PETE_NullCombiner());
      
	// Accumulate.
      
	PETE_apply(acc_op, ret, for_each(expr, EvalFunctor_0()));
      }
  }

  // Allow users to augment the reduction code.

  PETE_USER_REDUCTION_CODE
}


//=========================================================================
//
// UNARY OPERATIONS
// 
//=========================================================================

#define PETE_DefineUnary(Fun,Expr,Op)                                       \
template<class T>                                                           \
inline typename PETEUnaryReturn<T, Op>::type                                \
PETE_apply(Op, const T& a)                                                  \
{                                                                           \
  return Expr;                                                              \
}                                                                           \
template<class T>                                                           \
inline PETE_TUTree<Op, typename T::PETE_Expr_t>                             \
Fun(const PETE_Expr<T>& l)                                                  \
{                                                                           \
  return PETE_TUTree<Op, typename T::PETE_Expr_t>                           \
    (l.PETE_unwrap().MakeExpression());                                     \
}

PETE_DefineUnary(operator-, (-a), OpUnaryMinus)
PETE_DefineUnary(operator+, (+a), OpUnaryPlus)
PETE_DefineUnary(operator~, (~a), OpBitwiseNot)
PETE_DefineUnary(operator!, (!a), OpNot)
PETE_DefineUnary(PETE_identity,   (a), OpIdentity)

PETE_DefineUnary(acos, (acos(a)), FnArcCos)
PETE_DefineUnary(asin, (asin(a)), FnArcSin)
PETE_DefineUnary(atan, (atan(a)), FnArcTan)
PETE_DefineUnary(ceil, (ceil(a)), FnCeil)
PETE_DefineUnary(cos, (cos(a)), FnCos)
PETE_DefineUnary(cosh, (cosh(a)), FnHypCos)
PETE_DefineUnary(exp, (exp(a)), FnExp)
PETE_DefineUnary(fabs, (fabs(a)), FnFabs)
PETE_DefineUnary(floor, (floor(a)), FnFloor)
PETE_DefineUnary(log, (log(a)), FnLog)
PETE_DefineUnary(log10, (log10(a)), FnLog10)
PETE_DefineUnary(sin, (sin(a)), FnSin)
PETE_DefineUnary(sinh, (sinh(a)), FnHypSin)
PETE_DefineUnary(sqrt, (sqrt(a)), FnSqrt)
PETE_DefineUnary(tan, (tan(a)), FnTan)
PETE_DefineUnary(tanh, (tanh(a)), FnHypTan)
PETE_DefineUnary(erf, (erf(a)), FnErf)

//
// Define OpCast specially because it doesn't fit 
// the pattern that the #define needs.
//

template<class T1, class T2>
inline T1
PETE_apply(OpCast<T1>, const T2& a)
{
  return T1(a);
}

template<class T1, class Expr>
inline PETE_TUTree<OpCast<T1>, typename Expr::PETE_Expr_t>
pete_cast(const T1&, const PETE_Expr<Expr>& l)
{
  return
    PETE_TUTree<OpCast<T1>, typename Expr::PETE_Expr_t>
    (l.PETE_unwrap().MakeExpression());
}

//=========================================================================
//
// BINARY OPERATIONS
// 
//=========================================================================

#define PETE_DefineBinary(Fun,Expr,Op)                                      \
template<class T1, class T2>                                                \
inline typename PETEBinaryReturn<T1, T2, Op>::type                          \
PETE_apply(Op, const T1& a, const T2& b)                                    \
{                                                                           \
  return Expr;                                                              \
}                                                                           \
template<class T1, class T2>                                                \
inline PETE_TBTree<Op, typename T1::PETE_Expr_t, typename T2::PETE_Expr_t>  \
Fun(const PETE_Expr<T1>& l, const PETE_Expr<T2>& r)                         \
{                                                                           \
  typedef PETE_TBTree<Op,typename T1::PETE_Expr_t,                          \
    typename T2::PETE_Expr_t> ret;                                          \
  return ret(l.PETE_unwrap().MakeExpression(),                              \
    r.PETE_unwrap().MakeExpression());                                      \
}

#define PETE_DefineBinarySynonym(Fun,Op)                                    \
template<class T1, class T2>                                                \
inline PETE_TBTree<Op, typename T1::PETE_Expr_t, typename T2::PETE_Expr_t>  \
Fun(const PETE_Expr<T1>& l, const PETE_Expr<T2>& r)                         \
{                                                                           \
  typedef PETE_TBTree<Op,typename T1::PETE_Expr_t,typename T2::PETE_Expr_t> \
    ret;                                                                    \
  return ret(l.PETE_unwrap().MakeExpression(),                              \
    r.PETE_unwrap().MakeExpression());                                      \
}

PETE_DefineBinary(operator+, (a + b), OpAdd)
PETE_DefineBinary(operator-, (a - b), OpSubtract)
PETE_DefineBinary(operator*, (a * b), OpMultipply)
PETE_DefineBinary(operator/, (a / b), OpDivide)
PETE_DefineBinary(operator%, (a % b), OpMod)
PETE_DefineBinary(operator<, (a < b), OpLT)
PETE_DefineBinary(operator<=, (a <= b), OpLE)
PETE_DefineBinary(operator>, (a > b), OpGT)
PETE_DefineBinary(operator>=, (a >= b), OpGE)
PETE_DefineBinary(operator==, (a == b), OpEQ)
PETE_DefineBinary(operator!=, (a != b), OpNE)
PETE_DefineBinary(operator&&, (a && b), OpAnd)
PETE_DefineBinary(operator||, (a || b), OpOr)
PETE_DefineBinary(operator&, (a & b), OpBitwiseAnd)
PETE_DefineBinary(operator|, (a | b), OpBitwiseOr)
PETE_DefineBinary(operator^, (a ^ b), OpBitwiseXor)
  //PETE_DefineBinary(operator<<, (a << b), OpLeftShift)
  //PETE_DefineBinary(operator>>, (a >> b), OpRightShift)

PETE_DefineBinary(copysign, (copysign(a,b)), FnCopysign)
PETE_DefineBinary(ldexp, (ldexp(a,b)), FnLdexp)
PETE_DefineBinary(pow, (pow(a,b)), FnPow)
PETE_DefineBinary(fmod, (fmod(a,b)), FnFmod)
PETE_DefineBinary(atan2, (atan2(a,b)), FnArcTan2)

#define PETE_DefineBinaryWithScalars(Fun,Op,Sca)                            \
template<class T>                                                           \
inline PETE_TBTree<Op, PETE_Scalar<Sca>, typename T::PETE_Expr_t>           \
Fun(const Sca l, const PETE_Expr<T>& r)                                     \
{                                                                           \
  typedef PETE_TBTree<Op, PETE_Scalar<Sca>, typename T::PETE_Expr_t> ret;   \
  return ret(PETE_Scalar<Sca>(l), r.PETE_unwrap().MakeExpression());        \
}                                                                           \
template<class T>                                                           \
inline PETE_TBTree<Op, typename T::PETE_Expr_t, PETE_Scalar<Sca> >          \
Fun(const PETE_Expr<T>& l, const Sca r)                                     \
{                                                                           \
  typedef PETE_TBTree<Op, typename T::PETE_Expr_t, PETE_Scalar<Sca> > ret;  \
  return ret(l.PETE_unwrap().MakeExpression(), PETE_Scalar<Sca>(r));        \
}


//=========================================================================
//
// TRINARY OPERATORS
//
//=========================================================================

#define PETE_DefineTrinary(Fun,Expr,Op)                                     \
template<class T1, class T2, class T3>                                      \
inline typename PETETrinaryReturn<T1,T2,T3,Op>::type                        \
PETE_apply(Op, const T1& a, const T2& b, const T3& c)                       \
{                                                                           \
  return Expr;                                                              \
}                                                                           \
template<class Cond_t, class True_t, class False_t>                         \
inline PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                        \
  typename True_t::PETE_Expr_t, typename False_t::PETE_Expr_t>              \
Fun(const PETE_Expr<Cond_t>& c, const PETE_Expr<True_t>& t,                 \
  const PETE_Expr<False_t>& f)                                              \
{                                                                           \
  typedef PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                     \
    typename True_t::PETE_Expr_t, typename False_t::PETE_Expr_t> ret;       \
  return ret(c.PETE_unwrap().MakeExpression(),                              \
    t.PETE_unwrap().MakeExpression(),                                       \
    f.PETE_unwrap().MakeExpression());                                      \
}

template<class T1, class T2, class T3>
inline typename PETETrinaryReturn<T1,T2,T3,OpWhere>::type
PETE_apply(OpWhere, const T1& a, const T2& b, const T3& c)
{
  return a ? b : c;
}

template<class Cond_t, class True_t, class False_t>
inline PETE_TTTree<OpWhere, typename Cond_t::PETE_Expr_t, 
  typename True_t::PETE_Expr_t,
  typename False_t::PETE_Expr_t>
where(const PETE_Expr<Cond_t>& c, const PETE_Expr<True_t>& t,
  const PETE_Expr<False_t>& f)
{
  typedef PETE_TTTree<OpWhere, typename Cond_t::PETE_Expr_t, 
    typename True_t::PETE_Expr_t,
    typename False_t::PETE_Expr_t> ret;
  return ret(c.PETE_unwrap().MakeExpression(),
    t.PETE_unwrap().MakeExpression(),
    f.PETE_unwrap().MakeExpression());
}


#define PETE_DefineTrinaryWithScalars(Fun, Op, Sca)                         \
template<class Cond_t, class True_t>                                        \
inline PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                        \
  typename True_t::PETE_Expr_t, PETE_Scalar<Sca> >                          \
Fun(const PETE_Expr<Cond_t>& c, const PETE_Expr<True_t>& t,Sca f)           \
{                                                                           \
  typedef PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                     \
    typename True_t::PETE_Expr_t, PETE_Scalar<Sca> > ret;                   \
  return ret(c.PETE_unwrap().MakeExpression(),                              \
    t.PETE_unwrap().MakeExpression(), PETE_Scalar<Sca>(f));                 \
}                                                                           \
template<class Cond_t, class False_t>                                       \
inline PETE_TTTree<Op, typename Cond_t::PETE_Expr_t, PETE_Scalar<Sca>,      \
  typename False_t::PETE_Expr_t >                                           \
Fun(const PETE_Expr<Cond_t>& c, Sca t, const PETE_Expr<False_t>& f)         \
{                                                                           \
  typedef PETE_TTTree<Op, typename Cond_t::PETE_Expr_t, PETE_Scalar<Sca>,   \
    typename False_t::PETE_Expr_t > ret;                                    \
  return ret(c.PETE_unwrap().MakeExpression(),                              \
    PETE_Scalar<Sca>(t), f.PETE_unwrap().MakeExpression());                 \
}                                                                           \
template<class Cond_t>                                                      \
inline PETE_TTTree<Op, typename Cond_t::PETE_Expr_t, PETE_Scalar<Sca>,      \
  PETE_Scalar<Sca> >                                                        \
Fun(const PETE_Expr<Cond_t>& c, Sca t, Sca f)                               \
{                                                                           \
  typedef PETE_TTTree<Op, typename Cond_t::PETE_Expr_t, PETE_Scalar<Sca>,   \
    PETE_Scalar<Sca> > ret;                                                 \
  return ret(c.PETE_unwrap().MakeExpression(),                              \
    PETE_Scalar<Sca>(t), PETE_Scalar<Sca>(f));                              \
}


//=========================================================================
//
// Two argument where
//
//   The conventional where operator has three arguments:
//      where(conditional, true_expr, false_expr)
//   It acts much like the ?: operator in expressions.
//
//   A common usage of that structure is 
//      A = where(conditional,true_expr,A);
//   to do something only in the locations where the conditional is true.
//   This is inefficient.
//
//   The two argument where supports the following:
//      A = where(conditional,true_expr);
//   and means the same thing as the three argument version above.
//
//   This form of where cannot be used in expressions in which the where
//   is not the whole right hand side of the expression.
//
//======================================================================
//
// EXPLANATION
//
//   The three argument where operator above produces code something 
//   like this in the inner loop:
//
//      for (i=...)
//        a(i) = conditional(i) ? true_expr(i) : false_expr(i);
//
//   The two argument where has to produce something like:
//
//      for (i=...)
//        if ( conditional(i) ) 
//          a(i) = true_expr(i);
//
//   Two things are being made conditional here: evaluating true_expr
//   and assigning it to a.
//
//   In order to do that inside of expression templates, the following 
//   things need to happen when evaluating an expression:
//
//   1. Evaluating the binary OpWhere expression returns a
//   ConditionalAssign<T> with two components: A boolean for whether
//   the conditional was true, and the result of evaluating the
//   expression of type T.
//
//   2. The expression only gets evaluated if the conditional is true.
//   If it is not true, then the ConditionalAssign<T> uses a default ctor
//   to leave the T undefined.
//
//   3. The assigment operator from the ConditionalAssign<T> to a type
//   T only actually does the assignment if the bool is true.
//
//=========================================================================

PETE_DefineBinarySynonym(where, OpWhere)

//=========================================================================
//
// SCALARS
//
//=========================================================================

#define PETE_DefineScalar(Sca)                                              \
PETE_DefineBinaryWithScalars(operator+, OpAdd, Sca)                         \
PETE_DefineBinaryWithScalars(operator-, OpSubtract, Sca)                    \
PETE_DefineBinaryWithScalars(operator*, OpMultipply, Sca)                    \
PETE_DefineBinaryWithScalars(operator/, OpDivide, Sca)                      \
PETE_DefineBinaryWithScalars(operator%, OpMod, Sca)                         \
PETE_DefineBinaryWithScalars(operator<, OpLT, Sca)                          \
PETE_DefineBinaryWithScalars(operator<=, OpLE, Sca)                         \
PETE_DefineBinaryWithScalars(operator>, OpGT, Sca)                          \
PETE_DefineBinaryWithScalars(operator>=, OpGE, Sca)                         \
PETE_DefineBinaryWithScalars(operator==, OpEQ, Sca)                         \
PETE_DefineBinaryWithScalars(operator!=, OpNE, Sca)                         \
PETE_DefineBinaryWithScalars(operator&&, OpAnd, Sca)                        \
PETE_DefineBinaryWithScalars(operator||, OpOr, Sca)                         \
PETE_DefineBinaryWithScalars(operator&, OpBitwiseAnd, Sca)                  \
PETE_DefineBinaryWithScalars(operator|, OpBitwiseOr, Sca)                   \
PETE_DefineBinaryWithScalars(operator^, OpBitwiseXor, Sca)                  \
PETE_DefineBinaryWithScalars(where, OpWhere, Sca)                           \
PETE_DefineBinaryWithScalars(copysign, FnCopysign, Sca)                     \
PETE_DefineBinaryWithScalars(ldexp, FnLdexp, Sca)                           \
PETE_DefineBinaryWithScalars(pow, FnPow, Sca)                               \
PETE_DefineBinaryWithScalars(fmod, FnFmod, Sca)                             \
PETE_DefineBinaryWithScalars(atan2, FnArcTan2, Sca)                         \
PETE_DefineTrinaryWithScalars(where, OpWhere, Sca)

/*
PETE_DefineBinaryWithScalars(operator<<, OpLeftShift, Sca)                  \
PETE_DefineBinaryWithScalars(operator>>, OpRightShift, Sca)                 \
*/

PETE_DefineScalar(short)
PETE_DefineScalar(int)
PETE_DefineScalar(long)
PETE_DefineScalar(float)
PETE_DefineScalar(double)


//=========================================================================
//
// ASSIGNMENT OPERATIONS
// 
//=========================================================================

template<class Op, class T1, class T2> struct PETE_StructApply {};

#define PETE_DefineAssign(Expr,Cond,Op)				\
template<class T1, class T2>		 		        \
struct PETE_StructApply<Op,T1,T2>				\
{								\
  static void apply(T1& a,const T2& b) { Expr; }		\
};								\
								\
template<class T1, class T2>				        \
struct PETE_StructApply<Op,T1,ConditionalAssign<T2> >		\
{								\
  static void apply(T1& a, const ConditionalAssign<T2>& b)	\
    {								\
      if ( b.cond )						\
	Cond;							\
    }								\
};								\
								\
template<class T1, class T2>					\
inline void							\
PETE_apply(Op, T1 &a, const T2& b)				\
{								\
  PETE_StructApply<Op,T1,T2>::apply(a,b);			\
}

PETE_DefineAssign((a  = b) ,(a  = b.value) , OpAssign)
PETE_DefineAssign((a += b) ,(a += b.value) , OpAddAssign)
PETE_DefineAssign((a -= b) ,(a -= b.value) , OpSubtractAssign)
PETE_DefineAssign((a *= b) ,(a *= b.value) , OpMultipplyAssign)
PETE_DefineAssign((a /= b) ,(a /= b.value) , OpDivideAssign)
PETE_DefineAssign((a %= b) ,(a %= b.value) , OpModAssign)
PETE_DefineAssign((a |= b) ,(a |= b.value) , OpBitwiseOrAssign)
PETE_DefineAssign((a &= b) ,(a &= b.value) , OpBitwiseAndAssign)
PETE_DefineAssign((a ^= b) ,(a ^= b.value) , OpBitwiseXorAssign)
PETE_DefineAssign((a <<= b),(a <<= b.value), OpLeftShiftAssign)
PETE_DefineAssign((a >>= b),(a >>= b.value), OpRightShiftAssign)


//=========================================================================
//
// FUNCTOR NAME
//   Expressionize
//
// DESCRIPTION
//   Sometimes the result of MakeExpression on a user type isn't itself an
//   expression.  That is perfectly reasonable, but sometimes you need to
//   be make it one.  This just looks for that case and wraps it in an
//   identity operation if it is not already an expression.
// 
//=========================================================================

//
// If it is a general expression, wrap it.
//
template<class T>
struct Expressionize
{
  typedef PETE_TUTree< OpIdentity, T > type;
  static inline type
  apply(const T& t) { return type(t); }
};

//
// If it is a PETE_Expr, return it as is.
//
template<class T>
struct Expressionize< PETE_Expr<T> >
{
  typedef PETE_Expr<T> type;
  static inline const type&
  apply(const PETE_Expr<T>& t) { return t; }
};

//=========================================================================
//
// REDUCTIONS
// 
//=========================================================================

template<class T> 
inline typename T::PETE_Expr_t::PETE_Return_t
sum(const PETE_Expr<T>& expr)
{
  typename T::PETE_Expr_t::PETE_Return_t val ;
  Reduction(val, Expressionize<typename T::PETE_Expr_t>::apply( expr.PETE_unwrap().MakeExpression() ), 
    OpAssign(), OpAddAssign());
  return val;
}

template<class T> 
inline typename T::PETE_Expr_t::PETE_Return_t
prod(const PETE_Expr<T>& expr)
{
  typename T::PETE_Expr_t::PETE_Return_t val ;
  Reduction(val, Expressionize<typename T::PETE_Expr_t>::apply(expr.PETE_unwrap().MakeExpression()),
    OpAssign(), OpMultipplyAssign());
  return val;
}

//////////////////////////////////////////////////////////////////////
//
// Some macros to make it easier to define new operators.
//
//////////////////////////////////////////////////////////////////////

#define UNARY_FUNCTION(RET,FUNC,ARG)                                        \
struct FnUnary_ ## FUNC {                                                   \
  enum { tag = PETE_UnaryPassThruTag };                                     \
};                                                                          \
PETE_DefineUnary(FUNC,FUNC(a),FnUnary_ ## FUNC)                             \
template <>                                                                 \
struct PETEUnaryReturn<ARG, FnUnary_ ## FUNC> {                             \
  typedef RET type;                                                         \
};

#define BINARY_FUNCTION(RET,FUNC,ARG1,ARG2)                                 \
struct FnBinary_ ## FUNC {                                                  \
  enum { tag = PETE_BinaryPromoteTag };                                     \
};                                                                          \
PETE_DefineBinary(FUNC,(FUNC(a,b)),FnBinary_ ## FUNC)                       \
template<>                                                                  \
struct PETEBinaryReturn<ARG1,ARG2,FnBinary_ ## FUNC> {                      \
  typedef RET type;                                                         \
};


///////////////////////////////////////////////////////////////////////////
//
// END OF FILE
// 
///////////////////////////////////////////////////////////////////////////

#endif // PETE_H

