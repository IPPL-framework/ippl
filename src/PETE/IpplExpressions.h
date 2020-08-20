// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/


///////////////////////////////////////////////////////////////////////////
//
// FILE NAME
//    IpplExpressions.h
//
// CREATED
//    July 11, 1997
//
// DESCRIPTION
//    This header file defines custom objects and operators necessary to use
//    expression templates in IPPL.
//
///////////////////////////////////////////////////////////////////////////

#ifndef IPPL_EXPRESSIONS_H
#define IPPL_EXPRESSIONS_H


// We need to construct a custom version of Reduction. We must define
// this macro before including PETE.h.
#define PETE_USER_REDUCTION_CODE                                            \
  R global_ret;                                                             \
  reduce_masked(ret, global_ret, acc_op, 0 < n );                           \
  ret = global_ret;

// include files
#include "Message/Message.h"
#include "PETE/IpplTypeComputations.h"
#include "PETE/PETE.h"


//=========================================================================
//
// UNARY OPERATIONS
// 
//=========================================================================

// Abs is handled rather strangely. There appears to be two versions
// that do the same thing: abs and Abs.

PETE_DefineUnary(Abs, (0 < a ? a : -a), FnAbs)

inline double
PETE_apply(FnAbs, std::complex<double> a)
{
  return abs(a);
}

template<class T>
inline PETE_TUTree<FnAbs, typename T::PETE_Expr_t>
abs(const PETE_Expr<T>& l)
{
  return PETE_TUTree<FnAbs, typename T::PETE_Expr_t>
    (l.PETE_unwrap().MakeExpression());
}

PETE_DefineUnary(conj, (conj(a)), FnConj)
PETE_DefineUnary(arg, (arg(a)), FnArg)
PETE_DefineUnary(norm, (norm(a)), FnNorm)
PETE_DefineUnary(real, (real(a)), FnReal)
PETE_DefineUnary(imag, (imag(a)), FnImag)
PETE_DefineUnary(sign, (sign(a)), FnSign)
PETE_DefineUnary(trace, (trace(a)), FnTrace)
PETE_DefineUnary(transpose, (transpose(a)), FnTranspose)
PETE_DefineUnary(det, (det(a)), FnDet)
PETE_DefineUnary(cofactors, (cofactors(a)), FnCofactors)


//=========================================================================
//
// BINARY OPERATIONS
//
//=========================================================================

// define min/max for built-in scalar types
#define PETE_DefineScalarMinMax(Sca)                                      \
inline Sca                                                                \
Min(const Sca& a, const Sca& b)                                           \
{                                                                         \
  return (a<b ? a : b);                                                   \
}                                                                         \
inline Sca                                                                \
Max(const Sca& a, const Sca& b)                                           \
{                                                                         \
  return (b<a ? a : b);                                                   \
}

PETE_DefineScalarMinMax(short)
PETE_DefineScalarMinMax(int)
PETE_DefineScalarMinMax(long)
PETE_DefineScalarMinMax(float)
PETE_DefineScalarMinMax(double)

PETE_DefineBinary(Min, (Min(a,b)), FnMin)
PETE_DefineBinary(Max, (Max(a,b)), FnMax)

PETE_DefineBinary(dot, (dot(a,b)), FnDot)
PETE_DefineBinary(dotdot, (dotdot(a,b)), FnDotDot)
PETE_DefineBinary(outerProduct, (outerProduct(a,b)), FnOuterProduct)
PETE_DefineBinary(cross, (cross(a,b)), FnCross)

PETE_DefineBinarySynonym(lt, OpLT)
PETE_DefineBinarySynonym(gt, OpGT)
PETE_DefineBinarySynonym(le, OpLE)
PETE_DefineBinarySynonym(ge, OpGE)
PETE_DefineBinarySynonym(eq, OpEQ)
PETE_DefineBinarySynonym(ne, OpNE)

//tjw: make sure kosher to have "cross" down there (added in)
#define PETE_DefineIPPLScalar(Sca)                                         \
PETE_DefineBinaryWithScalars(Min, FnMin, Sca)                               \
PETE_DefineBinaryWithScalars(Max, FnMax, Sca)                               \
PETE_DefineBinaryWithScalars(dot, FnDot, Sca)                               \
PETE_DefineBinaryWithScalars(dotdot, FnDotDot, Sca)                         \
PETE_DefineBinaryWithScalars(outerProduct, FnOuterProduct, Sca)             \
PETE_DefineBinaryWithScalars(cross, FnDot, Sca)                             \
PETE_DefineBinaryWithScalars(lt, OpLT, Sca)                                 \
PETE_DefineBinaryWithScalars(le, OpLE, Sca)                                 \
PETE_DefineBinaryWithScalars(gt, OpGT, Sca)                                 \
PETE_DefineBinaryWithScalars(ge, OpGE, Sca)                                 \
PETE_DefineBinaryWithScalars(eq, OpEQ, Sca)                                 \
PETE_DefineBinaryWithScalars(ne, OpNE, Sca)

PETE_DefineIPPLScalar(short)
PETE_DefineIPPLScalar(int)
PETE_DefineIPPLScalar(long)
PETE_DefineIPPLScalar(float)
PETE_DefineIPPLScalar(double)

PETE_DefineScalar(std::complex<double>)
PETE_DefineBinaryWithScalars(eq, OpEQ, std::complex<double>)
PETE_DefineBinaryWithScalars(ne, OpNE, std::complex<double>)

#undef PETE_DefineIPPLScalar

// Now we need to provide special cases for Vektors, SymTenzors, and Tenzors
// so we can remove PETE_Expr base class.

#define PETE_DefineBinaryWithVSTScalars(Fun,Op,Sca)                         \
template<class T1, unsigned Dim, class T2>                                  \
inline PETE_TBTree<Op, PETE_Scalar< Sca<T1, Dim> >,                         \
  typename T2::PETE_Expr_t>                                                 \
Fun(const Sca<T1, Dim> &l, const PETE_Expr<T2>& r)                          \
{                                                                           \
  typedef PETE_TBTree<Op, PETE_Scalar< Sca<T1, Dim> >,                      \
    typename T2::PETE_Expr_t> ret;                                          \
  return ret(PETE_Scalar< Sca<T1, Dim> >(l),                                \
    r.PETE_unwrap().MakeExpression());                                      \
}                                                                           \
template<class T1, class T2, unsigned Dim>                                  \
inline PETE_TBTree<Op, typename T1::PETE_Expr_t,                            \
  PETE_Scalar< Sca<T2, Dim> > >                                             \
Fun(const PETE_Expr<T1>& l, const Sca<T2, Dim> &r)                          \
{                                                                           \
  typedef PETE_TBTree<Op, typename T1::PETE_Expr_t,                         \
    PETE_Scalar< Sca<T2, Dim> > > ret;                                      \
  return ret(l.PETE_unwrap().MakeExpression(),                              \
    PETE_Scalar< Sca<T2, Dim> >(r));                                        \
}

#define PETE_DefineTrinaryWithVSTScalars(Fun, Op, Sca)                      \
template<class Cond_t, class True_t, class T, unsigned Dim>                 \
inline PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                        \
  typename True_t::PETE_Expr_t, PETE_Scalar< Sca<T, Dim> > >                \
Fun(const PETE_Expr<Cond_t>& c, const PETE_Expr<True_t>& t,                 \
  const Sca<T, Dim> &f)                                                     \
{                                                                           \
  typedef PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                     \
    typename True_t::PETE_Expr_t, PETE_Scalar< Sca<T, Dim> > > ret;         \
  return ret(c.PETE_unwrap().MakeExpression(),                              \
    t.PETE_unwrap().MakeExpression(), PETE_Scalar< Sca<T, Dim> >(f));       \
}                                                                           \
template<class Cond_t, class T, unsigned Dim, class False_t>                \
inline PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                        \
  PETE_Scalar< Sca<T, Dim> >, typename False_t::PETE_Expr_t >               \
Fun(const PETE_Expr<Cond_t>& c, const Sca<T, Dim> &t,                       \
  const PETE_Expr<False_t>& f)                                              \
{                                                                           \
  typedef PETE_TTTree<Op, typename Cond_t::PETE_Expr_t, PETE_Scalar< Sca<T, Dim> >,  \
    typename False_t::PETE_Expr_t > ret;                                    \
  return ret(c.PETE_unwrap().MakeExpression(),                              \
    PETE_Scalar< Sca<T, Dim> >(t), f.PETE_unwrap().MakeExpression());       \
}                                                                           \
template<class Cond_t, class T, unsigned Dim>                               \
inline PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                        \
  PETE_Scalar< Sca<T, Dim> >, PETE_Scalar< Sca<T, Dim> > >                  \
Fun(const PETE_Expr<Cond_t>& c, const Sca<T, Dim> &t, const Sca<T, Dim> &f) \
{                                                                           \
  typedef PETE_TTTree<Op, typename Cond_t::PETE_Expr_t,                     \
    PETE_Scalar< Sca<T, Dim> >,  PETE_Scalar< Sca<T, Dim> > > ret;          \
  return ret(c.PETE_unwrap().MakeExpression(),                              \
    PETE_Scalar< Sca<T, Dim> >(t), PETE_Scalar< Sca<T, Dim> >(f));          \
}


//tjw: make sure kosher to have "cross" down there (added in)
#define PETE_DefineVSTScalar(Sca)                                           \
PETE_DefineBinaryWithVSTScalars(operator+, OpAdd, Sca)                      \
PETE_DefineBinaryWithVSTScalars(operator-, OpSubtract, Sca)                 \
PETE_DefineBinaryWithVSTScalars(operator*, OpMultipply, Sca)                 \
PETE_DefineBinaryWithVSTScalars(operator/, OpDivide, Sca)                   \
PETE_DefineBinaryWithVSTScalars(operator%, OpMod, Sca)                      \
PETE_DefineBinaryWithVSTScalars(operator<, OpLT, Sca)                       \
PETE_DefineBinaryWithVSTScalars(operator<=, OpLE, Sca)                      \
PETE_DefineBinaryWithVSTScalars(operator>, OpGT, Sca)                       \
PETE_DefineBinaryWithVSTScalars(operator>=, OpGE, Sca)                      \
PETE_DefineBinaryWithVSTScalars(operator==, OpEQ, Sca)                      \
PETE_DefineBinaryWithVSTScalars(operator!=, OpNE, Sca)                      \
PETE_DefineBinaryWithVSTScalars(operator&&, OpAnd, Sca)                     \
PETE_DefineBinaryWithVSTScalars(operator||, OpOr, Sca)                      \
PETE_DefineBinaryWithVSTScalars(operator&, OpBitwiseAnd, Sca)               \
PETE_DefineBinaryWithVSTScalars(operator|, OpBitwiseOr, Sca)                \
PETE_DefineBinaryWithVSTScalars(operator^, OpBitwiseXor, Sca)               \
PETE_DefineBinaryWithVSTScalars(copysign, FnCopysign, Sca)                  \
PETE_DefineBinaryWithVSTScalars(ldexp, FnLdexp, Sca)                        \
PETE_DefineBinaryWithVSTScalars(pow, FnPow, Sca)                            \
PETE_DefineBinaryWithVSTScalars(fmod, FnFmod, Sca)                          \
PETE_DefineBinaryWithVSTScalars(atan2, FnArcTan2, Sca)                      \
PETE_DefineTrinaryWithVSTScalars(where, OpWhere, Sca)                       \
PETE_DefineBinaryWithVSTScalars(Min, FnMin, Sca)                            \
PETE_DefineBinaryWithVSTScalars(Max, FnMax, Sca)                            \
PETE_DefineBinaryWithVSTScalars(dot, FnDot, Sca)                            \
PETE_DefineBinaryWithVSTScalars(dotdot, FnDotDot, Sca)                      \
PETE_DefineBinaryWithVSTScalars(outerProduct, FnOuterProduct, Sca)          \
PETE_DefineBinaryWithVSTScalars(cross, FnCross, Sca)                        \
PETE_DefineBinaryWithVSTScalars(lt, OpLT, Sca)                              \
PETE_DefineBinaryWithVSTScalars(le, OpLE, Sca)                              \
PETE_DefineBinaryWithVSTScalars(gt, OpGT, Sca)                              \
PETE_DefineBinaryWithVSTScalars(ge, OpGE, Sca)                              \
PETE_DefineBinaryWithVSTScalars(eq, OpEQ, Sca)                              \
PETE_DefineBinaryWithVSTScalars(ne, OpNE, Sca)

PETE_DefineVSTScalar(Vektor)
PETE_DefineVSTScalar(SymTenzor)
PETE_DefineVSTScalar(Tenzor)

#undef PETE_DefineVSTScalar


//=========================================================================
//
// ASSIGNMENT OPERATIONS
//
//=========================================================================

PETE_DefineAssign((a = Min(a,b)),(a = Min(a,b.value)), OpMinAssign)
PETE_DefineAssign((a = Max(a,b)),(a = Max(a,b.value)), OpMaxAssign)
PETE_DefineAssign((a = (a&&b))   ,(a = (a&&b.value))   , OpAndAssign)
PETE_DefineAssign((a = (a||b))   ,(a = (a||b.value))   , OpOrAssign)


//=========================================================================
//
// MIN and MAX REDUCTIONS
// 
//=========================================================================

template<class T> 
inline typename T::PETE_Expr_t::PETE_Return_t
min(const PETE_Expr<T>& expr)
{
  typename T::PETE_Expr_t::PETE_Return_t val ;
  Reduction(val,
	    Expressionize<typename T::PETE_Expr_t>::apply(expr.PETE_unwrap().MakeExpression()),
	    OpAssign(), 
	    OpMinAssign());
  return val;
}

template<class T> 
inline typename T::PETE_Expr_t::PETE_Return_t
max(const PETE_Expr<T>& expr)
{
  typename T::PETE_Expr_t::PETE_Return_t val ;
  Reduction(val,
	    Expressionize<typename T::PETE_Expr_t>::apply(expr.PETE_unwrap().MakeExpression()),
	    OpAssign(), 
	    OpMaxAssign());
  return val;
}

//=========================================================================
//
// MINMAX REDUCTION
// 
//=========================================================================

template<class T>
struct MinMaxHolder {
  T a;
  T b;
  MinMaxHolder() { }
  MinMaxHolder(const MinMaxHolder<T>& rhs) : a(rhs.a), b(rhs.b) { }
  const MinMaxHolder<T>& operator=(const MinMaxHolder<T>& rhs) {
    a = rhs.a;
    b = rhs.b;
    return *this;
  }
  const MinMaxHolder<T>& operator=(const T& rhs) {
    T c = rhs;
    a = c;
    b = c;
    return *this;
  }
  const MinMaxHolder<T>& operator*=(const MinMaxHolder<T>& rhs) {
    a = (a < rhs.a ? a : rhs.a);
    b = (rhs.b < b ? b : rhs.b);
    return *this;
  }
  const MinMaxHolder<T>& operator*=(const T& rhs) {
    T c = rhs;
    a = (a < c ? a : c);
    b = (c < b ? b : c);
    return *this;
  }
  Message& putMessage(Message& m) {
    m.put(a);
    m.put(b);
    return m;
  }
  Message& getMessage(Message& m) {
    m.get(a);
    m.get(b);
    return m;
  }
};

template<class T1, class T2>
inline void
minmax(const PETE_Expr<T1>& expr, T2& minval, T2& maxval)
{
  typedef typename T1::PETE_Expr_t::PETE_Return_t val_t;
  MinMaxHolder<val_t> ret;
  Reduction(ret,
	    Expressionize<typename T1::PETE_Expr_t>::apply(expr.PETE_unwrap().MakeExpression()),
	    OpAssign(),
	    OpMultipplyAssign());
  minval = static_cast<T2>(ret.a);
  maxval = static_cast<T2>(ret.b);
}


//////////////////////////////////////////////////////////////////////
//
// The 'any' function finds if there is any location in the expression
// where a condition is true.
//
//////////////////////////////////////////////////////////////////////

template<class T, class OP>
struct AnyHolder
{
  bool Test;
  T Val;
  OP Op;
  AnyHolder() : Test(false), Val(T(0)), Op(OP()) {}
  AnyHolder(const T& t, OP op) : Test(false), Val(t), Op(op) {}
  AnyHolder(const AnyHolder<T,OP>& rhs)
    : Test(rhs.Test), Val(rhs.Val), Op(rhs.Op) { }
  const AnyHolder<T,OP>& operator=(const T& rhs)
  {
    if ( PETE_apply(Op,rhs,Val) )
      Test = true;
    return *this;
  }
  const AnyHolder<T,OP>& operator=(const AnyHolder<T,OP>& rhs)
  {
    Test = rhs.Test;
    Val = rhs.Val;
    Op = rhs.Op;
    return *this;
  }
  const AnyHolder<T,OP>& operator*=(const T& rhs)
  {
    if ( PETE_apply(Op,rhs,Val) )
      Test = true;
    return *this;
  }
  const AnyHolder<T,OP>& operator*=(const AnyHolder<T,OP>& rhs)
  {
    Test = (Test || rhs.Test);
    return *this;
  }
  Message& putMessage(Message& m) 
  {
    m.put(Test);
    return m;
  }
  Message& getMessage(Message& m)
  {
    m.get(Test);
    return m;
  }
};

template<class T1, class T2>
inline bool
any(const PETE_Expr<T1>& expr, T2 val)
{
  AnyHolder<T2,OpEQ> ret(val,OpEQ());
  Reduction(ret,
	    Expressionize<typename T1::PETE_Expr_t>::apply(expr.PETE_unwrap().MakeExpression()),
	    OpAssign(),
	    OpMultipplyAssign());
  return ret.Test;
}

template<class T1, class T2, class Op>
inline bool
any(const PETE_Expr<T1>& expr, T2 val, Op op)
{
  AnyHolder<T2,Op> ret(val,op);
  Reduction(ret,
	    Expressionize<typename T1::PETE_Expr_t>::apply(expr.PETE_unwrap().MakeExpression()),
	    OpAssign(),
            OpMultipplyAssign());
  return ret.Test;
}


//=========================================================================
//
// BOUNDS REDUCTION - find bounding box of Vektor expression
// for scalars, use minmax
// for tensors, extend this code to include them as well
// 
//=========================================================================

template<class T, unsigned int D>
struct BoundsHolder {
  Vektor<T,D> a;
  Vektor<T,D> b;
  BoundsHolder() { }
  BoundsHolder(const BoundsHolder<T,D>& rhs) : a(rhs.a), b(rhs.b) { }
  const BoundsHolder<T,D>& operator=(const BoundsHolder<T,D>& rhs) {
    a = rhs.a;
    b = rhs.b;
    return *this;
  }
  const BoundsHolder<T,D>& operator=(const Vektor<T,D>& rhs) {
    Vektor<T,D> c(rhs);
    a = c;
    b = c;
    return *this;
  }
  const BoundsHolder<T,D>& operator=(const T& rhs) {
    Vektor<T,D> c(rhs);
    a = c;
    b = c;
    return *this;
  }
  const BoundsHolder<T,D>& operator*=(const BoundsHolder<T,D>& rhs) {
    for (unsigned int d=0; d < D; ++d) {
      a[d] = (a[d] < rhs.a[d] ? a[d] : rhs.a[d]);
      b[d] = (rhs.b[d] < b[d] ? b[d] : rhs.b[d]);
    }
    return *this;
  }
  const BoundsHolder<T,D>& operator*=(const Vektor<T,D>& rhs) {
    Vektor<T,D> c(rhs);
    for (unsigned int d=0; d < D; ++d) {
      a[d] = (a[d] < c[d] ? a[d] : c[d]);
      b[d] = (c[d] < b[d] ? b[d] : c[d]);
    }
    return *this;
  }
  const BoundsHolder<T,D>& operator*=(const T& rhs) {
    Vektor<T,D> c(rhs);
    for (unsigned int d=0; d < D; ++d) {
      a[d] = (a[d] < c[d] ? a[d] : c[d]);
      b[d] = (c[d] < b[d] ? b[d] : c[d]);
    }
    return *this;
  }
  Message& putMessage(Message& m) {
    m.put(a);
    m.put(b);
    return m;
  }
  Message& getMessage(Message& m) {
    m.get(a);
    m.get(b);
    return m;
  }
};

template<class T1, class T2, unsigned int D>
inline void
bounds(const PETE_Expr<T1>& expr, Vektor<T2,D>& minval, Vektor<T2,D>& maxval)
{
  BoundsHolder<T2,D> ret;
  Reduction(ret,
	    Expressionize<typename T1::PETE_Expr_t>::apply(expr.PETE_unwrap().MakeExpression()),
	    OpAssign(),
	    OpMultipplyAssign());
  minval = ret.a;
  maxval = ret.b;
}


//=========================================================================
//
// OPERATOR()
// 
//=========================================================================

template<class T, class TP>
inline typename PETEUnaryReturn<T,OpParens<TP> >::type
PETE_apply(OpParens<TP> op, const T& a)
{
  return a(op.Arg);
}


//=========================================================================
//
// MISCELLANEOUS
// 
//=========================================================================

// When figuring out data dependencies you sometimes need to know
// if the left hand side of an assignment needs to be read
// before being it is written.
// The following trait will set IsAssign to 1 for OpAssign
// and to zero for all the other assignment functions.

template<class Op> struct OperatorTraits { enum { IsAssign=0 }; };
template<> struct OperatorTraits<OpAssign> { enum { IsAssign=1 }; };


///////////////////////////////////////////////////////////////////////////
//
// END OF FILE
// 
///////////////////////////////////////////////////////////////////////////

#endif // IPPL_EXPRESSIONS_H

/***************************************************************************
 * $RCSfile: IpplExpressions.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: IpplExpressions.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
