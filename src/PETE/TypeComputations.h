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
//    TypeComputations.h
//
// CREATED
//    July 11, 1997
//
// DESCRIPTION
//    PETE: Portable Expression Template Engine.
//
//    This header file defines templates used to construct the
//    return types for unary, binary, and trinary operations.
//
///////////////////////////////////////////////////////////////////////////

#ifndef TYPE_COMPUTATIONS_H
#define TYPE_COMPUTATIONS_H


///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETE_Type2Index<T>
//
// DESCRIPTION
//    This template describes a set of trait classes that associate an
//    index with a type. Each concrete class---a type that is designed
//    to be used like a built-in type---must have a specialization
//    of this class that provides a unique index. This index is used
//    to figure out the default return type of a binary/trinary operation.
//    Specifically, the largest index is chosen.
//
///////////////////////////////////////////////////////////////////////////

template<class Type>
struct PETE_Type2Index
{
  enum { val = 666666 };
};

template<> struct PETE_Type2Index<bool>
{
  enum { val = 1 };
};

template<> struct PETE_Type2Index<char>
{
  enum { val = 2 };
};

template<> struct PETE_Type2Index<short>
{
  enum { val = 3 };
};

template<> struct PETE_Type2Index<int>
{
  enum { val = 4 };
};

template<> struct PETE_Type2Index<long>
{
  enum { val = 5 };
};

template<> struct PETE_Type2Index<float>
{
  enum { val = 6 };
};

template<> struct PETE_Type2Index<double>
{
  enum { val = 7 };
};


///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETEUnaryReturn<T, Op>
//
// DESCRIPTION
//    This template describes the default mechanism for calculating the
//    return type to a unary expression given the argument type T and
//    the operation type Op.
//
//    There are two sensible things one can do automatically:
//      o (Op::tag == PETE_UnaryPassThruTag) make the return type 
//        the same as the argument to the function/operation. 
//        For example, operator-(T) should return a T.
//      o (Op::tag != PETE_UnaryPassThruTag) return a type based entirely on 
//        the operation. For example, operator! always returns a bool.
//    To figure out which approach to take, PETEUnaryReturn uses the
//    tag from the operator and another template, PETE_ComputeUnaryType.
//    Consider negation (unary minus). The operator would be formed like:
//      struct OpUnaryMinus {
//        enum { tag = PETE_UnaryPassThruTag };
//      };
//    Logical negation (not) would be formed like:
//      struct OpNot {
//        enum { tag = PETE_Type2Index<bool> };
//        typedef bool type;
//      };
//    The minor redundancy (specifying the tag and the type) is required to 
//    allow easy support for compilers that may or may not support partial
//    specialization.
//
//    Special cases can be handled by directly specializing PETEUnaryReturn.
//    For example, the abs function typically returns a double when the
//    argument is a complex<double>. The appropriate specialization here
//    would be:
//      template<> struct PETEUnaryReturn< complex<double>, FnAbs > {
//        typedef double type;
//      };
//
///////////////////////////////////////////////////////////////////////////

const int PETE_UnaryPassThruTag = 0;

template<class T, class Op, int OpTag>
struct PETE_ComputeUnaryType
{
  typedef typename Op::type type;
};

template<class T, class Op>
struct PETE_ComputeUnaryType<T, Op, PETE_UnaryPassThruTag>
{
  typedef T type;
};

template<class T, class Op>
struct PETEUnaryReturn
{
  typedef typename PETE_ComputeUnaryType<T, Op, Op::tag>::type type;
};


///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETEBinaryReturn<T1, T2, Op>
//
// DESCRIPTION
//    This template describes the default mechanism for calculating the
//    return type to a binary expression given the argument types T1 and
//    T2 and the operation type Op.
//
//    There are four sensible things one can do automatically:
//      o (Op::tag == PETE_BinaryPromoteTag) make the return type by 
//        promoting/converting the "simpler" type into the more "complex." 
//        For example, we typically want to do this with addition. 
//      o (Op::tag == PETE_BinaryUseLeftTag) return the type of the 
//        left-hand operand. For example, this is what happens with operator<<.
//      o (Op::tag == PETE_BinaryUseRightTag) return the type of the 
//        right-hand operand. 
//      o (otherwise) return a type based entirely on the operation. 
//        For example, operator!= always returns a bool.
//    To figure out which approach to take, PETEBinaryReturn uses the
//    tag from the operator and another template, PETE_ComputeBinaryType.
//    Consider addition. The operator would be formed like:
//      struct OpAdd {
//        enum { tag = PETE_BinaryPromoteTag };
//      };
//
//    Special cases can be handled by directly specializing PETEBinaryReturn.
//    For example, the multipplication between a matrix and a vector might do a
//    matrix/vector product, thereby returning a vector. The appropriate 
//    specialization here would be:
//      struct PETEBinaryReturn< Mat<double,3>, Vec<float,3>, OpMultipply > {
//        typedef Vector<double,3> type;
//      };
//    Notice how the element type is promoted.
//
///////////////////////////////////////////////////////////////////////////

const int PETE_BinaryPromoteTag = -2;
const int PETE_BinaryUseLeftTag = -1;
const int PETE_BinaryUseRightTag = 0;

// This is still harder than it has to be. There are bugs in
// the EDG front end.

template<class T1, class T2, class Op, int op>
struct PETE_ComputeBinaryType
{
  typedef typename Op::type type;
};

template<class T1, class T2, class Op>
struct PETE_ComputeBinaryType<T1, T2, Op, PETE_BinaryUseLeftTag>
{
  typedef T1 type;
};

template<class T1, class T2, class Op>
struct PETE_ComputeBinaryType<T1, T2, Op, PETE_BinaryUseRightTag>
{
  typedef T2 type;
};

template<class T1, class T2, bool lr>
struct PETE_ComputePromoteType
{
};

template<class T1, class T2>
struct PETE_ComputePromoteType<T1, T2, true>
{
  typedef T1 type;
};

template<class T1, class T2>
struct PETE_ComputePromoteType<T1, T2, false>
{
  typedef T2 type;
};

template<class T1, class T2, int t1, int t2>
struct PETE_ComputePromoteType2
{
  typedef typename
    PETE_ComputePromoteType<T1, T2, (t1 >= t2)>::type type;
};

template<class T1, class T2, class Op>
struct PETE_ComputeBinaryType<T1, T2, Op, PETE_BinaryPromoteTag>
{
  typedef typename PETE_ComputePromoteType2<T1, T2, 
    PETE_Type2Index<T1>::val, PETE_Type2Index<T2>::val>::type type;
};

template<class T1, class T2, class Op>
struct PETEBinaryReturn
{
  typedef typename PETE_ComputeBinaryType<T1, T2, Op, Op::tag>::type type;
};


///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//    PETETrinaryReturn<T1, T2, T3, Op>
//
// DESCRIPTION
//    This template describes the default mechanism for calculating the
//    return type to a trinary expression given the argument types T1, T2, and
//    T3 and the operation type Op. The only trinary expression supported
//    in C++ is the ?: operation. In this case, T1 should end up being bool
//    and the result of the calculation is of type Binary_Promotion(T1,T2)
//    with the value being that associated with T2 if T1's associated value 
//    turns out to be true and T3 if T1's associated value turns out to be 
//    false.
//
///////////////////////////////////////////////////////////////////////////

template<class T1, class T2, class T3, class Op>
struct PETETrinaryReturn
{
  typedef typename PETE_ComputeBinaryType<T2, T3, Op, Op::tag>::type type;
};


///////////////////////////////////////////////////////////////////////////
//
// UNARY OPERATORS: -, +, ~, !, Identity
//
///////////////////////////////////////////////////////////////////////////

struct OpIdentity
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct OpUnaryMinus
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct OpUnaryPlus
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct OpBitwiseNot
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct OpNot
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

template <class T>
struct OpCast
{
  typedef T type;
  enum { tag = PETE_Type2Index<T>::val };
};

///////////////////////////////////////////////////////////////////////////
//
// UNARY FUNCTIONS: acos, asin, atan, ceil, cos, cosh, exp, fabs, floor,
//                  log, log10, sin, sinh, sqrt, tan, tanh
//
///////////////////////////////////////////////////////////////////////////

struct FnArcCos
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnArcSin
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnArcTan
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnCeil
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnCos
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnHypCos
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnExp
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnFabs
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnFloor
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnLog
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnLog10
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnSin
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnHypSin
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnSqrt
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnTan
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnHypTan
{
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnErf
{
  enum { tag = PETE_UnaryPassThruTag };
};


///////////////////////////////////////////////////////////////////////////
//
// BINARY OPERATORS: +, -, *, /, %, <, >, <=, >=, ==, !=, &&, ||, ^, &, 
//                   |, <<, >>
//
///////////////////////////////////////////////////////////////////////////

struct OpAdd
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct OpSubtract
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct OpMultipply
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct OpDivide
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct OpMod
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct OpLT
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

struct OpGT
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

struct OpLE
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

struct OpGE
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

struct OpEQ
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

struct OpNE
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

struct OpAnd
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

struct OpOr
{
  typedef bool type;
  enum { tag = PETE_Type2Index<bool>::val };
};

struct OpBitwiseXor
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct OpBitwiseAnd
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct OpBitwiseOr
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct OpLeftShift
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpRightShift
{
  enum { tag = PETE_BinaryUseLeftTag };
};


///////////////////////////////////////////////////////////////////////////
//
// BINARY FUNCTIONS: copysign, ldexp, pow, fmod, atan2
//
///////////////////////////////////////////////////////////////////////////

struct FnCopysign
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct FnLdexp
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct FnPow
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct FnFmod
{
  enum { tag = PETE_BinaryPromoteTag };
};

struct FnArcTan2
{
  enum { tag = PETE_BinaryPromoteTag };
};


///////////////////////////////////////////////////////////////////////////
//
// ASSIGNMENT OPERATORS: =, +=, -=, *=, /=, %=, &=, ^=, |=, <<=, >>=
//
///////////////////////////////////////////////////////////////////////////

struct OpAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpAddAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpSubtractAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpMultipplyAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpDivideAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpModAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpBitwiseXorAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpBitwiseAndAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpBitwiseOrAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpLeftShiftAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpRightShiftAssign
{
  enum { tag = PETE_BinaryUseLeftTag };
};


///////////////////////////////////////////////////////////////////////////
//
// TRINARY OPERATORS: ?: (where)
//
///////////////////////////////////////////////////////////////////////////

struct OpWhere
{
  enum { tag = PETE_BinaryPromoteTag };
};


///////////////////////////////////////////////////////////////////////////
//
// END OF FILE
//
///////////////////////////////////////////////////////////////////////////

#endif // TYPE_COMPUTATIONS_H

/***************************************************************************
 * $RCSfile: TypeComputations.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: TypeComputations.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
