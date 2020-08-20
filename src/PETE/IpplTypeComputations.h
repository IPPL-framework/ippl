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
//    IpplTypeComputations.h
//
// CREATED
//    July 11, 1997
//
// DESCRIPTION
//    PETE: Portable Expression Template Engine.
//
//    This header file contains IPPL-specific type computations.
//
///////////////////////////////////////////////////////////////////////////

#ifndef IPPL_TYPE_COMPUTATIONS_H
#define IPPL_TYPE_COMPUTATIONS_H


// include files
#include "PETE/TypeComputations.h"
#include <complex>


// forward declarations
template<class T, unsigned D> class Vektor;
template<class T, unsigned D> class Tenzor;
template<class T, unsigned D> class AntiSymTenzor;
template<class T, unsigned D> class SymTenzor;
template <class T> class RNGLattice;
class RNGXDiv;


// definition of global sign function
template<class T>
inline int sign(T a) { return ((a > 0) ? 1 : (a == 0 ? 0 : -1)); }


///////////////////////////////////////////////////////////////////////////
//
// PETE_Type2Index FOR USER TYPES
//
///////////////////////////////////////////////////////////////////////////

// Complex numbers.

template<> struct PETE_Type2Index<std::complex<double>> {
  enum { val = 8 };
};

// Return types for scalar ops with RNGs.

#define _SCALAR_RNG_OP_RETURNS_(GEN,SCA,OP)                             \
template <>                                                             \
struct PETEBinaryReturn<GEN,SCA,OP> {                                   \
  typedef PETEBinaryReturn<double,SCA,OP>::type type;                   \
};                                                                      \
template <>                                                             \
struct PETEBinaryReturn<SCA,GEN,OP> {                                   \
  typedef PETEBinaryReturn<SCA,double,OP>::type type;                   \
};

#define _SCALAR_RNG_RETURNS_(GEN,SCA)                                   \
_SCALAR_RNG_OP_RETURNS_(GEN,SCA,OpAdd)                                  \
_SCALAR_RNG_OP_RETURNS_(GEN,SCA,OpSubtract)                             \
_SCALAR_RNG_OP_RETURNS_(GEN,SCA,OpMultipply)                             \
_SCALAR_RNG_OP_RETURNS_(GEN,SCA,OpDivide)

#define _PETE_RNG_RETURNS_(GEN)                                         \
                                                                        \
template <> struct PETE_Type2Index< GEN > {                             \
  enum { val = PETE_Type2Index<double>::val };                          \
};                                                                      \
                                                                        \
_SCALAR_RNG_RETURNS_(GEN,short)                                         \
_SCALAR_RNG_RETURNS_(GEN,int)                                           \
_SCALAR_RNG_RETURNS_(GEN,long)                                          \
_SCALAR_RNG_RETURNS_(GEN,float)                                         \
_SCALAR_RNG_RETURNS_(GEN,double)                                        \
_SCALAR_RNG_RETURNS_(GEN,std::complex<double>)

_PETE_RNG_RETURNS_(RNGLattice<float>)
_PETE_RNG_RETURNS_(RNGLattice<double>)
_PETE_RNG_RETURNS_(RNGXDiv)

// Life is way easier with this feature.

template<class T, unsigned Dim>
struct PETE_Type2Index< Vektor<T, Dim> > {
  enum { val = 20 + 10 * Dim + PETE_Type2Index<T>::val };
};

template<class T, unsigned Dim>
struct PETE_Type2Index< SymTenzor<T, Dim> > {
  enum { val = 120 + 10 * Dim + PETE_Type2Index<T>::val };
};

template<class T, unsigned Dim>
struct PETE_Type2Index< Tenzor<T, Dim> > {
  enum { val = 220 + 10 * Dim + PETE_Type2Index<T>::val };
};

template<class T, unsigned Dim>
struct PETE_Type2Index< AntiSymTenzor<T, Dim> > {
  enum { val = 320 + 10 * Dim + PETE_Type2Index<T>::val };
};


///////////////////////////////////////////////////////////////////////////
//
// SPECIAL CASES FOR UNARY FUNCTIONS
//
///////////////////////////////////////////////////////////////////////////

// Abs function: special return for complex numbers.

struct FnAbs {
  enum { tag = PETE_UnaryPassThruTag };
};

template<> struct PETEUnaryReturn<std::complex<double>, FnAbs> {
  typedef double type;
};

// The conj, norm, arg, real, and imag functions for complex numbers.

struct FnConj {
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnNorm {
  typedef double type;
  enum { tag = PETE_Type2Index<double>::val };
};

template<> struct PETEUnaryReturn<std::complex<double>, FnNorm> {
  typedef double type;
};

struct FnArg {
  typedef double type;
  enum { tag = PETE_Type2Index<double>::val };
};

template<> struct PETEUnaryReturn<std::complex<double>, FnArg> {
  typedef double type;
};

struct FnReal {
  typedef double type;
  enum { tag = PETE_Type2Index<double>::val };
};

template<> struct PETEUnaryReturn<std::complex<double>, FnReal> {
  typedef double type;
};

struct FnImag {
  typedef double type;
  enum { tag = PETE_Type2Index<double>::val };
};

template<> struct PETEUnaryReturn<std::complex<double>, FnImag> {
  typedef double type;
};

// The sign function.

struct FnSign {
  typedef int type;
  enum { tag = PETE_Type2Index<int>::val };
};

template<class TP>
struct OpParens
{
  enum { tag = PETE_UnaryPassThruTag };
  TP Arg;
  OpParens() { Arg = TP(); }
  OpParens(const TP& a) : Arg(a) {}
};

// Tensor functions: trace, det (determinant),  and transpose

struct FnTrace {
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnDet {
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnTranspose {
  enum { tag = PETE_UnaryPassThruTag };
};

struct FnCofactors {
  enum { tag = PETE_UnaryPassThruTag };
};

// Life is pretty simple if we have partial specialization.

template<class T, unsigned Dim>
struct PETEUnaryReturn<Tenzor<T,Dim>, FnTrace> {
  typedef T type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<SymTenzor<T,Dim>, FnTrace> {
  typedef T type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<AntiSymTenzor<T,Dim>, FnTrace> {
  typedef T type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<Tenzor<T,Dim>, FnDet> {
  typedef T type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<SymTenzor<T,Dim>, FnDet> {
  typedef T type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<AntiSymTenzor<T,Dim>, FnDet> {
  typedef T type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<Tenzor<T,Dim>, FnTranspose> {
  typedef Tenzor<T,Dim> type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<SymTenzor<T,Dim>, FnTranspose> {
  typedef SymTenzor<T,Dim> type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<AntiSymTenzor<T,Dim>, FnTranspose> {
  typedef AntiSymTenzor<T,Dim> type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<Tenzor<T,Dim>, FnCofactors> {
  typedef Tenzor<T,Dim> type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<SymTenzor<T,Dim>, FnCofactors> {
  typedef SymTenzor<T,Dim> type;
};

template<class T, unsigned Dim>
struct PETEUnaryReturn<AntiSymTenzor<T,Dim>, FnCofactors> {
  typedef AntiSymTenzor<T,Dim> type;
};


///////////////////////////////////////////////////////////////////////////
//
// SPECIAL CASES FOR BINARY FUNCTIONS
//
///////////////////////////////////////////////////////////////////////////

// Min and Max functions.

struct FnMin {
  enum { tag = PETE_BinaryPromoteTag };
};

struct FnMax {
  enum { tag = PETE_BinaryPromoteTag };
};

// Dot, dot-dot, and outerProduct functions.

struct FnDot {
  enum { tag = PETE_BinaryPromoteTag };
};

struct FnDotDot {
  enum { tag = PETE_BinaryPromoteTag };
};

struct FnOuterProduct {
  enum { tag = PETE_BinaryPromoteTag };
};

// Cross-product:

struct FnCross {
  enum { tag = PETE_BinaryPromoteTag };
};

// Involving Vektors:

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Vektor<T1,Dim>,Vektor<T2,Dim>, FnCross> {
  typedef Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Vektor<T1,Dim>,Vektor<T2,Dim>, FnOuterProduct> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Vektor<T1,Dim>,Vektor<T2,Dim>, FnDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

// Involving Tenzors, but no combination with SymTenzors or AntiSymTenzors:

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,Tenzor<T2,Dim>,FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Vektor<T1,Dim>,Tenzor<T2,Dim>, FnDot> {
  typedef Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,Vektor<T2,Dim>, FnDot> {
  typedef Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,Tenzor<T2,Dim>,FnDotDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

// Involving SymTenzors, possibly combined with Tenzors:

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,SymTenzor<T2,Dim>, FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> 
    type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,SymTenzor<T2,Dim>, FnDotDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Vektor<T1,Dim>,SymTenzor<T2,Dim>, FnDot> {
  typedef Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,Vektor<T2,Dim>, FnDot> {
  typedef Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,SymTenzor<T2,Dim>,OpAdd> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpAdd>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,SymTenzor<T2,Dim>,OpSubtract> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpSubtract>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,SymTenzor<T2,Dim>,OpMultipply> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,SymTenzor<T2,Dim>, FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,SymTenzor<T2,Dim>, FnDotDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,Tenzor<T2,Dim>,OpAdd> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpAdd>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,Tenzor<T2,Dim>,OpSubtract> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpSubtract>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,Tenzor<T2,Dim>,FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,Tenzor<T2,Dim>,FnDotDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

// Involving AntiSymTenzors, possibly combined with Tenzors or SymTenzors:

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,AntiSymTenzor<T2,Dim>, FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> 
    type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,AntiSymTenzor<T2,Dim>, FnDotDot>
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Vektor<T1,Dim>,AntiSymTenzor<T2,Dim>, FnDot> {
  typedef Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,Vektor<T2,Dim>, FnDot> {
  typedef Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,AntiSymTenzor<T2,Dim>,OpAdd> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpAdd>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,AntiSymTenzor<T2,Dim>,OpSubtract> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpSubtract>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,AntiSymTenzor<T2,Dim>,OpMultipply> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,AntiSymTenzor<T2,Dim>, FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<Tenzor<T1,Dim>,AntiSymTenzor<T2,Dim>, FnDotDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,Tenzor<T2,Dim>,OpAdd> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpAdd>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,Tenzor<T2,Dim>,OpSubtract> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpSubtract>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,Tenzor<T2,Dim>,FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,Tenzor<T2,Dim>,FnDotDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,AntiSymTenzor<T2,Dim>,OpAdd> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpAdd>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,AntiSymTenzor<T2,Dim>,OpSubtract> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpSubtract>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,AntiSymTenzor<T2,Dim>,OpMultipply> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,AntiSymTenzor<T2,Dim>, FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<SymTenzor<T1,Dim>,AntiSymTenzor<T2,Dim>, FnDotDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,SymTenzor<T2,Dim>,OpAdd> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpAdd>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,SymTenzor<T2,Dim>,OpSubtract> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpSubtract>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,SymTenzor<T2,Dim>,FnDot> {
  typedef Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,Dim> type;
};

template<class T1, class T2, unsigned Dim>
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,SymTenzor<T2,Dim>,FnDotDot> {
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type type;
};

// Need to specify scalar operations directly.

#define _SCALAR_VST_RETURNS_(Sca)                                           \
template<class T1, unsigned Dim>                                            \
struct PETEBinaryReturn<Vektor<T1,Dim>,Sca,OpMultipply> {                    \
  typedef Vektor<typename PETEBinaryReturn<T1,Sca,OpMultipply>::type,Dim>    \
    type;                                                                   \
};                                                                          \
template<class T2, unsigned Dim>                                            \
struct PETEBinaryReturn<Sca,Vektor<T2,Dim>,OpMultipply> {                    \
  typedef Vektor<typename PETEBinaryReturn<Sca,T2,OpMultipply>::type,Dim>    \
    type;                                                                   \
};                                                                          \
template<class T1, unsigned Dim>                                            \
struct PETEBinaryReturn<Vektor<T1,Dim>,Sca,OpDivide> {                      \
  typedef Vektor<typename PETEBinaryReturn<T1,Sca,OpDivide>::type,Dim>      \
    type;                                                                   \
};                                                                          \
template<class T1, unsigned Dim>                                            \
struct PETEBinaryReturn<Tenzor<T1,Dim>,Sca,OpMultipply> {                    \
  typedef Tenzor<typename PETEBinaryReturn<T1,Sca,OpMultipply>::type,Dim>    \
    type;                                                                   \
};                                                                          \
template<class T2, unsigned Dim>                                            \
struct PETEBinaryReturn<Sca,Tenzor<T2,Dim>,OpMultipply> {                    \
  typedef Tenzor<typename PETEBinaryReturn<Sca,T2,OpMultipply>::type,Dim>    \
    type;                                                                   \
};                                                                          \
template<class T1, unsigned Dim>                                            \
struct PETEBinaryReturn<Tenzor<T1,Dim>,Sca,OpDivide> {                      \
  typedef Tenzor<typename PETEBinaryReturn<T1,Sca,OpDivide>::type,Dim>      \
    type;                                                                   \
};                                                                          \
template<class T1, unsigned Dim>                                            \
struct PETEBinaryReturn<SymTenzor<T1,Dim>,Sca,OpMultipply> {                 \
  typedef SymTenzor<typename PETEBinaryReturn<T1,Sca,OpMultipply>::type,Dim> \
    type;                                                                   \
};                                                                          \
template<class T2, unsigned Dim>                                            \
struct PETEBinaryReturn<Sca,SymTenzor<T2,Dim>,OpMultipply> {                 \
  typedef SymTenzor<typename PETEBinaryReturn<Sca,T2,OpMultipply>::type,Dim> \
    type;                                                                   \
};                                                                          \
template<class T1, unsigned Dim>                                            \
struct PETEBinaryReturn<SymTenzor<T1,Dim>,Sca,OpDivide> {                   \
  typedef SymTenzor<typename PETEBinaryReturn<T1,Sca,OpDivide>::type,Dim>   \
    type;                                                                   \
};                                                                          \
template<class T1, unsigned Dim>                                            \
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,Sca,OpMultipply> {             \
  typedef                                                                   \
  AntiSymTenzor<typename PETEBinaryReturn<T1,Sca,OpMultipply>::type,Dim>     \
    type;                                                                   \
};                                                                          \
template<class T2, unsigned Dim>                                            \
struct PETEBinaryReturn<Sca,AntiSymTenzor<T2,Dim>,OpMultipply> {             \
  typedef                                                                   \
  AntiSymTenzor<typename PETEBinaryReturn<Sca,T2,OpMultipply>::type,Dim>     \
    type;                                                                   \
};                                                                          \
template<class T1, unsigned Dim>                                            \
struct PETEBinaryReturn<AntiSymTenzor<T1,Dim>,Sca,OpDivide> {               \
  typedef                                                                   \
  AntiSymTenzor<typename PETEBinaryReturn<T1,Sca,OpDivide>::type,Dim>       \
    type;                                                                   \
};

_SCALAR_VST_RETURNS_(short)
_SCALAR_VST_RETURNS_(int)
_SCALAR_VST_RETURNS_(long)
_SCALAR_VST_RETURNS_(float)
_SCALAR_VST_RETURNS_(double)
_SCALAR_VST_RETURNS_(std::complex<double>)

#undef _SCALAR_VST_RETURNS_


///////////////////////////////////////////////////////////////////////////
//
// ASSIGNMENT OPERATORS: min=, max=, &&=, ||=
//
///////////////////////////////////////////////////////////////////////////

struct OpMinAssign {
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpMaxAssign {
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpAndAssign {
  enum { tag = PETE_BinaryUseLeftTag };
};

struct OpOrAssign {
  enum { tag = PETE_BinaryUseLeftTag };
};


///////////////////////////////////////////////////////////////////////////
//
// OPERATOR()
//
///////////////////////////////////////////////////////////////////////////

template<class T, class TP, unsigned Dim>
struct PETEUnaryReturn< Vektor<T, Dim>, OpParens<TP> > {
  typedef T type;
};

template<class T, class TP, unsigned Dim>
struct PETEUnaryReturn< AntiSymTenzor<T, Dim>, OpParens<TP> > {
  typedef T type;
};

template<class T, class TP, unsigned Dim>
struct PETEUnaryReturn< SymTenzor<T, Dim>, OpParens<TP> > {
  typedef T type;
};

template<class T, class TP, unsigned Dim>
struct PETEUnaryReturn< Tenzor<T, Dim>, OpParens<TP> > {
  typedef T type;
};

#endif // IPPL_TYPE_COMPUTATIONS_H

