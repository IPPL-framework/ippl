// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef	SYM_TENZOR_H
#define	SYM_TENZOR_H

// include files
#include "Message/Message.h"
#include "Utility/PAssert.h"
#include "PETE/IpplExpressions.h"
#include "AppTypes/TSVMeta.h"
#include "AppTypes/Tenzor.h"

#include <iostream>


//////////////////////////////////////////////////////////////////////
//
// Definition of class SymTenzor.
//
//////////////////////////////////////////////////////////////////////

//
//		| xOO x10 x20 |
//		| x10 x11 x21 |
//		| x20 x21 x22 |
//

template<class T, unsigned D>
class SymTenzor
{
public:

  typedef T Element_t;
  enum { ElemDim = 2 };
  enum { Size = D*(D+1)/2 };

  // Default Constructor 
  SymTenzor() {
    TSV_MetaAssignScalar< SymTenzor<T,D>,T,OpAssign>::apply(*this,T(0));
  } 

  // A noninitializing ctor.
  class DontInitialize {};
  SymTenzor(DontInitialize) {}

  // construct a SymTenzor from a single T
  SymTenzor(const T& x00) {
    TSV_MetaAssignScalar< SymTenzor<T,D>,T,OpAssign>::apply(*this,x00);
  }

  // construct a 2D SymTenzor
  SymTenzor(const T& x00, const T& x10, const T& x11) {
    PInsist(D==2, "Number of arguments does not match SymTenzor dimension!!");
    X[0] = x00; X[1] = x10; X[2] = x11; 
  }
  // construct a 3D SymTenzor
  SymTenzor(const T& x00, const T& x10, const T& x11, const T& x20,
            const T& x21, const T& x22) { 
    PInsist(D==3, "Number of arguments does not match SymTenzor dimension!!");
    X[0]= x00; X[1]= x10; X[2]= x11;
    X[3]= x20; X[4]= x21; X[5]= x22;
  }

  // Copy Constructor 
  SymTenzor(const SymTenzor<T,D> &rhs) {
    TSV_MetaAssign< SymTenzor<T,D> , SymTenzor<T,D> ,OpAssign > :: 
      apply(*this,rhs);
  }

  // Construct from a Tenzor.
  // Extract the symmetric part.
  SymTenzor(const Tenzor<T,D>& t) {
    for (unsigned int i=0; i<D; ++i) {
      (*this)(i,i) = t(i,i);
      for (unsigned int j=i+1; j<D; ++j)
	(*this)(i,j) = (t(i,j)+t(j,i))*0.5;
    }
  }

  // Dtor doesn't need to do anything.
  ~SymTenzor() { };

  // assignment operators
  const SymTenzor<T,D>& operator= (const SymTenzor<T,D> &rhs) {
    TSV_MetaAssign< SymTenzor<T,D> , SymTenzor<T,D> ,OpAssign > :: 
      apply(*this,rhs);
    return *this;
  }
  template<class T1>
  const SymTenzor<T,D>& operator= (const SymTenzor<T1,D> &rhs) {
    TSV_MetaAssign< SymTenzor<T,D> , SymTenzor<T1,D> ,OpAssign > :: 
      apply(*this,rhs);
    return *this;
  }
  const SymTenzor<T,D>& operator= (const T& rhs) {
    TSV_MetaAssignScalar< SymTenzor<T,D> , T ,OpAssign > :: apply(*this,rhs);
    return *this;
  }
  const SymTenzor<T,D>& operator= (const Tenzor<T,D> &rhs) {
    for (unsigned int i=0; i<D; ++i) {
      (*this)(i,i) = rhs(i,i);
      for (unsigned int j=i+1; j<D; ++j)
	(*this)(i,j) = (rhs(i,j)+rhs(j,i))*0.5;
    }
    return *this;
  }

  // accumulation operators
  template<class T1>
  SymTenzor<T,D>& operator+=(const SymTenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< SymTenzor<T,D> , SymTenzor<T1,D> , OpAddAssign > :: 
      apply(*this,rhs);
    return *this;
  }
  SymTenzor<T,D>& operator+=(const T& rhs)
  {
    TSV_MetaAssignScalar< SymTenzor<T,D> , T , OpAddAssign > :: 
      apply(*this,rhs);
    return *this;
  }

  template<class T1>
  SymTenzor<T,D>& operator-=(const SymTenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< SymTenzor<T,D> , SymTenzor<T1,D> , OpSubtractAssign > :: 
      apply(*this,rhs);
    return *this;
  }
  SymTenzor<T,D>& operator-=(const T& rhs)
  {
    TSV_MetaAssignScalar< SymTenzor<T,D> , T , OpSubtractAssign > :: 
      apply(*this,rhs);
    return *this;
  }

  template<class T1>
  SymTenzor<T,D>& operator*=(const SymTenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< SymTenzor<T,D> , SymTenzor<T1,D> , OpMultipplyAssign > :: 
      apply(*this,rhs);
    return *this;
  }
  SymTenzor<T,D>& operator*=(const T& rhs)
  {
    TSV_MetaAssignScalar< SymTenzor<T,D> , T , OpMultipplyAssign > :: 
      apply(*this,rhs);
    return *this;
  }

  template<class T1>
  SymTenzor<T,D>& operator/=(const SymTenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< SymTenzor<T,D> , SymTenzor<T1,D> , OpDivideAssign > :: 
      apply(*this,rhs);
    return *this;
  }
  SymTenzor<T,D>& operator/=(const T& rhs)
  {
    TSV_MetaAssignScalar< SymTenzor<T,D> , T , OpDivideAssign > :: 
      apply(*this,rhs);
    return *this;
  }

  // Methods

  void diagonal(const T& rhs) {
    for (unsigned int i = 0 ; i < D ; i++ ) {
      X[((i+1)*i/2) + i] = rhs;
    }
  }

  int len(void)  const { return Size; }
  int size(void) const { return sizeof(*this); }
  int get_Size(void) const { return Size; }

  // Operators
  
  Element_t operator()(unsigned int i, unsigned int j) const {
    int lo = i < j ? i : j;
    int hi = i > j ? i : j;
    return X[((hi+1)*hi/2) + lo];
  }

  Element_t& operator()(unsigned int i, unsigned int j) {
    int lo = i < j ? i : j;
    int hi = i > j ? i : j;
    return X[((hi+1)*hi/2) + lo];
  }

  Element_t& operator()(std::pair<int,int> a) {
    int i = a.first;
    int j = a.second;
    int lo = i < j ? i : j;
    int hi = i > j ? i : j;
    return X[((hi+1)*hi/2) + lo];
  }

  Element_t operator()( std::pair<int,int> a) const {
    int i = a.first;
    int j = a.second;
    int lo = i < j ? i : j;
    int hi = i > j ? i : j;
    return X[((hi+1)*hi/2) + lo];
  }

  Element_t HL(unsigned int hi, unsigned int lo) const {
    PAssert_GE( hi, lo );
    PAssert_LT( hi, D );
    return X[hi*(hi+1)/2 + lo];
  }
  Element_t& HL(unsigned int hi, unsigned int lo) {
    PAssert_GE( hi, lo );
    PAssert_LT( hi, D );
    return X[hi*(hi+1)/2 + lo];
  }

  Element_t& operator[](unsigned int i) { 
    PAssert_LT(i, Size);
    return X[i];
  }

  Element_t operator[](unsigned int i) const { 
    PAssert_LT(i, Size);
    return X[i];
  }

  //TJW: add these 12/16/97 to help with NegReflectAndZeroFace BC:
  // These are the same as operator[] but with () instead:

  Element_t& operator()(unsigned int i) { 
    PAssert (i < Size);
    return X[i];
  }

  Element_t operator()(unsigned int i) const { 
    PAssert (i < Size);
    return X[i];
  }
  //TJW.

  //----------------------------------------------------------------------
  // Comparison operators.
  bool operator==(const SymTenzor<T,D>& that) const {
    return TSV_MetaCompareArrays<T,T,D*(D+1)/2>::apply(X,that.X);
  }
  bool operator!=(const SymTenzor<T,D>& that) const {
    return !(*this == that);
  }

  //----------------------------------------------------------------------
  // parallel communication
  Message& putMessage(Message& m) const {
    m.setCopy(true);
    ::putMessage(m, X, X + ((D*(D + 1)/2)));
    return m;
  }

  Message& getMessage(Message& m) {
    ::getMessage(m, X, X + ((D*(D + 1)/2)));
    return m;
  }

private:

  // The elements themselves.
  T X[Size];

};


//////////////////////////////////////////////////////////////////////
//
// Free functions
//
//////////////////////////////////////////////////////////////////////

template <class T, unsigned D>
inline T trace(const SymTenzor<T,D> &rhs) {
  T result = 0.0;
  for (unsigned int i = 0 ; i < D ; i++ ) 
    result += rhs(i,i);
  return result;
}

template <class T, unsigned D>
inline SymTenzor<T,D> transpose(const SymTenzor<T,D> &rhs) {
  return rhs;
}

// Determinant: only implement for 1D, 2D, 3D:
template <class T, unsigned D>
inline T det(const SymTenzor<T,D>& /*rhs*/) {
  PInsist(D<3, "Tenzor det() function not implemented for D>3!");
  return T(-999999.999999);
}

template <class T>
inline T det(const SymTenzor<T,3>& rhs) {
  T result;
  result = 
    rhs(0,0)*(rhs(1,1)*rhs(2,2) - rhs(1,2)*rhs(2,1)) +
    rhs(0,1)*(rhs(1,2)*rhs(2,0) - rhs(1,0)*rhs(2,2)) +
    rhs(0,2)*(rhs(1,0)*rhs(2,1) - rhs(1,1)*rhs(2,0));
  return result;
}

template <class T>
inline T det(const SymTenzor<T,2>& rhs) {
  T result;
  result = rhs(0,0)*rhs(1,1) - rhs(0,1)*rhs(1,0);
  return result;
}

template <class T>
inline T det(const SymTenzor<T,1>& rhs) {
  T result = rhs(0,0);
  return result;
}

// cofactors() - pow(-1, i+j)*M(i,j), where M(i,j) is a minor of the tensor.
// See, for example, Arfken, Mathematical Methods for Physicists, 2nd Edition,
// p. 157 (the section where the determinant of a matrix is defined).

// Only implement for 1D, 2D, 3D:

template <class T, unsigned D>
inline Tenzor<T,D> cofactors(const SymTenzor<T,D>& /*rhs*/) {
  PInsist(D<4, "SymTenzor cofactors() function not implemented for D>3!");
  return Tenzor<T,D>(-999999.999999);
}

template <class T>
inline Tenzor<T,3> cofactors(const SymTenzor<T,3>& rhs) {
  Tenzor<T,3> result = typename Tenzor<T,3>::DontInitialize();

  result(0,0) = rhs(1,1)*rhs(2,2) - rhs(1,2)*rhs(2,1);
  result(1,0) = rhs(0,2)*rhs(2,1) - rhs(0,1)*rhs(2,2);
  result(2,0) = rhs(0,1)*rhs(1,2) - rhs(1,1)*rhs(0,2);
  result(0,1) = rhs(2,0)*rhs(1,2) - rhs(1,0)*rhs(2,2);
  result(1,1) = rhs(0,0)*rhs(2,2) - rhs(0,2)*rhs(2,0);
  result(2,1) = rhs(1,0)*rhs(0,2) - rhs(0,0)*rhs(1,2);
  result(0,2) = rhs(1,0)*rhs(2,1) - rhs(2,0)*rhs(1,1);
  result(1,2) = rhs(0,1)*rhs(2,0) - rhs(0,0)*rhs(2,1);
  result(2,2) = rhs(0,0)*rhs(1,1) - rhs(1,0)*rhs(0,1);
  return result;
}

template <class T>
inline Tenzor<T,2> cofactors(const SymTenzor<T,2>& rhs) {
  Tenzor<T,2> result = typename Tenzor<T,2>::DontInitialize();

  result(0,0) =  rhs(1,1);
  result(1,0) = -rhs(0,1);
  result(0,1) = -rhs(1,0);
  result(1,1) =  rhs(0,0);
  return result;
}

// For D=1, cofactor is the unit tensor, because det = single tensor element
// value:
template <class T>
inline Tenzor<T,1> cofactors(const SymTenzor<T,1>& /*rhs*/) {
  Tenzor<T,1> result = Tenzor<T,1>(1);
  return result;
}


//////////////////////////////////////////////////////////////////////
//
// Unary Operators
//
//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------
// unary operator-
template<class T, unsigned D>
inline SymTenzor<T,D> operator-(const SymTenzor<T,D> &op)
{
  return TSV_MetaUnary< SymTenzor<T,D> , OpUnaryMinus > :: apply(op);
}

//----------------------------------------------------------------------
// unary operator+
template<class T, unsigned D>
inline const SymTenzor<T,D> &operator+(const SymTenzor<T,D> &op)
{
  return op;
}

//////////////////////////////////////////////////////////////////////
//
// Binary Operators
//
//////////////////////////////////////////////////////////////////////

//
// Elementwise operators.
//

TSV_ELEMENTWISE_OPERATOR(SymTenzor,operator+,OpAdd)
TSV_ELEMENTWISE_OPERATOR(SymTenzor,operator-,OpSubtract)
TSV_ELEMENTWISE_OPERATOR(SymTenzor,operator*,OpMultipply)
TSV_ELEMENTWISE_OPERATOR(SymTenzor,operator/,OpDivide)
TSV_ELEMENTWISE_OPERATOR(SymTenzor,Min,FnMin)
TSV_ELEMENTWISE_OPERATOR(SymTenzor,Max,FnMax)

TSV_ELEMENTWISE_OPERATOR2(SymTenzor,Tenzor,operator+,OpAdd)
TSV_ELEMENTWISE_OPERATOR2(Tenzor,SymTenzor,operator+,OpAdd)
TSV_ELEMENTWISE_OPERATOR2(SymTenzor,Tenzor,operator-,OpSubtract)
TSV_ELEMENTWISE_OPERATOR2(Tenzor,SymTenzor,operator-,OpSubtract)
TSV_ELEMENTWISE_OPERATOR2(SymTenzor,Tenzor,operator*,OpMultipply)
TSV_ELEMENTWISE_OPERATOR2(Tenzor,SymTenzor,operator*,OpMultipply)
TSV_ELEMENTWISE_OPERATOR2(SymTenzor,Tenzor,operator/,OpDivide)
TSV_ELEMENTWISE_OPERATOR2(Tenzor,SymTenzor,operator/,OpDivide)

//----------------------------------------------------------------------
// dot products.
//----------------------------------------------------------------------

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const SymTenzor<T1,D> &lhs, const SymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< SymTenzor<T1,D> , SymTenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const SymTenzor<T1,D> &lhs, const Tenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< SymTenzor<T1,D> , Tenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const Tenzor<T1,D> &lhs, const SymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< Tenzor<T1,D> , SymTenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const Vektor<T1,D> &lhs, const SymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< Vektor<T1,D> , SymTenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const SymTenzor<T1,D> &lhs, const Vektor<T2,D> &rhs) 
{
  return TSV_MetaDot< SymTenzor<T1,D> , Vektor<T2,D> > :: apply(lhs,rhs);
}

//----------------------------------------------------------------------
// double dot products.
//----------------------------------------------------------------------

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const SymTenzor<T1,D> &lhs, const SymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDotDot< SymTenzor<T1,D> , SymTenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const SymTenzor<T1,D> &lhs, const Tenzor<T2,D> &rhs) 
{
  return TSV_MetaDotDot< SymTenzor<T1,D> , Tenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const Tenzor<T1,D> &lhs, const SymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDotDot< Tenzor<T1,D> , SymTenzor<T2,D> > :: apply(lhs,rhs);
}

//----------------------------------------------------------------------
// I/O
template<class T, unsigned D>
inline std::ostream& operator<<(std::ostream& out, const SymTenzor<T,D>& rhs) {
  if (D >= 1) {
    for (unsigned int i=0; i<D; i++) {
      out << "(";
      for (unsigned int j=0; j<D-1; j++) {
	out << rhs(i,j) << " , ";
      }
      out << rhs(i,D-1) << ")";
      // I removed this. --TJW: if (i < D - 1) out << endl;
    }
  } else {
    out << "( " << rhs(0,0) << " )";
  }
  return out;
}

#endif // SYM_TENZOR_H

/***************************************************************************
 * $RCSfile: SymTenzor.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: SymTenzor.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/

