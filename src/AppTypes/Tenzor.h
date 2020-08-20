// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef	TENZOR_H
#define	TENZOR_H

// include files
#include "Utility/PAssert.h"
#include "Message/Message.h"
#include "PETE/IpplExpressions.h"
#include "AppTypes/TSVMeta.h"

#include <iostream>

// forward declarations
template <class T, unsigned D> class SymTenzor;
template <class T, unsigned D> class AntiSymTenzor;


//////////////////////////////////////////////////////////////////////
//
// Definition of class Tenzor.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned D>
class Tenzor
{
public:

  typedef T Element_t;
  enum { ElemDim = 2 };
  enum { Size = D*D };

  // Default Constructor
  Tenzor() {
    TSV_MetaAssignScalar<Tenzor<T,D>,T,OpAssign>::apply(*this,T(0));
  }

  // A noninitializing ctor.
  class DontInitialize {};
  Tenzor(DontInitialize) {}

  // Copy Constructor
  Tenzor(const Tenzor<T,D> &rhs) {
    TSV_MetaAssign< Tenzor<T,D> , Tenzor<T,D> ,OpAssign >::apply(*this,rhs);
  }

  // constructor from a single T
  Tenzor(const T& x00) {
    TSV_MetaAssignScalar< Tenzor<T,D> , T ,OpAssign >::apply(*this,x00);
  }

  // constructors for fixed dimension
  Tenzor(const T& x00, const T& x10, const T& x01, const T& x11) {
    PInsist(D==2, "Number of arguments does not match Tenzor dimension!!");
    X[0] = x00;
    X[1] = x10;
    X[2] = x01;
    X[3] = x11;
  }
  Tenzor(const T& x00, const T& x10, const T& x20, const T& x01, const T& x11,
         const T& x21, const T& x02, const T& x12, const T& x22) {
    PInsist(D==3, "Number of arguments does not match Tenzor dimension!!");
    X[0] = x00;
    X[1] = x10;
    X[2] = x20;
    X[3] = x01;
    X[4] = x11;
    X[5] = x21;
    X[6] = x02;
    X[7] = x12;
    X[8] = x22;
  }

  // constructor from SymTenzor
  Tenzor(const SymTenzor<T,D>&);

  // constructor from AntiSymTenzor
  Tenzor(const AntiSymTenzor<T,D>&);

  // destructor
  ~Tenzor() { };

  // assignment operators
  const Tenzor<T,D>& operator= (const Tenzor<T,D> &rhs) {
    TSV_MetaAssign< Tenzor<T,D> , Tenzor<T,D> ,OpAssign> :: apply(*this,rhs);
    return *this;
  }
  template<class T1>
  const Tenzor<T,D>& operator= (const Tenzor<T1,D> &rhs) {
    TSV_MetaAssign< Tenzor<T,D> , Tenzor<T1,D> ,OpAssign> :: apply(*this,rhs);
    return *this;
  }
  const Tenzor<T,D>& operator= (const T& rhs) {
    TSV_MetaAssignScalar< Tenzor<T,D> , T ,OpAssign> :: apply(*this,rhs);
    return *this;
  }

  // accumulation operators
  template<class T1>
  Tenzor<T,D>& operator+=(const Tenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< Tenzor<T,D> , Tenzor<T1,D> , OpAddAssign > ::
      apply(*this,rhs);
    return *this;
  }
  Tenzor<T,D>& operator+=(const T& rhs)
  {
    TSV_MetaAssignScalar< Tenzor<T,D> , T , OpAddAssign > ::
      apply(*this,rhs);
    return *this;
  }

  template<class T1>
  Tenzor<T,D>& operator-=(const Tenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< Tenzor<T,D> , Tenzor<T1,D> , OpSubtractAssign > ::
      apply(*this,rhs);
    return *this;
  }
  Tenzor<T,D>& operator-=(const T& rhs)
  {
    TSV_MetaAssignScalar< Tenzor<T,D> , T , OpSubtractAssign > ::
      apply(*this,rhs);
    return *this;
  }

  template<class T1>
  Tenzor<T,D>& operator*=(const Tenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< Tenzor<T,D> , Tenzor<T1,D> , OpMultipplyAssign > ::
      apply(*this,rhs);
    return *this;
  }
  Tenzor<T,D>& operator*=(const T& rhs)
  {
    TSV_MetaAssignScalar< Tenzor<T,D> , T , OpMultipplyAssign > ::
      apply(*this,rhs);
    return *this;
  }

  template<class T1>
  Tenzor<T,D>& operator/=(const Tenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< Tenzor<T,D> , Tenzor<T1,D> , OpDivideAssign > ::
      apply(*this,rhs);
    return *this;
  }
  Tenzor<T,D>& operator/=(const T& rhs)
  {
    TSV_MetaAssignScalar< Tenzor<T,D> , T , OpDivideAssign > ::
      apply(*this,rhs);
    return *this;
  }

  // Methods

  void diagonal(const T& rhs) {
    for (unsigned int i = 0 ; i < D ; i++ )
      (*this)(i,i) = rhs;
  }

  int len(void)  const { return Size; }
  int size(void) const { return sizeof(*this); }

  // Operators

  Element_t &operator[]( unsigned int i )
  {
    PAssert_LT(i, Size);
    return X[i];
  }

  Element_t operator[]( unsigned int i ) const
  {
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

  Element_t operator()( unsigned int i,  unsigned int j ) const {
    PAssert ( (i<D) && (j<D) );
    return X[i*D+j];
  }

  Element_t& operator()( unsigned int i, unsigned int j ) {
    PAssert ( (i<D) && (j<D) );
    return X[i*D+j];
  }

  Element_t operator()(const std::pair<int,int> i) const {
    PAssert ( (i.first>=0) && (i.second>=0) && (i.first<(int)D) && (i.second<(int)D) );
    return (*this)(i.first,i.second);
  }

  Element_t& operator()(const std::pair<int,int> i) {
    PAssert ( (i.first>=0) && (i.second>=0) && (i.first<(int)D) && (i.second<(int)D) );
    return (*this)(i.first,i.second);
  }


  //----------------------------------------------------------------------
  // Comparison operators.
  bool operator==(const Tenzor<T,D>& that) const {
    return TSV_MetaCompareArrays<T,T,D*D>::apply(X,that.X);
  }
  bool operator!=(const Tenzor<T,D>& that) const {
    return !(*this == that);
  }

  //----------------------------------------------------------------------
  // parallel communication
  Message& putMessage(Message& m) const {
    m.setCopy(true);
    const T *p = X;
    ::putMessage(m, p, p + D*D);
    return m;
  }

  Message& getMessage(Message& m) {
    T *p = X;
    ::getMessage(m, p, p + D*D);
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

// trace() - sum of diagonal elements.

template <class T, unsigned D>
inline T trace(const Tenzor<T,D>& rhs) {
  T result = 0.0;
  for (unsigned int i = 0 ; i < D ; i++ )
    result += rhs(i,i);
  return result;
}

// transpose(). transpose(i,j) = input(j,i).

template <class T, unsigned D>
inline Tenzor<T,D> transpose(const Tenzor<T,D>& rhs) {
  Tenzor<T,D> result = typename Tenzor<T,D>::DontInitialize();

  for (unsigned int j = 0 ; j < D ; j++ )
    for (unsigned int i = 0 ; i < D ; i++ )
      result(i,j) = rhs(j,i);
  return result;
}

// Determinant: only implement for 1D, 2D, 3D:

template <class T, unsigned D>
inline T det(const Tenzor<T,D>& /*rhs*/) {
  PInsist(D<4, "Tenzor det() function not implemented for D>3!");
  return T(-999999.999999);
}

template <class T>
inline T det(const Tenzor<T,3>& rhs) {
  T result;
  result =
    rhs(0,0)*(rhs(1,1)*rhs(2,2) - rhs(1,2)*rhs(2,1)) +
    rhs(0,1)*(rhs(1,2)*rhs(2,0) - rhs(1,0)*rhs(2,2)) +
    rhs(0,2)*(rhs(1,0)*rhs(2,1) - rhs(1,1)*rhs(2,0));
  return result;
}

template <class T>
inline T det(const Tenzor<T,2>& rhs) {
  T result;
  result = rhs(0,0)*rhs(1,1) - rhs(0,1)*rhs(1,0);
  return result;
}

template <class T>
inline T det(const Tenzor<T,1>& rhs) {
  T result = rhs(0,0);
  return result;
}

// cofactors() - pow(-1, i+j)*M(i,j), where M(i,j) is a minor of the tensor.
// See, for example, Arfken, Mathematical Methods for Physicists, 2nd Edition,
// p. 157 (the section where the determinant of a matrix is defined).

// Only implement for 1D, 2D, 3D:

template <class T, unsigned D>
inline Tenzor<T,D> cofactors(const Tenzor<T,D>& /*rhs*/) {
  PInsist(D<4, "Tenzor cofactors() function not implemented for D>3!");
  return Tenzor<T,D>(-999999.999999);
}

template <class T>
inline Tenzor<T,3> cofactors(const Tenzor<T,3>& rhs) {
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
inline Tenzor<T,2> cofactors(const Tenzor<T,2>& rhs) {
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
inline Tenzor<T,1> cofactors(const Tenzor<T,1>& /*rhs*/) {
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
inline Tenzor<T,D> operator-(const Tenzor<T,D> &op)
{
  return TSV_MetaUnary< Tenzor<T,D> , OpUnaryMinus > :: apply(op);
}

//----------------------------------------------------------------------
// unary operator+
template<class T, unsigned D>
inline const Tenzor<T,D> &operator+(const Tenzor<T,D> &op)
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

TSV_ELEMENTWISE_OPERATOR(Tenzor,operator+,OpAdd)
TSV_ELEMENTWISE_OPERATOR(Tenzor,operator-,OpSubtract)
TSV_ELEMENTWISE_OPERATOR(Tenzor,operator*,OpMultipply)
TSV_ELEMENTWISE_OPERATOR(Tenzor,operator/,OpDivide)
TSV_ELEMENTWISE_OPERATOR(Tenzor,Min,FnMin)
TSV_ELEMENTWISE_OPERATOR(Tenzor,Max,FnMax)

//----------------------------------------------------------------------
// dot products.
//----------------------------------------------------------------------

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const Tenzor<T1,D> &lhs, const Tenzor<T2,D> &rhs)
{
  return TSV_MetaDot< Tenzor<T1,D> , Tenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const Vektor<T1,D> &lhs, const Tenzor<T2,D> &rhs)
{
  return TSV_MetaDot< Vektor<T1,D> , Tenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const Tenzor<T1,D> &lhs, const Vektor<T2,D> &rhs)
{
  return TSV_MetaDot< Tenzor<T1,D> , Vektor<T2,D> > :: apply(lhs,rhs);
}

//----------------------------------------------------------------------
// double dot product.
//----------------------------------------------------------------------

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const Tenzor<T1,D> &lhs, const Tenzor<T2,D> &rhs)
{
  return TSV_MetaDotDot< Tenzor<T1,D> , Tenzor<T2,D> > :: apply(lhs,rhs);
}

//----------------------------------------------------------------------
// Outer product.
//----------------------------------------------------------------------

template<class T1, class T2, unsigned int D>
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D >
outerProduct(const Vektor<T1,D>& v1, const Vektor<T2,D>& v2)
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  Tenzor<T0,D> ret = typename Tenzor<T0,D>::DontInitialize();

  for (unsigned int i=0; i<D; ++i)
    for (unsigned int j=0; j<D; ++j)
      ret(i,j) = v1[i]*v2[j];
  return ret;
}

//----------------------------------------------------------------------
// I/O
template<class T, unsigned D>
inline std::ostream& operator<<(std::ostream& out, const Tenzor<T,D>& rhs)
{
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

// include header files for SymTenzor and AntiSymTenzor in order
// to define constructors for Tenzor using these types
#include "AppTypes/SymTenzor.h"
#include "AppTypes/AntiSymTenzor.h"

template <class T, unsigned D>
Tenzor<T,D>::Tenzor(const SymTenzor<T,D>& rhs) {
  for (unsigned int i=0; i<D; ++i)
    for (unsigned int j=0; j<D; ++j)
      (*this)(i,j) = rhs(i,j);
}

template <class T, unsigned D>
Tenzor<T,D>::Tenzor(const AntiSymTenzor<T,D>& rhs) {
  for (unsigned int i=0; i<D; ++i)
    for (unsigned int j=0; j<D; ++j)
      (*this)(i,j) = rhs(i,j);
}


#endif // TENZOR_H

/***************************************************************************
 * $RCSfile: Tenzor.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: Tenzor.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $
 ***************************************************************************/
