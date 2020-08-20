// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef VEKTOR_H
#define VEKTOR_H

// include files
#include "Utility/PAssert.h"
#include "Message/Message.h"
#include "PETE/IpplExpressions.h"
#include "AppTypes/TSVMeta.h"

#include <iostream>
#include <iomanip>

//////////////////////////////////////////////////////////////////////
//
// Definition of class Vektor.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned D>
class Vektor
{
public:

  typedef T Element_t;
  enum { ElemDim = 1 };
  enum { Size = D };

  // Default Constructor initializes to zero.
  Vektor() {
    TSV_MetaAssignScalar<Vektor<T,D>,T,OpAssign>::apply(*this,T(0));
  }

  // Copy Constructor
  Vektor(const Vektor<T,D> &rhs) {
    TSV_MetaAssign< Vektor<T,D> , Vektor<T,D> ,OpAssign >::apply(*this,rhs);
  }

  // Templated Vektor constructor.
  template<class T1, unsigned D1>
  Vektor(const Vektor<T1,D1> &rhs) {
    for (unsigned d=0; d<D; ++d)
      X[d] = (d < D1) ? rhs[d] : T1(0);
  }

  // Constructor from a single T
  Vektor(const T& x00) {
    TSV_MetaAssignScalar<Vektor<T,D>,T,OpAssign>::apply(*this,x00);
  }

  // Constructors for fixed dimension
  Vektor(const T& x00, const T& x01) {
    PInsist(D==2, "Number of arguments does not match Vektor dimension!!");
    X[0] = x00;
    X[1] = x01;
  }

  Vektor(const T& x00, const T& x01, const T& x02) {
    PInsist(D==3, "Number of arguments does not match Vektor dimension!!");
    X[0] = x00;
    X[1] = x01;
    X[2] = x02;
  }

  Vektor(const T& x00, const T& x01, const T& x02, const T& x03) {
    PInsist(D==4, "Number of arguments does not match Vektor dimension!!");
    X[0] = x00;
    X[1] = x01;
    X[2] = x02;
    X[3] = x03;
  }

  // Destructor
  ~Vektor() { }

  // Assignment Operators
  const Vektor<T,D>& operator=(const Vektor<T,D> &rhs)
  {
    TSV_MetaAssign< Vektor<T,D> , Vektor<T,D> ,OpAssign> :: apply(*this,rhs);
    return *this;
  }
  template<class T1>
  const Vektor<T,D>& operator=(const Vektor<T1,D> &rhs)
  {
    TSV_MetaAssign< Vektor<T,D> , Vektor<T1,D> ,OpAssign> :: apply(*this,rhs);
    return *this;
  }
  const Vektor<T,D>& operator=(const T& rhs)
  {
    TSV_MetaAssignScalar< Vektor<T,D> , T ,OpAssign > :: apply(*this,rhs);
    return *this;
  }

  // Accumulation Operators
  template<class T1>
  Vektor<T,D>& operator+=(const Vektor<T1,D> &rhs)
  {
    TSV_MetaAssign< Vektor<T,D> , Vektor<T1,D> , OpAddAssign > :: apply(*this,rhs);
    return *this;
  }
  Vektor<T,D>& operator+=(const T& rhs)
  {
    TSV_MetaAssignScalar< Vektor<T,D> , T , OpAddAssign > :: apply(*this,rhs);
    return *this;
  }

  template<class T1>
  Vektor<T,D>& operator-=(const Vektor<T1,D> &rhs)
  {
    TSV_MetaAssign< Vektor<T,D> , Vektor<T1,D> , OpSubtractAssign > :: apply(*this,rhs);
    return *this;
  }
  Vektor<T,D>& operator-=(const T& rhs)
  {
    TSV_MetaAssignScalar< Vektor<T,D> , T , OpSubtractAssign > :: apply(*this,rhs);
    return *this;
  }

  template<class T1>
  Vektor<T,D>& operator*=(const Vektor<T1,D> &rhs)
  {
    TSV_MetaAssign< Vektor<T,D> , Vektor<T1,D> , OpMultipplyAssign > :: apply(*this,rhs);
    return *this;
  }
  Vektor<T,D>& operator*=(const T& rhs)
  {
    TSV_MetaAssignScalar< Vektor<T,D> , T , OpMultipplyAssign > :: apply(*this,rhs);
    return *this;
  }

  template<class T1>
  Vektor<T,D>& operator/=(const Vektor<T1,D> &rhs)
  {
    TSV_MetaAssign< Vektor<T,D> , Vektor<T1,D> , OpDivideAssign > ::
      apply(*this,rhs);
    return *this;
  }
  Vektor<T,D>& operator/=(const T& rhs)
  {
    TSV_MetaAssignScalar< Vektor<T,D> , T , OpDivideAssign > ::
      apply(*this,rhs);
    return *this;
  }

  // Get and Set Operations
  Element_t& operator[](unsigned int i);

  Element_t operator[](unsigned int i) const;

  Element_t& operator()(unsigned int i);

  Element_t operator()( unsigned int i) const;

  // Comparison operators.
  bool operator==(const Vektor<T,D>& that) const {
    return TSV_MetaCompareArrays<T,T,D>::apply(X,that.X);
  }
  bool operator!=(const Vektor<T,D>& that) const {
    return !(*this == that);
  }

  //----------------------------------------------------------------------
  // parallel communication
  Message& putMessage(Message& m) const {
    m.setCopy(true);
    ::putMessage(m, X, X + D);
    return m;
  }

  Message& getMessage(Message& m) {
    ::getMessage(m, X, X + D);
    return m;
  }

private:

  // Just store D elements of type T.
  T X[D];

};

template<class T, unsigned D>
typename Vektor<T,D>::Element_t& Vektor<T,D>::operator[](unsigned int i)
{
  PAssert (i<D);
  return X[i];
}

template<class T, unsigned D>
typename Vektor<T,D>::Element_t Vektor<T,D>::operator[](unsigned int i) const
{
  PAssert (i<D);
  return X[i];
}

template<class T, unsigned D>
typename Vektor<T,D>::Element_t& Vektor<T,D>::operator()(unsigned int i)
{
  PAssert (i<D);
  return X[i];
}

template<class T, unsigned D>
typename Vektor<T,D>::Element_t Vektor<T,D>::operator()( unsigned int i) const
{
  PAssert (i<D);
  return X[i];
}

//////////////////////////////////////////////////////////////////////
//
// Unary Operators
//
//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------
// unary operator-
template<class T, unsigned D>
inline Vektor<T,D> operator-(const Vektor<T,D> &op)
{
  return TSV_MetaUnary< Vektor<T,D> , OpUnaryMinus > :: apply(op);
}

//----------------------------------------------------------------------
// unary operator+
template<class T, unsigned D>
inline const Vektor<T,D> &operator+(const Vektor<T,D> &op)
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

TSV_ELEMENTWISE_OPERATOR(Vektor,operator+,OpAdd)
TSV_ELEMENTWISE_OPERATOR(Vektor,operator-,OpSubtract)
TSV_ELEMENTWISE_OPERATOR(Vektor,operator*,OpMultipply)
TSV_ELEMENTWISE_OPERATOR(Vektor,operator/,OpDivide)
TSV_ELEMENTWISE_OPERATOR(Vektor,Min,FnMin)
TSV_ELEMENTWISE_OPERATOR(Vektor,Max,FnMax)

//----------------------------------------------------------------------
// dot product
//----------------------------------------------------------------------

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dot(const Vektor<T1,D> &lhs, const Vektor<T2,D> &rhs)
{
  return TSV_MetaDot< Vektor<T1,D> , Vektor<T2,D> > :: apply(lhs,rhs);
}

//----------------------------------------------------------------------
// cross product
//----------------------------------------------------------------------

template < class T1, class T2, unsigned D >
inline Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
cross(const Vektor<T1,D> &lhs, const Vektor<T2,D> &rhs)
{
  return TSV_MetaCross< Vektor<T1,D> , Vektor<T2,D> > :: apply(lhs,rhs);
}

//----------------------------------------------------------------------
// I/O
template<class T, unsigned D>
inline std::ostream& operator<<(std::ostream& out, const Vektor<T,D>& rhs)
{
  std::streamsize sw = out.width();
  out << std::setw(1);
  if (D >= 1) {
    out << "( ";
    for (unsigned int i=0; i<D - 1; i++)
      out << std::setw(sw) << rhs[i] << " , ";
    out << std::setw(sw) << rhs[D - 1] << " )";
  } else {
    out << "( " << std::setw(sw) << rhs[0] << " )";
  }

  return out;
}

#endif // VEKTOR_H

/***************************************************************************
 * $RCSfile: Vektor.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: Vektor.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $
 ***************************************************************************/