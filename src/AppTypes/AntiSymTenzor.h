/***************************************************************************
 *
 * The IPPL Framework
 * 
 ***************************************************************************/

#ifndef	ANTI_SYM_TENZOR_H
#define	ANTI_SYM_TENZOR_H

// include files
#include "Utility/PAssert.h"
#include "Message/Message.h"
#include "PETE/IpplExpressions.h"
#include "AppTypes/TSVMeta.h"
#include "AppTypes/Tenzor.h"

#include <iostream>


//////////////////////////////////////////////////////////////////////
//
// Definition of class AntiSymTenzor.
//
//////////////////////////////////////////////////////////////////////

//
//		|  O -x10  -x20 |
//		| x10   0  -x21 |
//		| x20  x21   0  |
//

template<class T, unsigned D>
class AntiSymTenzor
{
public:

  typedef T Element_t;
  enum { ElemDim = 2 };
  enum { Size = D*(D-1)/2 };

  // Default Constructor 
  AntiSymTenzor() {
    TSV_MetaAssignScalar< AntiSymTenzor<T,D>,T,OpAssign>::apply(*this,T(0));
  } 

  // A noninitializing ctor.
  class DontInitialize {};
  AntiSymTenzor(DontInitialize) {}

  // Construct an AntiSymTenzor from a single T.
  // This doubles as the 2D AntiSymTenzor initializer.
  AntiSymTenzor(const T& x00) {
    TSV_MetaAssignScalar< AntiSymTenzor<T,D>,T,OpAssign>::apply(*this,x00);
  }
  // construct a 3D AntiSymTenzor
  AntiSymTenzor(const T& x10, const T& x20, const T& x21) { 
    PInsist(D==3,
            "Number of arguments does not match AntiSymTenzor dimension!!");
    X[0]= x10; X[1]= x20; X[2]= x21;
  }

  // Copy Constructor
  AntiSymTenzor(const AntiSymTenzor<T,D> &rhs) {
    TSV_MetaAssign< AntiSymTenzor<T,D> , AntiSymTenzor<T,D> ,OpAssign > :: 
      apply(*this,rhs);
  }

  // Construct from a Tenzor.
  // Extract the antisymmetric part.
  AntiSymTenzor( const Tenzor<T,D>& t ) {
    for (unsigned int i=1; i<D; ++i) {
      for (unsigned int j=0; j<i; ++j)
	(*this)[((i-1)*i/2)+j] = (t(i,j)-t(j,i))*0.5;
    }
  }

  ~AntiSymTenzor() { };

  // assignment operators
  const AntiSymTenzor<T,D>& operator= (const AntiSymTenzor<T,D> &rhs) {
    TSV_MetaAssign< AntiSymTenzor<T,D> , AntiSymTenzor<T,D> ,OpAssign > :: 
      apply(*this,rhs);
    return *this;
  }
  template<class T1>
  const AntiSymTenzor<T,D>& operator= (const AntiSymTenzor<T1,D> &rhs) {
    TSV_MetaAssign< AntiSymTenzor<T,D> , AntiSymTenzor<T1,D> ,OpAssign > :: 
      apply(*this,rhs);
    return *this;
  }
  const AntiSymTenzor<T,D>& operator= (const T& rhs) {
    TSV_MetaAssignScalar< AntiSymTenzor<T,D> , T ,OpAssign > :: 
      apply(*this,rhs);
    return *this;
  }

  // accumulation operators
  template<class T1>
  AntiSymTenzor<T,D>& operator+=(const AntiSymTenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< AntiSymTenzor<T,D> , AntiSymTenzor<T1,D> , OpAddAssign > 
      :: apply(*this,rhs);
    return *this;
  }

  template<class T1>
  AntiSymTenzor<T,D>& operator-=(const AntiSymTenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< AntiSymTenzor<T,D> , AntiSymTenzor<T1,D> , 
      OpSubtractAssign > :: apply(*this,rhs);
    return *this;
  }

  template<class T1>
  AntiSymTenzor<T,D>& operator*=(const AntiSymTenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< AntiSymTenzor<T,D> , AntiSymTenzor<T1,D> , 
      OpMultipplyAssign > :: apply(*this,rhs);
    return *this;
  }
  AntiSymTenzor<T,D>& operator*=(const T& rhs)
  {
    TSV_MetaAssignScalar< AntiSymTenzor<T,D> , T , OpMultipplyAssign > :: 
      apply(*this,rhs);
    return *this;
  }

  template<class T1>
  AntiSymTenzor<T,D>& operator/=(const AntiSymTenzor<T1,D> &rhs)
  {
    TSV_MetaAssign< AntiSymTenzor<T,D> , AntiSymTenzor<T1,D> , 
      OpDivideAssign > :: apply(*this,rhs);
    return *this;
  }
  AntiSymTenzor<T,D>& operator/=(const T& rhs)
  {
    TSV_MetaAssignScalar< AntiSymTenzor<T,D> , T , OpDivideAssign > :: 
      apply(*this,rhs);
    return *this;
  }

  // Methods

  int len(void)  const { return Size; }
  int size(void) const { return sizeof(*this); }
  int get_Size(void) const { return Size; }

  class AssignProxy {
  public:
    AssignProxy(Element_t &elem, int where)
      : elem_m(elem), where_m(where) { }
    AssignProxy(const AssignProxy &model)
      : elem_m(model.elem_m), where_m(model.where_m) { }
    const AssignProxy &operator=(const AssignProxy &a)
      {
        PAssert_EQ(where_m != 0 || (a.elem_m == -a.elem_m), true);
        elem_m = where_m < 0 ? -a.elem_m : a.elem_m;
        return *this;
      }
    const AssignProxy &operator=(const Element_t &e)
      {
        PAssert_EQ(where_m != 0 || (e == -e), true);
        elem_m = where_m < 0 ? -e : e;
        return *this;
      }

    operator Element_t() const
      {
	return (where_m < 0 ? -elem_m : elem_m);
      }

  private:

    Element_t &elem_m;
    int where_m;
  };

  // Operators
  
  Element_t operator()(unsigned int i, unsigned int j) const {
    if (i == j)
      return T(0.0);
    else if (i < j)
      return -X[((j-1)*j/2) + i];
    else
      return X[((i-1)*i/2) + j];
  }

  Element_t operator()( std::pair<int,int> a) const {
    int i = a.first;
    int j = a.second;
    return (*this)(i, j);
  }

  AssignProxy operator()(unsigned int i, unsigned int j) {
    if (i == j)
      return AssignProxy(AntiSymTenzor<T,D>::Zero, 0);
    else
      {
	int lo = i < j ? i : j;
	int hi = i > j ? i : j;
	return AssignProxy(X[((hi-1)*hi/2) + lo], i - j);
      }
  }

  AssignProxy operator()(std::pair<int,int> a) {
    int i = a.first;
    int j = a.second;
    return (*this)(i, j);
  }

  Element_t& operator[](unsigned int i) { 
    PAssert (i < Size);
    return X[i];
  }

  Element_t operator[](unsigned int i) const { 
    PAssert (i < Size);
    return X[i];
  }

  // These are the same as operator[] but with () instead:

  Element_t& operator()(unsigned int i) { 
    PAssert (i < Size);
    return X[i];
  }

  Element_t operator()(unsigned int i) const { 
    PAssert (i < Size);
    return X[i];
  }

  //----------------------------------------------------------------------
  // Comparison operators.
  bool operator==(const AntiSymTenzor<T,D>& that) const {
    return TSV_MetaCompareArrays<T,T,D*(D-1)/2>::apply(X,that.X);
  }
  bool operator!=(const AntiSymTenzor<T,D>& that) const {
    return !(*this == that);
  }

  //----------------------------------------------------------------------
  // parallel communication
  Message& putMessage(Message& m) const {
    m.setCopy(true);
    ::putMessage(m, X, X + Size);
    return m;
  }

  Message& getMessage(Message& m) {
    ::getMessage(m, X, X + Size);
    return m;
  }

private:

  friend class AssignProxy;

  // The elements themselves.
  T X[Size];

  // A place to store a zero element.
  // We need to return a reference to this
  // for the diagonal element.
  static T Zero;
};


// Assign the static zero element value:
template<class T, unsigned int D>
T AntiSymTenzor<T,D>::Zero = 0;



///////////////////////////////////////////////////////////////////////////
// Specialization for 1D  -- this is basically just the zero tensor
///////////////////////////////////////////////////////////////////////////

template <class T>
class AntiSymTenzor<T,1>
{
public:

  typedef T Element_t;
  enum { ElemDim = 2 };

  // Default Constructor 
  AntiSymTenzor() {} 

  // Copy Constructor 
  AntiSymTenzor(const AntiSymTenzor<T,1>&) {}

  // A noninitializing ctor.
  class DontInitialize {};
  AntiSymTenzor(DontInitialize) {}

  // Construct from a Tenzor: still a no-op here:
  AntiSymTenzor( const Tenzor<T,1>& /*t*/) { }

  ~AntiSymTenzor() {}

  // assignment operators
  const AntiSymTenzor<T,1>& operator=(const AntiSymTenzor<T,1>&) {
    return *this;
  }
  template <class T1>
  const AntiSymTenzor<T,1>& operator=(const AntiSymTenzor<T1,1>&) {
    return *this;
  }
  const AntiSymTenzor<T,1>& operator=(const T& rhs) {
    PInsist(rhs==0, "Cannot assign non-zero value to a 1D AntiSymTenzor!");
    return *this;
  }

  // accumulation operators
  template <class T1>
  AntiSymTenzor<T,1>& operator+=(const AntiSymTenzor<T1,1>&) {
    return *this;
  }

  template <class T1>
  AntiSymTenzor<T,1>& operator-=(const AntiSymTenzor<T1,1>&) {
    return *this;
  }

  template <class T1>
  AntiSymTenzor<T,1>& operator*=(const AntiSymTenzor<T1,1>&) {
    return *this;
  }
  AntiSymTenzor<T,1>& operator*=(const T&) {
    return *this;
  }

  template <class T1>
  AntiSymTenzor<T,1>& operator/=(const AntiSymTenzor<T1,1>&) {
    return *this;
  }
  AntiSymTenzor<T,1>& operator/=(const T&) {
    return *this;
  }

  // Methods

  int len(void)  const { return Size; }
  int size(void) const { return sizeof(*this); }
  int get_Size(void) const { return Size; }

  class AssignProxy {
  public:
    AssignProxy(Element_t &elem, int where)
      : elem_m(elem), where_m(where) {}
    AssignProxy(const AssignProxy& model)
      : elem_m(model.elem_m), where_m(model.where_m) {}
    const AssignProxy& operator=(const AssignProxy& a)
      {
        PAssert_EQ(where_m != 0 || (a.elem_m == -a.elem_m), true);
	elem_m = where_m < 0 ? -a.elem_m : a.elem_m;
	return *this;
      }
    const AssignProxy& operator=(const Element_t& e)
      {
        PAssert_EQ(where_m != 0 || (e == -e), true);
	elem_m = where_m < 0 ? -e : e;
	return *this;
      }

    operator Element_t() const
      {
	return (where_m < 0 ? -elem_m : elem_m);
      }

  private:

    Element_t &elem_m;
    int where_m;
  };

  // Operators
  
  Element_t operator()(unsigned int i, unsigned int j) const {
    PAssert_EQ(i, j);
    return T(0.0);
  }

  Element_t operator()( std::pair<int,int> a) const {
    int i = a.first;
    int j = a.second;
    return (*this)(i, j);
  }

  AssignProxy operator()(unsigned int i, unsigned int j) {
    PAssert_EQ(i, j);
    return AssignProxy(AntiSymTenzor<T,1>::Zero, 0);
  }

  AssignProxy operator()(std::pair<int,int> a) {
    int i = a.first;
    int j = a.second;
    return (*this)(i, j);
  }

  Element_t operator[](unsigned int i) const { 
    PAssert (i == 0);
    return AntiSymTenzor<T,1>::Zero;
  }

  // These are the same as operator[] but with () instead:

  Element_t operator()(unsigned int i) const { 
    PAssert (i == 0);
    return AntiSymTenzor<T,1>::Zero;
  }

  //----------------------------------------------------------------------
  // Comparison operators.
  bool operator==(const AntiSymTenzor<T,1>& /*that*/) const {
    return true;
  }
  bool operator!=(const AntiSymTenzor<T,1>& that) const {
    return !(*this == that);
  }

  //----------------------------------------------------------------------
  // parallel communication
  Message& putMessage(Message& m) const {
    m.setCopy(true);
    m.put(AntiSymTenzor<T,1>::Zero);
    return m;
  }

  Message& getMessage(Message& m) {
    T zero;
    m.get(zero);
    return m;
  }

private:

  friend class AssignProxy;

  // The number of elements.
  enum { Size = 0 };

  // A place to store a zero element.
  // We need to return a reference to this
  // for the diagonal element.
  static T Zero;
};


// Assign the static zero element value:
template<class T>
T AntiSymTenzor<T,1U>::Zero = 0;



//////////////////////////////////////////////////////////////////////
//
// Free functions
//
//////////////////////////////////////////////////////////////////////

template <class T, unsigned D>
inline T trace(const AntiSymTenzor<T,D>&) {
  return T(0.0);
}

template <class T, unsigned D>
inline AntiSymTenzor<T,D> transpose(const AntiSymTenzor<T,D>& rhs) {
  return -rhs;
}

// Determinant: only implement for 1D, 2D, 3D:

// For D=3, det is zero, because diagonal elements are zero:
template<class T>
inline T
det(const AntiSymTenzor<T,3>& /*t*/)
{
  return T(0.0);
}
// For D=2, det is nonzero; use linear indexing of only stored element:
template<class T>
inline T
det(const AntiSymTenzor<T,2>& t)
{
  T result;
  result = t(0)*t(0);
  return result;
}
// For D=1, det is zero, because diagonal elements are zero:
template<class T>
inline T
det(const AntiSymTenzor<T,1>& /*t*/)
{
  return T(0.0);
}

// cofactors() - pow(-1, i+j)*M(i,j), where M(i,j) is a minor of the tensor.
// See, for example, Arfken, Mathematical Methods for Physicists, 2nd Edition,
// p. 157 (the section where the determinant of a matrix is defined).

// Only implement for 1D, 2D, 3D:

template <class T, unsigned D>
inline Tenzor<T,D> cofactors(const AntiSymTenzor<T,D>& /*rhs*/) {
  PInsist(D<4, "AntiSymTenzor cofactors() function not implemented for D>3!");
  return Tenzor<T,D>(-999999.999999);
}

template <class T>
inline Tenzor<T,3> cofactors(const AntiSymTenzor<T,3>& rhs) {

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
inline Tenzor<T,2> cofactors(const AntiSymTenzor<T,2>& rhs) {

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
inline Tenzor<T,1> cofactors(const AntiSymTenzor<T,1>& /*rhs*/) {
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
inline AntiSymTenzor<T,D> operator-(const AntiSymTenzor<T,D> &op)
{
  return TSV_MetaUnary< AntiSymTenzor<T,D> , OpUnaryMinus > :: apply(op);
}

//----------------------------------------------------------------------
// unary operator+
template<class T, unsigned D>
inline const AntiSymTenzor<T,D> &operator+(const AntiSymTenzor<T,D> &op)
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

TSV_ELEMENTWISE_OPERATOR(AntiSymTenzor,operator+,OpAdd)
TSV_ELEMENTWISE_OPERATOR(AntiSymTenzor,operator-,OpSubtract)
TSV_ELEMENTWISE_OPERATOR(AntiSymTenzor,operator*,OpMultipply)
TSV_ELEMENTWISE_OPERATOR(AntiSymTenzor,operator/,OpDivide)
TSV_ELEMENTWISE_OPERATOR(AntiSymTenzor,Min,FnMin)
TSV_ELEMENTWISE_OPERATOR(AntiSymTenzor,Max,FnMax)

//----------------------------------------------------------------------
// dot products.
//----------------------------------------------------------------------

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const AntiSymTenzor<T1,D> &lhs, const AntiSymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< AntiSymTenzor<T1,D> , AntiSymTenzor<T2,D> > :: 
    apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const AntiSymTenzor<T1,D> &lhs, const Tenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< Tenzor<T1,D> , Tenzor<T2,D> > :: 
    apply(Tenzor<T1,D>(lhs),rhs);
}

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const Tenzor<T1,D> &lhs, const AntiSymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< Tenzor<T1,D> , Tenzor<T2,D> > :: 
    apply(lhs,Tenzor<T2,D>(rhs));
}

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const AntiSymTenzor<T1,D> &lhs, const SymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< Tenzor<T1,D> , Tenzor<T2,D> > :: 
    apply(Tenzor<T1,D>(lhs),Tenzor<T2,D>(rhs));
}

template < class T1, class T2, unsigned D >
inline Tenzor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const SymTenzor<T1,D> &lhs, const AntiSymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< Tenzor<T1,D> , Tenzor<T2,D> > :: 
    apply(Tenzor<T1,D>(lhs),Tenzor<T2,D>(rhs));
}

template < class T1, class T2, unsigned D >
inline Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const Vektor<T1,D> &lhs, const AntiSymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDot< Vektor<T1,D> , AntiSymTenzor<T2,D> > :: apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
dot(const AntiSymTenzor<T1,D> &lhs, const Vektor<T2,D> &rhs) 
{
  return TSV_MetaDot< AntiSymTenzor<T1,D> , Vektor<T2,D> > :: apply(lhs,rhs);
}

//----------------------------------------------------------------------
// double dot products.
//----------------------------------------------------------------------

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const AntiSymTenzor<T1,D> &lhs, const AntiSymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDotDot< AntiSymTenzor<T1,D> , AntiSymTenzor<T2,D> > :: 
    apply(lhs,rhs);
}

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const AntiSymTenzor<T1,D> &lhs, const Tenzor<T2,D> &rhs) 
{
  return TSV_MetaDotDot< Tenzor<T1,D> , Tenzor<T2,D> > :: 
    apply(Tenzor<T1,D>(lhs),rhs);
}

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const Tenzor<T1,D> &lhs, const AntiSymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDotDot< Tenzor<T1,D> , Tenzor<T2,D> > :: 
    apply(lhs,Tenzor<T2,D>(rhs));
}

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const AntiSymTenzor<T1,D> &lhs, const SymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDotDot< Tenzor<T1,D> , Tenzor<T2,D> > :: 
    apply(Tenzor<T1,D>(lhs),Tenzor<T2,D>(rhs));
}

template < class T1, class T2, unsigned D >
inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
dotdot(const SymTenzor<T1,D> &lhs, const AntiSymTenzor<T2,D> &rhs) 
{
  return TSV_MetaDotDot< Tenzor<T1,D> , Tenzor<T2,D> > :: 
    apply(Tenzor<T1,D>(lhs),Tenzor<T2,D>(rhs));
}

//----------------------------------------------------------------------
// I/O
template<class T, unsigned D>
inline std::ostream& operator<<(std::ostream& out, const AntiSymTenzor<T,D>& rhs) {
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

//////////////////////////////////////////////////////////////////////

#endif // ANTI_SYM_TENZOR_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
