// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SOFFSET_H
#define SOFFSET_H

// include files
#include "Index/NDIndex.h"
#include "Utility/PAssert.h"
#include "Message/Message.h"

#include <iostream>

// forward declarations
template <unsigned Dim> class SOffset;

template <unsigned Dim>
SOffset<Dim> operator+(const SOffset<Dim>&, const SOffset<Dim>&);
template <unsigned Dim>
SOffset<Dim> operator+(const SOffset<Dim>&, const int *);
template <unsigned Dim>
SOffset<Dim> operator+(const int *, const SOffset<Dim>&);
template <unsigned Dim>
NDIndex<Dim> operator+(const NDIndex<Dim>&, const SOffset<Dim>&);
template <unsigned Dim>
NDIndex<Dim> operator+(const SOffset<Dim>&, const NDIndex<Dim>&);

template <unsigned Dim>
SOffset<Dim> operator-(const SOffset<Dim>&, const SOffset<Dim>&);
template <unsigned Dim>
SOffset<Dim> operator-(const SOffset<Dim>&, const int *);
template <unsigned Dim>
SOffset<Dim> operator-(const int *, const SOffset<Dim>&);
template <unsigned Dim>
NDIndex<Dim> operator-(const NDIndex<Dim>&, const SOffset<Dim>&);
template <unsigned Dim>
NDIndex<Dim> operator-(const SOffset<Dim>&, const NDIndex<Dim>&);

template <unsigned Dim>
SOffset<Dim> operator*(const SOffset<Dim>&, const SOffset<Dim>&);
template <unsigned Dim>
SOffset<Dim> operator*(const SOffset<Dim>&, const int *);
template <unsigned Dim>
SOffset<Dim> operator*(const int *, const SOffset<Dim>&);
template <unsigned Dim>
NDIndex<Dim> operator*(const NDIndex<Dim>&, const SOffset<Dim>&);
template <unsigned Dim>
NDIndex<Dim> operator*(const SOffset<Dim>&, const NDIndex<Dim>&);

template <unsigned Dim>
SOffset<Dim> operator/(const SOffset<Dim>&, const SOffset<Dim>&);
template <unsigned Dim>
SOffset<Dim> operator/(const SOffset<Dim>&, const int *);
template <unsigned Dim>
SOffset<Dim> operator/(const int *, const SOffset<Dim>&);
template <unsigned Dim>
NDIndex<Dim> operator/(const NDIndex<Dim>&, const SOffset<Dim>&);

template <unsigned Dim>
SOffset<Dim>& operator-(const SOffset<Dim>&);

template <unsigned Dim>
std::ostream& operator<<(std::ostream&, const SOffset<Dim>&);

/***********************************************************************
 * 
 * SOffset is a simple class which stores an array of N integers
 * for use by the SIndex class.  It represents an integer offset
 * for the numbers in the SIndex set of indices.  Otherwise, it acts
 * just like a set of N integers.  The class is templated on the number
 * of elements N.
 *
 ***********************************************************************/

template <unsigned Dim>
class SOffset {

public:
  typedef int * iterator;
  typedef const int * const_iterator;

public:
  // constructors: note the default constructor initializes data to 0
  SOffset()       { for (unsigned int d=0; d < Dim; v[d++] = 0); }
  SOffset(int v0) { for (unsigned int d=0; d < Dim; v[d++] = v0); }
  SOffset(int v0, int v1);                 // only works for a 2D SOffset
  SOffset(int v0, int v1, int v2);         // only works for a 3D SOffset
  SOffset(int v0, int v1, int v2, int v3); // only works for a 4D SOffset
  SOffset(int v0, int v1, int v2, int v3, int v4); // only for 5D SOffset
  SOffset(int v0, int v1, int v2, int v3, int v4, int v5); // only for 6D
  SOffset(const int *vv);     // Takes array of offsets, for any dimensionality
  SOffset(const SOffset<Dim>&);

  // copy and index operators
  SOffset<Dim>& operator=(const SOffset<Dim>&);
  SOffset<Dim>& operator=(const int *);
  int& operator[](unsigned int d) { return v[d]; }
  const int& operator[](unsigned int d) const { return v[d]; }

  // comparison operators
  bool operator==(const SOffset<Dim>&)   const;
  bool operator<(const SOffset<Dim>&)    const;
  bool operator!=(const SOffset<Dim>& a) const { return !(*this == a); }
  bool operator<=(const SOffset<Dim>& a) const { return !(*this > a); }
  bool operator>=(const SOffset<Dim>& a) const { return !(*this < a); }
  bool operator>(const SOffset<Dim>& a)  const {
    return !(*this < a || *this == a);
  }

  // arithmetic operators
  SOffset<Dim>& operator+=(const SOffset<Dim>&);
  SOffset<Dim>& operator+=(const int *);
  SOffset<Dim>& operator-=(const SOffset<Dim>&);
  SOffset<Dim>& operator-=(const int *);
  SOffset<Dim>& operator*=(const SOffset<Dim>&);
  SOffset<Dim>& operator*=(const int *);
  SOffset<Dim>& operator/=(const SOffset<Dim>&);
  SOffset<Dim>& operator/=(const int *);

  // put or get to/from a Message
  Message& putMessage(Message &);
  Message& getMessage(Message &);

  // iterators for the offset, and information about the data
  iterator begin() { return v; }
  iterator end() { return (v + Dim); }
  const_iterator begin() const { return v; }
  const_iterator end() const { return (v + Dim); }
  unsigned int size() const { return Dim; }

  // is this point within an NDIndex object (includes the edges) ?
  bool inside(const NDIndex<Dim>&) const;
  
private:
  int v[Dim];
};


/***********************************************************************
 * 
 * Inlined constructors and SOffset functions.
 *
 ***********************************************************************/

template<unsigned int Dim>
inline 
SOffset<Dim>::SOffset(int v0, int v1) {
  CTAssert(Dim==2);
  v[0] = v0;
  v[1] = v1;
}

template<unsigned int Dim>
inline 
SOffset<Dim>::SOffset(int v0, int v1, int v2) {
  CTAssert(Dim==3);
  v[0] = v0;
  v[1] = v1;
  v[2] = v2;
}

template<unsigned int Dim>
inline 
SOffset<Dim>::SOffset(int v0, int v1, int v2, int v3) {
  CTAssert(Dim==4);
  v[0] = v0;
  v[1] = v1;
  v[2] = v2;
  v[3] = v3;
}

template<unsigned int Dim>
inline 
SOffset<Dim>::SOffset(int v0, int v1, int v2, int v3, int v4) {
  CTAssert(Dim==5);
  v[0] = v0;
  v[1] = v1;
  v[2] = v2;
  v[3] = v3;
  v[4] = v4;
}

template<unsigned int Dim>
inline 
SOffset<Dim>::SOffset(int v0, int v1, int v2, int v3, int v4, int v5) {
  CTAssert(Dim==6);
  v[0] = v0;
  v[1] = v1;
  v[2] = v2;
  v[3] = v3;
  v[4] = v4;
  v[5] = v5;
}

template<unsigned int Dim>
inline bool
SOffset<Dim>::operator==(const SOffset<Dim>& a) const {
  for (unsigned int i=0; i < Dim; ++i)
    if (v[i] != a.v[i])
      return false;
  return true;
}

template<unsigned int Dim>
inline bool
SOffset<Dim>::operator<(const SOffset<Dim>& a) const {
  for (unsigned int i=0; i < Dim; ++i) {
    if (v[i] < a.v[i])
      return true;
    else if (v[i] > a.v[i])
      return false;
  }

  // if we're here, they're exactly equal
  return false;
}

template<unsigned int Dim>
inline Message&
SOffset<Dim>::putMessage(Message &m) {
  ::putMessage(m, v, v + Dim);
  return m;
}

template<unsigned int Dim>
inline Message&
SOffset<Dim>::getMessage(Message &m) {
  ::getMessage_iter(m, v);
  return m;
}

template<unsigned int Dim>
inline std::ostream&
operator<<(std::ostream& o, const SOffset<Dim>& a) {
  o << "[";
  for (unsigned int i=0; i < Dim; ++i)
    o << a[i] << (i < (Dim-1) ? "," : "");
  o << "]";
  return o;
}

// is this point within an NDIndex object (includes the edges) ?
// NOTE: this intersects with the whole region of the NDIndex, basically
// assuming unit stride.
template<unsigned int Dim>
inline bool
SOffset<Dim>::inside(const NDIndex<Dim>& ndi) const {
  for (unsigned int d=0; d < Dim; ++d) {
    if (v[d] < ndi[d].first() || v[d] > ndi[d].last())
      return false;
  }
  return true;
}


/***********************************************************************
 * 
 * Template classes and functions to unroll the loops into subloops of
 * a maximum of 3 elements.
 * This is taken from the Utility/Vec.h class with modifications.
 *
 ***********************************************************************/

// a tag class used to indicate how many elements and when to split a loop
template<unsigned int Dim, int Flag>
class DivideSOffsetCopyTag {
};

// simple functors for performing operations on SOffset elements
struct SOffsetAssignOp {
  void operator()(int& a, int b) { a = b; }
};

struct SOffsetAddAssignOp {
  void operator()(int& a, int b) { a += b; }
};

struct SOffsetSubAssignOp {
  void operator()(int& a, int b) { a -= b; }
};

struct SOffsetMultAssignOp {
  void operator()(int& a, int b) { a *= b; }
};

struct SOffsetDivAssignOp {
  void operator()(int& a, int b) { a /= b; }
};


// Operate on a vector of dim 4 or greater.  Split it in half and do each half.
// Do it this way so that the depth of inlining is O(log(Dim)).
template<class T1, class T2, class Op, unsigned int L>
inline void
divide_soffset_op(T1 *p1, T2 *p2, Op op, DivideSOffsetCopyTag<L,4> ) {
  divide_soffset_op(p1, p2, op, DivideSOffsetCopyTag<L/2, (L>=8 ? 4 : L/2)>());
  divide_soffset_op(p1+(L/2), p2+(L/2), op,
                    DivideSOffsetCopyTag<L-L/2, (L>=8?4:L-L/2)>());
}


// Modify a vector of length 0, 1, 2 or 3.
template<class T1, class T2, class Op>
inline void
divide_soffset_op(T1 *p1, T2 *p2, Op op, DivideSOffsetCopyTag<3,3>) {
  op(p1[0], p2[0]);
  op(p1[1], p2[1]);
  op(p1[2], p2[2]);
}

template<class T1, class T2, class Op>
inline void
divide_soffset_op(T1 *p1, T2 *p2, Op op, DivideSOffsetCopyTag<2,2>) {
  op(p1[0], p2[0]);
  op(p1[1], p2[1]);
} 

template<class T1, class T2, class Op>
inline void
divide_soffset_op(T1 *p1, T2 *p2, Op op, DivideSOffsetCopyTag<1,1>) {
  op(*p1, *p2);
} 

template<class T1, class T2, class Op>
inline void
divide_soffset_op(T1 *,T2 *, Op, DivideSOffsetCopyTag<0,0>) {
}

//
// The SOffset operators just call divide_soffset_op to do the operation
//

template<unsigned int L>
inline 
SOffset<L>::SOffset(const SOffset<L>& SO)
{
  divide_soffset_op(v, SO.v, SOffsetAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
}

template<unsigned int L>
inline 
SOffset<L>::SOffset(const int *SO)
{
  divide_soffset_op(v, SO, SOffsetAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator=(const SOffset<L>& SO)
{
  if ( this != &SO )
    divide_soffset_op(v, SO.v, SOffsetAssignOp(),
                      DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator=(const int *SO)
{
  divide_soffset_op(v, SO, SOffsetAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator+=(const SOffset<L>& SO)
{
  divide_soffset_op(v, SO.v, SOffsetAddAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator+=(const int *SO)
{
  divide_soffset_op(v, SO, SOffsetAddAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator-=(const SOffset<L>& SO)
{
  divide_soffset_op(v, SO.v, SOffsetSubAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator-=(const int *SO)
{
  divide_soffset_op(v, SO, SOffsetSubAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator*=(const SOffset<L>& SO)
{
  divide_soffset_op(v, SO.v, SOffsetMultAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator*=(const int *SO)
{
  divide_soffset_op(v, SO, SOffsetMultAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator/=(const SOffset<L>& SO)
{
  divide_soffset_op(v, SO.v, SOffsetDivAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}

template<unsigned int L>
inline SOffset<L>&
SOffset<L>::operator/=(const int *SO)
{
  divide_soffset_op(v, SO, SOffsetDivAssignOp(),
                    DivideSOffsetCopyTag<L,( L>=4 ? 4 : L)>() );
  return *this;
}


//
// binary operators
// SOffset +-*/ SOffset  -->  SOffset
// SOffset +-*/ int *    -->  SOffset
// SOffset +-*/ NDIndex  -->  NDIndex
//

template<unsigned int L>
inline SOffset<L>
operator+(const SOffset<L>& a, const SOffset<L>& b)
{
  SOffset<L> retval(a);
  return (retval += b);
}

template<unsigned int L>
inline SOffset<L>
operator+(const SOffset<L>& a, const int *b)
{
  SOffset<L> retval(a);
  return (retval += b);
}

template<unsigned int L>
inline SOffset<L>
operator+(const int *b, const SOffset<L>& a)
{
  SOffset<L> retval(a);
  return (retval += b);
}

template<unsigned int L>
inline NDIndex<L>
operator+(const SOffset<L>& a, const NDIndex<L>& b)
{
  NDIndex<L> retval;
  for (unsigned int d=0; d < L; ++d)
    retval[d] = b[d] + a[d];
  return retval;
}

template<unsigned int L>
inline NDIndex<L>
operator+(const NDIndex<L>& b, const SOffset<L>& a)
{
  NDIndex<L> retval;
  for (unsigned int d=0; d < L; ++d)
    retval[d] = b[d] + a[d];
  return retval;
}

template<unsigned int L>
inline SOffset<L>
operator-(const SOffset<L>& a, const SOffset<L>& b)
{
  SOffset<L> retval(a);
  return (retval -= b);
}

template<unsigned int L>
inline SOffset<L>
operator-(const SOffset<L>& a, const int *b)
{
  SOffset<L> retval(a);
  return (retval -= b);
}

template<unsigned int L>
inline SOffset<L>
operator-(const int *b, const SOffset<L>& a)
{
  SOffset<L> retval(a);
  return (retval -= b);
}

template<unsigned int L>
inline NDIndex<L>
operator-(const SOffset<L>& a, const NDIndex<L>& b)
{
  NDIndex<L> retval;
  for (int d=0; d < L; ++d)
    retval[d] = b[d] - a[d];
  return retval;
}

template<unsigned int L>
inline NDIndex<L>
operator-(const NDIndex<L>& b, const SOffset<L>& a)
{
  NDIndex<L> retval;
  for (int d=0; d < L; ++d)
    retval[d] = b[d] - a[d];
  return retval;
}

template<unsigned int L>
inline SOffset<L>
operator*(const SOffset<L>& a, const SOffset<L>& b)
{
  SOffset<L> retval(a);
  return (retval *= b);
}

template<unsigned int L>
inline SOffset<L>
operator*(const SOffset<L>& a, const int *b)
{
  SOffset<L> retval(a);
  return (retval *= b);
}

template<unsigned int L>
inline SOffset<L>
operator*(const int *b, const SOffset<L>& a)
{
  SOffset<L> retval(a);
  return (retval *= b);
}

template<unsigned int L>
inline NDIndex<L>
operator*(const SOffset<L>& a, const NDIndex<L>& b)
{
  NDIndex<L> retval;
  for (unsigned int d=0; d < L; ++d)
    retval[d] = b[d] * a[d];
  return retval;
}

template<unsigned int L>
inline NDIndex<L>
operator*(const NDIndex<L>& b, const SOffset<L>& a)
{
  NDIndex<L> retval;
  for (unsigned int d=0; d < L; ++d)
    retval[d] = b[d] * a[d];
  return retval;
}

template<unsigned int L>
inline SOffset<L>
operator/(const SOffset<L>& a, const SOffset<L>& b)
{
  SOffset<L> retval(a);
  return (retval /= b);
}

template<unsigned int L>
inline SOffset<L>
operator/(const SOffset<L>& a, const int *b)
{
  SOffset<L> retval(a);
  return (retval /= b);
}

template<unsigned int L>
inline SOffset<L>
operator/(const int *b, const SOffset<L>& a)
{
  SOffset<L> retval(a);
  return (retval /= b);
}

template<unsigned int L>
inline NDIndex<L>
operator/(const NDIndex<L>& b, const SOffset<L>& a)
{
  NDIndex<L> retval;
  for (int d=0; d < L; ++d)
    retval[d] = b[d] / a[d];
  return retval;
}


//
// unary operators
//

template<unsigned int L>
inline SOffset<L>&
operator-(const SOffset<L>& a)
{
  SOffset<L> retval;
  return (retval -= a);
}

#endif // SOFFSET_H

/***************************************************************************
 * $RCSfile: SOffset.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: SOffset.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
