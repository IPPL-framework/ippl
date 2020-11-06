//
// Class Index
//   Define a slice in an array.
//
//   This essentially defines a list of evenly spaced numbers.
//   Most commonly this list will be increasing (positive stride)
//   but it can also have negative stride and be decreasing.
//
//   Index()      --> A null interval with no elements.
//   Index(n)     --> make an Index on [0..n-1]
//   Index(a,b)   --> make an Index on [a..b]
//   Index(a,b,s) --> make an Index on [a..b] with stride s
//
//   Example1:
//   --------
//   Index I(10);           --> Index on [0..9]
//   Index Low(5);          --> Index on [0..4]
//   Index High(5,9);       --> Index on [5..9]
//   Index IOdd(1,9,2);     --> Index on [1..9] stride 2
//   Index IEven(0,9,2);    --> Index on [0..9] stride 2
//
//   Given an Index I(a,n,s), and an integer j you can do the following:
//
//   I+j  : a+j+i*s        for i in [0..n-1]
//   j-I  : j-a-i*s
//   j*I  : j*a + i*j*s
//   I/j  : a/j + i*s/j
//
//   j/I we don't do because it is not a uniform stride, and we don't
//   allow strides that are fractions.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Utility/Unique.h"
#include "Utility/PAssert.h"


namespace ippl {

//////////////////////////////////////////////////////////////////////
// Null ctor.
//////////////////////////////////////////////////////////////////////
inline 
Index::Index()
: first_m(0),
  stride_m(0),
  length_m(0),
  BaseFirst(0),
  Base(Unique::get())
{
}

//////////////////////////////////////////////////////////////////////
// Ctor for [0..n-1]
//////////////////////////////////////////////////////////////////////

inline 
Index::Index(size_t n)
: first_m(0),
  stride_m(1),
  length_m(n),
  BaseFirst(0),
  Base(Unique::get())
{
}


//////////////////////////////////////////////////////////////////////
// ctor for range.
//////////////////////////////////////////////////////////////////////
inline 
Index::Index(int f, int l)
: first_m(f),
  stride_m(1),
  length_m(l-f+1),
  BaseFirst(0),
  Base(Unique::get())
{
  PAssert_GE(l - f + 1, 0);
}

inline 
Index::Index(int f, int l, int s)
: first_m(f),
  stride_m(s),
  BaseFirst(0),
  Base(Unique::get())
{
  PAssert_NE(s, 0);
  if ( f==l ) {
    length_m = 1;
  }
  else if ( (l>f) ^ (s<0) ) {
    length_m = (l-f)/s + 1;
  }
  else {
    length_m = 0;
  }
}

//////////////////////////////////////////////////////////////////////
// Some ctors for internal use.
//////////////////////////////////////////////////////////////////////

inline 
Index::Index(int m, int a, const Index &b)
: first_m(b.first_m*m+a),
  stride_m(b.stride_m*m),
  length_m(b.length_m),
  BaseFirst( b.BaseFirst ),
  Base(b.Base)
{
}

inline 
Index::Index(int f, int s, const Index *b)
: first_m(f),
  stride_m(s),
  length_m(b->length_m),
  BaseFirst(b->BaseFirst),
  Base(b->Base)
{
}

//////////////////////////////////////////////////////////////////////
//
// Informational functions about Index.
//
// first: return the first index.
// length: return the number of elements.
// stride: the stride between elements.
//
// min: the minimum value.
// max: the maximum value.
//
//////////////////////////////////////////////////////////////////////

inline int  Index::first()  const
{
  return first_m;
}

inline int  Index::stride() const
{
  return stride_m;
}

inline bool Index::empty()  const
{
  return length_m==0;
}

inline unsigned int  Index::length() const
{
  return length_m;
}

inline int Index::last() const
{
  return length_m==0 ? first_m : first_m + stride_m*(length_m-1);
}

inline int Index::min() const 
{
  return stride_m>=0 ? first_m : first_m+stride_m*(length_m-1);
}

inline int Index::max() const 
{
  return stride_m>=0 ? first_m+stride_m*(length_m-1) : first_m;
}

inline int Index::getBase() const
{
  return Base;
}

//////////////////////////////////////////////////////////////////////
//
// Operations on Index's.
//
//////////////////////////////////////////////////////////////////////

inline Index operator+(const Index& i, int off)
{
  return Index(1,off,i);
}

inline Index operator+(int off, const Index& i)
{
  return Index(1,off,i);
}

inline Index operator-(const Index& i, int off)
{
  return Index(1,-off,i);
}

inline Index operator-(int off, const Index& i)
{
  return Index(-1,off,i);
}

inline Index operator-(const Index& i)
{
  return Index(-1,0,i);
}

inline Index operator*(const Index& i, int m)
{
  return Index(m,0,i);
}

inline Index operator*(int m, const Index& i)
{
  return Index(m,0,i);
}

inline Index operator/(const Index& i, int d)
{
  return Index(i.first_m/d, i.stride_m/d, &i);
}

//////////////////////////////////////////////////////////////////////
//
// Comparison operators.
//
//////////////////////////////////////////////////////////////////////

inline bool Index::sameBase(const Index& i) const 
{
  return Base == i.Base;
}

//////////////////////////////////////////////////////////////////////

inline Index Index::plugBase(const Index &a) const
{
  Index ret;
  ret.BaseFirst = a.BaseFirst;
  ret.length_m = a.length_m;
  ret.stride_m = stride_m;
  ret.first_m = first_m + stride_m*(a.BaseFirst-BaseFirst);
  ret.Base = Base;
  return ret;
}

//////////////////////////////////////////////////////////////////////

inline Index Index::reverse() const
{
  Index j;
  j.first_m = last();
  j.length_m = length_m;
  j.stride_m = -stride_m;
  j.Base = Base;
  j.BaseFirst = BaseFirst;  
  return j;
}

//////////////////////////////////////////////////////////////////////

// Test to see if there is any overlap between two Indexes.
inline bool Index::touches(const Index&a) const
{
  return (min()<=a.max())&&(max()>=a.min());
}

// Test to see if one index completely contains another.
inline bool Index::contains(const Index&a) const
{
  return (min()<=a.min())&&(max()>=a.max());
}

inline bool Index::containsAllPoints(const Index &b) const
{
  // Find min and max values of type domains
  int a0 = min();
  int a1 = max();
  int  s = stride();
  int b0 = b.min();
  int b1 = b.max();
  int  t = b.stride();
  if (s < 0)
    s = -s;
  if (t < 0)
    t = -t;

  // We can do a quick short-circuit check to make sure they do not overlap
  // at all just from their endpoints.  If they don't even do this, we can
  // quit and say they do not touch.
  bool quicktest = (a0 <= b0 && a1 >= b1);
  if (!quicktest || s == 1)
    return quicktest;

  // OK, the endpoints of a contain those of b, and we must find out if
  // all the points in b are found in a.  This will be true if:
  //   1. The stride of b is a multipple of the stride of a
  //   2. The endpoints of b are found in a
  // If either of these conditions are false, a does not contain b 
  return (t % s == 0) && ((b0-a0) % s == 0) && ((a1-b1) % s == 0);
}



// Split an index into equal parts
inline bool Index::split(Index& l, Index& r) const
{
  PAssert_EQ(stride_m, 1);
  PAssert_GT(length_m, 1);
  //if ( length_m <= 1 )
  //  return false;
  //else
  //  {
      int first = first_m;
      int length = length_m;
      int mid = first + length/2 - 1;
      l = Index(first, mid);
      r = Index(mid+1,first+length-1);
      return true;
  //  }
}

// Split an index with the given ratio
inline bool Index::split(Index& l, Index& r, double a) const
{
  PAssert_EQ(stride_m, 1);
  PAssert_GT(length_m, 1);
  PAssert_LT(a, 1.0);
  PAssert_GT(a, 0.0);
  //if ( length_m <= 1 )
  //  return false;
  //else
  //  {
      int first = first_m;
      int length = length_m;
      int mid = first + static_cast<int>(length*a+0.5) - 1;
      l = Index(first, mid);
      r = Index(mid+1,first+length-1);
      return true;
  //  }
}
/*
//////////////////////////////////////////////////////////////////////

#define INDEX_PETE_DOUBLE_OPERATOR(OP,APP)			     \
								     \
inline								     \
PETE_TBTree< APP , Index::PETE_Expr_t , PETE_Scalar<double> >	     \
OP ( const Index& idx, double x )				     \
{								     \
  typedef							     \
    PETE_TBTree< APP , Index::PETE_Expr_t , PETE_Scalar<double> >    \
    Return_t;							     \
  return Return_t( idx.MakeExpression(), PETE_Scalar<double>(x) );   \
}								     \
								     \
inline								     \
PETE_TBTree< APP , PETE_Scalar<double> , Index::PETE_Expr_t >	     \
OP ( double x , const Index& idx )				     \
{								     \
  typedef							     \
    PETE_TBTree< APP , PETE_Scalar<double> , Index::PETE_Expr_t >    \
    Return_t;							     \
  return Return_t( PETE_Scalar<double>(x) , idx.MakeExpression());   \
}

INDEX_PETE_DOUBLE_OPERATOR(operator+,OpAdd)
INDEX_PETE_DOUBLE_OPERATOR(operator-,OpSubtract)
INDEX_PETE_DOUBLE_OPERATOR(operator*,OpMultipply)
INDEX_PETE_DOUBLE_OPERATOR(operator/,OpDivide)
INDEX_PETE_DOUBLE_OPERATOR(operator%,OpMod)

INDEX_PETE_DOUBLE_OPERATOR(lt,OpLT)
INDEX_PETE_DOUBLE_OPERATOR(le,OpLE)
INDEX_PETE_DOUBLE_OPERATOR(gt,OpGT)
INDEX_PETE_DOUBLE_OPERATOR(ge,OpGE)
INDEX_PETE_DOUBLE_OPERATOR(eq,OpEQ)
INDEX_PETE_DOUBLE_OPERATOR(ne,OpNE)

INDEX_PETE_DOUBLE_OPERATOR(Max,FnMax)
INDEX_PETE_DOUBLE_OPERATOR(Min,FnMin)*/
/*
#undef INDEX_PETE_DOUBLE_OPERATOR

//////////////////////////////////////////////////////////////////////

#define INDEX_PETE_FLOAT_OPERATOR(OP,APP)			     \
								     \
inline								     \
PETE_TBTree< APP , Index::PETE_Expr_t , PETE_Scalar<float> >	     \
OP ( const Index& idx, float x )				     \
{								     \
  typedef							     \
    PETE_TBTree< APP , Index::PETE_Expr_t , PETE_Scalar<float> >    \
    Return_t;							     \
  return Return_t( idx.MakeExpression(), PETE_Scalar<float>(x) );   \
}								     \
								     \
inline								     \
PETE_TBTree< APP , PETE_Scalar<float> , Index::PETE_Expr_t >	     \
OP ( float x , const Index& idx )				     \
{								     \
  typedef							     \
    PETE_TBTree< APP , PETE_Scalar<float> , Index::PETE_Expr_t >    \
    Return_t;							     \
  return Return_t( PETE_Scalar<float>(x) , idx.MakeExpression());   \
}

INDEX_PETE_FLOAT_OPERATOR(operator+,OpAdd)
INDEX_PETE_FLOAT_OPERATOR(operator-,OpSubtract)
INDEX_PETE_FLOAT_OPERATOR(operator*,OpMultipply)
INDEX_PETE_FLOAT_OPERATOR(operator/,OpDivide)
INDEX_PETE_FLOAT_OPERATOR(operator%,OpMod)

INDEX_PETE_FLOAT_OPERATOR(lt,OpLT)
INDEX_PETE_FLOAT_OPERATOR(le,OpLE)
INDEX_PETE_FLOAT_OPERATOR(gt,OpGT)
INDEX_PETE_FLOAT_OPERATOR(ge,OpGE)
INDEX_PETE_FLOAT_OPERATOR(eq,OpEQ)
INDEX_PETE_FLOAT_OPERATOR(ne,OpNE)

INDEX_PETE_FLOAT_OPERATOR(Max,FnMax)
INDEX_PETE_FLOAT_OPERATOR(Min,FnMin)

#undef INDEX_PETE_FLOAT_OPERATOR

//////////////////////////////////////////////////////////////////////

#define INDEX_PETE_INT_OPERATOR(OP,APP)			             \
								     \
inline								     \
PETE_TBTree< APP , Index::PETE_Expr_t , PETE_Scalar<int> >	     \
OP ( const Index& idx, int x )				             \
{								     \
  typedef							     \
    PETE_TBTree< APP , Index::PETE_Expr_t , PETE_Scalar<int> >       \
    Return_t;							     \
  return Return_t( idx.MakeExpression(), PETE_Scalar<int>(x) );      \
}								     \
								     \
inline								     \
PETE_TBTree< APP , PETE_Scalar<int> , Index::PETE_Expr_t >	     \
OP ( int x , const Index& idx )				             \
{								     \
  typedef							     \
    PETE_TBTree< APP , PETE_Scalar<int> , Index::PETE_Expr_t >       \
    Return_t;							     \
  return Return_t( PETE_Scalar<int>(x) , idx.MakeExpression());      \
}

INDEX_PETE_INT_OPERATOR(operator%,OpMod)

INDEX_PETE_INT_OPERATOR(lt,OpLT)
INDEX_PETE_INT_OPERATOR(le,OpLE)
INDEX_PETE_INT_OPERATOR(gt,OpGT)
INDEX_PETE_INT_OPERATOR(ge,OpGE)
INDEX_PETE_INT_OPERATOR(eq,OpEQ)
INDEX_PETE_INT_OPERATOR(ne,OpNE)

INDEX_PETE_INT_OPERATOR(Max,FnMax)
INDEX_PETE_INT_OPERATOR(Min,FnMin)

#undef INDEX_PETE_INT_OPERATOR*/

//////////////////////////////////////////////////////////////////////

}