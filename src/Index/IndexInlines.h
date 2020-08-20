// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INDEX_INLINES_H
#define INDEX_INLINES_H

// include files
#include "Utility/Unique.h"
#include "Utility/PAssert.h"

//////////////////////////////////////////////////////////////////////
// Null ctor.
//////////////////////////////////////////////////////////////////////
inline 
Index::Index()
: First(0),
  Stride(0),
  Length(0),
  BaseFirst(0),
  Base(Unique::get())
{
}

//////////////////////////////////////////////////////////////////////
// Ctor for [0..n-1]
//////////////////////////////////////////////////////////////////////

inline 
Index::Index(unsigned n)
: First(0),
  Stride(1),
  Length(n),
  BaseFirst(0),
  Base(Unique::get())
{
}


//////////////////////////////////////////////////////////////////////
// ctor for range.
//////////////////////////////////////////////////////////////////////
inline 
Index::Index(int f, int l)
: First(f),
  Stride(1),
  Length(l-f+1),
  BaseFirst(0),
  Base(Unique::get())
{
  PAssert_GE(l - f + 1, 0);
}

inline 
Index::Index(int f, int l, int s)
: First(f),
  Stride(s),
  BaseFirst(0),
  Base(Unique::get())
{
  PAssert_NE(s, 0);
  if ( f==l ) {
    Length = 1;
  }
  else if ( (l>f) ^ (s<0) ) {
    Length = (l-f)/s + 1;
  }
  else {
    Length = 0;
  }
}

//////////////////////////////////////////////////////////////////////
// Some ctors for internal use.
//////////////////////////////////////////////////////////////////////

inline 
Index::Index(int m, int a, const Index &b)
: First(b.First*m+a),
  Stride(b.Stride*m),
  Length(b.Length),
  BaseFirst( b.BaseFirst ),
  Base(b.Base)
{
}

inline 
Index::Index(int f, int s, const Index *b)
: First(f),
  Stride(s),
  Length(b->Length),
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
  return First;
}

inline int  Index::stride() const
{
  return Stride;
}

inline bool Index::empty()  const
{
  return Length==0;
}

inline unsigned int  Index::length() const
{
  return Length;
}

inline int Index::last() const
{
  return Length==0 ? First : First + Stride*(Length-1);
}

inline int Index::min() const 
{
  return Stride>=0 ? First : First+Stride*(Length-1); 
}

inline int Index::max() const 
{
  return Stride>=0 ? First+Stride*(Length-1) : First; 
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
  return Index(i.First/d, i.Stride/d, &i);
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
  ret.Length = a.Length;
  ret.Stride = Stride;
  ret.First = First + Stride*(a.BaseFirst-BaseFirst);
  ret.Base = Base;
  return ret;
}

//////////////////////////////////////////////////////////////////////

inline Index Index::reverse() const
{
  Index j;
  j.First = last();
  j.Length = Length;
  j.Stride = -Stride;
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
  PAssert_EQ(Stride, 1);
  PAssert_GT(Length, 1);
  //if ( Length <= 1 )
  //  return false;
  //else
  //  {
      int first = First;
      int length = Length;
      int mid = first + length/2 - 1;
      l = Index(first, mid);
      r = Index(mid+1,first+length-1);
      return true;
  //  }
}

// Split an index with the given ratio
inline bool Index::split(Index& l, Index& r, double a) const
{
  PAssert_EQ(Stride, 1);
  PAssert_GT(Length, 1);
  PAssert_LT(a, 1.0);
  PAssert_GT(a, 0.0);
  //if ( Length <= 1 )
  //  return false;
  //else
  //  {
      int first = First;
      int length = Length;
      int mid = first + static_cast<int>(length*a+0.5) - 1;
      l = Index(first, mid);
      r = Index(mid+1,first+length-1);
      return true;
  //  }
}

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
INDEX_PETE_DOUBLE_OPERATOR(Min,FnMin)

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

#undef INDEX_PETE_INT_OPERATOR

//////////////////////////////////////////////////////////////////////

#endif // INDEX_INLINES_H

/***************************************************************************
 * $RCSfile: IndexInlines.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: IndexInlines.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
