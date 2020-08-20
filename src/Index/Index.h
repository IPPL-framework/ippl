// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INDEX_H
#define INDEX_H

/***********************************************************************

Define a slice in an array.

This essentially defines a list of evenly spaced numbers.
Most commonly this list will be increasing (positive stride)
but it can also have negative stride and be decreasing.

Index()      --> A null interval with no elements.
Index(n)     --> make an Index on [0..n-1]
Index(a,b)   --> make an Index on [a..b]
Index(a,b,s) --> make an Index on [a..b] with stride s

Example1:
--------
Index I(10);			// Index on [0..9]
Index Low(5);			// Index on [0..4]
Index High(5,9);		// Index on [5..9]
Index IOdd(1,9,2);		// Index on [1..9] stride 2
Index IEven(0,9,2);		// Index on [0..9] stride 2

NDIndex<1> domain;
domain[0] = I;
FieldLayout<1> layout(domain)	// Construct the layout
Field<double, Dim> A(layout)	// Construct the field

A = 0;

cout << A << endl;

A[Low] = 1.0;

cout << A << endl;

A[High] = 2.0;

cout << A << endl;

A[IEven] -= 1.0;

cout << A << endl;

A[IOdd] -= 1.0;

cout << A << endl;

Output::
--------
0 0 0 0 0 0 0 0 0 0 
1 1 1 1 1 0 0 0 0 0 
1 1 1 1 1 2 2 2 2 2 
0 1 0 1 0 2 1 2 1 2 
0 0 0 0 0 1 1 1 1 1 

The same functionality can be reproduced without constucting additional
Index objects by making use of the binary operations.

--------
Index I(10);                    // Index on [0..9]
NDIndex<1> domain;              //
FieldLayout<1> layout(domain)   // Construct the layout
Field<double, Dim> A(layout)    // Construct the field

A = 0;

cout << A << endl;

A[I-5] = 1.0;

cout << A << endl;

A[I+5] = 2.0;

cout << A << endl;

A[I*2] -= 1.0;

cout << A << endl;

A[I*2+1] -= 1.0;

cout << A << endl;

Output::
--------
0 0 0 0 0 0 0 0 0 0
1 1 1 1 1 0 0 0 0 0
1 1 1 1 1 2 2 2 2 2
0 1 0 1 0 2 1 2 1 2
0 0 0 0 0 1 1 1 1 1


Given an Index I(a,n,s), and an integer j you can do the following:

I+j  : a+j+i*s        for i in [0..n-1]
j-I  : j-a-i*s        
j*I  : j*a + i*j*s    
I/j  : a/j + i*s/j

j/I we don't do because it is not a uniform stride, and we don't
allow strides that are fractions.

When performing these arithmetic operations the Index keeps
track of some information about a base index so we can deduce
what to do in array expressions. For example if you want to do

     A[I] = B[I+1];

you need to be able to be sure that you used the same Index on both
sides, and you need to be able to tell that the result of 
operator+(const Index&,int) on the right is offset 1 from I.

For the first requirement it keeps track of a pointer to the base 
Index that it came from.

The offset stuff is slightly trickier, as it does a number of
things to be sure it can do fairly arbitrary intersections and so on
efficiently.

Each Index has a "domain" of contiguous increasing integers. 
Each Index has a "range" of integers with constant stride.
If i is in the domain [a,b], the range is f+i*s, where f+a*s 
is the "first" element of the range and f+b*s is the "last".
Because we don't allow fractional strides, the range has no
repeated elements.

This "ordering" is imposed so that expressions like

   A[alpha+beta*I] = B[gamma + delta*I];

have a clearly defined meaning:

   for (i=a; i<=b; i++) 
      A[alpha+beta*(f+i*s)] = B[gamma+delta*(f+i*s)];

In practice we may choose whatever order we like for efficiency, 
but the definition remains the same.

This is evaluated as follows.

alpha + beta*I is evaluated to give I1 with the same domain as
I and the range given by f1 = alpha+beta*f, and s1=beta*s.
gamma + delta*I gives I2, with f2=gamma+delta*f, s2=delta*s.
Then we evaluate A[I1] = B[I2], which is equivalent to 

   for (i=a; i<=b; i++)
      A[f1+i*s1] = B[f2+i*s2];

With what has been defined up to now, we can always choose a=0.
The following will relax that.

All of this would be quite straightforward in serial, the fun
begins in parallel. Indexes are not parallel objects, but they
have operations defined for them for making regular section 
transfer calculations easier: intersect and plugBase.

The idea is that we represent the part of A and B that reside on
various processors with an Index. That index is the local domain 
of the array.

So, when we want to find what part of A[I1] is on processor p we do:

  Index LocalDestinationRange = I1.intersect(A.localDomain(p));

the Index LocalDestinationRange has a range which is those elments
in the range of I1 that are on processor p. (We restrict the possible
domains to contiguous integers -- no fair putting the odds on one
processor and the evens on another for now!)  That tells us where 
data is going to end up on this processor. The mapping between the 
Index's domain and its range is preserved under this operation.

From that we need to find where those elements come from. 

XXjr  Index LocalSourceRange = I2.plugBase(LocalSourceRange);
  Index LocalSourceRange = I2.plugBase(LocalDestinationRange); 

XXjr This plugs the domain of LocalSourceRange into I2, to get where
XXjr in I2 the elements will be coming from.
This plugs the domain of LocalDestinationRange into I2, to get where
in I2 the elements will be coming from.

Then for every candidate other processor pp, we intersect LocalSourceRange 
with that processor's domain for B and we have what needs to be
comminicated from that processor.

   Index RemoteSourceRange = LocalSourceRange.intersect(B.localDomain(pp));

Doing this operation for all the candidate pp's produces the get
schedule.

Finding the put schedule is very similar. Start by finding what parts
of I2 are on this processor:

   Index LocalSourceRange = I2.intersect(B.localDomain(p));

Plug the domain of that into I1 to find where they're going:

   Index LocalDestRange = I1.plugBase(LocalSourceRange);

Intersect that with all the candidate processors pp to find what
processor to send them to:

   Index RemoteDestRange = LocalDestRange.intersect(A.localDomain(pp));

One last plugBase puts that back in terms of the range of B:

   Index RemoteSourceRange = LocalSourceRange.plugBase(RemoteDestRange);

And that is how the put and get schedules are calculated.

***********************************************************************/


// include files
#include "PETE/IpplExpressions.h"
#include <iostream>

// forward declarations
class Index;
std::ostream& operator<<(std::ostream& out, const Index& I);


class Index : public PETE_Expr<Index>
{

public:
  class iterator
  {
  public:

    iterator()                          : Current(0)      , Stride(0)      {}
    iterator(int current, int stride=1) : Current(current), Stride(stride) {}

    int operator*() { return Current ; }
    iterator operator--(int)
    {
      iterator tmp = *this;
      Current -= Stride;             // Post decrement
      return tmp;
    }
    iterator& operator--()
    {
      Current -= Stride;
      return (*this);
    }
    iterator operator++(int)
    {
      iterator tmp = *this;
      Current += Stride;              // Post increment
      return tmp;
    }
    iterator& operator++()
    {
      Current += Stride;
      return (*this);
    }
    iterator& operator+=(int i)
    {
      Current += (Stride * i);
      return *this;
    }
    iterator& operator-=(int i)
    {
      Current -= (Stride * i);
      return *this;
    } 
    iterator operator+(int i) const
    {
      return iterator(Current+i*Stride,Stride);
    }
    iterator operator-(int i) const
    {
      return iterator(Current-i*Stride,Stride);
    }
    int operator[](int i) const
    {
      return Current + i * Stride;
    }
    bool operator==(const iterator &y) const 
    {
      return (Current == y.Current) && (Stride == y.Stride);
    }
    bool operator<(const iterator &y) const
    {
      return (Current < y.Current)||
	((Current==y.Current)&&(Stride<y.Stride));
    }
    bool operator!=(const iterator &y) const { return !((*this) == y); }
    bool operator> (const iterator &y) const { return y < (*this); }
    bool operator<=(const iterator &y) const { return !(y < (*this)); }
    bool operator>=(const iterator &y) const { return !((*this) < y); }
  private: 

    int Current;
    int Stride;
  };
 
  class cursor : public PETE_Expr<cursor>
  {
  private:
    int Current;
    int Stride;
    int First;
    unsigned Dim;
    const Index* I;
  public:
    cursor() {}
    cursor(const Index& i)
      : Current(i.first()),
	Stride(i.stride()),
	First(i.first()),
	Dim(0),
	I(&i)
      {
      }

    int operator*() const { return Current; }
    int offset() const { return Current; }
    int offset(int i) const
      {
	return Current +
	  ( Dim==0 ? i*Stride : 0 );
      }
    int offset(int i, int j)  const
      {
	return Current +
	  ( Dim==0 ? i*Stride : 0 ) +
	  ( Dim==1 ? j*Stride : 0 );
      }
    int offset(int i, int j, int k) const
      {
	return Current +
	  ( Dim==0 ? i*Stride : 0 ) +
	  ( Dim==1 ? j*Stride : 0 ) +
	  ( Dim==2 ? k*Stride : 0 );
      }
    void step(unsigned d)
      {
	if ( d==Dim )
	  Current += Stride;
      }
    void rewind(unsigned d)
      {
	if ( d==Dim )
	  Current = First;
      }
    bool plugBase(const Index& i, unsigned d=0)
      {
	Index plugged( I->plugBase(i) );
	Current = First = plugged.first();
	Stride = plugged.stride();
	Dim = d;
	return true;
      }
    int id() const { return I->id(); }

    // PETE interface.
    enum { IsExpr = 1 };
    typedef cursor PETE_Expr_t;
    typedef int PETE_Return_t;
    cursor MakeExpression() const { return *this; }
  };

  // Member functions.  Make almost all of these inline for efficiency.

  Index();		        // Null range.
  inline Index(unsigned n);	// [0..n-1]
  inline Index(int f, int l);		// [f..l]
  inline Index(int f, int l, int s);	// First to Last using Step.

  ~Index() {};		                // Don't need to do anything.
  int id() const { return Base; }

  inline int min() const;		// the smallest element.
  inline int max() const;		// the largest element.
  inline unsigned int length() const;   // the number of elems.
  inline int stride() const;		// the stride.
  inline int first() const;		// the first element.
  inline int last() const;		// the last element.
  inline bool empty() const;		// is it empty?
  inline int getBase() const;	// the id from the base index

  // Additive operations.
  friend inline Index operator+(const Index&,int);
  friend inline Index operator+(int,const Index&);
  friend inline Index operator-(const Index&,int);
  friend inline Index operator-(int,const Index&);

  // Multipplicative operations.
  friend inline Index operator-(const Index&);
  friend inline Index operator*(const Index&,int);
  friend inline Index operator*(int,const Index&);
  friend inline Index operator/(const Index&,int);

  // Intersect with another Index.
  Index intersect(const Index &) const;

  // Plug the base range of one into another.
  inline Index plugBase(const Index &) const; 

  // Test to see if two indexes are from the same base.
  inline bool sameBase(const Index&) const;

  // Test to see if there is any overlap between two Indexes.
  inline bool touches (const Index&a) const;
  // Test to see if one contains another (endpoints only)
  inline bool contains(const Index&a) const;
  // Test to see if one contains another (all points)
  inline bool containsAllPoints(const Index &b) const;
  // Split one into two.
  inline bool split(Index& l, Index& r) const;
  // Split index into two with a ratio between 0 and 1.
  inline bool split(Index& l, Index& r, double a) const;

  // iterator begin
  iterator begin() { return iterator(First,Stride); }
  // iterator end
  iterator end() { return iterator(First+Stride*Length,Stride); }

  // An operator< so we can impose some sort of ordering.
  bool operator<(const Index& r) const 
  {
    return (   (Length< r.Length) ||
	         ( (Length==r.Length) && (  (First<r.First) || 
                                      ( (First==r.First) && (Length>0) && (Stride<r.Stride) ) ) ) );
  }
  // Test for equality.
  bool operator==(const Index& r) const
  {
    return (Length==r.Length) && (First==r.First) && (Stride==r.Stride);
  }

  static void findPut(const Index&,const Index&, const Index&,Index&,Index&);

  // put data into a message to send to another node
  Message& putMessage(Message& m) const {
    int dbuf[3];
    int *d = dbuf;
    d[0] = first();
    d[1] = stride();
    d[2] = length();
    m.put(d, d + 3);
    return m;
  }

  // get data out from a message
  Message& getMessage(Message& m) {
    int dbuf[3];
    int *d = dbuf;
    m.get(d);
    *this = Index(d[0], d[0] + (d[2] - 1)*d[1], d[1]);
    return m;
  }

  // PETE interface.
  typedef cursor PETE_Expr_t;
  cursor MakeExpression() const { return cursor(*this); }

private: 

  // Here is the first element, the number of elements and the stride.
  int First;
  int Stride;
  unsigned Length;
  
  // Here we store the first element of the base index.
  // This gets updated whenever we do index or set operations
  // so we can do inverses quickly and easily.
  unsigned BaseFirst;
   
  // Keep id for the base so we can tell when two
  // indexes come from the same base.
  int Base;

  // Make an Index that interally counts the other direction.
  inline Index reverse() const;

  // Construct with a given base. This is private because
  // the interface shouldn't depend on how this is done.
  inline Index(int m, int a, const Index &b);
  inline Index(int f, int s, const Index *b);

  // Do a general intersect if the strides are not both 1.
  Index general_intersect(const Index&) const;

  // Provide a way to not initialize on construction. 
  class DontInitialize {};
  Index(DontInitialize) {}
};

//////////////////////////////////////////////////////////////////////

#include "Index/IndexInlines.h"

//////////////////////////////////////////////////////////////////////

#endif // INDEX_H

/***************************************************************************
 * $RCSfile: Index.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: Index.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
