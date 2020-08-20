// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef BRICK_ITERATOR_H
#define BRICK_ITERATOR_H

// include files
#include "Utility/Vec.h"
#include "Message/Message.h"
#include "PETE/IpplExpressions.h"

// forward declarations
template <unsigned Dim> class NDIndex;

//----------------------------------------------------------------------
// A set of counters for a brick iterator.
template<unsigned Dim>
class BrickCounter
{
public: 

  // Null ctor takes no action.
  BrickCounter() {}

  // Construct to count over a contiguous block of user allocated memory.
  BrickCounter(const NDIndex<Dim>&);

  // Go to the next element.
  BrickCounter& operator++() { op_pp(); return *this; }

  // Finished stepping in dimension d?
  bool done(unsigned d) const { return Counters[d] == Counts[d]; }

  // Step in dimension d.
  void step(unsigned d)   { Counters[d] += 1; }
  void rewind(unsigned d) { Counters[d] = 0;  }

  // How big is it in dimension d?
  inline int size(unsigned d) const { return Counts[d]; }

  // Where are we now?
  int GetOffset(unsigned d) const { return Counters[d]; }

protected: 
  void op_pp();
  vec<int,Dim> Counters; // Where are we now.
  vec<int,Dim> Counts;   // The number of elements in each direction.

};


//----------------------------------------------------------------------
// An iterator for the elements of a brick of data.
template<class T, unsigned Dim>
class BrickIterator : public BrickCounter<Dim>
{
public:

  // Null ctor fills the current ptr with null. 
  BrickIterator() : Current(0), Whole(true) {}

  // Construct w/ an alloc ptr, Counted NDIndex and Alloc NDIndex
  BrickIterator(T*, const NDIndex<Dim>&, const NDIndex<Dim>&);

  // Construct with ptr, sizes.
  BrickIterator(T*, const vec<int,Dim>&);

  // Go to the next element.
  BrickIterator& operator++() { op_pp(); return *this; }

  // Does this iterator iterate over the entire (whole) brick?
  bool whole() const { return Whole; }

  // Two iterators are the same if they point to the same thing.
  bool operator==(const BrickIterator<T,Dim>& a) const
    {
      return Current == a.Current;
    }
  bool operator!=(const BrickIterator<T,Dim>& a) const
    {
      return Current != a.Current;
    }

  // Return what youre pointing to.
  T& operator*() const
    {
      return *Current;
    }

  // Move to new places.
  void step(unsigned d)
    {
      BrickCounter<Dim>::step( d );
      Current += Strides[ d ];
    }
  void rewind(unsigned d)
    {
      BrickCounter<Dim>::rewind( d );
      Current -= Strides[d]*BrickCounter<Dim>::Counts[d];
    }

  // Return something given an offset in one, two or three dimensions.
  T& offset(int i) const
    {
      return Current[ i*Strides[0] ];
    }
  T& offset(int i, int j) const
    {
      return Current[ i*Strides[0] + j*Strides[1] ];
    }
  T& offset(int i,int j,int k) const
    {
      return Current[ i*Strides[0]+ j*Strides[1] + k*Strides[2] ];
    }
  T& offset(int *i) const
    {
      return Current[ vec<int,Dim>::dot(i,&Strides[0]) ];
    }
  T& unit_offset(int i) const
    {
      return Current[ i ];
    }
  T& unit_offset(int i, int j) const
    {
      return Current[ i + j*Strides[1] ];
    }
  T& unit_offset(int i,int j,int k) const
    {
      return Current[ i+ j*Strides[1] + k*Strides[2] ];
    }
  void moveBy(int i)
    {
      Current += i*Strides[0];
      BrickCounter<Dim>::Counters[0] += i;
    }
  void moveBy(int i, int j)
    {
      Current += (i*Strides[0] + j*Strides[1]);
      BrickCounter<Dim>::Counters[0] += i;
      BrickCounter<Dim>::Counters[1] += j;
    }
  void moveBy(int i, int j, int k)
    {
      Current += (i*Strides[0] + j*Strides[1] + k*Strides[2]);
      BrickCounter<Dim>::Counters[0] += i;
      BrickCounter<Dim>::Counters[1] += j;
      BrickCounter<Dim>::Counters[2] += k;
    }
  void moveBy(const int *i)
    {
      for (unsigned int d=0; d < Dim; ++d) {
	Current += i[d] * Strides[d];
	BrickCounter<Dim>::Counters[d] += i[d];
      }
    }
  int Stride(int i) const { return Strides[i]; }

  // message passing interface ... for putMessage, the second argument
  // is used for an optimization for when the entire brick is being
  // sent.  If that is so, do not copy data into a new buffer, just
  // put the pointer into the message.  USE WITH CARE.  The default is
  // tohave putMessage make a copy of the data in the brick before adding
  // it to the message.  In many situations, this is required since the
  // data referred to by the iterator is not contiguous.  getMessage
  // has no such option, it always does the most efficient thing it can.
  Message& putMessage(Message&, bool makecopy = true);
  Message& getMessage(Message&);

  // PETE interface
  typedef T PETE_Return_t;
  typedef BrickIterator<T,Dim> PETE_Expr_t;
  PETE_Expr_t MakeExpression() const { return *this; }

protected: 
  void op_pp();
  T* __restrict__ Current;		// The current datum.
  vec<int,Dim> Strides;		// The strides in the data.
  bool Whole;			// True if iterating over whole brick
};

//////////////////////////////////////////////////////////////////////

#include "Field/BrickIterator.hpp"

#endif // BRICK_ITERATOR_H
