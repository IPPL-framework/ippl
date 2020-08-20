// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef COMPRESSED_BRICK_ITERATOR_H
#define COMPRESSED_BRICK_ITERATOR_H

// include files
#include "Field/BrickIterator.h"

#include <iostream>

template<class T, unsigned Dim>
class CompressedBrickIterator;

//////////////////////////////////////////////////////////////////////

// Global function that lets us fake member templates.
template<class T, unsigned D1, unsigned D2>
CompressedBrickIterator<T,D2>
permute(const CompressedBrickIterator<T,D1>&,
	const NDIndex<D1>&, const NDIndex<D2>&);

//////////////////////////////////////////////////////////////////////






//////////////////////////////////////////////////////////////////////
//
// A version of BrickIterator that can do compression.
// If it detects that the block it is pointing to is constant,
// it can point to a single value and set its strides to zero.
// The location it uses for that single value must be passed in
// to the ctor.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
class CompressedBrickIterator : public BrickIterator<T,Dim>
{
public:
  // Replicate all the ctors from BrickIterator.

  // Construct w/
  //    a pointer to allocated memory
  //    owned NDIndex
  //    alloc NDIndex
  //    Location to store the compressed value.
  CompressedBrickIterator(T* t, const NDIndex<Dim>& c, const NDIndex<Dim>& a,
			  T& compressed)
    : BrickIterator<T,Dim>(t,c,a), CompressedData(&compressed)
    {
      if (!t) {
	BrickIterator<T,Dim>::Current = CompressedData;
	for (unsigned d=0; d<Dim; d++)
	  BrickIterator<T,Dim>::Strides[d] = 0;
      }
    }

  // Construct w/
  //    a pointer to allocated memory
  //    sizes
  //    Location to store the compressed value.
  CompressedBrickIterator(T* t, const vec<int,Dim>& v, T& compressed)
    : BrickIterator<T,Dim>(t,v),CompressedData(&compressed)
    {
      if (!t) {
	BrickIterator<T,Dim>::Current = CompressedData;
	for (unsigned d=0; d<Dim; d++)
	  BrickIterator<T,Dim>::Strides[d] = 0;
      }
    }
    
  // Construct with just a location for the compressed data.
  CompressedBrickIterator(T& t) : CompressedData(&t) {}

  // Null ctor for array allocations.
  // Not functional.  You have to overwrite this with a real one to use it.
  CompressedBrickIterator() : CompressedData(0) {}

  // Construct with just a domain and a compressed value.
  // This makes it compressed.
  CompressedBrickIterator(const NDIndex<Dim>& a, T& compressed);

  // Construct with a regular BrickIterator
  // and a place to store the compressed value.
  CompressedBrickIterator(const BrickIterator<T,Dim>& x, T& compressed)
    : BrickIterator<T,Dim>(x), CompressedData(&compressed)  {}

  // return true if it is currently compressed, false otherwise.
  bool IsCompressed() const
  {
    return BrickIterator<T,Dim>::Current == CompressedData;
  }

  // Check and see if it can compress.
  bool CanCompress(const T&) const;

  // Make it compress to a given value.
  void Compress(T& val);

  // Try to sparsify a CompressedBrickIterator.
  // Return true on success, false otherwise.
  // If it is already compressed it quickly returns true.
  bool TryCompress() { return TryCompress(**this); }

  // Here is a version that lets the user specify a value
  // to try sparsifying on.
  bool TryCompress(T val);

  // Since this has a potentially self-referential pointer,
  // we need the copy ctor and assignment operator to deal with it.
  const CompressedBrickIterator<T,Dim>&
  operator=(const CompressedBrickIterator<T,Dim>& rhs);

  CompressedBrickIterator(const CompressedBrickIterator<T,Dim>& X);

  // put data into a message to send to another node
  // ... for putMessage, the second argument
  // is used for an optimization for when the entire brick is being
  // sent.  If that is so, do not copy data into a new buffer, just
  // put the pointer into the message.  USE WITH CARE.  The default is
  // tohave putMessage make a copy of the data in the brick before adding
  // it to the message.  In many situations, this is required since the
  // data referred to by the iterator is not contiguous.  getMessage
  // has no such option, it always does the most efficient thing it can.
  Message& putMessage(Message& m, bool makecopy = true);

  // get data out from a message
  Message& getMessage(Message& m);

  // Permute the order of the loops (given by the first NDIndex) 
  // to correspond to the order in the second NDIndex.
  // Obviously this would be better done as a member template,
  // but we can fake it with a global function and some accessor functions.
  CompressedBrickIterator<T,1>
  permute(NDIndex<Dim>& current, NDIndex<1>& permuted) const
  { return ::permute(*this,current,permuted); }
  CompressedBrickIterator<T,2>
  permute(NDIndex<Dim>& current, NDIndex<2>& permuted) const
  { return ::permute(*this,current,permuted); }
  CompressedBrickIterator<T,3>
  permute(NDIndex<Dim>& current, NDIndex<3>& permuted) const
  { return ::permute(*this,current,permuted); }
  CompressedBrickIterator<T,4>
  permute(NDIndex<Dim>& current, NDIndex<4>& permuted) const
  { return ::permute(*this,current,permuted); }
  CompressedBrickIterator<T,5>
  permute(NDIndex<Dim>& current, NDIndex<5>& permuted) const
  { return ::permute(*this,current,permuted); }
  CompressedBrickIterator<T,6>
  permute(NDIndex<Dim>& current, NDIndex<6>& permuted) const
  { return ::permute(*this,current,permuted); }
  // The global function permute needs some special accessor functions.
  void SetCurrent(T* p) { BrickIterator<T,Dim>::Current = p; }
  void SetCount(int d, int count) { BrickCounter<Dim>::Counts[d] = count; }
  void ResetCounter(int d) { BrickCounter<Dim>::Counters[d] = 0; }
  void SetStride(int d, int stride) { BrickIterator<T,Dim>::Strides[d] = stride; }
  int GetStride(int d) const { return BrickIterator<T,Dim>::Strides[d]; }
  T& GetCompressedData() const { return *CompressedData; }
  void SetCompressedData(T *newData)
  { 
    CompressedData = newData;
  }

private:
  // If you are able to be compressed, put the constant value here.
  T* CompressedData;

};


#include "Field/CompressedBrickIterator.hpp"

#endif // COMPRESSED_BRICK_ITERATOR_H

/***************************************************************************
 * $RCSfile: CompressedBrickIterator.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: CompressedBrickIterator.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/

