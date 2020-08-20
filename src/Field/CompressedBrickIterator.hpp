// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "Field/CompressedBrickIterator.h"
#include "Utility/PAssert.h"


//////////////////////////////////////////////////////////////////////
//
// The constructor that produces a compressed CompressedBrickIterator.
// All it really does is initialize the BrickIterator like normal
// except with a pointer to the CompressedData, with strides equal
// to zero.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
CompressedBrickIterator<T,Dim>::
CompressedBrickIterator(const NDIndex<Dim>& o, T& compressed)
{

  // Point to the single data element.
  BrickIterator<T,Dim>::Current = CompressedData = &compressed;
  for (unsigned d=0; d<Dim; ++d)
    {
      // The counts start off at zero.
      BrickCounter<Dim>::Counters[d] = 0;
      // The counts are just the lengths.
      BrickCounter<Dim>::Counts[d] = o[d].length();
      // Set all the strides to zero.
      BrickIterator<T,Dim>::Strides[d] = 0;
    }
}

//////////////////////////////////////////////////////////////////////
//
// The routine that checks to see if all the values are the same.
// This is a regular function so that we can do compile time recursion.
//
//////////////////////////////////////////////////////////////////////

//
// CompressedLoopTag
//
// A tag that we can use to get the compile recursion to work.
//
// If B==true then we have an explicit function available.
// If B==false we need to use the general loop.
//
// If Dim>3 we use the general loop, but it calls the
// one for Dim-1 so the inner loops are always efficient.
//

template<unsigned Dim, bool B=(Dim<=3)>
class CompressedLoopTag
{
};

//
// Here is the one dimensional version that checks if all the values
// in a block are the same.
//

template<class T, unsigned Dim>
inline bool
all_values_equal( const CompressedBrickIterator<T,Dim>& iter, T val,
		  CompressedLoopTag<1,true> )
		  //mwerks		  CompressedLoopTag<1> )
{

  // Loop over all the elements.
  int n = iter.size(0);
  for (int i=0; i<n; ++i)
    // If it is not the same, return failure.
    if ( val != iter.offset(i) )
      return false;
  // If we get to here then all the values were the same.
  return true;
}

//
// Here is the two dimensional version that checks if all the values
// in a block are the same.
//

template<class T, unsigned Dim>
inline bool
all_values_equal( const CompressedBrickIterator<T,Dim>& iter, T val ,
		  CompressedLoopTag<2,true> )
  //mwerks		  CompressedLoopTag<2> )
{

  // Loop over all of the elements.
  int n0 = iter.size(0);
  int n1 = iter.size(1);

  if ( (n0>0)&&(n1>0) )
    for (int i1=0; i1<n1; ++i1)
      for (int i0=0; i0<n0; ++i0)
	{
	  // If it is not the same, return failure.
	  if ( !(val == iter.offset(i0,i1)) )
	    return false;
	}
  // If we get to here then all the values were the same.
  return true;
}

//
// Here is the three dimensional version that checks if all the values
// in a block are the same.
//

template<class T, unsigned Dim>
inline bool
all_values_equal( const CompressedBrickIterator<T,Dim>& iter, T val ,
		  CompressedLoopTag<3,true> )
  //mwerks		  CompressedLoopTag<3> )
{

  // Loop over all of the elements.
  int n0 = iter.size(0);
  int n1 = iter.size(1);
  int n2 = iter.size(2);
  if ( (n0>0)&&(n1>0)&&(n2>0) )
    for (int i2=0; i2<n2; ++i2)
      for (int i1=0; i1<n1; ++i1)
	for (int i0=0; i0<n0; ++i0)
	  // If it is not the same, return failure.
	  if ( !(val == iter.offset(i0,i1,i2)) )
	    return false;
  // If we get to here then all the values were the same.
  return true;
}

//
// Here is the N dimensional version that checks if all the values
// in a block are the same.
//
// Note that for this one we pass iter by value instead of by
// reference because we use the step() member function.
//

template<class T, unsigned Dim1, unsigned Dim2>
inline bool
all_values_equal(CompressedBrickIterator<T,Dim1> iter, T val,
		 CompressedLoopTag<Dim2,false>)
{

  // Loop over the outermost dimension.
  int n = iter.size(Dim2-1);
  for (int i=0; i<n; ++i)
    {
      // Check if the next innermost dimension is all equal.
      //mwerks      if ( ! all_values_equal(iter,val,CompressedLoopTag<(Dim2-1)>()) )
      if ( ! all_values_equal(iter,val,
			      CompressedLoopTag<(Dim2-1),((Dim2-1)<=3)>()) )
	// If not, we're done.
	return false;
      // Otherwise step one in the outermost dimension.
      iter.step(Dim2-1);
    }
  // If we get to here they were all equal.
  return true;
}

//////////////////////////////////////////////////////////////////////
//
// The function that compresses the iterator if all the
// data it points to are equal to the given value.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
bool CompressedBrickIterator<T,Dim>::CanCompress(const T& val) const
{


  if ( IsCompressed() )
    return *CompressedData == val;
  else
    return all_values_equal(*this,val,CompressedLoopTag<Dim,(Dim<=3)>());
  //mwerks    return all_values_equal(*this,val,CompressedLoopTag<Dim>());
}


// put data into a message to send to another node
// ... for putMessage, the second argument
// is used for an optimization for when the entire brick is being
// sent.  If that is so, do not copy data into a new buffer, just
// put the pointer into the message.  USE WITH CARE.  The default is
// tohave putMessage make a copy of the data in the brick before adding
// it to the message.  In many situations, this is required since the
// data referred to by the iterator is not contiguous.  getMessage
// has no such option, it always does the most efficient thing it can.
template<class T, unsigned Dim>
Message& CompressedBrickIterator<T,Dim>::putMessage(Message& m, bool makecopy)
{

  // Add in flag indicating if we're compressed.  Put it in as an integer.
  int compressed = (IsCompressed() ? 1 : 0);
  m.put(compressed);
  if (compressed == 1)
    {
      // If we are compressed, just add in the sizes and the value.
      int s[Dim];
      for (unsigned int i=0; i < Dim; ++i)
	s[i] = BrickCounter<Dim>::size(i);
      m.put(s, s + Dim);
      ::putMessage(m, BrickIterator<T,Dim>::Current, BrickIterator<T,Dim>::Current + 1);
    }
  else
    {
      // If uncompressed, just do as a normal BrickIterator.
      BrickIterator<T,Dim>::putMessage(m, makecopy);
    }
  return m;
}

// get data out from a message
template<class T, unsigned Dim>
Message& CompressedBrickIterator<T,Dim>::getMessage(Message& m)
{
  // Inform msg("CBI::getMessage", INFORM_ALL_NODES);
  int compressed = 0;
  m.get(compressed);
  // msg << "  Compressed = " << compressed << endl;
  if (compressed == 1)
    {
      int s[Dim];
      m.get((int*) s);
      for (unsigned int i=0; i < Dim; ++i) {
	BrickCounter<Dim>::Counts[i] = s[i];
	BrickIterator<T,Dim>::Strides[i] = 0;
	BrickCounter<Dim>::Counters[i] = 0;
      }
      PAssert(CompressedData != 0);
      BrickIterator<T,Dim>::Current = CompressedData;
      ::getMessage_iter(m, BrickIterator<T,Dim>::Current);
      // msg << "  Current value = " << *Current << endl;
      // msg << "  Compres value = " << *CompressedData << endl;
    }
  else
    {
      //    ((BrickIterator<T,Dim>*)this)->getMessage(m);
      BrickIterator<T,Dim>::getMessage(m);
    }
  return m;
}

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D1, unsigned D2>
CompressedBrickIterator<T,D2>
permute(const CompressedBrickIterator<T,D1>& iter,
	const NDIndex<D1>& current, const NDIndex<D2>& perm)
{

  unsigned int d1, d2;

  // This is the iterator we'll be building.
  CompressedBrickIterator<T,D2> permute(iter.GetCompressedData());
  if ( iter.IsCompressed() )
    {
      permute = CompressedBrickIterator<T,D2>( perm, *iter );
    }
  else
    {
      // Set the pointer to the same place as the one passed in.
      permute.SetCurrent( &*iter );

      // Loop over each dimension of the iterator.
      for (d2=0; d2<D2; ++d2)
	{
	  // The size of the loop comes from the permuted NDIndex.
	  permute.SetCount(d2,perm[d2].length());
	  // Set the counters to zero.
	  permute.ResetCounter(d2);
	  // Set the stride to zero in case we don't find a match below.
	  permute.SetStride(d2,0);
	  // Check each Index in current to find a match.
	  for (d1=0; d1<D1; ++d1)
	    {
	      if ( current[d1].sameBase( perm[d2] ) )
		{
		  // Found it.  Get the stride for this loop.
		  permute.SetStride(d2,iter.GetStride(d1));
		  // On to the next.
		  break;
		}
	    }
	}
    }
  // Done constructing permute.
  return permute;
}

//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
const CompressedBrickIterator<T,Dim>&
CompressedBrickIterator<T,Dim>::
operator=(const CompressedBrickIterator<T,Dim>& rhs)
{

  if ( this != &rhs )
    {
      *(dynamic_cast<BrickIterator<T,Dim>*>(this)) = rhs;
      CompressedData = rhs.CompressedData;
    }
  return *this;
}

//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
CompressedBrickIterator<T,Dim>::
CompressedBrickIterator(const CompressedBrickIterator<T,Dim>& X)
  : BrickIterator<T,Dim>(X), CompressedData(X.CompressedData)
{

}

//////////////////////////////////////////////////////////////////////

// Make it compress to a given value.
// NOTE!!!! This function can only be useful in the following two contexts:
//   1. the iterator was constructed with specific internal storage for
//      CompressedData; and this external storage will be modified (the new
//      value 'val' will be written into the storage.
//   2. the iterator was constructed with no internal storage, and you call
//      Compress with an external variable for which you can take the address
//      and have 'CompressedData' point to.
template<class T, unsigned Dim>
void
CompressedBrickIterator<T,Dim>::Compress(T& val)
{


  // Inform msg("CBI::Compress", INFORM_ALL_NODES);
  // msg << "Before storing value " << val << ": ";
  // msg << "CompressedData = " << (void *)CompressedData;
  if (CompressedData != 0) {
    // msg << ", old deref value = " << *CompressedData;
    *CompressedData = val;
    // msg << ", new deref value = " << *CompressedData;
  } else {
    CompressedData = &val;
  }
  // msg << endl;
  BrickIterator<T,Dim>::Current = CompressedData;
  for (unsigned d=0; d<Dim; ++d)
    BrickIterator<T,Dim>::Strides[d] = 0;
}

//////////////////////////////////////////////////////////////////////

// Here is a version that lets the user specify a value
// to try sparsifying on.
template<class T, unsigned Dim>
bool
CompressedBrickIterator<T,Dim>::TryCompress(T val)
{


  // Inform msg("CBI::TryCompress", INFORM_ALL_NODES);
  // msg << "Trying to compress to value " << val;
  // msg << " : IsCompressed = " << IsCompressed() << endl;
  if ( IsCompressed() )
    return true;
  if ( CanCompress(val) )
    {
      // msg << "  Compressing now." << endl;
      // NOTE!!!! This next call will ONLY work if this iterator was
      // constructed with some external storage for the CompressedData.
      // If at the time of this call CompressedData == 0, then this will
      // just not work, since CompressedData will be set to the address of
      // val which is a temporary variable only active within the scope of
      // this function. (bfh)
      Compress(val);
      return true;
    }
  // msg << "  Cannot compress." << endl;
  return false;
}

/***************************************************************************
 * $RCSfile: CompressedBrickIterator.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: CompressedBrickIterator.cpp,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $
 ***************************************************************************/
