//
// Class LField
//   Local Field class
//
// Copyright (c) 2003 - 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Field/LField.h"

#include "Utility/PAssert.h"
#include "Utility/IpplStats.h"
#include "Utility/Unique.h"
#include <cstdlib>

// the number of bytes in a single cache line, this is generally set
// by configuration options, but gets a default value if none is given
#ifndef IPPL_CACHE_LINE_SIZE
#define IPPL_CACHE_LINE_SIZE 32
#endif

// the number of "offset blocks" to use.  We will add a small offset
// to the beginning of where in each malloced storage block the LField
// data is stored, to try to avoid having several blocks all map to
// the same cache line.  This is the maximum number of blocks that we
// will add as an offset, where each block is the size of a cache line.
#ifndef IPPL_OFFSET_BLOCKS
#define IPPL_OFFSET_BLOCKS 16
#endif

// a debugging output message macro
#ifdef DEBUG_LFIELD
#define LFIELDMSG(x) x
#else
#define LFIELDMSG(x)
#endif


//////////////////////////////////////////////////////////////////////
//
// Initialize numeric types to zero.
// Everything else uses the default ctor.
//
//////////////////////////////////////////////////////////////////////

template<class T>
struct LFieldInitializer
{
  static void apply(T&) {}
};

#define MAKE_INITIALIZER(T)        \
template <>                        \
struct LFieldInitializer<T>        \
{                                  \
  static void apply(T& x) { x=0; } \
};

MAKE_INITIALIZER(bool)
MAKE_INITIALIZER(char)
MAKE_INITIALIZER(short)
MAKE_INITIALIZER(int)
MAKE_INITIALIZER(long)
MAKE_INITIALIZER(float)
MAKE_INITIALIZER(double)
MAKE_INITIALIZER(long long)

//////////////////////////////////////////////////////////////////////
//
// Construct given the sizes.
// This builds it compressed.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
LField<T,Dim>::LField(const NDIndex<Dim>& owned,
                      const NDIndex<Dim>& allocated,
                      int vnode)
: vnode_m(vnode),
  P(0),
  Pinned(false),
  Owned(owned),
  Allocated(allocated),
  Begin(owned, CompressedData),
  End(CompressedData),
  overlapCacheInited(false),
  allocCompressIndex(0),
  ownedCompressIndex(-1),
  offsetBlocks(Unique::get() % IPPL_OFFSET_BLOCKS)
{

  // Give the LField some initial (compressed) value
  LFieldInitializer<T>::apply(*Begin);

  // If we are not actually doing compression, expand the storage out,
  // and copy the initial value to all the elements
  if (IpplInfo::noFieldCompression)
    this->ReallyUncompress(true);

  //INCIPPLSTAT(incLFields);
}

//UL: for pinned mempory allocation
template<class T, unsigned Dim>
LField<T,Dim>::LField(const NDIndex<Dim>& owned,
                      const NDIndex<Dim>& allocated,
                      int vnode, bool p)
  : vnode_m(vnode),
    P(0),
    Pinned(p),
    Owned(owned),
    Allocated(allocated),
    Begin(owned, CompressedData),
    End(CompressedData),
    overlapCacheInited(false),
    allocCompressIndex(0),
    ownedCompressIndex(-1),
    offsetBlocks(Unique::get() % IPPL_OFFSET_BLOCKS)
{

  // Give the LField some initial (compressed) value
  LFieldInitializer<T>::apply(*Begin);

  // If we are not actually doing compression, expand the storage out,
  // and copy the initial value to all the elements
  if (IpplInfo::noFieldCompression)
    this->ReallyUncompress(true);

  //INCIPPLSTAT(incLFields);
}

//////////////////////////////////////////////////////////////////////
//
// Deep copy constructor.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
LField<T,Dim>::LField(const LField<T,Dim>& lf)
  : vnode_m(lf.vnode_m),
    P(0),
    Pinned(false),
    Owned(lf.Owned),
    Allocated(lf.Allocated),
    Begin(CompressedData),
    End(CompressedData),
    overlapCacheInited(false),
    allocCompressIndex(lf.allocCompressIndex),
    ownedCompressIndex(lf.ownedCompressIndex),
    offsetBlocks(Unique::get() % IPPL_OFFSET_BLOCKS)
{



  if ( lf.IsCompressed() )
    {
      // Build a compressed iterator.
      Begin = iterator(Owned,CompressedData);

      // get the constant value in lf.
      CompressedData = lf.CompressedData;
    }
  else
    {
      // Make sure we have something in this LField
      PAssert_NE(lf.Allocated.size(), 0);

      // If it is not compressed, allocate storage
      int n = lf.Allocated.size();
      allocateStorage(n);

      // Copy the data over.
      std::copy(lf.P, lf.P + n, P);

      // Build an iterator that counts over the real data.
      Begin = iterator(P,Owned,Allocated,CompressedData);
    }

  //INCIPPLSTAT(incLFields);
}


//////////////////////////////////////////////////////////////////////
//
// Destructor: just free the memory, if it's there.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
LField<T,Dim>::~LField()
{
  deallocateStorage();
}


//////////////////////////////////////////////////////////////////////
//
// Let the user tell us to try to compress.
// Return quickly if we already are compressed.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
bool
LField<T,Dim>::TryCompress(bool baseOnPhysicalCells)
{



  if (IsCompressed() || IpplInfo::noFieldCompression)
    return false;

  LFIELDMSG(Inform dbgmsg("LField::TryCompress", INFORM_ALL_NODES));
  LFIELDMSG(dbgmsg << "Trying to compress LField with domain = "<<getOwned());
  LFIELDMSG(dbgmsg << ", baseOnPhysicalCells = " << baseOnPhysicalCells<<endl);

  if (baseOnPhysicalCells)
    {
      if (CanCompressBasedOnPhysicalCells())
        {
          CompressBasedOnPhysicalCells();
          return true;
        }
    }
  else
    {
      if (CanCompress() )
        {
          Compress();
          return true;
        }
    }

  return false;
}


//////////////////////////////////////////////////////////////////////
//
// Look through the data and figure out if it can be compressed
// to the given value.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
bool
LField<T,Dim>::CanCompress(T val) const
{



  // Debugging macro
  LFIELDMSG(Inform dbgmsg("CanCompress"));

  // We definitely can't do this if compression is disabled.
  if (IpplInfo::noFieldCompression)
    return false;

  // If it is already compressed, we can compress it to any value.
  if (IsCompressed())
    //return *Begin == val;
    return true;

  // It is not currently compressed ... so go through and check
  // to see if all the elements are the same as the given argument.

  int sz = getAllocated().size();
  ADDIPPLSTAT(incCompressionCompareMax, sz);
  T *ptr1 = P;
  T *mid1 = P + allocCompressIndex;
  T *end1 = P + sz;

  PAssert_GT(sz, 0);
  PAssert(P != 0);
  PAssert_GE(allocCompressIndex, 0);
  PAssert_LT(allocCompressIndex, sz);

  // Quick short-cut check: compare to the last value in the
  // array that did not match before.

  if (IpplInfo::extraCompressChecks)
    {
      LFIELDMSG(dbgmsg << "Doing short-cut check, comparing " << *mid1);
      LFIELDMSG(dbgmsg << " to " << val << " at last-alloc-domain-failed");
      LFIELDMSG(dbgmsg << " index of " << allocCompressIndex << endl);
      ADDIPPLSTAT(incCompressionCompares, 1);

      if (!(*mid1 == val))
        {
          LFIELDMSG(dbgmsg << "Short-cut check determined we cannot ");
          LFIELDMSG(dbgmsg << "compress, by comparing " << *mid1<<" to ");
          LFIELDMSG(dbgmsg << val << " at last-alloc-domain-failed index");
          LFIELDMSG(dbgmsg << " of " << allocCompressIndex << endl);

          // It failed the test, so we can just keep the same index to
          // check next time, and return.
          return false;
        }
    }

  // Check from the beginning to the last-checked-index

  LFIELDMSG(dbgmsg << "Checking for compression for " << sz << " items, ");
  LFIELDMSG(dbgmsg << "comparing to value = " << val << endl);

  if (IpplInfo::extraCompressChecks)
    {
      // First check from last-failed-position to end, since we've
      // already looked at *mid1 and should have that section of memory
      // in cache
      T *checkptr = mid1 + 1;
      while (checkptr != end1)
        {
          if (!(*checkptr++ == val))
            {
              LFIELDMSG(dbgmsg << "Found that we cannot compress, after ");
              LFIELDMSG(dbgmsg << (checkptr - mid1) << " compares (");
              LFIELDMSG(dbgmsg << *(checkptr-1) << " != " << val << ")");
              LFIELDMSG(dbgmsg << endl);
              ADDIPPLSTAT(incCompressionCompares, (checkptr - mid1));
              allocCompressIndex = (checkptr - ptr1) - 1;
              return false;
            }
        }

      // Next, check from the first position to the last-failed-position.
      checkptr = ptr1;
      while (checkptr != mid1)
        {
          if (!(*checkptr++ == val))
            {
              LFIELDMSG(dbgmsg << "Found that we cannot compress, after ");
              LFIELDMSG(dbgmsg << (checkptr - ptr1) + (end1 - mid1));
              LFIELDMSG(dbgmsg << " compares (");
              LFIELDMSG(dbgmsg << *(checkptr-1) << " != " << val << ")");
              LFIELDMSG(dbgmsg << endl);
              ADDIPPLSTAT(incCompressionCompares,
                           (checkptr - ptr1) + (end1 - mid1));
              allocCompressIndex = (checkptr - ptr1) - 1;
              return false;
            }
        }
    }
  else
    {
      while (ptr1 != end1)
        {
          if (!(*ptr1++ == val))
            {
              LFIELDMSG(dbgmsg << "Found that we cannot compress, after ");
              LFIELDMSG(dbgmsg << (ptr1 - P) << " compares (");
              LFIELDMSG(dbgmsg << *(ptr1-1) << " != " << val << ")");
              LFIELDMSG(dbgmsg << endl);
              ADDIPPLSTAT(incCompressionCompares, (ptr1 - P));
              allocCompressIndex = (ptr1 - P) - 1;
              return false;
            }
        }
    }

  // If we are at this point, we did not find anything that did not
  // match, so we can compress (woo hoo).

  LFIELDMSG(dbgmsg << "Found that we CAN compress, after " << sz);
  LFIELDMSG(dbgmsg << " compares." << endl);
  ADDIPPLSTAT(incCompressionCompares, sz);
  allocCompressIndex = 0;
  return true;
}


//////////////////////////////////////////////////////////////////////
//
// Return true if this LField can be compressed based on physical
// cells only and false if it could not.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
bool LField<T,Dim>::CanCompressBasedOnPhysicalCells() const
{



  // Debugging macro

  LFIELDMSG(Inform dbgmsg("LField::CanCompressBasedOnPhysicalCells",
                          INFORM_ALL_NODES));

  // We definitely can't do this if compression is disabled.
  if (IpplInfo::noFieldCompression)
    return false;

  // If it is already compressed, we can compress it to any value.
  if (IsCompressed())
    return true;

  // Make an iterator over my owned domain. The cast is there because
  // this version of begin() is not a const member function.

  iterator p = const_cast<LField<T,Dim>*>(this)->begin(getOwned());

  // Get the value to compare against, either the first item or
  // an item from the last point where our compression check failed.

  T val = *p;
  int sz = getOwned().size();
  if (IpplInfo::extraCompressChecks && ownedCompressIndex > 0)
    {
      // There was a previous value, so get that one to compare against
      PAssert_LT((unsigned int) ownedCompressIndex, getAllocated().size());
      val = *(P + ownedCompressIndex);
      LFIELDMSG(dbgmsg << "Checking owned cells using previous ");
      LFIELDMSG(dbgmsg << "comparison value " << val << " from index = ");
      LFIELDMSG(dbgmsg << ownedCompressIndex << " against " << sz);
      LFIELDMSG(dbgmsg << " elements." << endl);
    }
  else
    {
      // We just use the first element, and will compare against
      // the rest, so we know we can skip comparing to this first element.
      ++p;
      --sz;
      LFIELDMSG(dbgmsg << "Checking owned cells using first element " << val);
      LFIELDMSG(dbgmsg << " for comparison against " << sz << " items."<<endl);
    }

  // Loop through the other physical cells until we encounter one that
  // doesn't match the 1st cell. If this occurs, we can't compress.

  ADDIPPLSTAT(incCompressionCompareMax, sz - 1);
  for (int i=0; i < sz; ++i, ++p)
    {
      if (!(*p == val))
        {
          LFIELDMSG(dbgmsg << "Found that we cannot compress, after ");
          LFIELDMSG(dbgmsg << i + 1 << " compares." << endl);
          ADDIPPLSTAT(incCompressionCompares, i + 1);
          ownedCompressIndex = (&(*p)) - P;
          LFIELDMSG(dbgmsg << "changed ownedCompressIndex to ");
          LFIELDMSG(dbgmsg << ownedCompressIndex << endl);
          return false;
        }
    }

  // Since we made it here, we can compress.

  LFIELDMSG(dbgmsg << "Found that we CAN compress, after ");
  LFIELDMSG(dbgmsg << sz << " compares." << endl);
  ADDIPPLSTAT(incCompressionCompares, sz);
  ownedCompressIndex = (-1);
  return true;
}


//////////////////////////////////////////////////////////////////////
//
// Force a compression to a specified value.  This version compresses
// the entire allocated domain.  If this is called when compression
// is turned off, it instead copies the given value into the whole
// domain's storage, so that it at least makes the whole domain
// equal to the value.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
LField<T,Dim>::Compress(const T& val)
{



  LFIELDMSG(Inform dbgmsg("LField::Compress", INFORM_ALL_NODES));
  LFIELDMSG(dbgmsg << "Compressing LField with domain = " << getOwned());
  LFIELDMSG(dbgmsg << " to new value = " << val << ", already compressed = ");
  LFIELDMSG(dbgmsg << (IsCompressed() ? 1 : 0) << endl);

  // When compression is disabled, interpret this to mean "assign every element
  // of the LField to the specified value," which is equivalent to compressing
  // the LField to the value then uncompressing it:

  if (IpplInfo::noFieldCompression)
    {
      for (iterator lit = begin(); lit != end(); ++lit)
        *lit = val;

      return;
    }

  // Compression is enabled if we're here, so save the compressed value and
  // free up memory if necessary.  We copy the value into the compressed
  // value storage, and then if we're currently compressed, we free up
  // that memory and update our iterators.

  CompressedData = val;
  if (!IsCompressed())
    {
      Begin.Compress(CompressedData);
      deallocateStorage();
    }

  //INCIPPLSTAT(incCompresses);
}


//////////////////////////////////////////////////////////////////////
//
// This function does a compressed based on physical cells only.
// It will compress to the value of the first element in the owned
// domain (instead of in the allocated domain).  If compression is
// turned off, this does nothing, it does not even attempt to fill
// in the owned domain with a value.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
LField<T,Dim>::CompressBasedOnPhysicalCells()
{



  // We do nothing in this case if compression is turned off.

  if (IpplInfo::noFieldCompression)
    return;

  // Set compression value to first element in owned domain, and free up
  // memory if necessary.

  CompressedData = *(begin(getOwned()));
  if (!IsCompressed())
    {
      Begin.Compress(CompressedData);
      deallocateStorage();
    }

  //INCIPPLSTAT(incCompresses);
}


//////////////////////////////////////////////////////////////////////
//
// We know this is compressed, so uncompress it.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void LField<T,Dim>::ReallyUncompress(bool fill_domain)
{



  PAssert_NE(Allocated.size(), 0);

  // Allocate the data.

  int n = Allocated.size();
  allocateStorage(n);

  LFIELDMSG(Inform dbgmsg("LField::ReallyUncompress", INFORM_ALL_NODES));
  LFIELDMSG(dbgmsg << "Uncompressing LField with domain = " << getOwned());
  LFIELDMSG(dbgmsg << ", fill_domain = " << (fill_domain ? 1 : 0) << endl);

  // Copy the constant value into the new space.

  if (fill_domain)
    {
      T val = *Begin;
      for (int i=0; i<n; i++)
        P[i] = val;
    }

  // Make the Begin iterator point to the new data.

  Begin = iterator(P,Owned,Allocated,CompressedData);

  // Indicate we've done one more decompress

  //INCIPPLSTAT(incDecompresses);
}


//////////////////////////////////////////////////////////////////////
//
// get an iterator over a subrange.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
typename LField<T,Dim>::iterator
LField<T,Dim>::begin(const NDIndex<Dim>& domain)
{
  // Remove this profiling because this is too lightweight.
  //
  //
  return iterator(P,domain,Allocated,CompressedData);
}


//////////////////////////////////////////////////////////////////////
//
// Get an iterator over a subrange, when we might want to try to
// compress the data in the subrange without affecting the rest of
// the LField data.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
typename LField<T,Dim>::iterator
LField<T,Dim>::begin(const NDIndex<Dim>& domain, T& compstore)
{

  if (IsCompressed())
    compstore = CompressedData;
  return iterator(P,domain,Allocated,compstore);
}


//////////////////////////////////////////////////////////////////////
//
// Swap the pointers between two LFields.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
LField<T,Dim>::swapData( LField<T,Dim>& a )
{



  // Swap the pointers to the data.
  {
    T *temp=P;
    P=a.P;
    a.P=temp;
  }

  // Swap the compressed data.
  {
    T temp = CompressedData;
    CompressedData = a.CompressedData;
    a.CompressedData = temp;
  }

  // Swap the last-compared-for-compression indices
  {
    int temp = allocCompressIndex;
    allocCompressIndex = a.allocCompressIndex;
    a.allocCompressIndex = temp;
    temp = ownedCompressIndex;
    ownedCompressIndex = a.ownedCompressIndex;
    a.ownedCompressIndex = temp;
  }

  // Swap the offset block value
  {
    int temp = offsetBlocks;
    offsetBlocks = a.offsetBlocks;
    a.offsetBlocks = temp;
  }

  // Reinitialize the begin iterators.
  Begin = iterator(P,Owned,Allocated,CompressedData);
  a.Begin = iterator(a.P,a.Owned,a.Allocated,a.CompressedData);

  // Make sure the domains agree.
  PAssert(Owned == a.Owned);
  PAssert(Allocated == a.Allocated);

  // Should we swap the overlap caches?
}


//////////////////////////////////////////////////////////////////////
//
// Actualy allocate storage for the LField data, doing any special
// memory tricks needed for performance.  Sets P pointer to new memory.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
LField<T,Dim>::allocateStorage(int newsize)
{
  PAssert(P == 0);
  PAssert_GT(newsize, 0);
  PAssert_GE(offsetBlocks, 0);

  // Determine how many blocks to offset the data, if we are asked to

  int extra = 0;
  if (IpplInfo::offsetStorage)
    extra = offsetBlocks*IPPL_CACHE_LINE_SIZE / sizeof(T);

  // Allocate the storage, creating some extra to account for offset, and
  // then add in the offset.
  P = new T[newsize + extra]();
  P += extra;

  ADDIPPLSTAT(incLFieldBytes, (newsize+extra)*sizeof(T));
}


//////////////////////////////////////////////////////////////////////
//
// Actually free the storage used in the LField, if any.  Resets P to zero.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
LField<T,Dim>::deallocateStorage()
{
  if (P != 0)
    {
      // Determine how many blocks to offset the data, if we are asked to.
      // If so, move the P pointer back.

      if (IpplInfo::offsetStorage)
        P -= (offsetBlocks*IPPL_CACHE_LINE_SIZE / sizeof(T));

      delete [] P;
      P = 0;
    }
}


//////////////////////////////////////////////////////////////////////
//
// print an LField out
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void LField<T,Dim>::write(std::ostream& out) const
{


  for (iterator p = begin(); p!=end(); ++p)
    out << *p << " ";
}