//
// Class Kokkos_LField
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
#include "Field/Kokkos_LField.h"

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
// to the beginning of where in each malloced storage block the Kokkos_LField
// data is stored, to try to avoid having several blocks all map to
// the same cache line.  This is the maximum number of blocks that we
// will add as an offset, where each block is the size of a cache line.
#ifndef IPPL_OFFSET_BLOCKS
#define IPPL_OFFSET_BLOCKS 16
#endif

// a debugging output message macro
#ifdef DEBUG_Kokkos_LField
#define Kokkos_LFieldMSG(x) x
#else
#define Kokkos_LFieldMSG(x)
#endif


//////////////////////////////////////////////////////////////////////
//
// Initialize numeric types to zero.
// Everything else uses the default ctor.
//
//////////////////////////////////////////////////////////////////////

template<class T>
struct Kokkos_LFieldInitializer
{
  static void apply(T&) {}
};

/*
#define MAKE_INITIALIZER(T)        \
template <>                        \
struct Kokkos_LFieldInitializer<T>        \
{                                  \
  static void apply(T& x) { x=0; } \
};
*/

// MAKE_INITIALIZER(bool)
// MAKE_INITIALIZER(char)
// MAKE_INITIALIZER(short)
// MAKE_INITIALIZER(int)
// MAKE_INITIALIZER(long)
// MAKE_INITIALIZER(float)
// MAKE_INITIALIZER(double)
// MAKE_INITIALIZER(long long)

//////////////////////////////////////////////////////////////////////
//
// Construct given the sizes.
// This builds it compressed.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::Kokkos_LField(const NDIndex<Dim>& owned,
                      const NDIndex<Dim>& allocated,
                      int vnode)
: vnode_m(vnode),
//   P(0),
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

  // Give the Kokkos_LField some initial (compressed) value
  Kokkos_LFieldInitializer<T>::apply(*Begin);

  // If we are not actually doing compression, expand the storage out,
  // and copy the initial value to all the elements
  if (IpplInfo::noFieldCompression)
    this->ReallyUncompress(true);

  //INCIPPLSTAT(incKokkos_LFields);
}

//UL: for pinned mempory allocation
template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::Kokkos_LField(const NDIndex<Dim>& owned,
                      const NDIndex<Dim>& allocated,
                      int vnode, bool p)
  : vnode_m(vnode),
//     P(0),
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

  // Give the Kokkos_LField some initial (compressed) value
  Kokkos_LFieldInitializer<T>::apply(*Begin);

  // If we are not actually doing compression, expand the storage out,
  // and copy the initial value to all the elements
  if (IpplInfo::noFieldCompression)
    this->ReallyUncompress(true);

  //INCIPPLSTAT(incKokkos_LFields);
}

//////////////////////////////////////////////////////////////////////
//
// Deep copy constructor.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::Kokkos_LField(const Kokkos_LField<T,Dim>& lf)
  : vnode_m(lf.vnode_m),
//     P(0),
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
      // Make sure we have something in this Kokkos_LField
      PAssert_NE(lf.Allocated.size(), 0);

      // If it is not compressed, allocate storage
      int n = lf.Allocated.size();
      allocateStorage(n);

      // Copy the data over.
      //FIXME
//       std::copy(lf.P, lf.P + n, P);

      // Build an iterator that counts over the real data.
      //FIXME
//       Begin = iterator(P,Owned,Allocated,CompressedData);
    }

  //INCIPPLSTAT(incKokkos_LFields);
}


//////////////////////////////////////////////////////////////////////
//
// Destructor: just free the memory, if it's there.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
Kokkos_LField<T,Dim>::~Kokkos_LField()
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
Kokkos_LField<T,Dim>::TryCompress(bool baseOnPhysicalCells)
{



  if (IsCompressed() || IpplInfo::noFieldCompression)
    return false;

  Kokkos_LFieldMSG(Inform dbgmsg("Kokkos_LField::TryCompress", INFORM_ALL_NODES));
  Kokkos_LFieldMSG(dbgmsg << "Trying to compress Kokkos_LField with domain = "<<getOwned());
  Kokkos_LFieldMSG(dbgmsg << ", baseOnPhysicalCells = " << baseOnPhysicalCells<<endl);

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
Kokkos_LField<T,Dim>::CanCompress(T val) const
{
/*
 * FIXME


  // Debugging macro
  Kokkos_LFieldMSG(Inform dbgmsg("CanCompress"));

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
      Kokkos_LFieldMSG(dbgmsg << "Doing short-cut check, comparing " << *mid1);
      Kokkos_LFieldMSG(dbgmsg << " to " << val << " at last-alloc-domain-failed");
      Kokkos_LFieldMSG(dbgmsg << " index of " << allocCompressIndex << endl);
      ADDIPPLSTAT(incCompressionCompares, 1);

      if (!(*mid1 == val))
        {
          Kokkos_LFieldMSG(dbgmsg << "Short-cut check determined we cannot ");
          Kokkos_LFieldMSG(dbgmsg << "compress, by comparing " << *mid1<<" to ");
          Kokkos_LFieldMSG(dbgmsg << val << " at last-alloc-domain-failed index");
          Kokkos_LFieldMSG(dbgmsg << " of " << allocCompressIndex << endl);

          // It failed the test, so we can just keep the same index to
          // check next time, and return.
          return false;
        }
    }

  // Check from the beginning to the last-checked-index

  Kokkos_LFieldMSG(dbgmsg << "Checking for compression for " << sz << " items, ");
  Kokkos_LFieldMSG(dbgmsg << "comparing to value = " << val << endl);

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
              Kokkos_LFieldMSG(dbgmsg << "Found that we cannot compress, after ");
              Kokkos_LFieldMSG(dbgmsg << (checkptr - mid1) << " compares (");
              Kokkos_LFieldMSG(dbgmsg << *(checkptr-1) << " != " << val << ")");
              Kokkos_LFieldMSG(dbgmsg << endl);
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
              Kokkos_LFieldMSG(dbgmsg << "Found that we cannot compress, after ");
              Kokkos_LFieldMSG(dbgmsg << (checkptr - ptr1) + (end1 - mid1));
              Kokkos_LFieldMSG(dbgmsg << " compares (");
              Kokkos_LFieldMSG(dbgmsg << *(checkptr-1) << " != " << val << ")");
              Kokkos_LFieldMSG(dbgmsg << endl);
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
              Kokkos_LFieldMSG(dbgmsg << "Found that we cannot compress, after ");
              Kokkos_LFieldMSG(dbgmsg << (ptr1 - P) << " compares (");
              Kokkos_LFieldMSG(dbgmsg << *(ptr1-1) << " != " << val << ")");
              Kokkos_LFieldMSG(dbgmsg << endl);
              ADDIPPLSTAT(incCompressionCompares, (ptr1 - P));
              allocCompressIndex = (ptr1 - P) - 1;
              return false;
            }
        }
    }

  // If we are at this point, we did not find anything that did not
  // match, so we can compress (woo hoo).

  Kokkos_LFieldMSG(dbgmsg << "Found that we CAN compress, after " << sz);
  Kokkos_LFieldMSG(dbgmsg << " compares." << endl);
  ADDIPPLSTAT(incCompressionCompares, sz);
  allocCompressIndex = 0;
  */
  return true;
}


//////////////////////////////////////////////////////////////////////
//
// Return true if this Kokkos_LField can be compressed based on physical
// cells only and false if it could not.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
bool Kokkos_LField<T,Dim>::CanCompressBasedOnPhysicalCells() const
{
/*
 * FIXME


  // Debugging macro

  Kokkos_LFieldMSG(Inform dbgmsg("Kokkos_LField::CanCompressBasedOnPhysicalCells",
                          INFORM_ALL_NODES));

  // We definitely can't do this if compression is disabled.
  if (IpplInfo::noFieldCompression)
    return false;

  // If it is already compressed, we can compress it to any value.
  if (IsCompressed())
    return true;

  // Make an iterator over my owned domain. The cast is there because
  // this version of begin() is not a const member function.

  iterator p = const_cast<Kokkos_LField<T,Dim>*>(this)->begin(getOwned());

  // Get the value to compare against, either the first item or
  // an item from the last point where our compression check failed.

  T val = *p;
  int sz = getOwned().size();
  if (IpplInfo::extraCompressChecks && ownedCompressIndex > 0)
    {
      // There was a previous value, so get that one to compare against
      PAssert_LT((unsigned int) ownedCompressIndex, getAllocated().size());
      val = *(P + ownedCompressIndex);
      Kokkos_LFieldMSG(dbgmsg << "Checking owned cells using previous ");
      Kokkos_LFieldMSG(dbgmsg << "comparison value " << val << " from index = ");
      Kokkos_LFieldMSG(dbgmsg << ownedCompressIndex << " against " << sz);
      Kokkos_LFieldMSG(dbgmsg << " elements." << endl);
    }
  else
    {
      // We just use the first element, and will compare against
      // the rest, so we know we can skip comparing to this first element.
      ++p;
      --sz;
      Kokkos_LFieldMSG(dbgmsg << "Checking owned cells using first element " << val);
      Kokkos_LFieldMSG(dbgmsg << " for comparison against " << sz << " items."<<endl);
    }

  // Loop through the other physical cells until we encounter one that
  // doesn't match the 1st cell. If this occurs, we can't compress.

  ADDIPPLSTAT(incCompressionCompareMax, sz - 1);
  for (int i=0; i < sz; ++i, ++p)
    {
      if (!(*p == val))
        {
          Kokkos_LFieldMSG(dbgmsg << "Found that we cannot compress, after ");
          Kokkos_LFieldMSG(dbgmsg << i + 1 << " compares." << endl);
          ADDIPPLSTAT(incCompressionCompares, i + 1);
          ownedCompressIndex = (&(*p)) - P;
          Kokkos_LFieldMSG(dbgmsg << "changed ownedCompressIndex to ");
          Kokkos_LFieldMSG(dbgmsg << ownedCompressIndex << endl);
          return false;
        }
    }

  // Since we made it here, we can compress.

  Kokkos_LFieldMSG(dbgmsg << "Found that we CAN compress, after ");
  Kokkos_LFieldMSG(dbgmsg << sz << " compares." << endl);
  ADDIPPLSTAT(incCompressionCompares, sz);
  ownedCompressIndex = (-1);
  */
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
Kokkos_LField<T,Dim>::Compress(const T& val)
{
    /*
     * FIXME

  Kokkos_LFieldMSG(Inform dbgmsg("Kokkos_LField::Compress", INFORM_ALL_NODES));
  Kokkos_LFieldMSG(dbgmsg << "Compressing Kokkos_LField with domain = " << getOwned());
  Kokkos_LFieldMSG(dbgmsg << " to new value = " << val << ", already compressed = ");
  Kokkos_LFieldMSG(dbgmsg << (IsCompressed() ? 1 : 0) << endl);

  // When compression is disabled, interpret this to mean "assign every element
  // of the Kokkos_LField to the specified value," which is equivalent to compressing
  // the Kokkos_LField to the value then uncompressing it:

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
  */
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
Kokkos_LField<T,Dim>::CompressBasedOnPhysicalCells()
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
void Kokkos_LField<T,Dim>::ReallyUncompress(bool /*fill_domain*/)
{
/*
 * FIXME


  PAssert_NE(Allocated.size(), 0);

  // Allocate the data.

  int n = Allocated.size();
  allocateStorage(n);

  Kokkos_LFieldMSG(Inform dbgmsg("Kokkos_LField::ReallyUncompress", INFORM_ALL_NODES));
  Kokkos_LFieldMSG(dbgmsg << "Uncompressing Kokkos_LField with domain = " << getOwned());
  Kokkos_LFieldMSG(dbgmsg << ", fill_domain = " << (fill_domain ? 1 : 0) << endl);

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
  */
}


//////////////////////////////////////////////////////////////////////
//
// get an iterator over a subrange.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
typename Kokkos_LField<T,Dim>::iterator
Kokkos_LField<T,Dim>::begin(const NDIndex<Dim>& domain)
{
    /*
     * FIXME
  // Remove this profiling because this is too lightweight.
  //
  //
  return iterator(P,domain,Allocated,CompressedData);
  */
  return nullptr;
}


//////////////////////////////////////////////////////////////////////
//
// Get an iterator over a subrange, when we might want to try to
// compress the data in the subrange without affecting the rest of
// the Kokkos_LField data.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
typename Kokkos_LField<T,Dim>::iterator
Kokkos_LField<T,Dim>::begin(const NDIndex<Dim>& domain, T& compstore)
{
/* FIXME
  if (IsCompressed())
    compstore = CompressedData;
  return iterator(P,domain,Allocated,compstore);
  */
    return nullptr;
}


//////////////////////////////////////////////////////////////////////
//
// Swap the pointers between two Kokkos_LFields.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
Kokkos_LField<T,Dim>::swapData( Kokkos_LField<T,Dim>& a )
{
/* FIXME


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
  */
}


//////////////////////////////////////////////////////////////////////
//
// Actualy allocate storage for the Kokkos_LField data, doing any special
// memory tricks needed for performance.  Sets P pointer to new memory.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
Kokkos_LField<T,Dim>::allocateStorage(int newsize)
{
    /* FIXME
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
  */
}


//////////////////////////////////////////////////////////////////////
//
// Actually free the storage used in the Kokkos_LField, if any.  Resets P to zero.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
Kokkos_LField<T,Dim>::deallocateStorage()
{
    /*
  if (P != 0)
    {
      // Determine how many blocks to offset the data, if we are asked to.
      // If so, move the P pointer back.

      if (IpplInfo::offsetStorage)
        P -= (offsetBlocks*IPPL_CACHE_LINE_SIZE / sizeof(T));

      delete [] P;
      P = 0;
    }
    */
}


//////////////////////////////////////////////////////////////////////
//
// print an Kokkos_LField out
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void Kokkos_LField<T,Dim>::write(std::ostream& out) const
{


  for (iterator p = begin(); p!=end(); ++p)
    out << *p << " ";
}