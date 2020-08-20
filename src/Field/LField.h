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
#ifndef LFIELD_H
#define LFIELD_H

// include files
#include "Field/CompressedBrickIterator.h"

#include <iostream>

// forward declarations
template <class T, unsigned Dim> class LField;
template <class T, unsigned Dim>
std::ostream& operator<<(std::ostream&, const LField<T,Dim>&);

// Template metaprogram to calculate Dim**N. Used to
// reserve space in the overlap cache below.

template <int Dim>
struct ToTheDim
{
  inline static int calc(int n)
  {
    return ToTheDim<Dim-1>::calc(n) * n;
  }
};

template<>
struct ToTheDim<1>
{
  inline static int calc(int n)
  {
    return n;
  }
};


//////////////////////////////////////////////////////////////////////

// This stores the local data for a Field.
template<class T, unsigned Dim>
class LField
{

public:
  // An iterator for the contents of this LField.
  typedef CompressedBrickIterator<T,Dim> iterator;

  // The type of domain stored here
  typedef NDIndex<Dim> Domain_t;

  //
  // Constructors and destructor
  //

  // Ctors for an LField.  Arguments:
  //     owned = domain of "owned" region of LField (without guards)
  //     allocated = domain of "allocated" region, which includes guards
  //     vnode = global vnode ID number (see below)
  LField(const NDIndex<Dim>& owned,
         const NDIndex<Dim>& allocated,
         int vnode = -1);

  //UL: for pinned memory allocation
  LField(const NDIndex<Dim>& owned,
         const NDIndex<Dim>& allocated,
         int vnode,
         bool p);

  // Copy constructor.
  LField(const LField<T,Dim>&);

  // Destructor: just free the memory, if it's there.

  ~LField();

  //
  // General information accessors
  //

  // Return information about the LField.
  int size(unsigned d) const { return Owned[d].length(); }
  const NDIndex<Dim>& getAllocated()   const { return Allocated; }
  const NDIndex<Dim>& getOwned()       const { return Owned; }
  T*    getP() { return P; }

  // Return global vnode ID number (between 0 and nvnodes - 1)
  int getVnode() const { return vnode_m; }

  //
  // iterator interface
  //

  // Return begin/end iterators for the LField data
  const iterator&     begin()          const { return Begin; }
  const iterator&     end()            const { return End; }

  // get an iterator over a subrange.
  iterator begin(const NDIndex<Dim>& domain);

  // Get an iterator over a subrange, when we might want to try to
  // compress the data in the subrange without affecting the rest of
  // the LField data.
  // The LField iterator here must be told about a specific
  // location into which to store a compressed value, since this
  // iterator is used only to create a message.  Since the intersect
  // region may not be the whole LField, we cannot use the LField's
  // storage, we need to provide our own (otherwise, when compressing
  // the Field, we'll write the compressed value for ALL current
  // iterators on the LField which use the LField's compression
  // storage).
  iterator begin(const NDIndex<Dim>& domain, T&);

  //
  // Compression handling.
  //

  // Let the user ask if we are already compressed.
  inline bool IsCompressed() const
  {
    return P==0;
  }

  // Let the user tell us to try to compress.
  // Return quickly if we already are compressed.
  // If the argument is true, then only examine the owned domain to determine
  // if all the values are the same.
  bool TryCompress(bool baseOnPhysicalCells = false);

  // Look through the data and figure out if it can be compressed.
  inline bool CanCompress() const
  {
    if (!IsCompressed())
      return CanCompress(*Begin);
    return true;
  }

  // Look through the data and figure out if it can be compressed
  // to the given value.  Return true if it can be compressed down to the
  // given value.  If this returns false, then the data is currently not
  // compressed and contains different values.
  bool CanCompress(T x) const;

  // Force a compress.  Delete the memory and make Begin compressed.
  // First is a version that uses the first value.
  inline void Compress()
  {
    if (!IsCompressed())
      Compress(*P);
  }

  // Here is version that lets the user specify a new value.
  void Compress(const T &val);

  // Let the user tell us to uncompress.
  // Return quickly if we are already uncompressed.
  inline void Uncompress(bool fill_domain = true)
  {
    if (IsCompressed())
      ReallyUncompress(fill_domain);
  }

  // Return a reference to the compressed data for debugging.
  T &getCompressedData()             { return CompressedData; }
  const T &getCompressedData() const { return CompressedData; }

  //
  // Overlap cache interface
  //

  bool OverlapCacheInitialized() { return overlapCacheInited; }

  void AddToOverlapCache(LField<T, Dim> *newCacheItem)
    {
      if (overlap.size() == 0)
        overlap.reserve(ToTheDim<Dim>::calc(3)-1);
      overlap.push_back(newCacheItem);
      overlapCacheInited = true;
    }

  typedef typename std::vector< LField<T, Dim> *>::iterator OverlapIterator;

  OverlapIterator BeginOverlap() { return overlap.begin(); }
  OverlapIterator EndOverlap() { return overlap.end(); }

  //
  // Swap the pointers between two LFields.
  //

  void swapData( LField<T,Dim>& a );

  //
  // I/O
  //

  // print an LField out
  void write(std::ostream&) const;

private:
  // Global vnode ID number for the associated Vnode (useful with more recent
  // FieldLayouts which store a logical "array" of vnodes; user specifies
  // numbers of vnodes along each direction). Classes or user codes that use
  // LField are responsible for setting and managing the values of this index;
  // if unset, it has the value -1. Generally, this parameter value is set on
  // construction of the vnode:

  int vnode_m;

  // The actual pointer to the LField data.

  T*             P;

  // UL: for pinned memory allocation

  bool           Pinned;

  // What domain in the data is owned by this LField.

  NDIndex<Dim>   Owned;

  // How total domain is actually allocated for thie LField (including guards)

  NDIndex<Dim>   Allocated;

  // An iterator pointing to the first element in the owned domain.

  iterator       Begin;

  // An iterator pointing one past the last element in the owned domain.

  iterator       End;

  // If compressed, put the data here.  If not compressed, this is not used.

  T              CompressedData;

  // The overlap cache, an whether it has been initialized.

  std::vector< LField<T, Dim> * > overlap;
  bool overlapCacheInited;

  // The index of the element to start comparing to the first element
  // in a "CanCompress" check.  This is generally set to the index of the
  // first element that failed a compression check last time.  There are
  // two versions, one for when we're checking using the entire allocated
  // domain, and another when we're checking using just the owned domain.

  mutable int allocCompressIndex;
  mutable int ownedCompressIndex;

  // A counter for an offset that is used to keep data from always aligning
  // at the same point in a memory page.

  long offsetBlocks;

  // Private methods used to implement compression

  bool CanCompressBasedOnPhysicalCells() const;
  void ReallyUncompress(bool fill_domain);
  void CompressBasedOnPhysicalCells();

  // Actualy allocate storage for the LField data, doing any special
  // memory tricks needed for performance.  Sets P pointer to new memory.
  void allocateStorage(int newsize);

  // Actually free the storage used in the LField, if any.  Resets P to zero.
  void deallocateStorage();


  // Disable default constructor and operator=

  LField();
  const LField<T,Dim> &operator=(const LField<T,Dim> &);
};


template<class T, unsigned Dim>
inline
std::ostream& operator<<(std::ostream& out, const LField<T,Dim>& a)
{


  a.write(out);
  return out;
}

//////////////////////////////////////////////////////////////////////

#include "Field/LField.hpp"

#endif // LFIELD_H