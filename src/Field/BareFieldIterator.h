// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef BARE_FIELD_ITERATOR_H
#define BARE_FIELD_ITERATOR_H

/***************************************************************************
 *
 * A iterator class used for BareField.  This will store a ref to a
 * BareField and keep track of a position within it's LField's.  Note that
 * this is not the most efficient way to loop through the elements of
 * a BareField, it is more efficient to do an explicit loop over LField's
 * and then loop over each LField separately (this avoids continual if-tests
 * in the increment operator checking to see if you are at the end of an
 * LField).
 *
 ***************************************************************************/

// include files
#include "Index/NDIndex.h"
#include "Field/LField.h"
#include "Field/FieldLoc.h"
#include "Field/CompressedBrickIterator.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"

// forward declarations
template<class T, unsigned Dim> class BareField;

// An iterator over all the elements of a BareField.
template<class T, unsigned Dim>
class BareFieldIterator
  : public CompressedBrickIterator<T,Dim>
{
public:
  // Typedef for the type of data iterated over by this iterator
  typedef T return_type;

  // Default constructor
  BareFieldIterator()
    : CompressedBrickIterator<T,Dim>(dummy_compressed_data),
      MyBareField(0),
      LFIndex(-1)
  {
    // Increment the statistic on how many iterators were created.
    //INCIPPLSTAT(incDefaultBareFieldIterators);
  }

  // Construct with a BareField, and a flag indicating if the field
  // is constant.  If it is constant, we do not try to decompress.
  BareFieldIterator(BareField<T,Dim> &df)
    : CompressedBrickIterator<T,Dim>(dummy_compressed_data),
      CurrentLField(df.begin_if()),
      MyBareField(&df),
      LFIndex(0)
  {
    // set iterator at first element
    beginLField();

    // Increment the statistic on how many iterators were created.
    //INCIPPLSTAT(incBareFieldIterators);
  }

  // Construct with a BareField and a FieldLoc pointing to some relative
  // location in the Field, perhaps in a different LField than the
  // first one.  If it is constant, we do not try to decompress.
  BareFieldIterator(const FieldLoc<Dim> &loc, BareField<T,Dim> &df)
    : CompressedBrickIterator<T,Dim>(dummy_compressed_data),
      CurrentLField(df.begin_if()),
      MyBareField(&df),
      LFIndex(0)
  {
    // set iterator location to loc
    SetCurrentLocation(loc);

    // Increment the statistic on how many iterators were created.
    //INCIPPLSTAT(incBareFieldIterators);
  }

  // Copy constructor.
  BareFieldIterator(const BareFieldIterator<T,Dim> &iter)
    : CompressedBrickIterator<T,Dim>(iter),
      CurrentLField(iter.CurrentLField),
      MyBareField(iter.MyBareField),
      LFIndex(iter.LFIndex)
  {
    // Increment the statistic on how many iterators were created.
    //INCIPPLSTAT(incBareFieldIterators);
  }

  // Destructor
   ~BareFieldIterator() 
  {
  }

  // Assignment operator
  const BareFieldIterator<T,Dim> &
  operator=(const BareFieldIterator<T,Dim> &rhs)
  {
    // if they are the same, just return
    if (this == &rhs)
      return *this;

    // invoke base class operator=
    *(dynamic_cast<CompressedBrickIterator<T,Dim>*>(this)) = rhs;

    // copy data members
    LFIndex = rhs.LFIndex;
    CurrentLField = rhs.CurrentLField;
    MyBareField = rhs.MyBareField;

    return *this;
  }

  // Reset where we are pointing
  void SetCurrentLocation(const FieldLoc<Dim> &loc)
  {
    // Whoops!  This assumes our iterator is pointing to the first LField!
    // We need to make sure this is true.
    PAssert(MyBareField != NULL);
    CurrentLField = MyBareField->begin_if();
    LFIndex = 0;

    // Now advance to the requested LField, indicated by getIndex, which
    // returns the relative index of the LField we want to interate over.
    int curr = loc.getIndex();
    while (!done() && curr-- > 0)
      nextLField();

    if (!done()) {
      // find the relative offset in this LField
      const NDIndex<Dim>& domain = (*CurrentLField).second->getOwned();
      int offloc[Dim];
      for (unsigned int d=0; d < Dim; ++d)
	offloc[d] = (loc[d] - domain[d].first()) / domain[d].stride();

      // set the position in the current LField
      beginLField();
      this->moveBy(offloc);
    } else {
      IpplInfo::abort("Inconsistent FieldLoc in SetCurrentLocation.");
    }
  }
  
  // Set the given FieldLoc to the current position of this iterator.
  void GetCurrentLocation(FieldLoc<Dim> &loc) const
  {
    const NDIndex<Dim>& domain = (*CurrentLField).second->getOwned();
    for (unsigned d=0; d<Dim; ++d)
      loc[d] = this->GetOffset(d)*domain[d].stride() + domain[d].first();
    loc.setIndex(LFIndex);
  }

  // Just return the relative offset within the current LField
  void GetCurrentLocation(int *loc) const
  {
    const NDIndex<Dim>& domain = (*CurrentLField).second->getOwned();
    for (unsigned d=0; d<Dim; ++d)
      loc[d] = this->GetOffset(d)*domain[d].stride() + domain[d].first();
  }

  // Increment the iterator
  BareFieldIterator<T,Dim>& operator++()
  {
    CompressedBrickIterator<T,Dim>::operator++();
    if ( *this == (*CurrentLField).second->end() ) 
      general_increment();
    return *this;
  }

  // Dereference the iterator, and return a ref to the data
  T& operator*() const
  {
    PAssert(MyBareField != NULL);
    // We fill guard cells before using iterator
    //      PAssert(MyBareField->isDirty() ? !isInGC() : true);
    return CompressedBrickIterator<T,Dim>::operator*();
  }

  // Move on to the LField
  void nextLField()
  {
    ++CurrentLField; ++LFIndex;
  }

  // Check if we are at the end of the total iteration space
  bool done() const
  {
    PAssert(MyBareField != NULL);
    return CurrentLField == MyBareField->end_if();
  }

  // Check if we are done with iteration over just the specified dimension
  bool done(unsigned d) const 
  { 
    return CompressedBrickIterator<T,Dim>::done(d); 
  }

  // Set ourselves up to point to the beginning of the first LField
  void beginLField()
  {
    if (!done())
      *(CompressedBrickIterator<T,Dim>*)this = 
	(*CurrentLField).second->begin();
    else
      LFIndex = (-1);
  }

  // Report true if the data being iterated over is currently compressed
  bool IsCompressed() const
  {
    bool is_compressed = CompressedBrickIterator<T,Dim>::IsCompressed();
    PAssert_EQ((*CurrentLField).second->IsCompressed(), is_compressed);
    return is_compressed;
  }

  // Report true if the data being iterated over can be compressed
  bool CanCompress() const
  {
    PAssert(MyBareField != NULL);
    if (!MyBareField->compressible()) return false;
    return (*CurrentLField).second->CanCompress();
  }

  // Go and compress the current LField
  void Compress()
  {
    (*CurrentLField).second->Compress();
  }

  // Go and compress the current LField to the given value
  void Compress(T val)
  {
    (*CurrentLField).second->Compress(val);
  }

  // Try to compress the current LField
  bool TryCompress()
  {
    return (*CurrentLField).second->TryCompress();
  }

  // Return a reference to the BareField we're iterating over
  BareField<T,Dim> &GetBareField()
  {
    PAssert(MyBareField != NULL);
    return *MyBareField;
  }

  // Same, just for const BareField's
  const BareField<T,Dim> &GetBareField() const
  {
    PAssert(MyBareField != NULL);
    return *MyBareField;
  }

  //
  // Scalar code interface
  //

  // Prepare this BareField for some future scalar code use where the
  // BareField may be modified by BareFieldIterators.  This means
  // to make sure that the field is properly uncompressed and guard cells
  // are filled.  The
  // one argument allows the user to explicitly avoid filling guard
  // cells if they do not need this.
  // If tryfill is true, this routine must be called in SPMD-fashion.  If
  // it is false, it can be called on a per-node basis.
  void prepareForScalarCode(bool tryfill = true)
  {
    PAssert(MyBareField != NULL);
    MyBareField->prepareForScalarCode(tryfill);
  }

  // Finish up after a scalar code section.  This means go back to
  // a compressed state if possible, and, if the user says so, indicate
  // that things have been modified so the dirty flag must be set.
  // If modified is true, this routine must be called in SPMD-fashion.  If
  // it is false, it can be called on a per-node basis.
  void finishScalarCode(bool modified = true)
  {
    PAssert(MyBareField != NULL);
    MyBareField->finishScalarCode(modified);
  }

protected:
  // The current LField we're iterating within right now
  typename BareField<T,Dim>::iterator_if CurrentLField;

  // The BareField we're iterating over
  BareField<T,Dim>* MyBareField;

  // Storage for some compressed value if needed.
  T dummy_compressed_data;

  // The index of our current LField, from 0 ... # LField's - 1
  int LFIndex;

  // Increment and go on to the next LField if necessary.
  // Put this here because the SGI compiler doesn't
  // recognize it out of line.
  void general_increment()
  {
    nextLField();
    beginLField();
  }

  // Check to see if our current iteration position is inside the owned
  // domain of the current LField.
  bool isInGC() const
  {
    // Get the owned domain for our current LField
    const NDIndex<Dim> &owned = (*CurrentLField).second->getOwned();

    // Check to see if our current position is within this owned domain
    for (unsigned d=0; d<Dim; ++d)
      {
	unsigned locd = this->GetOffset(d)*owned[d].stride() + owned[d].first();
	if (!owned[d].contains(Index(locd,locd)))
	  return true;
      }

    // If we're here, we must not be in the owned domain.
    return false;
  }
};


#endif // BARE_FIELD_ITERATOR_H

/***************************************************************************
 * $RCSfile: BareFieldIterator.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: BareFieldIterator.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
