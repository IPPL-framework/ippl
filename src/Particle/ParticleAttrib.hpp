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
#include "Particle/ParticleAttrib.h"
#include "Field/Field.h"
#include "Field/LField.h"
#include "Message/Message.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"
#include "Utility/IpplStats.h"
#include "AppTypes/AppTypeTraits.h"

/////////////////////////////////////////////////////////////////////
// Create a ParticleAttribElem to allow the user to access just the Nth
// element of the attribute stored here.
template <class T>
ParticleAttribElem<T,1U>
ParticleAttrib<T>::operator()(unsigned i) {
  PInsist(AppTypeTraits<T>::ElemDim > 0,
          "No operator()(unsigned) for this element type!!");
  return ParticleAttribElem<T,1U>(*this, vec<unsigned,1U>(i));
}


/////////////////////////////////////////////////////////////////////
// Same as above, but specifying two indices
template <class T>
ParticleAttribElem<T,2U>
ParticleAttrib<T>::operator()(unsigned i, unsigned j) {
  PInsist(AppTypeTraits<T>::ElemDim > 1,
          "No operator()(unsigned,unsigned) for this element type!!");
  return ParticleAttribElem<T,2U>(*this, vec<unsigned,2U>(i,j));
}


/////////////////////////////////////////////////////////////////////
// Same as above, but specifying three indices
template <class T>
ParticleAttribElem<T,3U>
ParticleAttrib<T>::operator()(unsigned i, unsigned j, unsigned k) {
  PInsist(AppTypeTraits<T>::ElemDim > 2,
          "No operator()(unsigned,unsigned,unsigned) for this element type!!");
  return ParticleAttribElem<T,3U>(*this, vec<unsigned,3U>(i,j,k));
}


/////////////////////////////////////////////////////////////////////
// Create storage for M particle attributes.  The storage is uninitialized.
// New items are appended to the end of the array.
template<class T>
void ParticleAttrib<T>::create(size_t M) {
  
  // make sure we have storage for M more items
  // and push back M items, using default value
  if (M > 0)
  {
	//ParticleList.insert(ParticleList.end(), M, T());
    ParticleList.insert(ParticleList.begin()+LocalSize, M, T());
    LocalSize+=M;
    attributeIsDirty_ = true;
  }

}


/////////////////////////////////////////////////////////////////////
// Delete the attribute storage for M particle attributes, starting at
// the position I.  Boolean flag indicates whether to use optimized
// destroy method.  This function really erases the data, which will
// change local indices of the data.  The optimized method just copies
// data from the end of the storage into the selected block.  Otherwise,
// we use the std::vector erase method, which preserves data ordering.

template<class T>
void ParticleAttrib<T>::destroy(size_t M, size_t I, bool optDestroy) {
  

  if (M == 0) return;
  if (optDestroy) {
    // get iterators for where the data to be deleted begins, and where
    // the data we copy from the end begins
    typename ParticleList_t::iterator putloc = ParticleList.begin() + I;
    typename ParticleList_t::iterator getloc = ParticleList.begin()+LocalSize - M;
    typename ParticleList_t::iterator endloc = ParticleList.begin()+LocalSize;

    // make sure we do not copy too much
    //if ((I + M) > (ParticleList.size() - M))
    if ((I + M) > (LocalSize - M))
      getloc = putloc + M;

    // copy over the data
    while (getloc != endloc)
      *putloc++ = *getloc++;
    // delete the last M items
    ParticleList.erase(endloc - M, endloc);
  }
  else {
    // just use the erase method
    typename ParticleList_t::iterator loc = ParticleList.begin() + I;
    ParticleList.erase(loc, loc + M);
  }
  LocalSize-=M;
  attributeIsDirty_ = true;
  return;
}


/////////////////////////////////////////////////////////////////////
// Delete the attribute storage for a list of particle destroy events
// This really erases the data, which will change local indices
// of the data.  If we are using the optimized destroy method,
// this just copies data from the end of the storage into the selected
// block.  Otherwise, we use a leading/trailing iterator semantic to
// copy data from below and  preserve data ordering.

template <class T>
void ParticleAttrib<T>::destroy(const std::vector< std::pair<size_t,size_t> >& dlist,
                                bool optDestroy)
{
  

  if (dlist.empty()) return;
  typedef std::vector< std::pair<size_t,size_t> > dlist_t;
  if (optDestroy) {
    // process list in reverse order, since we are backfilling
    dlist_t::const_reverse_iterator rbeg, rend = dlist.rend();
    // find point to copy data from
    typename ParticleList_t::iterator putloc, saveloc;
    typename ParticleList_t::iterator getloc = ParticleList.begin()+LocalSize;
    typename ParticleList_t::iterator endloc = ParticleList.begin()+LocalSize;
    // loop over destroy list and copy data from end of particle list
    size_t I, M, numParts=0;
    for (rbeg = dlist.rbegin(); rbeg != rend; ++rbeg) {
      I = (*rbeg).first;   // index number to begin destroy
      M = (*rbeg).second;  // number of particles to destroy
      numParts += M;       // running total of number of particles destroyed
      // set iterators for data copy
      putloc = ParticleList.begin() + I;
      // make sure we do not copy too much
      if ((I + M) > ((getloc - ParticleList.begin()) - M)) {
        // we cannot fill all M slots
        saveloc = getloc;  // save endpoint of valid particle data
        getloc = putloc + M;  // move to just past end of section to delete
        // copy over the data
        while (getloc != saveloc) {
          *putloc++ = *getloc++;
        }
        // reset getloc for next copy
        getloc = putloc;  // set to end of last copy
      }
      else {
        // fill all M slots using data from end of particle list
        getloc = getloc - M;
        saveloc = getloc;  // save new endpoint of valid particle data
        // copy over the data
        for (size_t m=0; m<M; ++m)
          *putloc++ = *getloc++;
        // reset getloc for next copy
        getloc = saveloc;  // set to new endpoint of valid particle data
      }
      LocalSize-=M;
    }
    // delete storage at end of particle list
    ParticleList.erase( endloc - numParts, endloc );
  }
  else {
    // just process destroy list using leading/trailing iterators
    dlist_t::const_iterator dnext = dlist.begin(), dend = dlist.end();
    size_t putIndex, getIndex, endIndex = LocalSize;
    putIndex = (*dnext).first;  // first index to delete
    getIndex = putIndex + (*dnext).second;  // move past end of destroy event
    ++dnext;  // move to next destroy event
    // make sure getIndex is not pointing to a deleted particle
    while (dnext != dend && getIndex == (*dnext).first) {
      getIndex += (*dnext).second;  // move past end of destroy event
      ++dnext;                      // move to next destroy event
    }
    while (dnext != dend) {
      // copy into deleted slot
      ParticleList[putIndex++] = ParticleList[getIndex++];
      // make sure getIndex points to next non-deleted particle
      while (dnext != dend && getIndex == (*dnext).first) {
        getIndex += (*dnext).second;  // move past end of destroy event
        ++dnext;                      // move to next destroy event
      }
    }
    // one more loop to do any remaining data copying beyond last destroy
    while (getIndex < endIndex) {
      // copy into deleted slot
      ParticleList[putIndex++] = ParticleList[getIndex++];
    }
    // now erase any data below last copy
    typename ParticleList_t::iterator loc = ParticleList.begin() + putIndex;
    ParticleList.erase(loc, ParticleList.begin()+LocalSize);
    LocalSize -= ParticleList.begin()+LocalSize - loc;
  }

  attributeIsDirty_ = true;
  return;
}


/////////////////////////////////////////////////////////////////////
// put the data for M particles into a message, starting from index I.
// This will either put in local or ghost particle data, but not both.
template<class T>
size_t
ParticleAttrib<T>::putMessage(Message& msg, size_t M, size_t I)
{

  if (M > 0) {
    if (isTemporary()) {
      ::putMessage(msg, M);
    }
    else {
      typename ParticleList_t::iterator currp = ParticleList.begin() + I;
      typename ParticleList_t::iterator endp = currp + M;
      ::putMessage(msg, currp, endp);
    }
  }

  return M;
}


/////////////////////////////////////////////////////////////////////
// put the data for particles in a list into a Message
// This will only work for local particle data right now.
template<class T>
size_t
ParticleAttrib<T>::putMessage(Message& msg,
			      const std::vector<size_t>& putList)
{

  std::vector<size_t>::size_type M = putList.size();

  if (M > 0) {
    if (isTemporary()) {
      ::putMessage(msg, M);
    }
    else {
      ::putMessage(msg, putList, ParticleList.begin());
    }
  }

  return M;
}


/////////////////////////////////////////////////////////////////////
// Get data out of a Message containing M particle's attribute data,
// and store it here.  Data is appended to the end of the list.  Return
// the number of particles retrieved.
template<class T>
size_t
ParticleAttrib<T>::getMessage(Message& msg, size_t M)
{

  if (M > 0) {
    if (isTemporary()) {
      size_t checksize;
      ::getMessage(msg, checksize);
      PAssert_EQ(checksize, M);
      create(M);
    }
    else {
      size_t currsize = size();
      create(M);
      ::getMessage_iter(msg, ParticleList.begin() + currsize);
    }
  }

  return M;
}

//~ virtual size_t ghostDestroy(size_t, size_t) {
    //~ return 0;
  //~ }
  //~
  //~ virtual void ghostCreate(size_t)
  //~ {
	  //~
  //~ }
  //~ // puts M particle's data starting from index I into a Message.
  //~ // Return the number of particles put into the message.  This is for
  //~ // when particles are being swapped to build ghost particle interaction
  //~ // lists.
  template<class T>
  size_t ParticleAttrib<T>::ghostPutMessage(Message&, size_t, size_t) {
    return 0;
  }
  // puts data for a list of particles into a Message, for interaction lists.
  // Return the number of particles put into the message.
  template<class T>
  size_t ParticleAttrib<T>::ghostPutMessage(Message&, const std::vector<size_t>&) {
    return 0;
  }
//~
  //~ // Get ghost particle data from a message.
  //~ virtual size_t ghostGetMessage(Message&, size_t) {
    //~ return 0;
  //~ }

template<class T>
void ParticleAttrib<T>::ghostCreate(size_t M) {
  
  // make sure we have storage for M more items
  // and push back M items, using default value
  if (M > 0)
  {
	//ParticleList.insert(ParticleList.end(), M, T());
    ParticleList.insert(ParticleList.end(), M, T());
  }

}

template<class T>
size_t ParticleAttrib<T>::ghostDestroy(size_t M, size_t I) {
  

  if (M > 0)
  {
	//ParticleList.insert(ParticleList.end(), M, T());
    ParticleList.erase(ParticleList.begin() + LocalSize + I, ParticleList.begin() + LocalSize + I + M);
  }
  return M;
}

template<class T>
size_t
ParticleAttrib<T>::ghostGetMessage(Message& msg, size_t M)
{

  if (M > 0) {
      size_t currsize = ParticleList.size();
      ghostCreate(M);
      ::getMessage_iter(msg, ParticleList.begin() + currsize);
  }


  return M;
}

/////////////////////////////////////////////////////////////////////
// Print out information for debugging purposes.  This version just
// prints out static information, so it is static
template<class T>
void ParticleAttrib<T>::printDebug(Inform& o)
{

  o << "PAttr: size = " << ParticleList.size()
    << ", capacity = " << ParticleList.capacity()
    << ", temporary = " << isTemporary();
}


template<class T>
struct PASortCompare
{
  static bool compare(const T &, const T &, bool)
  {
    // by default, just return false indicating "no change"
    return false;
  }
};

#define PA_SORT_COMPARE_SCALAR(SCALAR)					\
template<>								\
struct PASortCompare<SCALAR>						\
{									\
  static bool compare(const SCALAR &a, const SCALAR &b, bool ascending)	\
  {									\
    return (ascending ? (a < b) : (a > b));				\
  }									\
};

PA_SORT_COMPARE_SCALAR(char)
PA_SORT_COMPARE_SCALAR(unsigned char)
PA_SORT_COMPARE_SCALAR(short)
PA_SORT_COMPARE_SCALAR(unsigned short)
PA_SORT_COMPARE_SCALAR(int)
PA_SORT_COMPARE_SCALAR(unsigned int)
PA_SORT_COMPARE_SCALAR(long)
PA_SORT_COMPARE_SCALAR(unsigned long)
PA_SORT_COMPARE_SCALAR(float)
PA_SORT_COMPARE_SCALAR(double)


/////////////////////////////////////////////////////////////////////
// Calculate a "sort list", which is an array of data of the same
// length as this attribute, with each element indicating the
// (local) index wherethe ith particle shoulkd go.  For example,
// if there are four particles, and the sort-list is {3,1,0,2}, that
// means the particle currently with index=0 should be moved to the third
// position, the one with index=1 should stay where it is, etc.
// The optional second argument indicates if the sort should be ascending
// (true, the default) or descending (false).
template<class T>
void ParticleAttrib<T>::calcSortList(SortList_t &slist, bool ascending)
{
  unsigned int i;
  int j;

  //Inform dbgmsg("PA<T>::calcSortList");

  // Resize the sort list, if necessary, to our own size
  SortList_t::size_type slsize = slist.size();
  size_t mysize = size();
  if (slsize < mysize) {
    // dbgmsg << "Resizing provided sort-list: new size = ";
    slist.insert(slist.end(), mysize - slsize, (SortList_t::value_type) 0);
    // dbgmsg << slist.size() << ", attrib size = " << mysize << endl;
  }

  // Initialize the sort-list with a negative value, since we check
  // it later when determing what items to consider in the sort.  This
  // is done to avoid changing the attribute values.
  for (i=0; i < mysize; ++i)
    slist[i] = (-1);

  // OK, this is a VERY simple sort routine, O(N^2): Find min or max
  // of all elems, store where it goes, then for N-1 elems, etc.  I
  // am sure there is a better way.
  int firstindx = 0;
  int lastindx = (mysize - 1);
  for (i=0; i < mysize; ++i) {
    int currindx = firstindx;
    T currval = ParticleList[currindx];

    for (j=(firstindx + 1); j <= lastindx; ++j) {
      // skip looking at this item if we already know where it goes
      if (slist[j] < 0) {
	// compare current to jth item, if the new one is different
	// in the right way, save that index
	if (PASortCompare<T>::compare(ParticleList[j], currval, ascending)) {
	  currindx = j;
	  currval = ParticleList[currindx];
	}
      }
    }

    // We've found the min or max element, it has index = currindx.
    // So the currindx's item in the sort-list should say "this will be
    // the ith item".
    slist[currindx] = i;
    // dbgmsg << "Found min/max value " << i << " at position " << currindx;
    // dbgmsg << " with value = " << currval << endl;

    // Adjust the min/max index range to look at next time, if necessary
    while (slist[firstindx] >= 0 && firstindx < lastindx)
      firstindx++;
    while (slist[lastindx] >= 0 && firstindx < lastindx)
      lastindx--;
    // dbgmsg << " firstindx = " << firstindx << ", lastindx = " << lastindx;
    // dbgmsg << endl;
  }
}


/////////////////////////////////////////////////////////////////////
// Process a sort-list, as described for "calcSortList", to reorder
// the elements in this attribute.  All indices in the sort list are
// considered "local", so they should be in the range 0 ... localnum-1.
// The sort-list does not have to have been calculated by calcSortList,
// it could be calculated by some other means, but it does have to
// be in the same format.  Note that the routine may need to modify
// the sort-list temporarily, but it will return it in the same state.
template<class T>
void ParticleAttrib<T>::sort(SortList_t &slist)
{
    // Make sure the sort-list has the proper length.
    PAssert_GE(slist.size(), size());
    
    // Inform dbgmsg("PA<T>::sort");
    // dbgmsg << "Sorting " << size() << " items." << endl;
    
    // Go through the sort-list instructions, and move items around.
    int i = 0, j = 0, k = -1, mysize = size();
    while ( i < mysize ) {
        PAssert_LT(slist[i], mysize);
        
        // skip this swap if the swap-list value is negative.  This
        // happens when we've already put the item in the proper place.
        if ( i == k || slist[i] < 0 ) {
            ++i; k = -1;
            // dbgmsg << "Skipping item " << i << " in sort: slist[" << i << "] = ";
            // dbgmsg << slist[i] << endl;
            continue;
        }
        
        j = ( k > 0 ) ? k : slist[i];
        k = slist[j];
        
        // We should not have a negative slist value for the destination
        PAssert_GE(k, 0);
        
        // OK, swap the items
        std::iter_swap(ParticleList.begin() + i, ParticleList.begin() + j);
        // dbgmsg << "Swapping item " << i << " to position " << slist[i] << endl;
        
        
        // then indicate that we've put this
        // item in the proper location.
        slist[j] -= mysize;
    }
    

    // Restore the sort-list
    for (i=0; i < mysize; ++i) {
        if (slist[i] < 0)
        slist[i] += mysize;
        // dbgmsg << "At end of sort: restored slist[" << i << "] = " << slist[i];
        // dbgmsg << ", data[" << i << "] = " << ParticleList[i] << endl;
    }
}

//////////////////////////////////////////////////////////////////////
// scatter functions
//////////////////////////////////////////////////////////////////////
//mwerks Moved into class definition (.h file).


/////////////////////////////////////////////////////////////////////
// gather functions
/////////////////////////////////////////////////////////////////////
//mwerks Moved into class definition (.h file).

/////////////////////////////////////////////////////////////////////
// Global function templates
/////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////
// scatter functions for number density
/////////////////////////////////////////////////////////////////////

template <class FT, unsigned Dim, class M, class C, class PT, class IntOp>
void
scatter(Field<FT,Dim,M,C>& f, const ParticleAttrib< Vektor<PT,Dim> >& pp,
        const IntOp& /*intop*/, FT val) {

  // make sure field is uncompressed and guard cells are zeroed
  f.Uncompress();
  FT zero = 0;
  f.setGuardCells(zero);

  const M& mesh = f.get_mesh();
  // iterate through particles and call scatter operation
  typename ParticleAttrib< Vektor<PT,Dim> >::const_iterator ppiter;
  size_t i = 0;
  for (ppiter = pp.cbegin(); i < pp.size(); ++i,++ppiter)
    IntOp::scatter(val,f,*ppiter,mesh);

  // accumulate values in guard cells
  f.accumGuardCells();

  INCIPPLSTAT(incParticleScatters);
  return;
}

template <class FT, unsigned Dim, class M, class C, class PT,
          class IntOp, class CacheData>
void
scatter(Field<FT,Dim,M,C>& f, const ParticleAttrib< Vektor<PT,Dim> >& pp,
  const IntOp& /*intop*/, ParticleAttrib<CacheData>& cache, FT val) {

  // make sure field is uncompressed and guard cells are zeroed
  f.Uncompress();
  FT zero = 0;
  f.setGuardCells(zero);

  const M& mesh = f.get_mesh();
  // iterate through particles and call scatter operation
  typename ParticleAttrib< Vektor<PT,Dim> >::iterator ppiter;
  typename ParticleAttrib<CacheData>::iterator citer=cache.begin();
  size_t i = 0;
  for (ppiter = pp.begin(); i < pp.size(); ++i, ++ppiter, ++citer)
    IntOp::scatter(val,f,*ppiter,mesh,*citer);

  // accumulate values in guard cells
  f.accumGuardCells();

  INCIPPLSTAT(incParticleScatters);
  return;
}

template <class FT, unsigned Dim, class M, class C,
          class IntOp, class CacheData>
void
scatter(Field<FT,Dim,M,C>& f, const IntOp& /*intop*/,
        const ParticleAttrib<CacheData>& cache, FT val) {

  // make sure field is uncompressed and guard cells are zeroed
  f.Uncompress();
  FT zero = 0;
  f.setGuardCells(zero);

  // iterate through particles and call scatter operation
  typename ParticleAttrib<CacheData>::iterator citer, cend=cache.begin()+cache.size();//not sure jp
  for (citer = cache.begin(); citer != cend; ++citer)
    IntOp::scatter(val,f,*citer);

  // accumulate values in guard cells
  f.accumGuardCells();

  INCIPPLSTAT(incParticleScatters);
  return;
}


/***************************************************************************
 * $RCSfile: ParticleAttrib.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: ParticleAttrib.cpp,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $
 ***************************************************************************/
