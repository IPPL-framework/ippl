// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef LSINDEX_H
#define LSINDEX_H

// include files
#include "Index/SOffset.h"
#include "FieldLayout/Vnode.h"
#include "Utility/Vec.h"

#include <vector>

/***********************************************************************
 *
 * LSIndex represents a set of single-point indices for a Field, just
 * for a single vnode.  SIndex contains a list of these LSIndex objects.
 * Expressions involving sparse indices are constrained to have the same
 * number of indices in the corresponding LSIndex objects, even though
 * the indices themselves do not have to be the same.
 *
 ***********************************************************************/

template<unsigned int Dim>
class LSIndex {

public:
  // useful typedefs
  typedef std::vector< SOffset<Dim> >             container_t;
  typedef typename container_t::iterator     iterator;
  typedef typename container_t::const_iterator const_iterator;
  typedef typename container_t::size_type    size_type;

public:
  // constructors
  LSIndex(Vnode<Dim>* vn)
    : VN(vn), compressed(false)
    {
      Strides[0] = 1;
      for (unsigned int d=1; d < Dim; ++d)
        Strides[d] = Strides[d-1] * vn->getDomain()[d-1].length();
    }
  LSIndex(const LSIndex<Dim>& lsi)
    : VN(lsi.VN), IndexList(lsi.IndexList), compressed(lsi.compressed),
      Strides(lsi.Strides) { }

  // destructor
  ~LSIndex() { }

  // change this LSIndex to store the values in the given LSIndex instead
  LSIndex& operator=(const LSIndex& lsi) {
    VN = lsi.VN;
    IndexList = lsi.IndexList;
    compressed = lsi.compressed;
    Strides = lsi.Strides;
    return *this;
  }

  // are we compressed?
  bool IsCompressed() const { return compressed; }

  // compress this list
  void Compress(bool docompress) {
    clear();
    compressed = docompress;
  }

  // check to see if the given point would go here
  bool contains(const SOffset<Dim> &indx) {
    return indx.inside( (VN->getDomain()) );
  }

  // add a new point ... to do explicit checking for errors, call contains()
  // before adding a point.
  void addIndex(const SOffset<Dim> &indx) {
    compressed = false;
    IndexList.push_back(indx);
  }

  // return the Nth index
  SOffset<Dim>& getIndex(unsigned int n) {
    if (compressed) {
      int mval = n;
      for (unsigned int d=(Dim-1); d >= 1; --d) {
        int dval = mval / Strides[d];
        mval -= dval * Strides[d];
        CompressedPoint[d] = dval + VN->getDomain()[d].first();
      }
      CompressedPoint[0] = mval + VN->getDomain()[0].first();

      //Inform dbgmsg("LSIndex::getIndex", INFORM_ALL_NODES);
      //dbgmsg << "For dom=" << VN->getDomain() << ": mapped n=" << n;
      //dbgmsg << " to SOffset=" << CompressedPoint << endl;

      return CompressedPoint;
    }

    // if we're here, not compressed, so just return Nth point
    return IndexList[n];
  }

  // return a copy of the Nth index
  SOffset<Dim> getIndex(unsigned int n) const {
    if (compressed) {
      SOffset<Dim> retval;
      int mval = n;
      for (unsigned int d=(Dim-1); d >= 1; --d) {
        int dval = mval / Strides[d];
        mval -= dval * Strides[d];
        retval[d] = dval + VN->getDomain()[d].first();
      }
      retval[0] = mval + VN->getDomain()[0].first();

      //Inform dbgmsg("LSIndex::getIndex", INFORM_ALL_NODES);
      //dbgmsg << "For dom=" << VN->getDomain() << ": mapped n=" << n;
      //dbgmsg << " to SOffset=" << retval << endl;

      return retval;
    }

    // if we're here, not compressed, so just return Nth point
    return IndexList[n];
  }

  // remove the given point if we have it.  Just move the last element
  // up, the order does not matter
  void removeIndex(const SOffset<Dim> &indx) {
    iterator loc = (*this).find(indx);
    if (loc != end()) {
      *loc = IndexList.back();
      IndexList.pop_back();
    }
  }

  // clear out the existing indices
  void clear() { IndexList.erase(IndexList.begin(), IndexList.end()); }

  // reserve enough space to hold at least n points
  void reserve(size_type n) { IndexList.reserve(n); }

  //
  // container methods
  //

  // return begin/end iterators
  iterator begin() { return IndexList.begin(); }
  iterator end() { return IndexList.end(); }
  const_iterator begin() const { return IndexList.begin(); }
  const_iterator end() const { return IndexList.end(); }

  // return size information about the number of index points here
  size_type capacity() const { return IndexList.capacity(); }
  size_type size() const {
    return (compressed ? VN->getDomain().size() : IndexList.size());
  }

  // return an iterator to the given point, if we have it; otherwise
  // return the end iterator
  iterator find(const SOffset<Dim>& indx) {
    for (iterator a = begin(); a != end(); ++a)
      if (*a == indx)
        return a;
    return end();
  }

  // just return a boolean indicating if we have the given point
  bool hasIndex(const SOffset<Dim>& indx) const {
    for (const_iterator a = begin(); a != end(); ++a)
      if (*a == indx)
        return true;
    return false;
  }

  //
  // vnode information methods
  //

  // return the local domain of vnode this LSIndex contains points for
  const NDIndex<Dim>& getDomain() const { return VN->getDomain(); }

  // return the local procssor of vnode this LSIndex contains points for
  int getNode() const { return VN->getNode(); }

private:
  // vnode on which this LSIndex stores points
  Vnode<Dim> *VN;

  // list of points
  container_t IndexList;

  // are we compressed?  If so, IndexList should be empty, and all points
  // in the domain assumed to be here.
  bool compressed;
  SOffset<Dim> CompressedPoint;

  // strides for computing mapping from N --> (i,j,k)
  vec<int,Dim> Strides;
};

#endif // LSINDEX_H

/***************************************************************************
 * $RCSfile: LSIndex.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: LSIndex.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $
 ***************************************************************************/
