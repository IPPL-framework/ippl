// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SINDEX_H
#define SINDEX_H

// include files
#include "Index/NDIndex.h"
#include "Index/SOffset.h"
#include "Index/LSIndex.h"
#include "FieldLayout/FieldLayoutUser.h"
#include "Utility/Inform.h"
#include <memory>

#include <vector>
#include <iostream>

// forward declarations
template <unsigned Dim> class FieldLayout;
template <unsigned Dim> class SIndex;
template <unsigned Dim, unsigned Brackets> class IndexedSIndex;
template <unsigned Dim>
std::ostream& operator<<(std::ostream&, const SIndex<Dim>&);

/***********************************************************************
 * 
 * SIndex represents a set of single-point indices for a Field, which
 * are used to implement sparse-index operations.  The user creates an
 * SIndex object through a where statement or by just adding individual
 * points, and then performs operations using the SIndex object just as
 * is done with a regular Index object.  In that case, operations are
 * only performed on those elements of the LHS which are in the SIndex
 * list.  Stencil operations are represented by using SOffset objects
 * to indicate offsets from the SIndex indices.
 *
 * Initially, an SIndex object is empty; points must be added to it by
 * calling addIndex.  An SIndex has an offset, set (or changed) by
 * calling setOffset(const SOffset&).  Only indices for local vnodes are
 * stored in this object; adding other points will return an error flag.
 *
 * Constructing an SIndex requires a FieldLayout object, since SIndex
 * needs to know the range of index values for which is can store points,
 * and the location of vnodes.
 *
 * A note about offsets: the offset factor is used primarily to make
 * it possible to specify index offsets in expressions for stenciling
 * purposes.  The method for adding an offset to an SIndex is via the
 * + or - operators, e.g., A[si] = B[si + SOffset<Dim>(1,-1)], or via
 * the () operator, e.g., A[si] = B[si(1,-1)] (which means the same as
 * above).  Otherwise, the user should not specify an offset, and the
 * operations such as union, intersection, etc. implicitly assume that
 * the offset is zero for the LHS and the RHS.
 *
 ***********************************************************************/

template <unsigned Dim>
class SIndex : public FieldLayoutUser {

public:
  //# public typedefs
  typedef std::vector< std::shared_ptr<LSIndex<Dim> > >  container_t;
  typedef unsigned int                          size_type;
  typedef typename container_t::iterator        iterator_iv;
  typedef typename container_t::const_iterator  const_iterator_iv;
  typedef typename container_t::size_type       size_type_iv;
  typedef typename LSIndex<Dim>::iterator       iterator_indx;
  typedef typename LSIndex<Dim>::const_iterator  const_iterator_indx;

  // default constructor: this requires the user to call 'initialize'
  // before any other actions are carried out with this SIndex
  SIndex();

  // constructor: requires a FieldLayout
  SIndex(FieldLayout<Dim>&);

  // copy constructor
  SIndex(const SIndex<Dim>&);

  // destructor: frees memory used to store indices, and check out from Layout
  virtual ~SIndex();

  // initialize the object, if it was constructed with the default
  // constructor
  void initialize(FieldLayout<Dim>&);

  // report if we need initialization
  bool needInitialize() const { return (Layout == 0); }

  // a templated operator= taking a PETE expression
  template<class T1>
  SIndex<Dim>& operator=(const PETE_Expr<T1>& rhs) {
    assign(*this, rhs);
    return *this;
  }

  // assignment operator, with another SIndex or SOffset object.  If an
  // NDIndex object is given, all the points in the NDIndex are used.
  SIndex<Dim>& operator=(const SIndex<Dim>&);
  SIndex<Dim>& operator=(const SOffset<Dim>&);
  SIndex<Dim>& operator=(const NDIndex<Dim>&);

  // intersection operator, with another SIndex or SOffset object.
  // intersected SIndexes must have the same layout; intersection with an
  // SOffset will leave this object with at most one point
  SIndex<Dim>& operator&=(const SIndex<Dim>&);
  SIndex<Dim>& operator&=(const SOffset<Dim>&);
  SIndex<Dim>& operator&=(const NDIndex<Dim>&);

  // union operator, with another SIndex or SOffset object.  This will
  // append the point if it is not already present.
  SIndex<Dim>& operator|=(const SIndex<Dim>&);
  SIndex<Dim>& operator|=(const SOffset<Dim>&);
  SIndex<Dim>& operator|=(const NDIndex<Dim>&);

  // add a new index point, specified as an Offset or as a single-point NDIndex
  // return success (this can fail if the point is outsize the field's domain)
  bool addIndex(const SOffset<Dim>&);
  bool addIndex(iterator_iv&, const SOffset<Dim>&);
  void addIndex(const NDIndex<Dim>&);

  // remove an index point, specified as an Offset or as a single-point NDIndex
  // return success (this can fail if the point is not on this node)
  bool removeIndex(const SOffset<Dim>&);
  bool removeIndex(iterator_iv&, const SOffset<Dim>&);
  void removeIndex(const NDIndex<Dim>&);

  // reserve storage space equal to the given fraction of the size of
  // each vnode.  if fraction=1.0, reserve storage for the entire vnode.
  void reserve(double = 1.0);

  // clear out the existing indices
  void clear();

  // get the offset for this sparse index.  You can change the offset by
  // retrieving it this way and then adding to it.
  SOffset<Dim>& getOffset() { return Offset; }
  const SOffset<Dim>& getOffset() const { return Offset; }

  // get the FieldLayout we are using
  FieldLayout<Dim>& getFieldLayout() const { return *Layout; };

  // change to using a new layout.
  void setFieldLayout(FieldLayout<Dim>&);

  // get or change the 'bounding box' domain of this SIndex
  const NDIndex<Dim> &getDomain() const { return BoundingBox; }
  void setDomain(const NDIndex<Dim> &ndi) { BoundingBox = ndi; }

  //
  // SIndex <--> SOffset operations
  //

  // add or subtract a given offset
  // SIndex<Dim>& operator+=(const SOffset<Dim>& so) {Offset+=so;return *this;}
  // SIndex<Dim>& operator-=(const SOffset<Dim>& so) {Offset-=so;return *this;}
  friend
  SIndex<Dim> operator+(const SIndex<Dim>& si, const SOffset<Dim>& so) {
    return SIndex(si, so);
  }

  friend
  SIndex<Dim> operator+(const SOffset<Dim> &so, const SIndex<Dim>& si) {
    return SIndex(si, so);
  }

  friend
  SIndex<Dim> operator+(const SIndex<Dim>& si, const int *so) {
    return SIndex(si, so);
  }

  friend
  SIndex<Dim> operator+(const int *so, const SIndex<Dim>& si) {
    return SIndex(si, so);
  }

  friend
  SIndex<Dim> operator-(const SIndex<Dim>& si, const SOffset<Dim>& so) {
    return SIndex(si, -so);
  }

  friend
  SIndex<Dim> operator-(const SOffset<Dim> &so, const SIndex<Dim>& si) {
    return SIndex(si, -so);
  }

  friend
  SIndex<Dim> operator-(const SIndex<Dim>& si, const int *so) {
    return SIndex(si, -SOffset<Dim>(so));
  }

  friend
  SIndex<Dim> operator-(const int *so, const SIndex<Dim>& si) {
    return SIndex(si, -SOffset<Dim>(so));
  }

  // () operators which make a copy of this SIndex with an extra offset.
  // These are functionally identical to the operator+, but provide a
  // nicer syntax.  That is, si(1,1) means  si + SOffset<Dim>(1,1)
  SIndex<Dim> operator()(int);
  SIndex<Dim> operator()(int,int);
  SIndex<Dim> operator()(int,int,int);
  SIndex<Dim> operator()(int,int,int,int);
  SIndex<Dim> operator()(int,int,int,int,int);
  SIndex<Dim> operator()(int,int,int,int,int,int);
  SIndex<Dim> operator()(const SOffset<Dim>&);
  SIndex<Dim> operator()(const int *);

  // operator[], which is used with Index or NDIndex objects to further
  // subset the data.  This will only work if the dimension of the Index
  // arguments + Brackets is <= Dim.  Otherwise, too many dimensions worth
  // of Index objects are being applied
  IndexedSIndex<Dim,1> operator[](const Index &);

  template<unsigned int Dim2>
  IndexedSIndex<Dim,Dim2> operator[](const NDIndex<Dim2> &ndi) {
     
    
    
    CTAssert(Dim >= Dim2);
    NDIndex<Dim> dom;
    for (unsigned int i=0; i < Dim2; ++i)
      dom[i] = ndi[i];
    return IndexedSIndex<Dim,Dim2>(*this, dom);
  }

  //
  // SIndex <--> NDIndex operations
  //

  // convert from the given SOffset value to an NDIndex, with offset added
  void toNDIndex(const SOffset<Dim>&, NDIndex<Dim>&);

  //
  // container methods
  //

  // return begin/end iterators for the LSIndex container
  iterator_iv  begin_iv()      { return IndexList.begin(); }
  iterator_iv  end_iv()        { return IndexList.end(); }
  const_iterator_iv  begin_iv() const { return IndexList.begin(); }
  const_iterator_iv  end_iv()   const { return IndexList.end(); }
  size_type_iv size_iv() const { return IndexList.size(); }

  // return the total size, which is the sum of the individual sizes
  size_type_iv size() const;

  // return whether the given point is contained here
  bool hasIndex(const SOffset<Dim>&) const;

  //
  // virtual functions for FieldLayoutUser's
  //

  // Repartition onto a new layout
  virtual void Repartition(UserList *);

  // Tell this object that an object is being deleted
  virtual void notifyUserOfDelete(UserList *);

  //
  // I/O
  //

  // print out debugging info
  void printDebug(Inform&) const;

private:
  // our FieldLayout, indicating the extent and distrib. of index space
  FieldLayout<Dim>* Layout;

  // our current offset; by default, this is zero.  We keep a special flag
  // to indicate whether it is zero or not.
  SOffset<Dim> Offset;

  // our list of indices for each local vnode
  container_t IndexList;

  // our 'bounding box', which is the region that is or should be iterated
  // over to determine what points are in this sparse index list.  By default,
  // this is the domain of the FieldLayout
  NDIndex<Dim> BoundingBox;

  // a special constructor, taking another SIndex and an Offset.  This
  // version is almost like a copy constructor, except that the given Offset
  // is added in to the offset from the copied SIndex.
  SIndex(const SIndex<Dim>&, const SOffset<Dim>&);
  SIndex(const SIndex<Dim>&, const int *);

  // set up our internal data structures from the constructor.  Assumes
  // the Layout and Offset have been set.
  void setup();
};

#include "Index/SIndex.hpp"

#endif // SINDEX_H

/***************************************************************************
 * $RCSfile: SIndex.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: SIndex.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
