//
// Class SubFieldIter
//   Iterator for a subset of a BareField
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
#ifndef SUB_FIELD_ITER_H
#define SUB_FIELD_ITER_H

/***************************************************************************
  SubFieldIter is a class is required for each type of object which is used to
  store a subset for a BareField.  For example, NDIndex specifies a rect-
  angular block, SIndex specifies a (sparse) list of individual elements,
  and SOffset specifies a single point (which is assumed to be the same
  on all nodes).  SubFieldIter acts as the iterator for the SubBareField
  class which uses that particular type of subset object to select a view of
  the field.

  To add a new subset object and define a new SubFieldIter class:
    * Add a new item to the enumeration at the top of this file.
    * Define a new SubFieldIter class, templated on the type of object (e.g.,
       NDIndex<Dim>), following the pattern for the examples below.  The
       required elements in SubFieldIter are
       (which can be set up by copying an existing class and modifying it):

       - derive from SubFieldIterBase

       - two construction options, one with needed iterator args, the other
         a default constructor (this is needed by PETE).  You can also
         define a copy constructor, or else make sure that element-wise
         copying will suffice for your iterator class.

       - override the versions of all the functions in the base class which
         need to be overridden, to supply the special functionality for that
         subset.  For example, some classes need special initialization when
         they are used in an expression - if so, make a specific version of
         'initialize' in your subset specialization class.

       - add specialization in SubFieldTraits to indicate how this subset
         object can be constructed from other subset objects, and what kind
         of combinations of subset objects will work.

       - static bool matchType(int t) { return (t == Type_u); } ... return
         whether the type of subset object matches the given type.  Some
         objects, like SOffset, match all subset types (since SOffset used
         as a subset object appears to the rest of the expression like a
         single value, e.g., as a scalar.

       - static void makeNDIndex(Subset_t& s, NDIndex<Dim>& i) ... convert
         the given subset object data to an NDIndex

       - bool findIntersection() { } ... find the intersection between a
         the current component and a given NDIndex, and return the intersection
         as an NDIndex.  Also return a boolean flag indicating if this
         intersection did indeed contain some points (true).  This is then used
         in a BrickExpression to take data from the RHS and store it into the
         LHS.  findIntersection is only called for an iterator which occurs
         on the LHS; if something is on the RHS, then plugBase will be called
         instead, with an argument of the domain as calculated by
         findIntersection.

       - void nextLField() { } ... for subsets which must keep track of their
         current vnode (e.g., SIndex), this increments a vnode iterator.

       - NDIndex<Dim> plugBase(const NDIndex<Dim>& i) const ... this is the
         function which takes the NDIndex component on the LHS, and plugs it
         in to the RHS.  This results in each SubBareField on the RHS setting
         internal iterators to point to the proper section of data
         based on any offsets it may have and on the subset from the LHS.
         For some subset objects, this will not depend on the given subdomain.

       - setLFieldData(LField<T,Dim>*, NDIndex<Dim>&) ... for iterators
         which occur on the LHS of an expression, the LField referred to by
         its internal LField iterator may need to be changed to a new one
         which is a copy of the existing LField (to handle the case where a
         Field occurs on the LHS and the RHS).  This function, which has a
         default definition, changes the iterator to use a different LField
         and to iterate over the region specified in the second argument.
         It is only called for iterators on the LHS.

       - bool CanTryCompress() ... Some subset objects cannot easily be
         used on the LHS with compressed LFields.  If the new one can not
         (for example, if it is not possible to determine if all the points
         referred to by the subset have the same value), this should return
         false, and all the other compression routines can just be no-ops.

 ***************************************************************************/


// include files
#include "Index/NDIndex.h"
#include "Index/SIndex.h"
#include "Field/BareField.h"
#include "Field/LField.h"
#include "Utility/IpplException.h"
#include "Utility/PAssert.h"
#include "PETE/IpplExpressions.h"


///////////////////////////////////////////////////////////////////////////
// type codes for different subset objects
enum { NoSubsetType, NDIndexSubsetType, SIndexSubsetType, SOffsetSubsetType };


///////////////////////////////////////////////////////////////////////////
// A base class for all specialized versions of the SubFieldIter class.
// This provides common functionality and storage, and default versions of
// functions which may be overridden in the specialized classes.

template<class T, unsigned int Dim, class S, unsigned int ExprDim>
class SubFieldIterBase {

public:
  // How many indices we must loop over when doing an expression involving
  // this object
  enum { ExprDim_u = ExprDim };

  // Construct with a SubField and the domain to use
  SubFieldIterBase(const BareField<T,Dim>& df,
                   const typename BareField<T,Dim>::iterator_if& ldf,
                   const S& s,
                   unsigned int B)
    : MyBareField(&(const_cast<BareField<T,Dim> &>(df))),
      MyDomain(&(const_cast<S&>(s))),
      CurrentLField(ldf),
      MyBrackets(B) {
      if (CurrentLField != getBareField().end_if()) {
          LFPtr = (*CurrentLField).second.get();
      } else {
          LFPtr = nullptr;
      }
  }

  // Default constructor
  SubFieldIterBase() { }

  // destructor: nothing to do
  ~SubFieldIterBase() { }

  // Return the BareField, for which this iterator represents a particular
  // subset or view of
  BareField<T,Dim> &getBareField() { return *MyBareField; }
  const BareField<T,Dim> &getBareField() const { return *MyBareField; }

  // Return a copy of the current iterator pointing to the active LField
  typename BareField<T,Dim>::iterator_if getLFieldIter() const {
    return CurrentLField;
  }

  // Return the subset object
  S &getDomain() { return *MyDomain; }
  const S &getDomain() const { return *MyDomain; }

  // Check if our brackets are properly balanced
  bool checkBrackets() const { return MyBrackets == Dim; }
  unsigned int getBrackets() const { return MyBrackets; }

  // Go to the next LField.
  typename BareField<T,Dim>::iterator_if nextLField() {
      if (CurrentLField != getBareField().end_if()) {
          ++CurrentLField;
      } else {
          throw IpplException("SubFieldIterBase::nextLField()", "Reached the container end, no next LField!");
      }
      if (CurrentLField != getBareField().end_if()) {
          LFPtr = (*CurrentLField).second.get();
      } else {
          LFPtr = nullptr;
      }
      return CurrentLField;
  }

  // Return the LField pointed to by LFPtr
  LField<T,Dim>* getLField() { return LFPtr; }
  const LField<T,Dim>* getLField() const { return LFPtr; }

  // Use a new LField
  void setLField(LField<T,Dim>* p) { LFPtr = p; }

  // Use a new LField, where we use data on the given NDIndex region
  void setLFieldData(LField<T,Dim>* p, NDIndex<Dim>&)  { LFPtr = p; }

  /* tjw 3/3/99: try to mimic changes made in
     IndexedBareFieldIterator::FillGCIfNecessary:

  // Fill the guard cells for our field, if necessary.  We punt on
  // trying to check for a stencil op, just fill GC's if the Field has
  // it's dirty flag set.
  // If we were trying to check for a stencil op, the arguments are
  // the domain we are doing the expression on, and the NDIndex domain
  // which is the bounding box for this subset object.
  void FillGCIfNecessary(const NDIndex<Dim> &, const NDIndex<Dim> &) {
    bool isdirty = getBareField().isDirty();
    // bool isstencil = isStencil(i, j);
    // if (isdirty && isstencil) {
    if (isdirty)
      getBareField().fillGuardCells();
  }
  tjw 3/3/99 */

  //tjw 3/3/99:
  // Fill the guard cells for our field, if necessary.  We punt on
  // trying to check for a stencil op, just fill GC's if the Field has
  // it's dirty flag set.
  // If we were trying to check for a stencil op, the arguments are
  // the domain we are doing the expression on, and the NDIndex domain
  // which is the bounding box for this subset object.
  void FillGCIfNecessary() const {
    bool isdirty = this->getBareField().isDirty();
    if (isdirty)
      ( const_cast<BareField<T,Dim> &>(this->getBareField()) ).fillGuardCells();
  }
  //tjw 3/3/99.

private:
  // ptr to the subfield we are iterating over
  BareField<T,Dim>* MyBareField;

  // pointer to the subset object
  S* MyDomain;

  // a pointer to the LField we're working with, which generally starts out
  // as the LField pointed to by CurrentLField, but can change if, say, we're
  // using a copy of the LField on the LHS of an expression
  LField<T,Dim>* LFPtr;

  // iterator pointing to the current LField we're looping over
  typename BareField<T,Dim>::iterator_if CurrentLField;

  // how many brackets have been applied so far
  unsigned int MyBrackets;
};


///////////////////////////////////////////////////////////////////////////
// The general subset-object-templated class definition for the SubField
// iterator.

template<class T, unsigned int Dim, class S>
class SubFieldIter { };


///////////////////////////////////////////////////////////////////////////
// A specialized versions of the SubFieldIter class for an NDIndex
// subset object.  This overrides certain functions from the base class,
// and provides definitions of needed functions not available in the base.

template<class T, unsigned int Dim>
class SubFieldIter<T, Dim, NDIndex<Dim> >
  : public SubFieldIterBase<T, Dim, NDIndex<Dim>, Dim>,
    public PETE_Expr<SubFieldIter<T, Dim, NDIndex<Dim> > >
{

public:
  // public typedefs
  typedef NDIndex<Dim>                   Subset_t;
  typedef SubFieldIter<T, Dim, Subset_t> SFI;

  // Construct with a SubField and the domain to use
  SubFieldIter(const BareField<T,Dim>& df,
               const typename BareField<T,Dim>::iterator_if& ldf,
               const NDIndex<Dim>& s, unsigned int B)
    : SubFieldIterBase<T,Dim,Subset_t,Dim>(df, ldf, s, B) { }

  // Default constructor
  SubFieldIter() { }

  // destructor: nothing to do
  ~SubFieldIter() { }

  //
  // Methods overriding base class behavior
  //

  // Use a new LField, where we use data on the given NDIndex region
  void setLFieldData(LField<T,Dim>* p, NDIndex<Dim>& n)  {
    SubFieldIterBase<T,Dim,Subset_t,Dim>::setLFieldData(p, n);
    P = this->getLField()->begin(n);
  }

  // Return a special code indicating what the subset type is, and match the
  // types together.
  static int  getSubsetType()  { return NDIndexSubsetType; }
  static bool matchType(int t) { return (t == NDIndexSubsetType); }

  // If necessary, this routine will distribute any data it needs to
  // among the processors.  For example, single-value subset objects must
  // broadcast the single value to all the nodes.
  void initialize() { }

  // Calculate the intersection with the current component and the given
  // subdomain.  Return true if an intersection exists, and the intersection
  // domain in the second argument.
  bool findIntersection(const NDIndex<Dim>& loc, NDIndex<Dim>& inter) {
    inter = this->getDomain().intersect(loc);
    return ( ! inter.empty() );
  }

  // convert this subset object into an NDIndex object
  static void makeNDIndex(const Subset_t& s, NDIndex<Dim>& i) { i = s; }

  // The LHS tells this guy about a given local domain.  Not all
  // subsetting operations will care about this.  The LField iterator P
  // will be set in this call to iterate through the values it needs to.
  bool plugBase(const NDIndex<Dim>& i) {
    // make sure we have a fully indexed object
    PInsist(this->checkBrackets(),"Field not fully indexed!!");

    // Find the modified domain after we plug in the information from what
    // domain is being iterated over on the LHS
    NDIndex<Dim> plugged = this->getDomain().plugBase(i);

    // Try to find a single local array that has all of the rhs.
    // Loop over all the local arrays.
    typename BareField<T,Dim>::iterator_if lf_i = this->getBareField().begin_if();
    typename BareField<T,Dim>::iterator_if lf_e = this->getBareField().end_if();
    for ( ; lf_i != lf_e; ++lf_i) {
      // is the search domain completely within the LField we're examining?
      if ((*lf_i).second->getAllocated().contains(plugged)) {
        // Found it.  Make this one current and go.
        setLFieldData((*lf_i).second.get(), plugged);
        return true;
      }
    }

    // Didn't find it.
    return false;
  }

  // Finished this dimension, rewind.
  void rewind(unsigned d) { P.rewind(d); }

  // Step one or n in a given dimension.
  void step(unsigned int d)        { P.step(d); }
  void step(unsigned int d, int n) { P.step(d, n); }

  // How big in this dimension.
  int size(unsigned d) const { return P.size(d); }

  // return the value currently pointed at, or offset in 1,2, or 3 dims
  T& operator*()                 { return *P; }
  T& offset()                    { return *P; }
  T& offset(int i)               { return P.offset(i); }
  T& offset(int i, int j)        { return P.offset(i, j); }
  T& offset(int i, int j, int k) { return P.offset(i, j, k); }
  T& unit_offset(int i)               { return P.unit_offset(i); }
  T& unit_offset(int i, int j)        { return P.unit_offset(i, j); }
  T& unit_offset(int i, int j, int k) { return P.unit_offset(i, j, k); }
  int Stride(int d) const { return P.Stride(d); }

  // Compression interface
  bool CanCompress() const    { return this->getLField()->CanCompress(); }
  void Compress(T v)          { return this->getLField()->Compress(v);   }
  bool TryCompress()          { return this->getLField()->TryCompress(); }
  bool TryCompress(T v)       { return this->getLField()->TryCompress(v);}
  bool IsCompressed() const   {
      PAssert_EQ(this->getLField()->IsCompressed(), P.IsCompressed());
      return this->getLField()->IsCompressed();
  }
  bool DomainCompressed() const { return true; }

  //
  // PETE Interface
  //

  enum        { IsExpr = 1 };
  typedef SFI PETE_Expr_t;
  typedef T   PETE_Return_t;
  PETE_Expr_t MakeExpression() const { return *this; }

private:
  // where in this LField are we
  typename LField<T,Dim>::iterator P;
};


///////////////////////////////////////////////////////////////////////////
// A specialized versions of the SubFieldIter class for an SIndex
// subset object.  This overrides certain functions from the base class,
// and provides definitions of needed functions not available in the base.

template<class T, unsigned int Dim>
class SubFieldIter<T, Dim, SIndex<Dim> >
  : public SubFieldIterBase<T, Dim, SIndex<Dim>, 1U>,
    public PETE_Expr<SubFieldIter<T, Dim, SIndex<Dim> > >
{
public:
  // public typedefs
  typedef SIndex<Dim>                    Subset_t;
  typedef SubFieldIter<T, Dim, Subset_t> SFI;

  // Construct with a SubField and the domain to use
  SubFieldIter(const BareField<T,Dim>& df,
               const typename BareField<T,Dim>::iterator_if& ldf,
               const SIndex<Dim>& s, unsigned int B)
    : SubFieldIterBase<T,Dim,Subset_t,1U>(df, ldf, s, B) {
    ComponentLF = this->getDomain().begin_iv();
    computeLSOffset();
  }

  // Default constructor
  SubFieldIter() { }

  // destructor: nothing to do
  ~SubFieldIter() { }

  //
  // Methods overriding base class behavior
  //

  // Go to the next LField.
  typename BareField<T,Dim>::iterator_if nextLField() {
    typename BareField<T,Dim>::iterator_if lfi =
      SubFieldIterBase<T,Dim,Subset_t,1U>::nextLField();
    ++ComponentLF;
    computeLSOffset();
    return lfi;
  }

  // Return a special code indicating what the subset type is, and match the
  // types together.
  static int  getSubsetType()  { return SIndexSubsetType; }
  static bool matchType(int t) { return (t == SIndexSubsetType); }

  // If necessary, this routine will distribute any data it needs to
  // among the processors.  For example, single-value subset objects must
  // broadcast the single value to all the nodes.
  void initialize() { }

  // Calculate the intersection with the current LField domain and the
  // given subdomain.  Return true if an intersection exists, and the
  // intersection domain in the second argument.
  bool findIntersection(const NDIndex<Dim>&, NDIndex<Dim>& inter) {
    // If there are any points in the LField, we return the owned domain
    // for the LField.
    if ((*ComponentLF)->size() > 0) {
      inter = this->getLField()->getOwned();
      return true;
    }
    return false;
  }

  // convert this subset object into an NDIndex object
  static void makeNDIndex(const Subset_t& s, NDIndex<Dim>& i) {
    //i = s.getFieldLayout().getDomain();
    i = s.getDomain();
  }

  // The LHS tells this guy about a given local domain.  Not all
  // subsetting operations will care about this.
  bool plugBase(const NDIndex<Dim>&) { return true; }

  // Finished this dimension, rewind.
  void rewind(unsigned) { }

  // Step one or n in a given dimension.
  void step(unsigned int)      { }
  void step(unsigned int, int) { }

  // How big in this dimension
  int size(unsigned d) const { return (d == 0 ? (*ComponentLF)->size() : 0); }

  // return the value currently pointed at, or the Nth value.  We only provide
  // options for 1D, since SIndex looks like a 1D array of values to the
  // BrickExpression object.
  T& operator*()                 { return offset(0); }
  T& offset()                    { return offset(0); }
  T& offset(int i)               {
    SOffset<Dim> loc = (*ComponentLF)->getIndex(i);
    loc -= LFOffset;
    return this->getLField()->begin().offset(loc.begin());
  }
  T& unit_offset(int i)          { return offset(i); }
  int Stride(int)                { return 1; }

  // Compression interface
  bool CanCompress() const      { return false; }
  void Compress(T)              { }
  bool TryCompress()            { return false; }
  bool TryCompress(T)           { return false; }
  bool IsCompressed() const     { return this->getLField()->IsCompressed(); }
  bool DomainCompressed() const { return (*ComponentLF)->IsCompressed(); }

  //
  // PETE Interface
  //

  enum        { IsExpr = 1 };
  typedef SFI PETE_Expr_t;
  typedef T   PETE_Return_t;
  PETE_Expr_t MakeExpression() const { return *this; }

private:
  typename Subset_t::iterator_iv ComponentLF;
  SOffset<Dim>                   LFOffset;

  // calculate the offset for the current LField.  This is the position of
  // the lower-left corner of owned domain, minus the offset we are adding to
  // our sparse indices.
  void computeLSOffset() {
    if (this->getLFieldIter() != this->getBareField().end_if()) {
      NDIndex<Dim> owned = this->getLField()->getOwned();
      for (unsigned int d=0; d < Dim; ++d)
        LFOffset[d] = (owned[d].first() - this->getDomain().getOffset()[d]);
    }
  }
};


///////////////////////////////////////////////////////////////////////////
// A specialized versions of the SubFieldIter class for an SOffset, which
// is used to act as a single-value
// subset object.  This overrides certain functions from the base class,
// and provides definitions of needed functions not available in the base.
template<class T, unsigned int Dim>
class SubFieldIter<T, Dim, SOffset<Dim> >
  : public SubFieldIterBase<T, Dim, SOffset<Dim>, 1U>,
    public PETE_Expr<SubFieldIter<T, Dim, SOffset<Dim> > >
{

public:
  // public typedefs
  typedef SOffset<Dim>                   Subset_t;
  typedef SubFieldIter<T, Dim, Subset_t> SFI;

  // Construct with a SubField and the domain to use
  SubFieldIter(const BareField<T,Dim>& df,
               const typename BareField<T,Dim>::iterator_if& ldf,
               const SOffset<Dim>& s, unsigned int B)
    : SubFieldIterBase<T,Dim,Subset_t,1U>(df, ldf, s, B), SingleValPtr(0) { }

  // Default constructor
  SubFieldIter() { }

  // destructor: nothing to do
  ~SubFieldIter() { }

  //
  // Methods overriding base class behavior
  //

  // Use a new LField, where we use data on the given NDIndex region
  void setLFieldData(LField<T,Dim>* p, NDIndex<Dim>& n)  {
    SubFieldIterBase<T,Dim,Subset_t,1U>::setLFieldData(p, n);
    // the following if test COULD be removed if we assume this function is
    // called after 'findIntersection' is called and while we're still working
    // on the same LField.
    if (n.contains(Component)) {
      SOffset<Dim> s = this->getDomain();
      NDIndex<Dim> owned = p->getOwned();
      for (unsigned int d=0; d < Dim; ++d)
        s[d] -= owned[d].first();
      SingleValPtr = &(p->begin().offset(s.begin()));
    }
  }

  // Return a special code indicating what the subset type is, and match the
  // types together.
  static int  getSubsetType()  { return SOffsetSubsetType; }
  static bool matchType(int)   { return true; }

  // initialization routines
  void initialize() {
    // make an NDIndex with the point in it
    makeNDIndex(this->getDomain(), Component);

    // distribute the value to all the nodes
    this->getBareField().getsingle(Component, SingleValStore);
  }

  // Calculate the intersection with the current component and the given
  // subdomain.  Return true if an intersection exists, and the intersection
  // domain in the second argument.
  bool findIntersection(const NDIndex<Dim>& loc, NDIndex<Dim>& inter) {
    inter = Component.intersect(loc);
    return ( ! inter.empty() );
  }

  // convert this subset object into an NDIndex object
  static void makeNDIndex(const Subset_t& s, NDIndex<Dim>& i) {
    for (unsigned int d=0; d < Dim; ++d)
      i[d] = Index(s[d], s[d]);
  }

  // The LHS tells this guy about a given local domain.  Not all
  // subsetting operations will care about this.
  bool plugBase(const NDIndex<Dim>&) { return true; }

  // Finished this dimension, rewind.
  void rewind(unsigned) { }

  // Step one or n in a given dimension.
  void step(unsigned int)      { }
  void step(unsigned int, int) { }

  // How big in this dimension.
  int size(unsigned int d) const { return (d == 0 ? 1 : 0); }

  // return the value currently pointed at, or the Nth value.  We only provide
  // options for 1D, since SOffset looks like a 1D array of values to the
  // BrickExpression object.
  T& operator*()                 { return offset(0); }
  T& offset()                    { return offset(0); }
  T& offset(int)                 {
    return (SingleValPtr == 0 ? SingleValStore : *SingleValPtr);
  }
  T& unit_offset(int)            { return offset(0); }
  int Stride(int /*d*/)          { return 0; }
  // Compression interface
  bool CanCompress() const      { return false; }
  void Compress(T)              { }
  bool TryCompress()            { return false; }
  bool TryCompress(T)           { return false; }
  bool IsCompressed() const     { return false; }
  bool DomainCompressed() const { return true; }

  //
  // PETE Interface
  //

  enum        { IsExpr = 1 };
  typedef SFI PETE_Expr_t;
  typedef T   PETE_Return_t;
  PETE_Expr_t MakeExpression() const { return *this; }

private:
  T* SingleValPtr;
  T  SingleValStore;
  NDIndex<Dim> Component;
};

#endif // SUB_FIELD_ITER_H