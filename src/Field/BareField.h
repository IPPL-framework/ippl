// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef BARE_FIELD_H
#define BARE_FIELD_H

/***************************************************************************
 *
 * This is the user visible BareField of type T.
 * It doesn't even really do expression evaluation; that is
 * handled with the templates in Expressions.h
 *
 ***************************************************************************/

// include files
#include "Field/LField.h"
#include "Field/IndexedBareField.h"
#include "Field/GuardCellSizes.h"
#include "Field/FieldLoc.h"
#include "Field/BareFieldIterator.h"
#include "FieldLayout/FieldLayout.h"
#include "FieldLayout/FieldLayoutUser.h"
#include "PETE/IpplExpressions.h"
#include "Index/SIndex.h"
#include "SubField/SubBareField.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"
#include "Utility/Unique.h"
#include "Utility/my_auto_ptr.h"
#include "Utility/vmap.h"

#include <iostream>
#include <cstdlib>

// forward declarations
class Index;
template<unsigned Dim> class NDIndex;
template<unsigned Dim> class FieldLayout;
template<class T, unsigned Dim> class LField;
template<class T, unsigned Dim> class BareField;
template<class T, unsigned Dim>
std::ostream& operator<<(std::ostream&, const BareField<T,Dim>&);

// class definition
template<class T,  unsigned Dim>
class BareField : public FieldLayoutUser,
		  public PETE_Expr< BareField<T,Dim> >
{

public: 
  // Some externally visible typedefs and enums
  typedef T T_t;
  typedef FieldLayout<Dim> Layout_t;
  typedef LField<T,Dim> LField_t;
  enum { Dim_u = Dim };

public:
  // A default constructor, which should be used only if the user calls the
  // 'initialize' function before doing anything else.  There are no special
  // checks in the rest of the BareField methods to check that the field has
  // been properly initialized.
  BareField();

  // Copy ctor.  Deep copy.
  BareField(const BareField<T,Dim>&);

  // Create a new BareField with a given layout and optional guard cells.
  BareField(Layout_t &);
  BareField(Layout_t &, const GuardCellSizes<Dim>&);

  // Destroy the BareField.
  ~BareField();

  // Initialize the field, if it was constructed from the default constructor.
  // This should NOT be called if the field was constructed by providing
  // a FieldLayout.
  void initialize(Layout_t &);
  void initialize(Layout_t &, const bool); //UL: for pinned memory allocation
  void initialize(Layout_t &, const GuardCellSizes<Dim>&);

  // Some typedefs to make access to the maps a bit simpler.
  typedef vmap< typename Unique::type, my_auto_ptr< LField<T,Dim> > > 
    ac_id_larray;
  typedef typename ac_id_larray::iterator iterator_if;
  typedef typename ac_id_larray::const_iterator const_iterator_if;
  typedef typename LField<T,Dim>::iterator LFI;

  // An iterator over the elements of the BareField.
  typedef BareFieldIterator<T,Dim> iterator;

  // Let the user iterate over the larrays.
  iterator_if begin_if() { return Locals_ac.begin(); }
  iterator_if end_if()   { return Locals_ac.end(); }
  const_iterator_if begin_if() const { return Locals_ac.begin(); }
  const_iterator_if end_if()   const { return Locals_ac.end(); }
  typename ac_id_larray::size_type size_if() const { return Locals_ac.size(); }

  // If you make any modifications using an iterator, you must call this.
  virtual void fillGuardCells(bool reallyFill = true) const;

  // For use before scatters into guard cells
  void setGuardCells(const T&) const;

  // For use after scatters into guard cells
  void accumGuardCells();

  // Dirty flag maintenance.
  bool isDirty() const { return dirty_m; }
  void setDirtyFlag() { if (IpplInfo::deferGuardCellFills) dirty_m = true; }
  void clearDirtyFlag() { dirty_m = false; }

  // If the dirty flag is not set, fill guard cells; otherwise,
  // don't fill guard cells, but do call boundary conditions.
  void fillGuardCellsIfNotDirty() const
  {
    if (!isDirty())
      fillGuardCells();
    else
      fillGuardCells(false);
  }

  // Access to the layout.
  Layout_t &getLayout() const
  {
    PAssert(Layout != 0);
    return *Layout;
  }

  // When we apply a bracket it converts the type
  // to IndexedBareField so that we can check at compile time
  // that we have the right number of indexes and brackets.
  IndexedBareField<T,Dim,1>   operator[](const Index& idx);
  IndexedBareField<T,Dim,1>   operator[](int i);
  IndexedBareField<T,Dim,Dim> operator[](const NDIndex<Dim>& nidx);
  SubBareField<T,Dim,SIndex<Dim> >  operator[](const SIndex<Dim>&);

  // Boundary condition handling.
  const GuardCellSizes<Dim>& getGC() const { return Gc; }
  const GuardCellSizes<Dim>& getGuardCellSizes() const { return Gc; }
  unsigned leftGuard(unsigned d) const   { return getGC().left(d); }
  unsigned rightGuard(unsigned d) const  { return getGC().right(d); }

  const Index& getIndex(unsigned d) const {return getLayout().getDomain()[d];}
  const NDIndex<Dim>& getDomain() const { return getLayout().getDomain(); }

  // Assignment from a constant.
  const BareField<T,Dim>& operator=(T x)
  {
    assign(*this,x);
    return *this;
  }

  // Assign another array.
  const BareField<T,Dim>&
  operator=(const BareField<T,Dim>& x)
  {
    assign(*this,x);
    return *this;
  }

  template<class X>
  const BareField<T,Dim>&
  operator=(const BareField<X,Dim>& x)
  {
    assign(*this,x);
    return *this;
  }

  // If we have member templates available, assign a generic expression.
  template<class B>
  const BareField<T,Dim>&
  operator=(const PETE_Expr<B>& x)
  {
    assign(*this,x);
    return *this;
  }

  //
  // Methods to deal with compression
  //

  // Tell whether BareField may be compressed or not
  bool compressible() const
  {
    return compressible_m;
  }

  // Report what fraction of the elements are compressed.
  // Completely compressed is 1.0, completely uncompressed is 0.0.
  double CompressedFraction() const;

  // Tell a BareField to compress/uncompress itself.

  void Compress() const;
  void Uncompress() const;

  // dictate whether BareField may be compressed or not.  This will set
  // a flag, and set the state to be consistent with the flag.
  void setCompression(bool compress)
  {
    if (!Ippl::noFieldCompression) {
      compressible_m = compress;
      if (compress)
        Compress();
      else
        Uncompress();
    }
  }

  //
  // virtual functions for FieldLayoutUser's (and other UserList users)
  //

  // Repartition onto a new layout
  virtual void Repartition(UserList *);

  // Tell this object that an object is being deleted
  virtual void notifyUserOfDelete(UserList *);

  //
  // Scalar code interface
  //

  // Let the user iterate over each element.  Return a begin iterator
  // for the whole BareField.
  iterator begin() const
  {
    return iterator(const_cast<BareField<T,Dim> &>(*this));
  }

  // Let the user iterate over each element.  Return an end iterator
  // for the whole BareField.
  iterator end() const
  {
    return iterator();
  }

  // Let the user iterate over each element, but specify where they should
  // start (other than the beginning).  We don't need an end, since this just
  // affect where we start, not how we traverse the data.
  iterator beginLoc(const FieldLoc<Dim> &loc) const
  {
    return iterator(loc, const_cast<BareField<T,Dim> &>(*this));
  }

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
    // Turn off compression for now
    setCompression(false);

    // Fill the guard cells if they are dirty
    if (tryfill && isDirty())
      fillGuardCells(true);

    // Increment the counter for scalar code starts
    INCIPPLSTAT(incBeginScalarCodes);
  }

  // Finish up after a scalar code section.  This means go back to
  // a compressed state if possible, and, if the user says so, indicate
  // that things have been modified so the dirty flag must be set.
  // If modified is true, this routine must be called in SPMD-fashion.  If
  // it is false, it can be called on a per-node basis.
  void finishScalarCode(bool modified = true)
  {
    // If the user says the field has been modified, either we have
    // to set the dirty flag or go and actually fill GC's (if we are not
    // deferring GC fills, we need to update things now).  So, we try to
    // set the dirty flag; if it gets set, then fillGuardCellsIfNotDirty
    // will just leave it set, if it does not get set, then that routine
    // will go and actually do a GC fill.
    if (modified)
      {
	setDirtyFlag();
	fillGuardCellsIfNotDirty();
      }

    // Turn compression back on, and try to compress if possible
    setCompression(true);

    // Increment the counter for scalar code stops
    INCIPPLSTAT(incEndScalarCodes);
  }

  //
  // Single-element access
  //

  // Get a ref to a single element of the Field; if it is not local to our
  // processor, print an error and exit.  This allows the user to provide
  // different index values on each node, instead of using the same element
  // and broadcasting to all nodes.
  T& localElement(const NDIndex<Dim>&) const;

  // get a single value and return it in the given storage.  Whichever
  // node owns the value must broadcast it to the other nodes.
  void getsingle(const NDIndex<Dim>&, T&) const;

  //
  // I/O routines for a BareField
  // 

  void write(std::ostream&);

  //
  // PETE interface.
  //

  enum { IsExpr = 0 };
  typedef iterator PETE_Expr_t;
  iterator MakeExpression() const { return begin(); }

protected:
  // The container of local arrays.
  ac_id_larray Locals_ac;

private:
  friend class BareFieldIterator<T,Dim>;

  // Setup allocates all the LFields.  The various ctors call this.
  void setup();

  // How the local arrays are laid out.
  Layout_t *Layout;

  // The specification of how many guard cells this array needs.
  GuardCellSizes<Dim> Gc;

  // A version of get_single that uses a slower but more
  // robust method.  The externally visible get_single
  // calls this when it determines it needs it.
  void getsingle_bc(const NDIndex<Dim>&, T&) const;

  // Dirty flag. Tells whether or not we need to fill guard cells.
  bool dirty_m;

  // compression flags
  bool compressible_m;          // are we allowed to compress this BareField?

  //UL: for pinned memory allocation
  bool pinned;
};

//////////////////////////////////////////////////////////////////////

//
// Construct a BareField from nothing ... default case.
//

template< class T, unsigned Dim >
inline
BareField<T,Dim>::
BareField()
: Layout(0),			 // No layout yet.
  Gc( GuardCellSizes<Dim>(0U) ), // No guard cells.
  compressible_m(!Ippl::noFieldCompression),
  pinned(false) //UL: for pinned memory allocation
{
}


//
// Construct a BareField from a FieldLayout.
//

template< class T, unsigned Dim >
inline
BareField<T,Dim>::
BareField(Layout_t & l)
: Layout(&l),			 // Just record the layout.
  Gc( GuardCellSizes<Dim>(0U) ), // No guard cells.
  compressible_m(!Ippl::noFieldCompression),
  pinned(false) //UL: for pinned memory allocation
{
  setup();			// Do the common setup chores.
}


//
// Construct a BareField from a FieldLayout and guard cell sizes.
//

template< class T, unsigned Dim >
inline
BareField<T,Dim>::
BareField(Layout_t & l, const GuardCellSizes<Dim>& g)
: Layout(&l),			// Just record the layout.
  Gc(g),			// Just record guard cells.
  compressible_m(!Ippl::noFieldCompression),
  pinned(false) //UL: for pinned memory allocation
{
  setup();			// Do the common setup chores.
}


//////////////////////////////////////////////////////////////////////

template< class T, unsigned Dim >
inline IndexedBareField<T,Dim,1> 
BareField<T,Dim>::operator[](const Index& idx)
{
  return IndexedBareField<T,Dim,1>(*this,idx);
}

template< class T, unsigned Dim >
inline IndexedBareField<T,Dim,1>
BareField<T,Dim>::operator[](int i) 
{
  return IndexedBareField<T,Dim,1>(*this,i);
}

template< class T, unsigned Dim >
inline IndexedBareField<T,Dim,Dim> 
BareField<T,Dim>::operator[](const NDIndex<Dim>& nidx)
{
  return IndexedBareField<T,Dim,Dim>(*this,nidx);
}

template< class T, unsigned Dim >
inline SubBareField<T,Dim,SIndex<Dim> >
BareField<T,Dim>::operator[](const SIndex<Dim>& s)
{
  return SubBareField<T,Dim,SIndex<Dim> >(*this, s);
}

template< class T, unsigned Dim >
inline
std::ostream& operator<<(std::ostream& out, const BareField<T,Dim>& a)
{
  
  

  BareField<T,Dim>& nca = const_cast<BareField<T,Dim>&>(a);
  nca.write(out);
  return out;
}


//////////////////////////////////////////////////////////////////////

#include "Field/BareField.hpp"

#endif // BARE_FIELD_H

/***************************************************************************
 * $RCSfile: BareField.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: BareField.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
