/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef INDEXED_BARE_FIELD_H
#define INDEXED_BARE_FIELD_H

// Include files
#include "Field/FieldLoc.h"
#include "Utility/PAssert.h"
#include "PETE/IpplExpressions.h"

#include <iostream>
#include "Field/BareFieldIterator.h"

// forward declarations
class Index;
template<unsigned Dim> class NDIndex;
template<class T, unsigned Dim> class LField;
template<class T, unsigned Dim> class BareField;
template<class T, unsigned Dim, unsigned Brackets> class IndexedBareField;
template<class T, unsigned Dim, unsigned Brackets>
std::ostream& operator<<(std::ostream&, const IndexedBareField<T,Dim,Brackets>&);


//
// Because of a limitation in the template mechanism on SGI
// we need a tag class to get it to match the Dim parameter
// in the IndexedBareField.
//
template<unsigned D> class DimTag {};

//----------------------------------------------------------------------

template<class T, unsigned Dim>
class IndexedBareFieldIterator : public BareField<T,Dim>::iterator
{
public:
  enum { Dim_u = Dim };
  typedef T return_type;

  // Null ctor for arrays of these things.
  IndexedBareFieldIterator()
    {
    }

  // Construct with a BareField and an NDIndex.
  // Point to the beginning of the BareField & record the index.
  IndexedBareFieldIterator(BareField<T,Dim>& df, const NDIndex<Dim> &idx)
    : BareField<T,Dim>::iterator(df), I(idx)
    {
      // make sure we're at the start of the PROPER LField (not necessarily
      // the first)
      beginLField(); 
    }

  // Destructor
  ~IndexedBareFieldIterator()
    {
    }

  // Return the subdomain we are indexing into here
  const NDIndex<Dim>& getDomain() const
    {
      return I;
    }

  // Some other accsssor functions:
  const BareField<T,Dim>& getBareField() const { return *(this->MyBareField); }
  BareField<T,Dim>& getBareField()       { return *(this->MyBareField); }

  // Fills the guard cells if necessary.
  template<unsigned D1, class T1>
  void FillGCIfNecessary(const BareField<T1,D1> &lhs) const
  {
    // If the dirty flag is set, we need to decide whether we need
    // to fill the guard cells. A safe way to do this is to pass
    // the domain of the lhs and our domain to 'isStencil', which
    // will check to see if the domains are offset, indicating a
    // stencil. This doesn't necessarily mean that we *must* fill
    // (as in the case where the stencil is fully contained in an
    // LField), but filling is the safe thing to do short of a more
    // detailed analysis.

    if (this->GetBareField().isDirty())
      {
	if ( isStencil(lhs.getDomain(), getDomain()) ||
	     !(lhs.getLayout() == getBareField().getLayout()) )
	  {
	    BareField<T,Dim> &bf = 
	      const_cast<BareField<T,Dim>&>(this->GetBareField());
	    bf.fillGuardCells();
	  }
      }
  }


  // The LHS tells this guy about a given local domain.
  //MWERKS: Moved this member template function definition into here.
  //MWERKS template<unsigned int D1>
  //MWERKS bool plugBase(const NDIndex<D1>& i);
  template<unsigned int D2>
  bool
  plugBase(const NDIndex<D2>& i)
  {
    
    
    
    //  Inform msg("plugBase", INFORM_ALL_NODES);
    
    // Substitute the index expression from the lhs into this one.
    NDIndex<Dim> plugged ( I.plugBase(i) );

    // Try to find a single local array that has all of the rhs.
    // Loop over all the local arrays.
    typedef typename BareField<T,Dim>::iterator_if BFI;
    BFI e = BareFieldIterator<T,Dim>::MyBareField->end_if();
    for (BFI lf_i = BareFieldIterator<T,Dim>::MyBareField->begin_if(); lf_i !=  e ;  ++lf_i) {
      if ( (*lf_i).second->getAllocated().contains( plugged ) ) {
	// Found it.  Make this one current and go.
	BareFieldIterator<T,Dim>::CurrentLField = lf_i;
	(CompressedBrickIterator<T,Dim>&)(*this) = (*lf_i).second->begin(plugged);
	return true;
      }
    }
    // Didn't find it.
    return false;
  }

  // From a given FieldLoc, set where we're pointing.  Since an
  // IndexedBareField iterates over only a subset, we must make sure the
  // requested point is within the subset domain.
  void SetCurrentLocation(const FieldLoc<Dim>& loc)
    {
      BareField<T,Dim>::SetCurrentLocation(loc);
      if (!this->done()) {
	FieldLoc<Dim> checkloc;
	GetCurrentLocation(checkloc);
	PAssert(I.contains(checkloc.getDomain()));
      }
    }

  // Increment this iterator.
  IndexedBareFieldIterator<T,Dim>& operator++()
    {
      CompressedBrickIterator<T,Dim>::operator++();
      if (*this == (*BareFieldIterator<T,Dim>::CurrentLField).second->end())
	general_increment();
      return *this;
    }

  // Set ourselves up to point to the first item in the current LField
  // that we'll be accessing.  Since we're iterating over a subset of
  // the domain, we might need to skip over some LFields first.
  void beginLField()
    {
      // get a pointer to the current LField, and ourselves cast as a
      // compressed brick iterator.
      CompressedBrickIterator<T,Dim> *c=(CompressedBrickIterator<T,Dim>*)this;

      // advance to the first LField that intersects our subdomain
      while (!this->done() && ! I.touches((*BareFieldIterator<T,Dim>::CurrentLField).second->getOwned()))
	this->nextLField();

      // now that we're at a new LField with some points, set the pointer
      // to the first item we'll need.  Set ourselves equal to a
      // compressed brick iterator that iterates over the subdomain formed
      // from the intersection of the total indexed subdomain and the
      // current LField's owned domain.
      if (!this->done())
	*c = (*BareFieldIterator<T,Dim>::CurrentLField).second->begin(
           I.intersect((*BareFieldIterator<T,Dim>::CurrentLField).second->getOwned()));
      else
	BareFieldIterator<T,Dim>::LFIndex = (-1);
    }

protected:
  // User supplied offsets.
  NDIndex<Dim> I;			    

  // Increment and go on to the next LField if necessary.
  void general_increment()
    {
      this->nextLField();
      beginLField();
    }
}; 

//----------------------------------------------------------------------

template < class T, unsigned Dim, unsigned Brackets >
class IndexedBareField : public PETE_Expr< IndexedBareField<T,Dim,Brackets> >
{

  friend class IndexedBareField<T,Dim,Brackets-1>;
  friend class BareField<T,Dim>;

public:

  typedef T T_t;
  enum { Dim_u = Dim };
  typedef IndexedBareFieldIterator<T,Dim> iterator;

  constexpr IndexedBareField (const IndexedBareField<T, Dim, Brackets>&) = default;
  IndexedBareField<T,Dim,Brackets+1> operator[](const Index& idx)
  {
    CTAssert(Brackets<Dim);
    return IndexedBareField<T,Dim,Brackets+1> (A,Indexes,idx);
  }

  IndexedBareField<T,Dim,Brackets+1> operator[](int i)
  {
    CTAssert(Brackets<Dim);
    return IndexedBareField<T,Dim,Brackets+1> (A,Indexes,Index(i,i));
  }
  // Also allow using a single NDIndex instead of N Index objects:
  IndexedBareField<T,Dim,Brackets+1> operator[](const NDIndex<Dim>& ndidx)
  {
    CTAssert(Brackets<Dim);
    return IndexedBareField<T,Dim,Brackets+1> (A,Indexes,ndidx);
  }

  void operator=(T x)
  {
    assign(*this,x);
  }

  IndexedBareField<T,Dim,Brackets>&
  operator=(const IndexedBareField<T,Dim,Brackets>& x)
  {
    CTAssert(Dim==Brackets);
    assign(*this,x);
    return *this;
  }

  template<class X, unsigned Dim1, unsigned Brackets1>
  IndexedBareField<T,Dim,Brackets>&
  operator=(const IndexedBareField<X,Dim1,Brackets1>& x)
  {
    CTAssert(Dim1==Brackets1);
    assign(*this,x);
    return *this;
  }

  IndexedBareField<T,Dim,Brackets>&
  operator=(const Index& x)
  {
    assign(*this,x);
    return *this;
  }

  template<class B>
  IndexedBareField<T,Dim,Brackets>&
  operator=(const PETE_Expr<B>& x)
  {
    assign(*this,x);
    return *this;
  }

  const NDIndex<Dim>& getDomain() const { return Indexes; }
        BareField<T,Dim>& getBareField()       { return A; }
  const BareField<T,Dim>& getBareField() const { return A; }

  // Pass operator() down to each element of type T.
  // Could also build versions with 2 or more arguments...
  // Without member templates we are restricted to passing an integer down.
  PETE_TUTree< OpParens<int>, iterator >
  operator()(int arg)
  {
    CTAssert(Dim==Brackets);
    typedef PETE_TUTree< OpParens<int> , iterator > Elem_t;
    return Elem_t(arg,begin());
  }
  PETE_TUTree< OpParens< std::pair<int,int> >, iterator >
  operator()(int a1, int a2)
  {
    CTAssert(Dim==Brackets);
    typedef PETE_TUTree< OpParens< std::pair<int,int> > , iterator > Elem_t;
    return Elem_t(std::pair<int,int>(a1,a2),begin());
  }

  // Return a single value.
  // operator T() { T r; getsingle(r); return r; }
  //T get() { T r ; getsingle(r); return r; }
  T get();

  // Return the beginning and end iterators for this class.
  iterator begin() const { return iterator( A, Indexes ); }
  iterator end() const   { return iterator(); }

  // Print out the values for this object to the given output stream.
  void write(std::ostream&);

  // PETE interface
  enum { IsExpr = 0 };
  typedef iterator PETE_Expr_t;
  iterator MakeExpression() const { return begin(); }

protected: 
  BareField<T,Dim> &A;
  NDIndex<Brackets> Indexes;

  // Make the constructors private so that only BareField and IndexedBareField
  // can construct them.
  IndexedBareField(BareField<T,Dim> &a, const Index& idx)
    : A(a), Indexes(idx) {}
  IndexedBareField(BareField<T,Dim> &a, int i)
    : A(a), Indexes(Index(i,i)) {}
  IndexedBareField(BareField<T,Dim> &a,const NDIndex<Brackets-1>& idx, const Index& i)
    : A(a), Indexes(idx,i) {}

  // Also allow using a single NDIndex instead of N Index objects:
  IndexedBareField(BareField<T,Dim> &a, const NDIndex<Dim>& ndidx)
    : A(a), Indexes(ndidx) {}

  // read a single value from the array.
  void getsingle(T&);
};


template < class T, unsigned Dim, unsigned Brackets >
inline
std::ostream& operator<<(std::ostream& out, const IndexedBareField<T,Dim,Brackets>& i) {
  
  
  IndexedBareField<T,Dim,Brackets>& nci =
    const_cast<IndexedBareField<T,Dim,Brackets>&>(i);
  nci.write(out);
  return out;
}

//////////////////////////////////////////////////////////////////////

#include "Field/IndexedBareField.hpp"

#endif // INDEXED_BARE_FIELD_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
