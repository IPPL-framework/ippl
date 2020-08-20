// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INDEXED_FIELD_H
#define INDEXED_FIELD_H

// include files
#include "Field/IndexedBareField.h"
#include "Utility/PAssert.h"

#include <iostream>

// forward declarations
template<class T, unsigned D, class M, class C> class Field;

//----------------------------------------------------------------------

template < class T, unsigned Dim, unsigned Brackets, class M, class C>
class IndexedField : public IndexedBareField<T,Dim,Brackets>
{

  friend class IndexedField<T,Dim,Brackets-1,M,C>;
  friend class Field<T,Dim,M,C>;

public:

  IndexedField<T,Dim,Brackets+1,M,C> operator[](const Index& idx)
  {
    CTAssert(Brackets<Dim);
    return IndexedField<T,Dim,Brackets+1,M,C> (F,IndexedBareField<T,Dim,Brackets>::Indexes,idx);
  }
  IndexedField<T,Dim,Brackets+1,M,C> operator[](int i)
  {
    CTAssert(Brackets<Dim);
    return IndexedField<T,Dim,Brackets+1,M,C> (F,IndexedBareField<T,Dim,Brackets>::Indexes,Index(i,i));
  }
  // Also allow using a single NDIndex instead of N Index objects:
  IndexedField<T,Dim,Dim,M,C> operator[](const NDIndex<Dim>& ndidx)
  {
    CTAssert(Brackets<Dim);
    return IndexedField<T,Dim,Dim,M,C> (F,IndexedBareField<T,Dim,Brackets>::Indexes,ndidx);
  }

  void operator=(T x)
  {
    assign(*this,x);
  }

  IndexedField<T,Dim,Brackets,M,C>&
  operator=(const IndexedField<T,Dim,Brackets,M,C>& x)
  {
    CTAssert(Dim==Brackets);
    assign(*this,x);
    return *this;
  }

  IndexedField<T,Dim,Brackets,M,C>&
  operator=(const Index& x)
  {
    assign(*this,x);
    return *this;
  }

  template<class T1, unsigned Dim1, unsigned Brackets1, class M1, class C1>
  IndexedField<T,Dim,Brackets,M,C>&
  operator=(const IndexedField<T1,Dim1,Brackets1,M1,C1>& x)
  {
    CTAssert(Dim1==Brackets1);
    assign(*this,x);
    return *this;
  }

  template<class B>
  IndexedField<T,Dim,Brackets,M,C>&
  operator=(const PETE_Expr<B>& x)
  {
    assign(*this,x);
    return *this;
  }


        Field<T,Dim,M,C>& getField()       { return F; }
  const Field<T,Dim,M,C>& getField() const { return F; }

protected: 

  Field<T,Dim,M,C> &F;

  // Make the constructors private so that only Field and IndexedField
  // can construct them.
  IndexedField(Field<T,Dim,M,C> &f, const Index& idx)
    : IndexedBareField<T,Dim,Brackets>(f,idx), F(f) {}
  IndexedField(Field<T,Dim,M,C> &f, int i)
    : IndexedBareField<T,Dim,Brackets>(f,i), F(f) {}
  IndexedField(Field<T,Dim,M,C> &f, const NDIndex<Brackets-1>& idx, const Index& i)
    : IndexedBareField<T,Dim,Brackets>(f,idx,i), F(f) {}
  // Also allow using a single NDIndex instead of N Index objects:
  IndexedField(Field<T,Dim,M,C> &f, const NDIndex<Dim>& ndidx)
    : IndexedBareField<T,Dim,Dim>(f,ndidx), F(f) {}
  
};

//////////////////////////////////////////////////////////////////////

#endif // INDEXED_FIELD_H

/***************************************************************************
 * $RCSfile: IndexedField.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: IndexedField.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
