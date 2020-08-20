// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SUB_FIELD_H
#define SUB_FIELD_H

// include files
#include "SubField/SubBareField.h"
#include "SubField/SubFieldTraits.h"


// forward declarations
template <class T> class PETE_Expr;
template<class T, unsigned D, class M, class C> class Field;


/***************************************************************************
  SubField - represent a view on a given Field, referring to a
  subset of the original field data.  This is meant as an eventual
  replacement for IndexedField.  It is the same as SubBareField, but also
  includes the Mesh and Centering template parameters.
 ***************************************************************************/

template<class T, unsigned int Dim, class M, class C, class S>
class SubField : public SubBareField<T,Dim,S> {

  friend class Field<T,Dim,M,C>;

public:
  //
  // accessor functions
  //

  // return a reference to the field we are subsetting
  Field<T,Dim,M,C>&       getField() const { return F; }

  //
  // bracket operators
  //

  // bracket operator, which select subsets of the BareField.
  //mwerks template<class S2>
  //mwerks SubField<T,Dim,M,C,typename SubFieldTraits<T,Dim,S,S2>::Return_t>
  //mwerks  operator[](const S2&);
  //////////////////////////////////////////////////////////////////////
  // bracket operators, which select subsets of the Field.  This
  // further subsets from the current SubField based on the type of
  // input subset object.
  template<class S2>
  SubField<T,Dim,M,C,typename SubFieldTraits<T,Dim,S,S2>::Return_t>
  operator[](const S2& s) {
    // create a new instance of the resulting subset object
    typename SubFieldTraits<T,Dim,S,S2>::Return_t newdomain;
    
    

    // make sure we can subset by the number of dimensions requested, then
    // combine the current subset value with the new one
    int B = SubFieldTraits<T,Dim,S,S2>::Brackets_u;
    if (this->checkAddBrackets(B)) {
        SubFieldTraits<T,Dim,S,S2>::combine(
            SubBareField<T,Dim,S>::MyDomain, 
            s, 
            newdomain, 
            SubBareField<T,Dim,S>::Brackets, 
            SubBareField<T,Dim,S>::A);
        SubBareField<T,Dim,S>::Brackets += B;
    }

    // return a new SubField
    return SubField<T,Dim,M,C,
      typename SubFieldTraits<T,Dim,S,S2>::Return_t>(F, newdomain);
  }


  //
  // assignment operators
  //

  // assignment of a scalar
  void operator=(T);

  // assignment of another subfield
  SubField<T,Dim,M,C,S>& operator=(const SubField<T,Dim,M,C,S> &);

  // assignment of an arbitrary expression
  //mwerks  template<class B>
  //mwerks  SubField<T,Dim,M,C,S>& operator=(const PETE_Expr<B> &);
  //////////////////////////////////////////////////////////////////////
  // assignment of an arbitrary expression
  template<class B>
  SubField<T,Dim,M,C,S>&
  operator=(const PETE_Expr<B> &b) {

    assign(*this, b);
    return *this;
  }



protected: 
  // the field we are subsetting
  Field<T,Dim,M,C>& F;

public:

  // Make the constructor private so that only this class and it's friends
  // can construct them.
  //mwerks  template<class S2>
  //mwerks  SubField(Field<T,Dim,M,C>&, const S2&);
  //////////////////////////////////////////////////////////////////////
  // Make the constructor private so that only this class and it's friends
  // can construct them.
  template<class S2>
  SubField(Field<T,Dim,M,C>& f, const S2& s)
    : SubBareField<T,Dim,S>(f, s), F(f) { }

};

#include "SubField/SubField.hpp"

#endif // SUB_FIELD_H

/***************************************************************************
 * $RCSfile: SubField.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubField.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
