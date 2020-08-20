/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef SUB_BARE_FIELD_H
#define SUB_BARE_FIELD_H

/***************************************************************************
  SubBareField - represent a view on a given BareField, referring to a
  subset of the original field data.  This is meant as an eventual
  replacement for IndexedBareField.

  SubBareField is templated on the T and Dim of the Field, and also on the
  type of subset object used to refer to the field subsection.  This can
  range from NDIndex for a rectangular block, to SIndex for an arbitrary
  list of points, to SOffset to store a single point.  The behavior of this
  class, and in particular the iterator for this class, is specialized to
  each subset object using a traits class 'SFTraits' and a couple of global
  functions.

  From a BareField, you create a SubBareField by using the bracket
  operators [], giving a particular type of data to further index the object.
  Indexing with and Index or NDIndex object results in a SubBareField object
  templated on NDIndex, etc.  A SubBareField keeps track of the number of
  'Brackets' its been given; an expression can only involve a SubBareField
  if Brackets == Dim.  Using [] on a SubBareField will generally increase the
  number of brackets, and result in a new SubBareField incorporating the
  info about how to select the subset as specified by the object in brackets.
  For example, Field<T,Dim>[Index] results in a SubBareField<T,Dim,NDIndex<Dim>>
  object with Brackets==1; further application of [Index] operators increases
  the number of Brackets and stores each Index in the internal NDIndex at the
  proper location.  SFTraits provides the information on what types of objects
  can be used to subset, and what type of subset object they produce.

  For the future, it might be useful to make this 'SubBareField' be the actual
  class employed by the user, with the current 'BareField' some form of
  internal class used only to store the data in whatever representation is
  most efficient.
 ***************************************************************************/

// include files
#include "SubField/SubFieldIter.h"
#include "SubField/SubFieldTraits.h"
#include "PETE/IpplExpressions.h"
#include <iostream>

// forward declarations
template <class T, unsigned Dim, class S> class SubBareField;
template <class T, unsigned Dim, class S> class SubFieldIter;
template <class T, unsigned Dim, class S>
std::ostream& operator<<(std::ostream&,const SubBareField<T,Dim,S>&);


template <class T, unsigned Dim, class S>
class SubBareField : public PETE_Expr< SubBareField<T,Dim,S> > {

  friend class BareField<T,Dim>;

public:
  //# public typedefs
  typedef T T_t;
  typedef S Index_t;
  typedef SubFieldIter<T,Dim,S> iterator;

  //# public enumerations
  enum { Dim_u = Dim };

  constexpr SubBareField(const SubBareField<T, Dim, S>&) = default;

  // Return the beginning and end iterators for this class.
  iterator begin() const;
  iterator end() const;

  //
  // accessor functions
  //

  // return the 'domain', that is, the information which subsets the field
  const S& getDomain() const { return MyDomain; }

  // fill in the second argument with the data for the 'bounding box' of
  // the domain.  This could be the whole field domain, a single point, or
  // perhaps just a subset of the whole domain.
  void makeNDIndex(NDIndex<Dim> &i) { iterator::makeNDIndex(MyDomain, i); }

  // return a reference to the field we are subsetting
  BareField<T,Dim>& getBareField() const { return A; }

  // Return a single value.
  T    get() { T r; get(r); return r; }
  void get(T& r);

  // Return a typecode for the subset object
  static int getSubsetType() { return iterator::getSubsetType(); }

  // check to make sure Dim == Brackets.
  bool checkBrackets() const { return Brackets == Dim; }

  //
  // bracket operators
  //

  // bracket operator, which select subsets of the BareField.
  //mwerks template<class S2>
  //mwerks SubBareField<T,Dim,typename SubFieldTraits<T,Dim,S,S2>::Return_t>
  //mwerks operator[](const S2&);
  //////////////////////////////////////////////////////////////////////
  // bracket operators, which select subsets of the BareField.  This
  // further subsets from the current SubBareField based on the type of
  // input subset object.
  template<class S2>
  SubBareField<T,Dim,typename SubFieldTraits<T,Dim,S,S2>::Return_t>
  operator[](const S2& s) {
    // create a new instance of the resulting subset object
    typename SubFieldTraits<T,Dim,S,S2>::Return_t newdomain;
    
    

    // make sure we can subset by the number of dimensions requested, then
    // combine the current subset value with the new one
    int B = SubFieldTraits<T,Dim,S,S2>::Brackets_u;
    if (checkAddBrackets(B)) {
      SubFieldTraits<T,Dim,S,S2>::combine(MyDomain, s, newdomain, Brackets, A);
      Brackets += B;
    }
 
    // return a new SubBareField
    return SubBareField<T,Dim,
      typename SubFieldTraits<T,Dim,S,S2>::Return_t>(A,newdomain);
  }


  //
  // assignment operators
  //

  // assignment of another SubBareField
  SubBareField<T,Dim,S>& operator=(const SubBareField<T,Dim,S> &);

  // assignment of a scalar
  SubBareField<T,Dim,S>& operator=(T);

  // assignment of an arbitrary expression
  //mwerks template<class B>
  //mwerks SubBareField<T,Dim,S>& operator=(const PETE_Expr<B> &);
  //////////////////////////////////////////////////////////////////////
  // assignment of an arbitrary expression
  template<class B>
  SubBareField<T,Dim,S>&
  operator=(const PETE_Expr<B> &b) {
    
    
    assign(*this, b);
    return *this;
  }


  //
  // I/O
  //

  void write(std::ostream&);

  //
  // PETE interface
  //

  enum { IsExpr = 1 };
  typedef iterator PETE_Expr_t;
  iterator MakeExpression() const { return begin(); }

  // Pass operator() down to each element of type T.
  // Could also build versions with 2 or more arguments...
  // Without member templates we are restricted to passing an integer down.
  PETE_TUTree<OpParens<int>, iterator> operator()(int arg) {
    checkBrackets();
    typedef PETE_TUTree<OpParens<int>, iterator> Elem_t;
    return Elem_t(arg, begin());
  }
  PETE_TUTree< OpParens< std::pair<int,int> >, iterator> operator()(int a1,int a2){
    checkBrackets();
    typedef PETE_TUTree<OpParens< std::pair<int,int> >, iterator> Elem_t;
    return Elem_t(std::pair<int,int>(a1,a2), begin());
  }

protected:
  // the field we are subsetting
  BareField<T,Dim>& A;

  // the 'domain', that is, the information which subsets the field
  S MyDomain;

  // the current number of dimensions we have indexed via bracket operators.
  // An expression can only be carried out if the number of brackets == Dim.
  unsigned int Brackets;

  // check to see if it is ok to add the given number of brackets to our
  // current number
  bool checkAddBrackets(unsigned int);

public:
  // the class constructor, used by BareField or SubBareField to make a new one
  //mwerks template<class S2>
  //mwerks SubBareField(BareField<T,Dim>&, const S2&);
  //////////////////////////////////////////////////////////////////////
  // Make the constructor private so that only this class and it's friends
  // can construct them.
  template<class S2>
  SubBareField(BareField<T,Dim>& f, const S2& s) : A(f) {
    
    
    
    // initialize the subset object, to a state where it can be combined
    // with the given input data.  Then, put in data from given subset object.
    Brackets = SubFieldTraits<T,Dim,S,S2>::construct(MyDomain, s, A);
  }

};

// I/O

// write a subfield to the given ostream
template<class T, unsigned int Dim, class S>
inline
std::ostream& operator<<(std::ostream& o, const SubBareField<T,Dim,S>& s) {
  
  
  SubBareField<T,Dim,S>& ncs = const_cast<SubBareField<T,Dim,S>&>(s);
  ncs.write(o);
  return o;
}


#include "SubField/SubBareField.hpp"

#endif

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
