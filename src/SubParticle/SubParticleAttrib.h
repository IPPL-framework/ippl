// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SUB_PARTICLE_ATTRIB_H
#define SUB_PARTICLE_ATTRIB_H

/***************************************************************************
  SubParticleAttrib - represents a view on a given ParticleAttrib or
  ParticleAttribElem, given an SIndex.  It is used in gathering or
  scattering data from SIndex-listed points in a Field to a ParticleAttrib.

  SubParticleAttrib is templated on the general attribute type (either
  ParticleAttrib or ParticleAttribElem), the type of the data, and the
  Dim of the data.  It will only work with these cases indexed by an
  SIndex object.  In that case, the SIndex represents a 1-D "serialization"
  of any other N-D object, such as a BareField, so that ParticleAttrib's
  can be involved directly with BareField's in expressions.  The syntax is
  like this:
                SIndex<Dim> S = (some expression);
                BareField<T,Dim> A = (some value);
                ParticleAttrib<T> PA;
                PA[S] = A[S];
  In this case, S refers to a set of points in the field, and the expression
  loops through all the LOCAL points in the sparse index object S and
  assigns the value of the expression to PA.  PA will be changed in length
  to have the same number of elements, on each processor, as there are local
  sparse index points.  The RHS can be any expression, as long as all Field
  and ParticleAttrib's are indexed by a sparse index.  This is basically
  a random 'gather' from the field to the attribute storage.  The opposite
  operation is like this:
                A[S] = PA[S] + (other expression terms);
  In that case, PA[S] can act like any other SIndex-indexed object in the
  expression.  It just needs to have the proper number of elements.  This
  is then a 'scatter' of values from the attribute back to the field.

  SubParticleAttrib does not define any further [] operators, etc. as does
  SubBareField.  It is used when you apply [] with an SIndex object to a
  ParticleAttrib or ParticleAttribElem.  It has the extra interface to
  supply the necessary iterators to participate in a SIndex PETE expression.
  The user will not typically make any of these objects themselves.

 ***************************************************************************/

// include files
#include "SubField/SubFieldIter.h"
#include "PETE/IpplExpressions.h"

#include <iostream>

// forward declarations
template <class PA, class T, unsigned Dim> class SubParticleAttrib;
template <class PA, class T, unsigned Dim>
std::ostream& operator<<(std::ostream &, const SubParticleAttrib<PA,T,Dim> &);


// an iterator for a SubParticleAttrib.  This must be defined outside the
// class for proper templated function resolution.
template <class PA, class T, unsigned Dim>
class SubParticleAttribIter
  : public PETE_Expr< SubParticleAttribIter<PA,T,Dim> >
{
public:
  // useful typedefs
  typedef SIndex<Dim>                         Index_t;
  typedef SubFieldIter<T, Dim, Index_t>       SFI;
  typedef typename Index_t::const_iterator_iv const_iterator_iv;

  // constructor
  SubParticleAttribIter(PA &a, const_iterator_iv lf, int o, const Index_t &s)
    : dom(&s), attrib(&a), lfield(lf), attribOffset(o) { }

  // copy constructor
  SubParticleAttribIter(const SubParticleAttribIter<PA,T,Dim> &i)
    : dom(i.dom),
      attrib(const_cast<PA *>(i.attrib)),
      lfield(i.lfield),
      attribOffset(i.attribOffset) { }

  // default constructor
  SubParticleAttribIter() : dom(0), attrib(0), attribOffset(0) { }

  // destructor
  ~SubParticleAttribIter() { }

  bool operator!=(const SubParticleAttribIter<PA,T,Dim> &i) {
    return (attribOffset != i.attribOffset);
  }

  PA             &getAttrib()       { return *attrib; }
  const Index_t  &getDomain() const { return *dom; }

  static int  getSubsetType()       { return SFI::getSubsetType(); }
  static bool matchType(int t)      { return SFI::matchType(t); }

  void initialize()                 { }

  int size() const                  { return attrib->size(); }
  int size(int) const               { return (*lfield)->size(); }
  T& offset(int i)                  { return (*attrib)[attribOffset + i]; }
  T& unit_offset(int i)             { return (*attrib)[attribOffset + i]; }

  static void makeNDIndex(const Index_t &s, NDIndex<Dim>& i) {
    i = s.getDomain();
  }

  const_iterator_iv nextLField() {
    attribOffset += size(0);
    ++lfield;
    return lfield;
  }

  const_iterator_iv getLFieldIter() const {
    return lfield;
  }

  bool plugBase(const NDIndex<Dim>&) {
    return true;
  }

  //
  // PETE Interface
  //

  enum                                    { IsExpr = 1 };
  typedef SubParticleAttribIter<PA,T,Dim> PETE_Expr_t;
  typedef T                               PETE_Return_t;
  PETE_Expr_t MakeExpression() const { return *this; }

private:
  // the domain, the attrib, and the iterator to the first vnode to use
  const Index_t     *dom;
  PA                *attrib;
  const_iterator_iv  lfield;

  // our current starting offset for indexing into the attrib array.  This
  // is changed as we switch among LFields.
  int attribOffset;
};


// The SubParticleAttrib class declaration.
template <class PA, class T, unsigned Dim>
class SubParticleAttrib : public PETE_Expr< SubParticleAttrib<PA,T,Dim> > {

public:
  // useful typedefs
  typedef PA                              Attrib_t;
  typedef T                               T_t;
  typedef SubParticleAttribIter<PA,T,Dim> iterator;
  typedef typename iterator::Index_t      Index_t;

  // useful enums
  enum { Dim_u = Dim };

  // constructor
  SubParticleAttrib(PA &a, const Index_t &dom)
    : A(a), MyDomain(dom) { }

  // copy constructor
  SubParticleAttrib(const SubParticleAttrib<PA,T,Dim> &spa)
    : A(const_cast<PA &>(spa.A)), MyDomain(spa.MyDomain) { }

  // destructor
  ~SubParticleAttrib() { }

  //
  // accessor functions
  //

  // return the 'domain', that is, the information which subsets the field
  const Index_t &getDomain() const { return MyDomain; }

  // return a reference to the attribute we are subsetting
  PA &getAttrib() const { return A; }

  // Return a typecode for the subset object
  static int getSubsetType() { return iterator::getSubsetType(); }

  // Return an NDIndex representing the bounding box of this object
  void makeNDIndex(NDIndex<Dim>& i) { iterator::makeNDIndex(MyDomain, i); }

  // Return the beginning and end iterators for this class.
  iterator begin() const;
  iterator end() const;

  // Make sure the LHS ParticleAttrib has the proper length.  It should
  // have the same length as the number of LOCAL sparse index points.
  // If it does not, we readjust the length.  Return the size of the result.
  int adjustSize();

  //
  // assignment operators
  //

  // assignment of another SubParticleAttrib
  SubParticleAttrib<PA,T,Dim>& operator=(const SubParticleAttrib<PA,T,Dim> &);

  // assignment of a scalar
  SubParticleAttrib<PA,T,Dim>& operator=(T);

  // assignment of an arbitrary expression
  //mwerks  template<class B>
  //mwerks  SubParticleAttrib<PA,T,Dim>& operator=(const PETE_Expr<B> &);
  //////////////////////////////////////////////////////////////////////
  // assignment of an arbitrary expression
  template<class B>
  SubParticleAttrib<PA,T,Dim>&
  operator=(const PETE_Expr<B> &b) {

    // invoke the complex expression assignment, which gathers items from
    // the expression and puts them in the attribute
    assign(*this, b);
    return *this;
  }

  //
  // I/O
  //

  void write(std::ostream &) const;

  //
  // PETE interface
  //

  enum { IsExpr = 1 };
  typedef iterator PETE_Expr_t;
  PETE_Expr_t MakeExpression() const { return begin(); }

private:
  // the attribute we are subsetting
  PA &A;

  // the 'domain', that is, the information which subsets the object
  const Index_t &MyDomain;
};

// I/O

// write to the given ostream
template<class PA, class T, unsigned Dim>
inline
std::ostream& operator<<(std::ostream &o, const SubParticleAttrib<PA,T,Dim> &n) {

  n.write(o);
  return o;
}


#include "SubParticle/SubParticleAttrib.hpp"

#endif

/***************************************************************************
 * $RCSfile: SubParticleAttrib.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubParticleAttrib.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
