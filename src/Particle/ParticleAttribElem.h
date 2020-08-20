// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_ATTRIB_ELEM_H
#define PARTICLE_ATTRIB_ELEM_H

/*
 * ParticleAttribElem - Templated class for indexed particle attrib elements.
 *
 * This templated class is used to access individual elements of a particle
 * attribute in an expression, such as the Nth component of a Vektor or
 * the (i,j) element of a Tenzor.  It employs the () operator to access the
 * desired quantity, and may participate in particle expressions.
 *
 * Use of this capability requires the following:
 *   1) the data type for the attribute must have the operator() defined,
 *      with the proper number of indices (for example, Vektor has an
 *      operator(unsigned), and Tenzor has an operator(unsigned, unsigned)
 *      to access the (i,j)th element).
 *   2) the data type must have a typedef Return defined within it specifying
 *      the type of the individual elements; for Vektor<T, Dim>, you need a
 *      'typedef T Return' statement in the Vektor code.
 */

// include files
#include "SubParticle/SubParticleAttrib.h"
#include "PETE/IpplExpressions.h"
#include "Utility/Vec.h"
#include "AppTypes/AppTypeTraits.h"
#include <cstddef>

// forward declarations
template <class T, unsigned Dim> class ParticleAttribElemIterator;
template <class T>               class ParticleAttrib;


// a simple templated function to extract the Nth element of an object
// which has several elements accessed using the () operator.  We define
// versions for 1, 2, and 3D
template<class T>
inline typename AppTypeTraits<T>::Element_t&
get_PETE_Element(T& elem, const vec<unsigned,1U>& indx) {
  return elem(indx[0]);
}
template<class T>
inline typename AppTypeTraits<T>::Element_t&
get_PETE_Element(T& elem, const vec<unsigned,2U>& indx) {
  return elem(indx[0], indx[1]);
}
template<class T>
inline typename AppTypeTraits<T>::Element_t&
get_PETE_Element(T& elem, const vec<unsigned,3U>& indx) {
  return elem(indx[0], indx[1], indx[2]);
}



// ParticleAttribElem - templated on type of data and number of indices
template<class T, unsigned Dim>
class ParticleAttribElem : public PETE_Expr< ParticleAttribElem<T,Dim> > {

  friend class ParticleAttribElemIterator<T,Dim>;

public:
  // type of data contained in the attribute elements
  typedef typename AppTypeTraits<T>::Element_t Element_t;
  typedef ParticleAttribElemIterator<T,Dim> iterator;

  //
  // PETE interface.
  //
  enum { IsExpr = 0 };
  typedef iterator PETE_Expr_t;
  PETE_Expr_t MakeExpression() const { return begin(); }

public:
  // constructor: arguments = begin and end iterators (expected to be
  // pointers), and indices
  ParticleAttribElem(ParticleAttrib<T> &pa, const vec<unsigned,Dim>& i)
    : Attrib(pa), indx(i) { }

  // copy constructor
  ParticleAttribElem(const ParticleAttribElem<T,Dim>& pae)
    : Attrib((ParticleAttrib<T> &)(pae.Attrib)), indx(pae.indx) { }

  // return begin and end iterators for this container
  iterator begin() const {
    return iterator((ParticleAttribElem<T,Dim>&) *this, 0);
  }
  iterator end()   const {
    return iterator((ParticleAttribElem<T,Dim>&) *this, size());
  }

  // return the size of this container
  size_t size() const {
    return Attrib.size();
  }

  //
  // methods to make this object have an interface similar to ParticleAttrib
  //

  // get the Nth element
  Element_t &operator[](size_t);

  // Create storage for M particle attributes.  The storage is uninitialized.
  // New items are appended to the end of the array.
  void create(size_t);

  // Delete the attribute storage for M particle attributes, starting at
  // the position I.
  void destroy(size_t M, size_t I, bool optDestroy=true);

  // This version takes a list of particle destroy events
  // Boolean flag indicates whether to use optimized destroy method
  void destroy(const std::vector< std::pair<size_t,size_t> >& dlist,
	       bool optDestroy=true);

  //
  // bracket operator to refer to an attrib and an SIndex object
  //

  template<unsigned SDim>
  SubParticleAttrib<ParticleAttribElem<T,Dim>, Element_t, SDim>
  operator[](const SIndex<SDim> &s) const {
    return SubParticleAttrib<ParticleAttribElem<T,Dim>, Element_t, SDim>(
	(ParticleAttribElem<T,Dim> &)(*this), s);
  }

  //
  // Assignment operators
  //

  // assign a general expression
  template<class T1>
  const ParticleAttribElem<T,Dim>& operator=(const PETE_Expr<T1>& rhs) {
    assign(*this,rhs);
    return *this;
  }

  // assignment of a ParticleAttribElem
  const ParticleAttribElem<T,Dim>&
  operator=(const ParticleAttribElem<T,Dim>& rhs) {
    if (size() > rhs.size()) {
      ERRORMSG("Attempting to copy particle attributes with unequal sizes.");
      ERRORMSG("\n" << size() << " != " << rhs.size() << endl);
    }
    assign(*this,rhs);
    return *this;
  }

  // assignment of a scalar
  const ParticleAttribElem<T,Dim>& operator=(Element_t rhs) {
    assign(*this,PETE_Scalar<Element_t>(rhs),OpAssign());
    return *this;
  }

private:
  // the attribute we're finding the Nth element of
  ParticleAttrib<T> &Attrib;

  // the desired index
  vec<unsigned,Dim> indx;
};


// an iterator for the elements in this particle attribute
template <class T, unsigned Dim>
class ParticleAttribElemIterator
  : public PETE_Expr< ParticleAttribElemIterator<T,Dim> > {

public:
  ParticleAttribElemIterator() : PAE(0), aptr(0) { }
  ParticleAttribElemIterator(ParticleAttribElem<T,Dim>& pae, int p)
    : PAE(&pae), aptr(p) { }
  ParticleAttribElemIterator(const ParticleAttribElemIterator<T,Dim>& i)
    : PAE(i.PAE), aptr(i.aptr) { }

  // PETE interface
  typedef ParticleAttribElemIterator<T,Dim> PETE_Expr_t;
  typedef typename AppTypeTraits<T>::Element_t PETE_Return_t;
  PETE_Expr_t MakeExpression() const { return *this; }
  PETE_Return_t& operator*() { return (*PAE)[aptr]; }

  ParticleAttribElemIterator<T,Dim>& operator++() {
    ++aptr;
    return *this;
  }
  ParticleAttribElemIterator<T,Dim>& rewind() {
    aptr = 0;
    return *this;
  }

  bool operator!=(const ParticleAttribElemIterator<T,Dim>& a) const {
    return (PAE != a.PAE || aptr != a.aptr);
  }
  bool operator==(const ParticleAttribElemIterator<T,Dim>& a) const {
    return (PAE == a.PAE && aptr == a.aptr);
  }

  const ParticleAttribElem<T,Dim>& getParticleAttribElem() const {
    return *PAE;
  }

private:
  ParticleAttribElem<T,Dim> *PAE;
  int aptr;
};

// definitions of routines used to make ParticleAttribElem have an interface
// similar to ParticleAttrib
#include "Particle/ParticleAttrib.h"

template<class T, unsigned Dim>
inline
typename ParticleAttribElem<T,Dim>::Element_t &
ParticleAttribElem<T,Dim>::operator[](size_t n) {
  return get_PETE_Element(Attrib[n], indx);
}

template<class T, unsigned Dim>
inline void
ParticleAttribElem<T,Dim>::create(size_t M) {
  Attrib.create(M);
}

template<class T, unsigned Dim>
inline void
ParticleAttribElem<T,Dim>::destroy(size_t M, size_t I,
				   bool optDestroy) {
  Attrib.destroy(M, I, optDestroy);
}

template<class T, unsigned Dim>
inline void
ParticleAttribElem<T,Dim>::destroy(const std::vector< std::pair<size_t,size_t> > &d,
				   bool optDestroy) {
  Attrib.destroy(d, optDestroy);
}


#endif // PARTICLE_ATTRIB_ELEM_H

/***************************************************************************
 * $RCSfile: ParticleAttribElem.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: ParticleAttribElem.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
