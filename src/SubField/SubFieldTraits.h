// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SUB_FIELD_TRAITS_H
#define SUB_FIELD_TRAITS_H


///////////////////////////////////////////////////////////////////////////
// a templated traits class describing how to combine different subset
// objects together.  The general case generates errors if you try to use
// subset objects in pairs which are not supported; partial specializations
// of this class indicate which combinations actually work and do the
// right thing in those cases.
//
// This class will deal with two classes, call them A and B.  The class
// tells how to construct B from A (A --> B), or how to combine A into B.
// So you can always think of it as  A --> B.  The order of these classes in
// the template parameter list is always Dim(B), Dim(A), B, A.
//
// There are two functions in this traits class:
// a) construct:
//    static method used to set up the subset object based on an input object.
//    Arguments:
//      Subset object (B) which is being constructed.  Assumed to be empty.
//      Input subset object (A) to be used to construct B.
//      BareField with data we are subsetting, if needed.
//    Returns:
//      Number of brackets which this construction adds.
// b) combine:
//    static methods to combine subset objects.
//    Arguments:
//      Existing subset object (B)
//      The subset object (A) to add to the first argument.
//      The return object, constructed from A and B (of type Return_t)
//      The current number of brackets already used
//      BareField with data we are subsetting, if needed.

// include files
#include "Index/NDIndex.h"
#include "Index/SIndex.h"
#include "Field/BareField.h"
#include "Field/LField.h"


template<class T, unsigned int Dim, class S1, class S2>
struct SubFieldTraits {
  // static contruct method
  static int construct(S1&, const S2&, BareField<T,Dim>&) {
    PInsist(false,"Unsupported indexing attempted.");
    return 0;
  }

  // a typedef for the return type when you combine two subset objects
  typedef S2 Return_t;

  // an enum for how many brackets this combination will add
  enum { Brackets_u = 0 };

  // static combine method
  template<class S3>
  static void combine(const S1&, const S2&, S3&,
		      unsigned int&, BareField<T,Dim>&) {
    PInsist(false,"Unsupported indexing attempted.");
  }
};


///////////////////////////////////////////////////////////////////////////
// NDIndex related specializations

// construct NDIndex<Dim> from NDIndex<Dim2>
// combine   [NDIndex<Dim>][NDIndex<Dim2>] --> NDIndex<Dim>
template<class T, unsigned int Dim, unsigned int Dim2>
struct SubFieldTraits<T, Dim, NDIndex<Dim>, NDIndex<Dim2> > {
  static int construct(NDIndex<Dim>& out, const NDIndex<Dim2>& s,
		       BareField<T,Dim>&) {
    CTAssert(Dim2 <= Dim);
    for (unsigned int d=0; d < Dim2; ++d)
      out[d] = s[d];
    return Dim2;
  }
  typedef NDIndex<Dim> Return_t;
  enum { Brackets_u = Dim2 };
  static void combine(const NDIndex<Dim>& s1, const NDIndex<Dim2>& s2,
		      Return_t& out, unsigned int B, BareField<T,Dim>&) {
    CTAssert(Dim2 <= Dim);
    unsigned int d;
    for (d=0; d < B; ++d)
      out[d]   = s1[d];
    for (d=0; d < Dim2; ++d)
      out[d+B] = s2[d];
  }
};


// construct NDIndex<Dim> from Index (B from A)
// combine   [NDIndex<Dim>][Index] --> NDIndex<Dim>
template<class T, unsigned int Dim>
struct SubFieldTraits<T, Dim, NDIndex<Dim>, Index> {
  static int construct(NDIndex<Dim>& out, const Index& s,
		       BareField<T,Dim>&) {
    out[0] = s;
    return 1;
  }
  typedef NDIndex<Dim> Return_t;
  enum { Brackets_u = 1 };
  static void combine(const NDIndex<Dim>& s1, const Index& s2,
		      Return_t& out, unsigned int B, BareField<T,Dim>&) {
    unsigned int d;
    for (d=0; d < B; ++d)
      out[d] = s1[d];
    out[d] = s2;
  }
};


// construct NDIndex<Dim> from int (B from A)
// combine   [NDIndex<Dim>][int] --> NDIndex<Dim>
template<class T, unsigned int Dim>
struct SubFieldTraits<T, Dim, NDIndex<Dim>, int> {
  static int construct(NDIndex<Dim>& out, const int& s,
		       BareField<T,Dim>&) {
    out[0] = Index(s,s);
    return 1;
  }
  typedef NDIndex<Dim> Return_t;
  enum { Brackets_u = 1 };
  static void combine(const NDIndex<Dim>& s1, const int& s2,
		      Return_t& out, unsigned int B, BareField<T,Dim>&) {
    unsigned int d;
    for (d=0; d < B; ++d)
      out[d] = s1[d];
    out[d] = Index(s2, s2);
  }
};


// construct NDIndex<Dim> from SOffset<Dim2> (B from A)
// combine   [NDIndex<Dim>][SOffset<Dim2>] --> NDIndex<Dim>
template<class T, unsigned int Dim, unsigned int Dim2>
struct SubFieldTraits<T, Dim, NDIndex<Dim>, SOffset<Dim2> > {
  static int construct(NDIndex<Dim>& out, const SOffset<Dim2>& s,
		       BareField<T,Dim>&) {
    CTAssert(Dim2 <= Dim);
    for (unsigned int d=0; d < Dim2; ++d)
      out[d] = Index(s[d], s[d]);
    return Dim;
  }
  typedef NDIndex<Dim> Return_t;
  enum { Brackets_u = Dim2 };
  static void combine(const NDIndex<Dim>& s1, const SOffset<Dim2>& s2,
		      Return_t& out, unsigned int B, BareField<T,Dim>&) {
    CTAssert(Dim2 <= Dim);
    unsigned int d;
    for (d=0; d < B; ++d)
      out[d] = s1[d];
    for (d=0; d < Dim2; ++d)
      out[d+B] = Index(s2[d], s2[d]);
  }
};


///////////////////////////////////////////////////////////////////////////
// SIndex related specializations

// construct SIndex<Dim> from SIndex<Dim> (B from A)
// combine   [SIndex<Dim>][SIndex<Dim>] --> SIndex<Dim> (intersection)
template<class T, unsigned int Dim>
struct SubFieldTraits<T, Dim, SIndex<Dim>, SIndex<Dim> > {
  static int construct(SIndex<Dim>& out, const SIndex<Dim>& s,
		       BareField<T,Dim>&) {
    // assigning SIndex to SIndex also will initialize things if needed
    out = s;
    return Dim;
  }
  typedef SIndex<Dim> Return_t;
  enum { Brackets_u = 0 };
  static void combine(const SIndex<Dim>& s1, const SIndex<Dim>& s2,
		      Return_t& out, unsigned int, BareField<T,Dim>& /*A*/) {
    out  = s1;
    out &= s2;
  }
};


// construct SIndex<Dim> from NDIndex<Dim> (B from A)
// combine   [SIndex<Dim>][NDIndex<Dim>] --> SIndex<Dim> (intersection)
template<class T, unsigned int Dim>
struct SubFieldTraits<T, Dim, SIndex<Dim>, NDIndex<Dim> > {
  static int construct(SIndex<Dim>& out, const NDIndex<Dim>& s,
		       BareField<T,Dim>& A) {
    if (out.needInitialize())
      out.initialize(A.getLayout());
    out = s;
    return Dim;
  }
  typedef SIndex<Dim> Return_t;
  enum { Brackets_u = 0 };
  static void combine(const SIndex<Dim>& s1, const NDIndex<Dim>& s2,
		      Return_t& out, unsigned int, BareField<T,Dim>& /*A*/) {
    out  = s1;
    out &= s2;
  }
};


// construct SIndex<Dim> from SOffset<Dim> (B from A)
// combine   [SIndex<Dim>][SOffset<Dim>] --> SIndex<Dim> (intersection)
template<class T, unsigned int Dim>
struct SubFieldTraits<T, Dim, SIndex<Dim>, SOffset<Dim> > {
  static int construct(SIndex<Dim>& out, const SOffset<Dim>& s,
		       BareField<T,Dim>& A) {
    if (out.needInitialize())
      out.initialize(A.getLayout());
    out = s;
    return Dim;
  }
  typedef SIndex<Dim> Return_t;
  enum { Brackets_u = 0 };
  static void combine(const SIndex<Dim>& s1, const SOffset<Dim>& s2,
		      Return_t& out, unsigned int, BareField<T,Dim>& /*A*/) {
    out  = s1;
    out &= s2;
  }
};


///////////////////////////////////////////////////////////////////////////
// SOffset related specializations

// construct SOffset<Dim> from SOffset<Dim2>
// combine   [SOffset<Dim>][SOffset<Dim2>] --> SOffset<Dim>
template<class T, unsigned int Dim, unsigned int Dim2>
struct SubFieldTraits<T, Dim, SOffset<Dim>, SOffset<Dim2> > {
  static int construct(SOffset<Dim>& out, const SOffset<Dim2>& s,
		       BareField<T,Dim>&) {
    CTAssert(Dim2 <= Dim);
    for (unsigned int d=0; d < Dim2; ++d)
      out[d] = s[d];
    return Dim;
  }
  typedef SOffset<Dim> Return_t;
  enum { Brackets_u = Dim2 };
  static void combine(const SOffset<Dim>& s1, const SOffset<Dim2>& s2,
		      Return_t& out, unsigned int B, BareField<T,Dim>&) {
    CTAssert(Dim2 <= Dim);
    unsigned int d;
    for (d=0; d < B; ++d)
      out[d]   = s1[d];
    for (d=0; d < Dim2; ++d)
      out[d+B] = s2[d];
  }
};


// construct SOffset<Dim> from int
// combine   [SOffset<Dim>][int] --> SOffset<Dim>
template<class T, unsigned int Dim>
struct SubFieldTraits<T, Dim, SOffset<Dim>, int> {
  static int construct(SOffset<Dim>& out, const int& s,
		       BareField<T,Dim>&) {
    out[0] = s;
    return 1;
  }
  typedef SOffset<Dim> Return_t;
  enum { Brackets_u = 1 };
  static void combine(const SOffset<Dim>& s1, const int& s2,
		      Return_t& out, unsigned int B, BareField<T,Dim>&) {
    unsigned int d;
    for (d=0; d < B; ++d)
      out[d] = s1[d];
    out[d] = s2;
  }
};


// combine   [SOffset<Dim>][Index] --> NDIndex<Dim>
template<class T, unsigned int Dim>
struct SubFieldTraits<T, Dim, SOffset<Dim>, Index> {
  typedef NDIndex<Dim> Return_t;
  enum { Brackets_u = 1 };
  static void combine(const SOffset<Dim>& s1, const Index& s2,
		      Return_t& out, unsigned int B, BareField<T,Dim>&) {
    unsigned int d;
    for (d=0; d < B; ++d)
      out[d] = Index(s1[d], s1[d]);
    out[d] = s2;
  }
};


// combine   [SOffset<Dim>][NDIndex<Dim2>] --> NDIndex<Dim>
template<class T, unsigned int Dim, unsigned int Dim2>
struct SubFieldTraits<T, Dim, SOffset<Dim>, NDIndex<Dim2> > {
  typedef NDIndex<Dim> Return_t;
  enum { Brackets_u = Dim2 };
  static void combine(const SOffset<Dim>& s1, const NDIndex<Dim2>& s2,
		      Return_t& out, unsigned int B, BareField<T,Dim>&) {
    CTAssert(Dim2 <= Dim);
    unsigned int d;
    for (d=0; d < B; ++d)
      out[d]   = Index(s1[d], s1[d]);
    for (d=0; d < Dim2; ++d)
      out[d+B] = s2[d];
  }
};


#endif // SUB_FIELD_TRAITS_H

/***************************************************************************
 * $RCSfile: SubFieldTraits.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubFieldTraits.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
