// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INDEXED_SINDEX_H
#define INDEXED_SINDEX_H

// include files
#include "Index/SIndex.h"
#include "Index/NDIndex.h"
#include "Index/SIndexAssign.h"
#include "Utility/PAssert.h"


/***********************************************************************
 * 
 * IndexedSIndex represents an SIndex object + an NDIndex object which
 * selects just a subset of the original field domain of the SIndex
 * object.  When the user uses the [] operator on an SIndex object, it
 * returns an IndexedSIndex object which knows it should refer to points
 * only in the index space of the Index objects provided in the [] operator.
 * This is used to modify the original SIndex to let it know that it refers
 * only to a subset, and so when it is used in an SIndex assignment or
 * expression, only that subset should be looped over.
 *
 * The first template parameter is the dimension of the SIndex object; the
 * second is the number of dimensions worth of bracket operators which have
 * been applied so far.  This is the sum of the dimensions which have been
 * specified by Index or NDIndex objects inside the bracket operators of
 * SIndex or IndexedSIndex.  By the time the SIndex object is needed, this
 * should have Dim == Brackets.  If not, it is an error.
 *
 ***********************************************************************/


template <unsigned Dim, unsigned Brackets>
class IndexedSIndex {

public:
  // Initialize this object with the SIndex it refers to, and the NDIndex
  // it should use.
  IndexedSIndex(SIndex<Dim> &s, const NDIndex<Dim> &i)
    : sIndex(s), domain(i) { }

  IndexedSIndex(const IndexedSIndex<Dim,Brackets> &isi)
    : sIndex(const_cast<SIndex<Dim> &>(isi.sIndex)), domain(isi.domain) { }

  // destructor: nothing to do
  ~IndexedSIndex() { }

  // get the SIndex we are using
  const SIndex<Dim> &getSIndex() const { return sIndex; }

  // get the domain we're Indexing
  const NDIndex<Dim> &getDomain() const { return domain; }

  // assignment operators.  First make sure we have fully specified
  // what domain to use by checking that Dim == Brackets, then call
  // assign but with the smaller domain.
  template<class T>
  IndexedSIndex<Dim,Brackets>& operator=(const PETE_Expr<T>& rhs) {
    CTAssert(Brackets == Dim);
    assign(sIndex, rhs, domain);
    return *this;
  }

  // operator[], which is used with Index or NDIndex objects to further
  // subset the data.  This will only work if the dimension of the Index
  // arguments + Brackets is <= Dim.  Otherwise, too many dimensions worth
  // of Index objects are being applied
  IndexedSIndex<Dim,Brackets + 1>
  operator[](const Index &i) {
    CTAssert((Brackets + 1) <= Dim);
    NDIndex<Dim> dom = domain;
    dom[Brackets] = i;
    return IndexedSIndex<Dim,Brackets + 1>(sIndex, dom);
  }

  template<unsigned Dim2>
  IndexedSIndex<Dim,Brackets + Dim2>
  operator[](const NDIndex<Dim2> &ndi) {
    CTAssert((Brackets + Dim2) <= Dim);
    NDIndex<Dim> dom = domain;
    for (unsigned int i=0; i < Dim2; ++i)
      dom[Brackets + i] = ndi[i];
    return IndexedSIndex<Dim,Brackets + Dim2>(sIndex, dom);
  }

private:
  // the SIndex we refer to
  SIndex<Dim> &sIndex;

  // the NDIndex we are subsetting to
  NDIndex<Dim> domain;

  // copy constructor and operator=, made private since we do not want them
  IndexedSIndex<Dim,Brackets>& operator=(const IndexedSIndex<Dim,Brackets> &);
};

#endif // INDEXED_SINDEX_H

/***************************************************************************
 * $RCSfile: IndexedSIndex.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: IndexedSIndex.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
