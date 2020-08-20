// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 * This program was prepared by PSI. 
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "SubField/SubBareField.h"
#include "SubField/SubFieldTraits.h"
#include "SubField/SubFieldAssign.h"
#include "Field/BareField.h"
#include "Field/LField.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"



//////////////////////////////////////////////////////////////////////
// check to see if it is ok to add the given number of brackets to our
// current number
template<class T, unsigned int Dim, class S>
bool SubBareField<T,Dim,S>::checkAddBrackets(unsigned int B) {
  
  

  if ((Brackets + B) > Dim) {
    ERRORMSG("Too many dimensions indexed in SubBareField: ");
    ERRORMSG((Brackets + B) << " > " << Dim << endl);
    return false;
  }
  return true;
}



//////////////////////////////////////////////////////////////////////
// assignment of another SubBareField
template<class T, unsigned int Dim, class S>
SubBareField<T,Dim,S>&
SubBareField<T,Dim,S>::operator=(const SubBareField<T,Dim,S> &t) {
  
  
  assign(*this, t);
  return *this;
}


//////////////////////////////////////////////////////////////////////
// assignment of a scalar
template<class T, unsigned int Dim, class S>
SubBareField<T,Dim,S>&
SubBareField<T,Dim,S>::operator=(T t) {
  
  
  assign(*this, t);
  return *this;
}


//////////////////////////////////////////////////////////////////////
// get a single value and return it in the given storage.  Whichever
// node owns the value must broadcast it to the other nodes.
template<class T, unsigned int Dim, class S>
void SubBareField<T,Dim,S>::get(T& r) {
  
  
  // make sure we have a properly bracketed object
  PAssert_EQ(checkBrackets(), true);

  // construct an NDIndex which refers to the single point.
  NDIndex<Dim> Indexes;
  iterator::makeNDIndex(MyDomain, Indexes);

  // get the value from the BareField
  A.get(Indexes, r);
}


//////////////////////////////////////////////////////////////////////
// Return the beginning and end iterators for this class.
template<class T, unsigned int Dim, class S>
typename SubBareField<T,Dim,S>::iterator SubBareField<T,Dim,S>::begin() const {
  
  

  return iterator(A, A.begin_if(), MyDomain, Brackets);
}

template<class T, unsigned int Dim, class S>
typename SubBareField<T,Dim,S>::iterator SubBareField<T,Dim,S>::end() const {
  
  

  return iterator(A, A.end_if(), MyDomain, Brackets);
}


//////////////////////////////////////////////////////////////////////
// write this subfield to the given ostream
template<class T, unsigned int Dim, class S>
void SubBareField<T,Dim,S>::write(std::ostream& o) {
  
  
  // make sure we have the right number of brackets
  PAssert_EQ(Dim, Brackets);

  // determine the range of values which we need to print
  NDIndex<Dim> testIndex;
  iterator::makeNDIndex(MyDomain, testIndex);

  // finally, print out the subfield
  o << A[testIndex];
}


//////////////////////////////////////////////////////////////////////

/***************************************************************************
 * $RCSfile: SubBareField.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubBareField.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
