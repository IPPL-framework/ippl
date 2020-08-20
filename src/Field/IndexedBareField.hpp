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
#include "Field/IndexedBareField.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"

#include "FieldLayout/FieldLayout.h"

#include <vector>
#include <iostream>

//----------------------------------------------------------------------

//MWERKS: moved this member template into class definition, in 
//MWERKS: IndexedBareField.h
// template<class T, unsigned D1>
// template<unsigned int D2>
// bool
// IndexedBareFieldIterator<T,D1>::plugBase(const NDIndex<D2>& i)

//////////////////////////////////////////////////////////////////////


template< class T, unsigned Dim, unsigned Brackets >
void 
IndexedBareField<T,Dim,Brackets>::write(std::ostream& out)
{
  
  
  // make sure we have the right number of brackets
  PInsist(Dim == Brackets,
          "Field not fully indexed in IndexedBareField::write!!");
  NDIndex<Dim> testIndex;
  for (unsigned d=0; d<Dim; d++)
    testIndex[d] = Indexes[d];

  // make a BareField which will store the subset
  FieldLayout<Dim> subfl(testIndex);
  BareField<T,Dim> subset(subfl);

  // assign values to this subfield
  //  my_indexed_assign(subset, A, testIndex);
  assign(subset[testIndex], A[testIndex]);

  // finally, print out the subfield
  out << subset;
}


//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim, unsigned Brackets>
void
IndexedBareField<T,Dim,Brackets>::getsingle(T& r)
{
  
  
  A.getsingle(Indexes, r);
}

//////////////////////////////////////////////////////////////////////
// Return a single value.
template<class T, unsigned Dim, unsigned Brackets>
T
IndexedBareField<T,Dim,Brackets>::get()
{
  T r;
  
  
  getsingle(r);
  return r;
}

/***************************************************************************
 * $RCSfile: IndexedBareField.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: IndexedBareField.cpp,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
