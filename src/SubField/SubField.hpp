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
#include "SubField/SubField.h"
#include "SubField/SubFieldTraits.h"
#include "Field/Field.h"



//////////////////////////////////////////////////////////////////////
// assignment of a scalar
template<class T, unsigned int Dim, class M, class C, class S>
void SubField<T,Dim,M,C,S>::operator=(T t) {

  assign(*this, t);
}


//////////////////////////////////////////////////////////////////////
// assignment of another subfield
template<class T, unsigned int Dim, class M, class C, class S>
SubField<T,Dim,M,C,S>&
SubField<T,Dim,M,C,S>::operator=(const SubField<T,Dim,M,C,S> &s) {

  assign(*this, s);
  return *this;
}

/***************************************************************************
 * $RCSfile: SubField.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubField.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
