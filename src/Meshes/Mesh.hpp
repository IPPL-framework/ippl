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
#include "Meshes/Mesh.h"

// static member data
template<unsigned int Dim>
std::string Mesh<Dim>::MeshBC_E_Names[3] = {"Reflective","Periodic  ","No BC     "};


//////////////////////////////////////////////////////////////////////////
// default constructor for Mesh
template<unsigned int Dim>
Mesh<Dim>::Mesh() { }


//////////////////////////////////////////////////////////////////////////
// destructor for Mesh
template<unsigned int Dim>
Mesh<Dim>::~Mesh() { }


//////////////////////////////////////////////////////////////////////////
// notify all the registered FieldLayoutUser's that this Mesh has
// changed.  This is done by called the 'Repartition' virtual function
// in FieldLayoutUser
template<unsigned int Dim>
void Mesh<Dim>::notifyOfChange() {
  // Repartition each registered user.
  for (iterator_if p=begin_if(); p!=end_if(); ++p) {
    FieldLayoutUser *user = (FieldLayoutUser *)((*p).second);
    user->Repartition(this);
  }
}


/***************************************************************************
 * $RCSfile: Mesh.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: Mesh.cpp,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
