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
#include "Particle/ParticleLayout.h"
#include "Particle/ParticleBConds.h"



/////////////////////////////////////////////////////////////////////
// constructor
template<class T, unsigned Dim>
ParticleLayout<T, Dim>::ParticleLayout() {
   
  
  setUpdateFlag(ALL, true);
}

/***************************************************************************
 * $RCSfile: ParticleLayout.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:29 $
 * IPPL_VERSION_ID: $Id: ParticleLayout.cpp,v 1.1.1.1 2003/01/23 07:40:29 adelmann Exp $ 
 ***************************************************************************/


