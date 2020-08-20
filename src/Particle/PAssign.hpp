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
#include "Particle/PAssign.h"
#include "Particle/PAssignDefs.h"
#include "Particle/ParticleAttrib.h"
#include "Particle/ParticleAttribElem.h"
#include "PETE/IpplExpressions.h"
#include "Utility/IpplStats.h"



//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim, class RHS, class OP>
void
assign(const ParticleAttribElem<T,Dim>& ca, RHS b, OP op)
{
  // Cast away Const-ness.  Aarrgghh!!
  ParticleAttribElem<T,Dim>& a = (ParticleAttribElem<T,Dim>&) ca;
  typename RHS::Wrapped& bb = b.PETE_unwrap();

  // Begin/End iterators for the ParticleAttribElem on the left hand side.
  typename ParticleAttribElem<T,Dim>::iterator pa = a.begin(), aend = a.end();
  for ( ; pa != aend ; ++pa) {
    PETE_apply( op, *pa, for_each(bb,EvalFunctor_0()) );
    for_each(bb,PETE_Increment(),PETE_NullCombiner());
  }

  INCIPPLSTAT(incParticleExpressions);
}

template<class T, class RHS, class OP>
void
assign(const ParticleAttrib<T>& ca, RHS b, OP op)
{
  // Cast away Const-ness.  Aarrgghh!!
  ParticleAttrib<T>& a = (ParticleAttrib<T>&) ca;
  typename RHS::Wrapped& bb = b.PETE_unwrap();

  // Begin and end iterators for the ParticleAttrib on the left hand side.
  typename ParticleAttrib<T>::iterator pa = a.begin(), aend = a.end();
  for ( ; pa != aend ; ++pa) {
    PETE_apply( op, *pa, for_each(bb,EvalFunctor_0()) );
    for_each(bb,PETE_Increment(),PETE_NullCombiner());
  }

  INCIPPLSTAT(incParticleExpressions);
}

//////////////////////////////////////////////////////////////////////

/***************************************************************************
 * $RCSfile: PAssign.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: PAssign.cpp,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
