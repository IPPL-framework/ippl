// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef GEN_PARTICLE_H
#define GEN_PARTICLE_H

/*
 * GenParticle - Specialized Particle container with just one more attribute.
 *
 * GenParticle is a special version of a IpplParticleBase class, specialized
 * to have one more attribute of type T2 in addition to the normal attributes
 * defined in IpplParticleBase.
 */

// include files
#include "Particle/IpplParticleBase.h"


// GenParticle class definition
template<class PLayout, class T>
class GenParticle : public IpplParticleBase<PLayout> {

public:
  // the extra attribute for this class
  ParticleAttrib<T> data;

  // constructor: user-provided Layout object must be supplied
  GenParticle(PLayout* L) : IpplParticleBase<PLayout>(L) {
    addAttribute(data);
  }

private:
  // disable default constructor
  GenParticle();

};

#endif // GEN_PARTICLE_H

/***************************************************************************
 * $RCSfile: GenParticle.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: GenParticle.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
