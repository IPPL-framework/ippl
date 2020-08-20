// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef GEN_ARRAY_PARTICLE_H
#define GEN_ARRAY_PARTICLE_H

/*
 * GenArrayParticle - Specialized Particle container for N data elements.
 *
 * GenArrayParticle is a special version of a IpplParticleBase class, specialized
 * to have D-dimensional coordinates of type T1, and N extra data items of
 * type T2.
 */

// include files
#include "Particle/IpplParticleBase.h"


// GenArrayParticle class definition
template<class PLayout, class T, unsigned N>
class GenArrayParticle : public IpplParticleBase<PLayout> {

public:
  // attributes for this class: an array with N attributes
  ParticleAttrib<T> data[N];

  // constructor: user-provided Layout object must be supplied
  GenArrayParticle(PLayout* L) : IpplParticleBase<PLayout>(L) {
    for (unsigned int i = 0; i < N; i++)
      this->addAttribute(data[i]);
  }

private:
  // disable default constructor
  GenArrayParticle();

};

#endif // GEN_ARRAY_PARTICLE_H