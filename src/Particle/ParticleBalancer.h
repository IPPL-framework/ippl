// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_BALANCER_H
#define PARTICLE_BALANCER_H

/***************************************************************************
 *
 * ParticleBalancer - functions for performing load balancing for
 *    particles using a particle density field.
 *
 * Load balancing for particles is performed by constructing a BareField
 * object over the domain occupied by the particles, and filling this field
 * with the density of the particles.  The Field is then repartitioned by
 * using one of the Field repartitioning algorithms, and the resulting
 * Field layout is given to the Particle object, which then redistributes
 * the particles.
 *
 ***************************************************************************/

// forward declarations
template<class T, unsigned Dim, class Mesh, class CachingPolicy> class ParticleSpatialLayout;
template<class T, unsigned Dim> class ParticleUniformLayout;
template<class PLayout> class IpplParticleBase;


// calculate a new RegionLayout for a given IpplParticleBase, and distribute the
// new RegionLayout to all the nodes.  This uses a Field BinaryBalancer.
template<class T, unsigned Dim, class Mesh, class CachingPolicy>
bool
BinaryRepartition(IpplParticleBase<ParticleSpatialLayout<T,Dim,Mesh,CachingPolicy> >&, double = 0.0);

// the same, but taking a uniform layout (this will not actually do anything)
template<class T, unsigned Dim>
bool
BinaryRepartition(IpplParticleBase<ParticleUniformLayout<T,Dim> >&, double = 0.0);

#include "Particle/ParticleBalancer.hpp"

#endif // PARTICLE_BALANCER_H

/***************************************************************************
 * $RCSfile: ParticleBalancer.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: ParticleBalancer.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/


