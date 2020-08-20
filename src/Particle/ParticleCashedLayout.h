//
// Class ParticleCashedLayout
//   Please note: for the time being this class is *not* used! But since it
//   might be used in future projects, we keep this file.
//
// Copyright (c) 2003 - 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//

#ifndef PARTICLE_CASHED_LAYOUT_H
#define PARTICLE_CASHED_LAYOUT_H

/*
 * ParticleCashedLayout - particle layout based on spatial decomposition,
 *   with particle interaction (nearest-neighbor with cutoff)
 *
 * This is a specialized version of ParticleLayout, which places particles
 * on processors based on their spatial location relative to a fixed grid.
 * In particular, this can maintain particles on processors based on a
 * specified FieldLayout or RegionLayout, so that particles are always on
 * the same node as the node containing the Field region to which they are
 * local.  This may also be used if there is no associated Field at all,
 * in which case a grid is selected based on an even distribution of
 * particles among processors.
 */

// include files
#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleInteractAttrib.h"
#include "Particle/IpplParticleBase.h"
#include "Region/RegionLayout.h"

#include <vector>
#include <iostream>

// forward declarations
class UserList;
template <unsigned Dim> class FieldLayout;
template <unsigned Dim, class T> class UniformCartesian;
template <class T, unsigned Dim, class Mesh> class ParticleCashedLayout;
template <class T, unsigned Dim, class Mesh>
std::ostream& operator<<(std::ostream&, const ParticleCashedLayout<T,Dim,Mesh>&);


// ParticleCashedLayout class definition.  Template parameters are the type
// and dimension of the ParticlePos object used for the particles.  The
// dimension of the position must match the dimension of the FieldLayout
// object used in this particle layout, if any.
// Optional template parameter for the mesh type
template < class T, unsigned Dim, class Mesh=UniformCartesian<Dim,T> >
class ParticleCashedLayout : public ParticleSpatialLayout<T, Dim, Mesh> {

public:

  typedef typename ParticleLayout<T, Dim>::SingleParticlePos_t
    SingleParticlePos_t;
  typedef typename ParticleLayout<T, Dim>::Index_t Index_t;

  // type of attributes this layout should use for position, ID, and rad
  typedef ParticleInteractAttrib<SingleParticlePos_t> ParticlePos_t;
  typedef ParticleInteractAttrib<Index_t>             ParticleIndex_t;
  typedef ParticleInteractAttrib<T>                   ParticleInterRadius_t;

public:
  // constructor: The Field layout to which we match our particle's
  // locations.
  ParticleCashedLayout(FieldLayout<Dim>&);

  // constructor: this one also takes a Mesh
  ParticleCashedLayout(FieldLayout<Dim>&, Mesh&);

  // a similar constructor, but this one takes a RegionLayout.
  ParticleCashedLayout(const RegionLayout<T,Dim,Mesh>&);

  // a default constructor ... in this case, no layout will
  // be assumed by this class.  A layout may be given later via the
  // 'setLayout' method, either as a FieldLayout or as a RegionLayout.
  ParticleCashedLayout();

  // destructor
  ~ParticleCashedLayout();

  //
  // Particle swapping/update routines
  //

  // Update the location and indices of all atoms in the given IpplParticleBase
  // object.  This handles swapping particles among processors if
  // needed, and handles create and destroy requests.  When complete,
  // all nodes have correct layout information.
  void update(IpplParticleBase< ParticleCashedLayout<T,Dim,Mesh> >& p,
              const ParticleAttrib<char>* canSwap=0);


  // Retrieve a Forward-style iterator for the beginning and end of the
  // Nth (local) particle's nearest-neighbor pairlist.
  // If this is the first call of this
  // method after update(), this must make sure up-to-date info on particles
  // from neighboring nodes is available.
  void getCashedParticles(IpplParticleBase< ParticleCashedLayout<T,Dim,Mesh> >&);

  // specify the interaction radius ... two versions, one which gives a
  // single value for all atoms
  void setInteractionRadius(const T& r) {
    InterRadius = r;
    return;
  }

  // Return the maximum interaction radius of the entire system.  This is
  // the value from the most recent call to update()
  T getMaxInteractionRadius() { return MaxGlobalInterRadius; }

  // Return the interaction radius of atom i.
    T getInteractionRadius(unsigned /*i*/) {
    return InterRadius;
  }

  // directly set NeedGhostSwap flag to indicate whether this is needed
  // useful when a ParticleAttrib other than position R is modified and
  // we do not need to call update().
  void setNeedGhostSwap(bool cond=true) {
    NeedGhostSwap = cond;
  }

  //
  // virtual functions for FieldLayoutUser's (and other UserList users)
  //

  // Repartition onto a new layout
  virtual void Repartition(UserList *);

private:
  // information needed to compute which ghost particles to send/receive
  bool NeedGhostSwap;
  bool* InterNodeList;
  bool* SentToNodeList;
  unsigned InteractionNodes;

  // interaction radius data.  If the attribute pointer is null, use the
  // scalar value instead.
  // also, the maximum interaction radius for the local particles, and for all
  // the particles
  T InterRadius, MaxGlobalInterRadius;

  // perform common constructor tasks
  void setup();

  // recalculate where we need to send ghost particles for building
  // nearest-neighbor interaction lists
  void rebuild_interaction_data();

  // recalculate where we need to send ghost particles for building
  // nearest-neighbor interaction lists
  // special version which accounts for periodic boundary conditions
  void rebuild_interaction_data(const bool periodicBC[2*Dim]);

  // copy particles to other nodes for pairlist computation.  The arguments
  // are the current number of local particles, and the IpplParticleBase object.
  // This will also calculate the pairlists if necessary.
  void swap_ghost_particles(unsigned,
		    IpplParticleBase< ParticleCashedLayout<T,Dim,Mesh> >&);

  // copy particles to other nodes for pairlist computation.  The arguments
  // are the current number of local particles, and the IpplParticleBase object.
  // This will also calculate the pairlists if necessary.
  // special version to take account of periodic boundaries
  void swap_ghost_particles(unsigned,
		    IpplParticleBase< ParticleCashedLayout<T,Dim,Mesh> >&,
                    const bool periodicBC[2*Dim]);

  // change the value of the maximum local interaction radius
  void setMaxInteractionRadius(T maxval) { MaxGlobalInterRadius = maxval; }

  // Return the maximum interaction radius of the local particles.
  T getMaxLocalInteractionRadius();
};

#include "Particle/ParticleCashedLayout.hpp"

#endif

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End: