// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_LAYOUT_H
#define PARTICLE_LAYOUT_H

/*
 * ParticleLayout - base class for all particle layout classes.
 *
 * This class is used as the generic base class for all classes
 * which maintain the information on where all the particles are located
 * on a parallel machine.  It is responsible for performing particle
 * swapping and load balancing, and for determining pair lists.
 *
 * If more general layout information is needed, such as the global -> local
 * mapping for each particle, then derived classes must provide this info.
 *
 * When particles are created or destroyed, this class is also responsible
 * for determining where particles are to be created, gathering this
 * information, and recalculating the global indices of all the particles.
 * For consistency, creation and destruction requests are cached, and then
 * performed all in one step when the update routine is called.
 *
 * Derived classes must provide the following:
 *   1) Specific version of update and loadBalance.  These are not virtual,
 *      as this class is used as a template parameter (instead of being
 *      assigned to a base class pointer).
 *   2) Internal storage to maintain their specific layout mechanism
 *   3) the definition of a class pair_iterator, and a function
 *      void getPairlist(int, pair_iterator&, pair_iterator&) to get a
 *      begin/end iterator pair to access the local neighbor of the Nth
 *      local atom.  This is not a virtual function, it is a requirement of
 *      the templated class for use in other parts of the code.
 */

// include files
#include "Particle/ParticleBConds.h"


// forward declarations
template<class T, unsigned Dim> class Vektor;


// ParticleLayout class definition.  Template parameters are the type
// and dimension of the ParticlePos object used for the particles.
template<class T, unsigned Dim>
class ParticleLayout {

public:
  // useful enumerations
  enum { Dimension = Dim };	                      // dim of coord data
  enum UpdateFlags { SWAP, BCONDS, NUMFLAGS, OPTDESTROY, ALL }; // update opts

  // useful typedefs common to all layouts
  typedef T                                   Position_t;
  typedef unsigned                            Index_t;
  typedef Vektor<T,Dim>                       SingleParticlePos_t;

  // in addition, all subclasses must provide a typedef or class definition
  // for a type 'pair_iterator', which iterates over lists of 'pair_t' objs,
  // and typedefs for the ParticlePos_t and ParticleIndex_t types.

public:
  // constructor and destructor: no arguments
  ParticleLayout();
  ~ParticleLayout() { }

  // set the flags used to indicate what to do during the update
  void setUpdateFlag(UpdateFlags f, bool val) {
    if (f == ALL) {
      UpdateOptions = (~0);
    } else {
      unsigned int mask = (1 << f);
      UpdateOptions = (val ? (UpdateOptions|mask) : (UpdateOptions&(~mask)));
    }
  }

  // get the flags used to indicate what to do during the update
  bool getUpdateFlag(UpdateFlags f) const {
    return (f==ALL ? (UpdateOptions==(~0u)) : ((UpdateOptions & (1u << f))!=0));
  }

  // get the boundary conditions container
  ParticleBConds<T,Dim>& getBConds() { return BoundConds; }

  // copy over the given boundary conditions
  void setBConds(const ParticleBConds<T,Dim>& bc) { BoundConds = bc; }

protected:
  // apply the given boundary conditions to the given set of n particle
  // positions.  This modifies the position values based on the BC type.
  //mwerks  template<class PPT, class NDI>
  //mwerks  void apply_bconds(unsigned, PPT&, const ParticleBConds<T,Dim>&, const NDI&);
  /////////////////////////////////////////////////////////////////////
  // Apply the given boundary conditions to the current particle positions.
  // PPT is the type of particle position attribute container, and NDI is
  // the type of index object (NDIndex or NDRegion)
  template<class PPT, class NDI>
  void apply_bconds(unsigned n, PPT& R,
		    const ParticleBConds<T,Dim>& bcs,
		    const NDI& nr) {

    // apply boundary conditions to the positions
    for (unsigned int i=0; i < n; ++i)
      for (unsigned int j=0; j < Dim; j++)
	R[i][j] = bcs.apply(R[i][j], j, nr);
  }

private:
  // the list of boundary conditions for this set of particles
  ParticleBConds<T,Dim> BoundConds;

  // flags indicating what should be updated
  unsigned int UpdateOptions;
};

#include "Particle/ParticleLayout.hpp"

#endif // PARTICLE_LAYOUT_H

/***************************************************************************
 * $RCSfile: ParticleLayout.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:29 $
 * IPPL_VERSION_ID: $Id: ParticleLayout.h,v 1.1.1.1 2003/01/23 07:40:29 adelmann Exp $ 
 ***************************************************************************/
