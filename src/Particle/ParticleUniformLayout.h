// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_UNIFORM_LAYOUT_H
#define PARTICLE_UNIFORM_LAYOUT_H

/*
 * ParticleUniformLayout - particle layout based on uniform distribution.
 *
 * This is a specialized version of ParticleLayout, which places particles
 * on processors simply based on their global index.  The total number
 * of atoms on each processor is kept as uniform as possible, with no
 * regard as to the relative location of each atom.
 */

// include files
#include "Particle/ParticleLayout.h"
#include "Particle/IpplParticleBase.h"

#include <iostream>

// forward declarations
class Inform;
template<class T, unsigned Dim> class ParticleUniformLayout;
template<class T, unsigned Dim>
std::ostream& operator<<(std::ostream&, const ParticleUniformLayout<T,Dim>&);


// ParticleUniformLayout class definition.  Template parameters are the type
// and dimension of the ParticlePos object used for the particles.
template<class T, unsigned Dim>
class ParticleUniformLayout : public ParticleLayout<T, Dim> {

public:
  // pair iterator definition ... this layout does not allow for pairlists
  typedef int pair_t;
  typedef pair_t* pair_iterator;

  typedef typename ParticleLayout<T, Dim>::SingleParticlePos_t
    SingleParticlePos_t;
  typedef typename ParticleLayout<T, Dim>::Index_t Index_t;

  // type of attributes this layout should use for position and ID
  typedef ParticleAttrib<SingleParticlePos_t> ParticlePos_t;
  typedef ParticleAttrib<Index_t>             ParticleIndex_t;

public:
  // constructor and destructor
  ParticleUniformLayout();
  ~ParticleUniformLayout();

  //
  // Particle swapping/update routines
  //

  // Update the location and indices of all atoms in the given IpplParticleBase
  // object.  This handles swapping particles among processors if
  // needed, and handles create and destroy requests.  When complete,
  // all nodes have correct layout information.
  void update(IpplParticleBase< ParticleUniformLayout<T,Dim> >& p,
              const ParticleAttrib<char>* canSwap = 0);

  //
  // I/O
  //

  // Print out information for debugging purposes.
  void printDebug(Inform&);

private:
  // Particle redistribution data for each node; used in update
  int *LocalSize;
  int *Change;
  int *MsgCount;
};

#include "Particle/ParticleUniformLayout.hpp"

#endif // PARTICLE_UNIFORM_LAYOUT_H

/***************************************************************************
 * $RCSfile: ParticleUniformLayout.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:29 $
 * IPPL_VERSION_ID: $Id: ParticleUniformLayout.h,v 1.1.1.1 2003/01/23 07:40:29 adelmann Exp $ 
 ***************************************************************************/
