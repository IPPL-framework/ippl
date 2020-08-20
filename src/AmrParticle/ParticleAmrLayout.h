//
// Class ParticleAmrLayout
//   Particle layout for AMR particles.
//
// Copyright (c) 2016 - 2020, Matthias Frey, Uldis Locans, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// Implemented as part of the PhD thesis
// "Precise Simulations of Multibunches in High Intensity Cyclotrons"
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
#ifndef PARTICLE_AMR_LAYOUT_H
#define PARTICLE_AMR_LAYOUT_H

#include "Particle/ParticleLayout.h"

template<class T, unsigned Dim>
class ParticleAmrLayout : public ParticleLayout<T, Dim> {
    
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
    
    ParticleAmrLayout();
    
    /*!
     * @param finestLevel of current simulation state
     */
    void setFinestLevel(int finestLevel);
    
    /*!
     * @param maxLevel allowed during simulation run
     */
    void setMaxLevel(int maxLevel);
    
    /*!
     * Set the computational domain of the base level. E.g. the computational
     * domain is [-1, 1]^3. With dh = 4, we get a new domain of [-1.04, 1.04]^3.
     * @param dh is the mesh enlargement in [%]
     */
    virtual void setBoundingBox(double dh) = 0;
    
protected:
    int finestLevel_m;                  ///< Current finest level of simluation
    int maxLevel_m;                     ///< Maximum level allowed
};


// ============================================================================


template <class T, unsigned Dim>
ParticleAmrLayout<T, Dim>::ParticleAmrLayout()
    : finestLevel_m(0),
      maxLevel_m(0)
{ }


template <class T, unsigned Dim>
void ParticleAmrLayout<T, Dim>::setFinestLevel(int finestLevel) {
    finestLevel_m = finestLevel;
}


template <class T, unsigned Dim>
void ParticleAmrLayout<T, Dim>::setMaxLevel(int maxLevel) {
    maxLevel_m = maxLevel;
}

#endif
