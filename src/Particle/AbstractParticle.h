//
// Class AbstractParticle
//   Abstract base class for IpplParticleBase and AmrParticleBase
//
// Copyright (c) 2017, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef ABSTRACT_PARTICLE_H
#define ABSTRACT_PARTICLE_H

#include "Particle/ParticleLayout.h"
#include "Particle/ParticleAttrib.h"

template <class T, unsigned Dim>
class AbstractParticle {

public:
    typedef typename ParticleLayout<T, Dim>::SingleParticlePos_t
    SingleParticlePos_t;
    typedef typename ParticleLayout<T, Dim>::Index_t Index_t;
    typedef ParticleAttrib<SingleParticlePos_t> ParticlePos_t;
    typedef ParticleAttrib<Index_t>             ParticleIndex_t;
    typedef typename ParticleLayout<T, Dim>::UpdateFlags UpdateFlags;
    typedef typename ParticleLayout<T, Dim>::Position_t Position_t;
    typedef ParticleLayout<T, Dim> Layout_t;

public:

    AbstractParticle() : R_p(0), ID_p(0) {}

    virtual ~AbstractParticle() { }
//     AbstractParticle(ParticlePos_t& R,
//                      ParticleIndex_t& ID) : R_p(&R), ID_p(&ID)
//                      {
//                          std::cout << "AbstractParticle()" << std::endl;
//                     }

    virtual void addAttribute(ParticleAttribBase& pa) = 0;

    virtual size_t getTotalNum() const = 0;
    virtual size_t getLocalNum() const = 0;
    virtual size_t getDestroyNum() const = 0;
    virtual size_t getGhostNum() const = 0;
    virtual void setTotalNum(size_t n) = 0;
    virtual void setLocalNum(size_t n) = 0;

    virtual unsigned int getMinimumNumberOfParticlesPerCore() const = 0;
    virtual void setMinimumNumberOfParticlesPerCore(unsigned int n) = 0;

    virtual Layout_t& getLayout() = 0;
    virtual const Layout_t& getLayout() const = 0;

    virtual bool getUpdateFlag(UpdateFlags f) const = 0;

    virtual void setUpdateFlag(UpdateFlags f, bool val) = 0;

    virtual ParticleBConds<Position_t, Dim>& getBConds() = 0;

    virtual void setBConds(const ParticleBConds<Position_t, Dim>& bc) = 0;

    virtual bool singleInitNode() const = 0;

    virtual void resetID() = 0;


    virtual void update() = 0;
    virtual void update(const ParticleAttrib<char>& canSwap) = 0;

    virtual void createWithID(unsigned id) = 0;
    virtual void create(size_t) = 0;
    virtual void globalCreate(size_t np) = 0;

    virtual void destroy(size_t, size_t, bool = false) = 0;

    virtual void performDestroy(bool updateLocalNum = false) = 0;

    virtual void ghostDestroy(size_t M, size_t I) = 0;

public:
    ParticlePos_t* R_p;
    ParticleIndex_t* ID_p;
};

#endif