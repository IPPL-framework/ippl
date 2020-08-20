//
// Class AmrParticleBase
//   Ippl interface for AMR particles.
//   The derived classes need to extend the base class by subsequent methods.
//
//   template <class FT, unsigned Dim, class PT>
//   void scatter(const ParticleAttrib<FT>& attrib, AmrField_t& f,
//                const ParticleAttrib<Vektor<PT, Dim> >& pp,
//                int lbase = 0, int lfine = -1) const;
//
//
//   gather the data from the given Field into the given attribute, using
//   the given Position attribute
//
//   template <class FT, unsigned Dim, class PT>
//   void gather(ParticleAttrib<FT>& attrib, const AmrField_t& f,
//               const ParticleAttrib<Vektor<PT, Dim> >& pp,
//               int lbase = 0, int lfine = -1) const;
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
#ifndef AMR_PARTICLE_BASE_HPP
#define AMR_PARTICLE_BASE_HPP

#include <numeric>
#include <algorithm>

template<class PLayout>
AmrParticleBase<PLayout>::AmrParticleBase() : forbidTransform_m(false),
                                              scale_m(1.0),
                                              lorentzFactor_m(1.0, 1.0, 1.0),
//                                               isLorentzTransformed_m(false),
                                              LocalNumPerLevel_m()
{
    updateParticlesTimer_m = IpplTimings::getTimer("AMR update particles");
    sortParticlesTimer_m = IpplTimings::getTimer("AMR sort particles");
    domainMappingTimer_m = IpplTimings::getTimer("AMR map particles");
}


template<class PLayout>
AmrParticleBase<PLayout>::AmrParticleBase(PLayout* layout)
    : IpplParticleBase<PLayout>(layout),
      forbidTransform_m(false),
      scale_m(1.0),
      lorentzFactor_m(1.0, 1.0, 1.0),
//       isLorentzTransformed_m(false),
      LocalNumPerLevel_m()
{
    updateParticlesTimer_m = IpplTimings::getTimer("AMR update particles");
    sortParticlesTimer_m = IpplTimings::getTimer("AMR sort particles");
    domainMappingTimer_m = IpplTimings::getTimer("AMR map particles");
}


template<class PLayout>
const typename AmrParticleBase<PLayout>::ParticleLevelCounter_t&
    AmrParticleBase<PLayout>::getLocalNumPerLevel() const
{
    return LocalNumPerLevel_m;
}


template<class PLayout>
typename AmrParticleBase<PLayout>::ParticleLevelCounter_t&
    AmrParticleBase<PLayout>::getLocalNumPerLevel()
{
    return LocalNumPerLevel_m;
}


template<class PLayout>
void AmrParticleBase<PLayout>::setLocalNumPerLevel(
    const ParticleLevelCounter_t& LocalNumPerLevel)
{
    LocalNumPerLevel_m = LocalNumPerLevel;
}


template<class PLayout>
void AmrParticleBase<PLayout>::destroy(size_t M, size_t I, bool doNow) {
    /* if the particles are deleted directly
     * we need to update the particle level count
     */
    if (M > 0) {
        if ( doNow ) {
            for (size_t ip = I; ip < M + I; ++ip)
                --LocalNumPerLevel_m[ Level[ip] ];
        }
        IpplParticleBase<PLayout>::destroy(M, I, doNow);
    }
}


template<class PLayout>
void AmrParticleBase<PLayout>::performDestroy(bool updateLocalNum) {
    // nothing to do if destroy list is empty
    if ( this->DestroyList.empty() )
        return;
    
    if ( updateLocalNum ) {
        typedef std::vector< std::pair<size_t,size_t> > dlist_t;
        dlist_t::const_iterator curr = this->DestroyList.begin();
        const dlist_t::const_iterator last = this->DestroyList.end();
        
        while ( curr != last ) {
            for (size_t ip = curr->first;
                 ip < curr->first + curr->second;
                 ++ip)
            {
                --LocalNumPerLevel_m[ Level[ip] ];
            }
            ++curr;
        }
    }
    IpplParticleBase<PLayout>::performDestroy(updateLocalNum);
}


template<class PLayout>
void AmrParticleBase<PLayout>::create(size_t M) {
    
//     size_t localnum = LocalNumPerLevel_m[0];
    
    // particles are created at the coarsest level
    LocalNumPerLevel_m[0] += M;
    
    IpplParticleBase<PLayout>::create(M);
    
//     for (size_t i = localnum; i < LocalNumPerLevel_m[0]; ++i) {
//         this->Grid[i] = 0;
//         this->Level[i] = 0;
//     }
}


template<class PLayout>
void AmrParticleBase<PLayout>::createWithID(unsigned id) {
    
//     size_t localnum = LocalNumPerLevel_m[0];
    
    ++LocalNumPerLevel_m[0];
    
    IpplParticleBase<PLayout>::createWithID(id);
    
//     this->Grid[localnum] = 0;
//     this->Level[localnum] = 0;
}


template<class PLayout>
void AmrParticleBase<PLayout>::update() {
    // update all level
    this->update(0, -1);
}


template<class PLayout>
void AmrParticleBase<PLayout>::update(int lev_min, int lev_max, bool isRegrid) {
    
    IpplTimings::startTimer(updateParticlesTimer_m);

    // make sure we've been initialized
    PLayout *Layout = &this->getLayout();

    PAssert(Layout != 0);
    
    // ask the layout manager to update our atoms, etc.
    Layout->update(*this, lev_min, lev_max, isRegrid);
    
    //sort the particles by grid and level
    sort();
    
    INCIPPLSTAT(incParticleUpdates);
    
    IpplTimings::stopTimer(updateParticlesTimer_m);
}

template<class PLayout>
void AmrParticleBase<PLayout>::update(const ParticleAttrib<char>& canSwap) {
    
    IpplTimings::startTimer(updateParticlesTimer_m);

    // make sure we've been initialized
    PLayout *Layout = &this->getLayout();
    PAssert(Layout != 0);
    
    // ask the layout manager to update our atoms, etc.
    Layout->update(*this, &canSwap);
    
    //sort the particles by grid and level
    sort();
    
    INCIPPLSTAT(incParticleUpdates);
    
    IpplTimings::stopTimer(updateParticlesTimer_m);
}


template<class PLayout>
void AmrParticleBase<PLayout>::sort() {
    
    IpplTimings::startTimer(sortParticlesTimer_m);
    size_t LocalNum = this->getLocalNum();
    SortList_t slist1(LocalNum); //slist1 holds the index of where each element should go
    SortList_t slist2(LocalNum); //slist2 holds the index of which element should go in this position

    //sort the lists by grid and level
    //slist1 hold the index of where each element should go in the list
    std::iota(slist1.begin(), slist1.end(), 0);
    std::sort(slist1.begin(), slist1.end(), [this](const SortListIndex_t &i, 
                                                   const SortListIndex_t &j)
    {
        return (this->Level[i] < this->Level[j] ||
               (this->Level[i] == this->Level[j] && this->Grid[i] < this->Grid[j]));
    });

    //slist2 holds the index of which element should go in this position
    for (unsigned int i = 0; i < LocalNum; ++i)
        slist2[slist1[i]] = i;

    //sort the array according to slist2
    this->sort(slist2);

    IpplTimings::stopTimer(sortParticlesTimer_m);
}


template<class PLayout>
void AmrParticleBase<PLayout>::sort(SortList_t &sortlist) {
    attrib_container_t::iterator abeg = this->begin();
        attrib_container_t::iterator aend = this->end();
        for ( ; abeg != aend; ++abeg )
            (*abeg)->sort(sortlist);
}


template<class PLayout>
void AmrParticleBase<PLayout>::setForbidTransform(bool forbidTransform) {
    forbidTransform_m = forbidTransform;
}


template<class PLayout>
bool AmrParticleBase<PLayout>::isForbidTransform() const {
    return forbidTransform_m;
}


template<class PLayout>
const double& AmrParticleBase<PLayout>::domainMapping(bool inverse) {
    IpplTimings::startTimer(domainMappingTimer_m);
    
    double scale = scale_m;
    
    Vector_t gamma = lorentzFactor_m;
    
    if ( !inverse ) {
//         if ( !this->DestroyList.empty() ) {
//             this->performDestroy(true);
//         }

        Vector_t rmin = Vector_t(0.0, 0.0, 0.0);
        Vector_t rmax = Vector_t(0.0, 0.0, 0.0);
        
        getGlobalBounds_m(rmin, rmax);
        
        /* in case of 1 particle, the bunch is rotated
         * transformed to the local frame such that this
         * particle lies on the origin (0, 0, 0).
         */
        if ( this->getTotalNum() == 1 ||
             (rmin == Vector_t(0.0, 0.0, 0.0) && rmax == Vector_t( 0.0,  0.0,  0.0)) ) {
            rmin = Vector_t(-1.0, -1.0, -1.0);
            rmax = Vector_t( 1.0,  1.0,  1.0);
        }

        /* Lorentz transfomration factor
         * is not equal 1.0 only in longitudinal
         * direction
         */
        rmin *= gamma;
        rmax *= gamma;

        PLayout& layout = this->getLayout();

        const auto& lo = layout.lowerBound;
        const auto& hi = layout.upperBound;

        Vector_t tmp = Vector_t(std::max( std::abs(rmin[0] / lo[0]), std::abs(rmax[0] / hi[0]) ),
                                std::max( std::abs(rmin[1] / lo[1]), std::abs(rmax[1] / hi[1]) ),
                                std::max( std::abs(rmin[2] / lo[2]), std::abs(rmax[2] / hi[2]) )
                               );
        
        scale = std::max( tmp[0], tmp[1] );
        scale = std::max( scale, tmp[2] );
    } else {
        // inverse Lorentz transform
        gamma = 1.0 / gamma;
    }
    
    if ( std::isnan(scale) || std::isinf(scale) ) {
        if ( !Ippl::myNode() )
            throw IpplException("AmrParticleBase::domainMapping()",
                                "Scale factor is Nan or Inf");
    }

    Vector_t vscale = Vector_t(scale, scale, scale);
    
    // Lorentz transform + mapping to [-1, 1]
    for (unsigned int i = 0; i < this->getLocalNum(); ++i) {
        this->R[i] = this->R[i] * gamma / vscale;
    }
    
    scale_m = 1.0 / scale;
    
    IpplTimings::stopTimer(domainMappingTimer_m);
    
    return scale_m;
}


template<class PLayout>
const double& AmrParticleBase<PLayout>::getScalingFactor() const {
    return scale_m;
}

template<class PLayout>
void AmrParticleBase<PLayout>::setLorentzFactor(const Vector_t& lorentzFactor) {
    lorentzFactor_m = lorentzFactor;
}


template<class PLayout>
void AmrParticleBase<PLayout>::getLocalBounds_m(Vector_t &rmin, Vector_t &rmax) {
    const size_t localNum = this->getLocalNum();
    if (localNum == 0) {
        double max = 1e10;
        rmin = Vector_t( max,  max,  max);
        rmax = Vector_t(-max, -max, -max);
        return;
    }

    rmin = this->R[0];
    rmax = this->R[0];
    for (size_t i = 1; i < localNum; ++ i) {
        for (unsigned short d = 0; d < 3u; ++ d) {
            if (rmin(d) > this->R[i](d)) rmin(d) = this->R[i](d);
            else if (rmax(d) < this->R[i](d)) rmax(d) = this->R[i](d);
        }
    }
}


template<class PLayout>
void AmrParticleBase<PLayout>::getGlobalBounds_m(Vector_t &rmin, Vector_t &rmax) {
    this->getLocalBounds_m(rmin, rmax);

    double min[6];
    for (unsigned int i = 0; i < 3; ++i) {
        min[2*i] = rmin[i];
        min[2*i + 1] = -rmax[i];
    }

    allreduce(min, 6, std::less<double>());

    for (unsigned int i = 0; i < 3; ++i) {
        rmin[i] = min[2*i];
        rmax[i] = -min[2*i + 1];
    }
}


// template<class PLayout>
// void AmrParticleBase<PLayout>::lorentzTransform(bool inverse) {
//     
//     if ( isLorentzTransformed_m && !inverse ) {
//         return;
//     }
//         
//     isLorentzTransformed_m = true;
//     
//     Vector_t gamma = lorentzFactor_m;
//     
//     if ( inverse ) {
//         gamma = 1.0 / gamma;
//         isLorentzTransformed_m = false;
//     }
//     
//     for (std::size_t i = 0; i < this->getLocalNum(); ++i)
//         this->R[i] *= gamma;
// }

#endif
