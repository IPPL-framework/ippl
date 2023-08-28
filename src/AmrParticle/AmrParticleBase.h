//
// Class AmrParticleBase
//   Ippl interface for AMR particles.
//   The derived classes need to extend the base class by subsequent methods.
//
// Matthias Frey (2016 - 2020)
// Implemented as part of the PhD thesis
// "Precise Simulations of Multibunches in High Intensity Cyclotrons"
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
//
#ifndef AMR_PARTICLE_BASE_H
#define AMR_PARTICLE_BASE_H

#include "Ippl.h"

#include "AmrParticleLevelCounter.h"
#include "Particle/IpplParticleBase.h"

template <class PLayout>
class AmrParticleBase : public IpplParticleBase<PLayout> {
public:
    typedef typename PLayout::ParticlePos_t ParticlePos_t;
    typedef typename PLayout::ParticleIndex_t ParticleIndex_t;
    typedef typename PLayout::SingleParticlePos_t SingleParticlePos_t;
    typedef typename PLayout::AmrField_t AmrField_t;
    typedef typename PLayout::AmrVectorField_t AmrVectorField_t;
    typedef typename PLayout::AmrScalarFieldContainer_t AmrScalarFieldContainer_t;
    typedef typename PLayout::AmrVectorFieldContainer_t AmrVectorFieldContainer_t;

    typedef long SortListIndex_t;
    typedef std::vector<SortListIndex_t> SortList_t;
    typedef std::vector<ParticleAttribBase*> attrib_container_t;

    ParticleIndex_t Level;  // m_lev
    ParticleIndex_t Grid;   // m_grid

    typedef AmrParticleLevelCounter<size_t, size_t> ParticleLevelCounter_t;

public:
    AmrParticleBase();

    AmrParticleBase(PLayout* layout);

    ~AmrParticleBase() {}

    // initialize AmrParticleBase class - add level and grid variables to attribute list
    void initializeAmr() {
        this->addAttribute(Level);
        this->addAttribute(Grid);
    }

    const ParticleLevelCounter_t& getLocalNumPerLevel() const;

    ParticleLevelCounter_t& getLocalNumPerLevel();

    void setLocalNumPerLevel(const ParticleLevelCounter_t& LocalNumPerLevel);

    /* Functions of IpplParticleBase<PLayout> adpated to
     * work with AmrParticleLevelCounter:
     * - createWithID()
     * - create()
     * - destroy()
     * - performDestroy()
     */

    void createWithID(unsigned id);

    void create(size_t M);

    void destroy(size_t M, size_t I, bool doNow = false);

    void performDestroy(bool updateLocalNum = false);

    // Update the particle object after a timestep.  This routine will change
    // our local, total, create particle counts properly.
    void update();

    /*!
     * There's is NO check performed if lev_min <= lev_max and
     * lev_min >= 0.
     * @param lev_min is the start level to update
     * @param lev_max is the last level to update
     * @param isRegrid is true if we are updating the grids (default: false)
     */
    void update(int lev_min, int lev_max, bool isRegrid = false);

    // Update the particle object after a timestep.  This routine will change
    // our local, total, create particle counts properly.
    void update(const ParticleAttrib<char>& canSwap);

    // sort particles based on the grid and level that they belong to
    void sort();

    // sort the particles given a sortlist
    void sort(SortList_t& sortlist);

    PLayout& getAmrLayout() { return this->getLayout(); }
    const PLayout& getAmrLayout() const { return this->getLayout(); }

    /*!
     * This method is used in the AmrPartBunch::boundp() function
     * in order to avoid multpile particle mappings during the
     * mesh regridding process.
     *
     * @param forbidTransform true if we don't want to map particles onto
     * \f$[-1, 1]^3\f$
     */
    inline void setForbidTransform(bool forbidTransform);

    /*!
     * @returns true if we are not mapping the particles onto
     * \f$[-1, 1]^3\f$ during an update call.
     */
    inline bool isForbidTransform() const;

    /*!
     * Linear mapping to AMReX computation domain [-1, 1]^3 including the Lorentz
     * transform. All dimensions
     * are mapped by the same scaling factor.
     * The potential and electric field need to be scaled afterwards appropriately.
     * @param PData is the particle data
     * @param inverse is true if we want to do the inverse operation
     * @returns scaling factor
     */
    const double& domainMapping(bool inverse = false);

    /*!
     * This function is used during the cell tagging routines.
     * @returns the scaling factor of the particle domain mapping.
     */
    inline const double& getScalingFactor() const;

    void setLorentzFactor(const Vector_t& lorentzFactor);

    //     void lorentzTransform(bool inverse = false);

private:
    void getLocalBounds_m(Vector_t& rmin, Vector_t& rmax);
    void getGlobalBounds_m(Vector_t& rmin, Vector_t& rmax);

protected:
    IpplTimings::TimerRef updateParticlesTimer_m;
    IpplTimings::TimerRef sortParticlesTimer_m;
    IpplTimings::TimerRef domainMappingTimer_m;

    bool forbidTransform_m;  ///< To avoid multiple transformations during regrid

    /*!
     * Scaling factor for particle coordinate transform
     * (used for Poisson solve and particle-to-core distribution)
     */
    double scale_m;

    /*!
     * Lorentz factor used for the domain mapping.
     * Is updated in AmrBoxLib
     *
     */
    Vector_t lorentzFactor_m;

    //     bool isLorentzTransformed_m;

private:
    ParticleLevelCounter_t LocalNumPerLevel_m;
};

#include "AmrParticleBase.hpp"

#endif
