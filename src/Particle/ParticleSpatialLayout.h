//
// Class ParticleSpatialLayout
//   Particle layout based on spatial decomposition.
//
//   This is a specialized version of ParticleLayout, which places particles
//   on processors based on their spatial location relative to a fixed grid.
//   In particular, this can maintain particles on processors based on a
//   specified FieldLayout or RegionLayout, so that particles are always on
//   the same node as the node containing the Field region to which they are
//   local.  This may also be used if there is no associated Field at all,
//   in which case a grid is selected based on an even distribution of
//   particles among processors.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_PARTICLE_SPATIAL_LAYOUT_H
#define IPPL_PARTICLE_SPATIAL_LAYOUT_H


#include "FieldLayout/FieldLayout.h"
#include "Particle/ParticleLayout.h"
#include "Particle/ParticleBase.h"

#include "Region/RegionLayout.h"

namespace ippl {
    /*!
     * ParticleSpatialLayout class definition.
     * @tparam T value type
     * @tparam Dim dimension
     * @tparam Mesh type
     */
    template <typename T, unsigned Dim,
              class Mesh = UniformCartesian<T, Dim>
              >
    class ParticleSpatialLayout : public detail::ParticleLayout<T, Dim>
//                                 ,  public FieldLayoutUser
    {
    public:
        using hash_type = typename ParticleBase<ParticleSpatialLayout<T, Dim, Mesh> >::hash_type;
        using locate_type = typename detail::ViewType<int, 1>::view_type;
        using bool_type = typename detail::ViewType<bool, 1>::view_type;
        using RegionLayout_t = detail::RegionLayout<T, Dim, Mesh>;

    public:
        // constructor: this one also takes a Mesh
        ParticleSpatialLayout(FieldLayout<Dim>&, Mesh&);

        ParticleSpatialLayout() : detail::ParticleLayout<T, Dim>() { }

        ~ParticleSpatialLayout() = default;

    //
    // spatial decomposition layout information
    //

    // retrieve a reference to the FieldLayout object in use.  This may be used,
    // e.g., to construct a Field with the same layout as the Particles.  Note
    // that if this object was constructed by providing a RegionLayout in the
    // constructor, then this generated FieldLayout will not necessarily match
    // up with the Region (it will be offset by some amount).  But, if this
    // object was either 1) created with a FieldLayout to begin with, or 2)
    // created with no layout, and one was generated internally, then the
    // returned FieldLayout will match and can be used to make new Fields or
    // Particles.
//     FieldLayout<Dim>& getFieldLayout()
//     {
//         return RLayout.getFieldLayout();
//     }

//     // retrieve a reference to the RegionLayout object in use
//     RegionLayout<T,Dim,Mesh>& getLayout()
//     {
//         return RLayout;
//     }
//     const RegionLayout<T,Dim,Mesh>& getLayout() const
//     {
//         return RLayout;
//     }

//     // get number of particles on a physical node
//     int getNodeCount(unsigned i) const
//     {
//         PAssert_LT(i, (unsigned int) Ippl::Comm->getNodes());
//         return NodeCount[i];
//     }

//     // get flag for empty node domain
//     bool getEmptyNode(unsigned i) const
//     {
//         PAssert_LT(i, (unsigned int) Ippl::Comm->getNodes());
//         return EmptyNode[i];
//     }

    //
    // Particle swapping/update routines
    //

//     // Update the location and indices of all atoms in the given IpplParticleBase
//     // object.  This handles swapping particles among processors if
//     // needed, and handles create and destroy requests.  When complete,
//     // all nodes have correct layout information.

        void update(ParticleBase<ParticleSpatialLayout<T, Dim, Mesh>>& pdata);

        const RegionLayout_t& getRegionLayout() const { return rlayout_m; }
    //
    // virtual functions for FieldLayoutUser's (and other UserList users)
    //

    // Repartition onto a new layout
//     virtual void Repartition(UserList *);


    protected:
    //! The RegionLayout which determines where our particles go.
    RegionLayout_t rlayout_m;

    // The number of particles located on each physical node.
//     size_t *NodeCount;

    // Flag for which nodes have no local domain
//     bool* EmptyNode;

//     unsigned NeighborNodes[Dim];


    // perform common constructor tasks
        void setup();

    public:
        void locateParticles(const ParticleBase<ParticleSpatialLayout<T, Dim, Mesh>>& pdata,
                             locate_type& ranks,
                             bool_type& invalid) const;

        void fillHash(int rank, const locate_type& ranks, hash_type& hash);

        size_t numberOfSends(int rank, const locate_type& ranks);

    };
}

#include "Particle/ParticleSpatialLayout.hpp"

#endif
