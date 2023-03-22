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
//   After each 'time step' in a calculation, which is defined as a period
//   in which the particle positions may change enough to affect the global
//   layout, the user must call the 'update' routine, which will move
//   particles between processors, etc.  After the Nth call to update, a
//   load balancing routine will be called instead.  The user may set the
//   frequency of load balancing (N), or may supply a function to
//   determine if load balancing should be done or not.
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

#include "Types/IpplTypes.h"

#include "FieldLayout/FieldLayout.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleLayout.h"
#include "Region/RegionLayout.h"

namespace ippl {
    /*!
     * ParticleSpatialLayout class definition.
     * @tparam T value type
     * @tparam Dim dimension
     * @tparam Mesh type
     */
    template <typename T, unsigned Dim, class Mesh = UniformCartesian<T, Dim>>
    class ParticleSpatialLayout : public detail::ParticleLayout<T, Dim> {
    public:
        using hash_type   = typename ParticleBase<ParticleSpatialLayout<T, Dim, Mesh>>::hash_type;
        using locate_type = typename detail::ViewType<int, 1>::view_type;
        using bool_type   = typename detail::ViewType<bool, 1>::view_type;
        using vector_type = typename detail::ParticleLayout<T, Dim>::vector_type;
        using RegionLayout_t = detail::RegionLayout<T, Dim, Mesh>;
        using Mesh_t         = UniformCartesian<double, Dim>;

        using size_type = detail::size_type;

    public:
        // constructor: this one also takes a Mesh
        ParticleSpatialLayout(FieldLayout<Dim>&, Mesh&);

        ParticleSpatialLayout()
            : detail::ParticleLayout<T, Dim>() {}

        ~ParticleSpatialLayout() = default;
        //~ParticleSpatialLayout() {}

        void updateLayout(FieldLayout<Dim>&, Mesh&);

        template <class BufferType>
        void update(BufferType& pdata, BufferType& buffer);

        const RegionLayout_t& getRegionLayout() const { return rlayout_m; }

    protected:
        //! The RegionLayout which determines where our particles go.
        RegionLayout_t rlayout_m;

    public:
        void locateParticles(const ParticleBase<ParticleSpatialLayout<T, Dim, Mesh>>& pdata,
                             locate_type& ranks, bool_type& invalid) const;

        /*!
         * @param rank we sent to
         * @param ranks a container specifying where a particle at the i-th index should go.
         * @param hash a mapping to fill the send buffer contiguously
         */
        void fillHash(int rank, const locate_type& ranks, hash_type& hash);

        /*!
         * @param rank we sent to
         * @param ranks a container specifying where a particle at the i-th index should go.
         */
        size_t numberOfSends(int rank, const locate_type& ranks);
    };
}  // namespace ippl

#include "Particle/ParticleSpatialLayout.hpp"

#endif
