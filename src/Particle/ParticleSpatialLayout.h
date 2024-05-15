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
    template <typename T, unsigned Dim, class Mesh = UniformCartesian<T, Dim>,
              typename... PositionProperties>
    class ParticleSpatialLayout : public detail::ParticleLayout<T, Dim, PositionProperties...> {
    public:
        using Base = detail::ParticleLayout<T, Dim, PositionProperties...>;
        using typename Base::position_memory_space, typename Base::position_execution_space;

        using hash_type   = detail::hash_type<position_memory_space>;
        using locate_type = typename detail::ViewType<int, 1, position_memory_space>::view_type;
        using bool_type   = typename detail::ViewType<bool, 1, position_memory_space>::view_type;

        using vector_type = typename Base::vector_type;
        using RegionLayout_t =
            typename detail::RegionLayout<T, Dim, Mesh, position_memory_space>::uniform_type;
        using FieldLayout_t = typename ippl::FieldLayout<Dim>;

        using size_type = detail::size_type;

    public:
        // constructor: this one also takes a Mesh
        ParticleSpatialLayout(FieldLayout<Dim>&, Mesh&);

        ParticleSpatialLayout()
            : detail::ParticleLayout<T, Dim, PositionProperties...>() {}

        ~ParticleSpatialLayout() = default;

        void updateLayout(FieldLayout<Dim>&, Mesh&);

        template <class ParticleContainer>
        void update(ParticleContainer& pc);

        const RegionLayout_t& getRegionLayout() const { return rlayout_m; }

    protected:
        //! The RegionLayout which determines where our particles go.
        RegionLayout_t rlayout_m;

        //! The FieldLayout containing information on nearest neighbors
        FieldLayout_t flayout_m;

        //! Type of the Kokkos view containing the local regions.
        using region_view_type = typename RegionLayout_t::view_type;
        //! Type of a single Region object.
        using region_type = typename region_view_type::value_type;
        //! Array of N rank lists, where N = number of hypercubes for the dimension Dim.
        using neighbor_list = typename FieldLayout_t::neighbor_list;

        template <size_t... Idx>
        KOKKOS_INLINE_FUNCTION constexpr static bool positionInRegion(
            const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region);

    public:
        /*!
         * For each particle in the bunch, determine the rank on which it should
         * be stored based on its location
         * @tparam ParticleContainer the particle container type
         * @param pc the particle container
         * @param ranks the integer view in which to store the destination ranks
         * @param invalid the boolean view in which to store whether each particle
         * is invalidated, i.e. needs to be sent to another rank
         * @return The total number of invalidated particles
         */
        template <typename ParticleContainer>
        size_type locateParticles(const ParticleContainer& pc, locate_type& ranks,
                                  bool_type& invalid) const;

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
