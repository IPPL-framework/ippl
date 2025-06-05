//
// Class ParticleSpatialOverlapLayout
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
#ifndef IPPL_PARTICLE_SPATIAL_OVERLAP_LAYOUT_H
#define IPPL_PARTICLE_SPATIAL_OVERLAP_LAYOUT_H

#include "Types/IpplTypes.h"

#include "FieldLayout/FieldLayout.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleLayout.h"
#include "Region/RegionLayout.h"

namespace ippl {
    /*!
     * ParticleSpatialOverlapLayout class definition.
     * @tparam T value type
     * @tparam Dim dimension
     * @tparam Mesh type
     */
    template<typename T, unsigned Dim, class Mesh = UniformCartesian<T, Dim>,
        typename... PositionProperties>
    class ParticleSpatialOverlapLayout : public ParticleSpatialLayout<T, Dim, PositionProperties...> {
    public:
        using overlap_type = std::array<T, Dim>;
        using Base = ParticleSpatialLayout<T, Dim, PositionProperties...>;
        using typename Base::position_memory_space, typename Base::position_execution_space;

        using typename Base::hash_type;
        using typename Base::locate_type;
        using index_t = typename hash_type::value_type;
        using locate_type_nd = Kokkos::View<index_t *[1 << Dim], position_memory_space>;
        // the maximum number of overlapping ranks
        using typename Base::bool_type;

        using typename Base::vector_type;
        using typename Base::RegionLayout_t;
        using NDRegion_t = typename RegionLayout_t::NDRegion_t;
        using typename Base::FieldLayout_t;

        using size_type = detail::size_type;

        using neighbor_list_type = hash_type;
        using typename Base::particle_position_type;

    private:
        const T rcutoff_m;
        std::array<size_type, Dim> numCells_m;
        std::array<size_type, Dim> cellStrides_m;
        std::array<T, Dim> cellWidth_m;
        size_type totalCells_m, numGhostCells_m, numLocalCells_m;
        static constexpr size_type numGhostCellsPerDim_m = 1;
        hash_type cellPermutationForward_m; // given index from flattened indices gives cell index
        hash_type cellPermutationBackward_m; // given index in range [0, numLocCells) gives global index corresponding to flattened index
        hash_type cellStartingIdx_m;
        hash_type cellIndex_m;
        hash_type cellParticleCount_m;

        using CellIndex_t = Vector_t<size_type, Dim>;

    public:
        struct NeighborData {
            std::array<size_type, Dim> cellStrides;
            std::array<size_type, Dim> numCells;
            std::array<T, Dim> cellWidth;
            NDRegion_t region;
            hash_type cellStartingIdx;
            hash_type cellIndex;
            hash_type cellParticleCount;
            hash_type cellPermutationForward;
            hash_type cellPermutationBackward;
        };

    private:
        template<class ParticleContainer>
        void particleExchange1(ParticleContainer &pc);

        template<class ParticleContainer>
        void particleExchange2(ParticleContainer &pc);

        template<class ParticleContainer>
        void buildCells(ParticleContainer &pc);

        //! Type of the Kokkos view containing the local regions.
        using typename Base::region_view_type;
        //! Type of a single Region object.
        using typename Base::region_type;

        KOKKOS_INLINE_FUNCTION constexpr static bool isLocalCellIndex(size_type index,
                                                                      const std::array<size_type, Dim> &numCells);

        KOKKOS_INLINE_FUNCTION constexpr static bool positionInRegion(const vector_type &pos, const region_type &region,
                                                                      T overlap);

        KOKKOS_INLINE_FUNCTION constexpr static size_type getCellIndex(const vector_type &pos, const NDRegion_t &region,
                                                                       const std::array<size_type, Dim> &strides,
                                                                       const std::array<T, Dim> &cellWidth);

        KOKKOS_INLINE_FUNCTION constexpr static CellIndex_t getCellIndex(size_type index,
            const std::array<size_type, Dim> &strides);

        using neighbor_info_type = Kokkos::Array<index_t, detail::countHypercubes(Dim)>;

        KOKKOS_INLINE_FUNCTION constexpr static neighbor_info_type getNeighborCells(
            index_t cellIndex, const std::array<size_type, Dim> &numCells,
            const hash_type &cellPermutation);

    public:
        //
        // <----------------------- TODO add template parameter for local ordering or should it be separate class?
        //

        // constructor: this one also takes a Mesh
        ParticleSpatialOverlapLayout(FieldLayout<Dim> &fl, Mesh &mesh, const T &rcutoff);

        ParticleSpatialOverlapLayout()
            : ParticleSpatialLayout<T, Dim, PositionProperties...>() {
        }

        ~ParticleSpatialOverlapLayout() = default;

        template<class ParticleContainer>
        void update(ParticleContainer &pc);

        //        template <typename ParticleContainer>
        //        size_type locateParticles(const ParticleContainer& pc, locate_type& ranks,
        //                                  bool_type& invalid) const;

        template<typename ParticleContainer>
        size_type locateParticles(const ParticleContainer &pc, locate_type_nd &ranks,
                                  bool_type &invalid) const;

        template<typename ParticleContainer>
        size_type locateParticles(const ParticleContainer &pc, locate_type &ranks, locate_type &offsets,
                                  bool_type &invalid) const;


        size_t numberOfSends(int rank, const locate_type_nd &ranks);

        size_t numberOfSends(int rank, const locate_type &ranks);

        size_t getNumCells() const; // returns local number of cels

        neighbor_list_type getParticlesOfCell(size_type cellIndex) const;

        KOKKOS_FUNCTION static constexpr neighbor_list_type getParticlesOfCell(
            const NeighborData &neighborData, size_type cellIndex);


        void fillHash(int rank, const locate_type_nd &ranks, hash_type &hash);


        void fillHash(int rank, const locate_type &ranks, const locate_type &offsets, hash_type &hash);

        NeighborData getNeighborData() const;

        KOKKOS_FUNCTION static neighbor_list_type getNeighbors(size_type i, NeighborData &neighborData);

        KOKKOS_FUNCTION static neighbor_list_type getNeighbors(const vector_type &pos, const NeighborData &neighborData);


        template<typename ExecutionSpace, typename Functor>
        void forEachPair(Functor &&f) const;
    };
} // namespace ippl

#include "Particle/ParticleSpatialOverlapLayout.hpp"

#endif  // IPPL_PARTICLE_SPATIAL_OVERLAP_LAYOUT_H
