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
    class ParticleSpatialOverlapLayout : public ParticleSpatialLayout<T, Dim, Mesh, PositionProperties...> {
    public:
        using Base = ParticleSpatialLayout<T, Dim, Mesh, PositionProperties...>;
        using typename Base::position_memory_space, typename Base::position_execution_space;

        using typename Base::hash_type;
        using typename Base::locate_type;
        using index_t = typename hash_type::value_type;
        using locate_type_nd = Kokkos::View<index_t *[1 << Dim], position_memory_space>;
        // the maximum number of overlapping ranks
        using typename Base::bool_type;

        using typename Base::vector_type;
        using typename Base::RegionLayout_t;
        using typename Base::FieldLayout_t;

        using size_type = detail::size_type;

        using particle_neighbor_list_type = hash_type;
        using typename Base::particle_position_type;

    private:
        const T rcutoff_m;
        Vector_t<size_type, Dim> numCells_m;
        Vector_t<size_type, Dim> cellStrides_m;
        Vector_t<T, Dim> cellWidth_m;
        size_type totalCells_m, numGhostCells_m, numLocalCells_m, numLocalParticles_m;
        static constexpr size_type numGhostCellsPerDim_m = 1;
        hash_type cellPermutationForward_m; // given index from flattened indices gives cell index
        hash_type cellPermutationBackward_m;
        // given index in range [0, numLocCells) gives global index corresponding to flattened index
        hash_type cellStartingIdx_m;
        hash_type cellIndex_m;
        hash_type cellParticleCount_m;

        using CellIndex_t = Vector_t<size_type, Dim>;
        using FlatCellIndex_t = typename CellIndex_t::value_type;

        //! Type of the Kokkos view containing the local regions.
        using typename Base::region_view_type;
        //! Type of a single Region object.
        using typename Base::region_type;
    public:
        class NeighborData {
        private:
            friend class ParticleSpatialOverlapLayout;

            NeighborData(size_type numLocalParticles,
                         Vector_t<size_type, Dim> cellStrides,
                         Vector_t<size_type, Dim> numCells,
                         Vector_t<T, Dim> cellWidth,
                         region_type region,
                         hash_type cellStartingIdx,
                         hash_type cellIndex,
                         hash_type cellParticleCount,
                         hash_type cellPermutationForward,
                         hash_type cellPermutationBackward) : numLocalParticles(numLocalParticles),
                                                              cellStrides(cellStrides), numCells(numCells),
                                                              cellWidth(cellWidth), region(region),
                                                              cellStartingIdx(cellStartingIdx), cellIndex(cellIndex),
                                                              cellParticleCount(cellParticleCount),
                                                              cellPermutationForward(cellPermutationForward),
                                                              cellPermutationBackward(cellPermutationBackward) {
            }

            size_type numLocalParticles;
            Vector_t<size_type, Dim> cellStrides;
            Vector_t<size_type, Dim> numCells;
            Vector_t<T, Dim> cellWidth;
            region_type region;
            hash_type cellStartingIdx;
            hash_type cellIndex;
            hash_type cellParticleCount;
            hash_type cellPermutationForward;
            hash_type cellPermutationBackward;
        };

    public:
        template<class ParticleContainer>
        void particleExchange1(ParticleContainer &pc);

        template<class ParticleContainer>
        void particleExchange2(ParticleContainer &pc);

        template<class ParticleContainer>
        void buildCells(ParticleContainer &pc);

    protected:
        KOKKOS_INLINE_FUNCTION constexpr static FlatCellIndex_t toFlatCellIndex(
            const CellIndex_t &cellIndex, const Vector_t<size_type, Dim> &cellStrides,
            hash_type cellPermutationForward);

        KOKKOS_INLINE_FUNCTION constexpr static CellIndex_t toCellIndex(FlatCellIndex_t nonPermutedIndex,
                                                                        const Vector_t<size_type, Dim> &numCells);

        KOKKOS_INLINE_FUNCTION constexpr static bool isLocalCellIndex(const CellIndex_t &index,
                                                                      const Vector_t<size_type, Dim> &numCells);

        KOKKOS_INLINE_FUNCTION constexpr static bool positionInRegion(const vector_type &pos, const region_type &region,
                                                                      T overlap);

        KOKKOS_INLINE_FUNCTION constexpr static CellIndex_t getCellIndex(
            const vector_type &pos, const region_type &region,
            const Vector_t<T, Dim> &cellWidth);


        using cell_particle_neighbor_list_type = Kokkos::Array<size_type, detail::countHypercubes(Dim)>;

        KOKKOS_INLINE_FUNCTION constexpr static cell_particle_neighbor_list_type getCellNeighbors(
            const CellIndex_t &cellIndex, const Vector_t<size_type, Dim> &cellStrides,
            const hash_type &cellPermutationForward);

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

        template<typename ParticleContainer>
        size_type locateParticles(const ParticleContainer &pc, locate_type_nd &ranks,
                                  bool_type &invalid) const;

        template<typename ParticleContainer>
        size_type locateParticles(const ParticleContainer &pc, locate_type &ranks, locate_type &offsets,
                                  bool_type &invalid) const;


        size_t numberOfSends(int rank, const locate_type_nd &ranks);

        size_t numberOfSends(int rank, const locate_type &ranks);


        void fillHash(int rank, const locate_type_nd &ranks, hash_type &hash);


        void fillHash(int rank, const locate_type &ranks, const locate_type &offsets, hash_type &hash);

        NeighborData getNeighborData() const;

        KOKKOS_FUNCTION static particle_neighbor_list_type getParticleNeighbors(
            index_t particleIndex, const NeighborData &neighborData);

        KOKKOS_FUNCTION static particle_neighbor_list_type getParticleNeighbors(
            const vector_type &pos, const NeighborData &neighborData);

        template<typename ExecutionSpace, typename Functor>
        void forEachPair(Functor &&f) const;
    };
} // namespace ippl

#include "Particle/ParticleSpatialOverlapLayout.hpp"

#endif  // IPPL_PARTICLE_SPATIAL_OVERLAP_LAYOUT_H
