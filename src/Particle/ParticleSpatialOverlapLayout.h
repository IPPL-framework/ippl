//
// Class ParticleSpatialOverlapLayout
//   Particle layout based on spatial decomposition.
//
//   This is a specialized version of ParticleSpatialLayout, which allows
//   particles to be on multiple processes if they are close to the respective
//   region.
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
    // This additional namespace is needed to allow for Mesh to have default argument
    // UniformCartesian<T, Dim> via typedef at the end of the file. Compiling for cuda led to error
    // when trying to do so directly...
    /*!
     * ParticleSpatialOverlapLayout class definition.
     * @tparam T value type
     * @tparam Dim dimension
     * @tparam Mesh type
     */
    template <typename T, unsigned Dim, class Mesh = UniformCartesian<T, Dim>,
              typename... PositionProperties>
    class ParticleSpatialOverlapLayout
        : public ParticleSpatialLayout<T, Dim, Mesh, PositionProperties...> {
    public:
        using Base = ParticleSpatialLayout<T, Dim, Mesh, PositionProperties...>;
        using typename Base::position_execution_space;
        using typename Base::position_memory_space;

        using typename Base::bool_type;
        using typename Base::hash_type;
        using typename Base::locate_type;

        using typename Base::FieldLayout_t;
        using typename Base::RegionLayout_t;
        using typename Base::vector_type;

        using size_type = detail::size_type;

        using index_t                     = typename hash_type::value_type;
        using particle_neighbor_list_type = hash_type;
        using typename Base::particle_position_type;

        //! Type of the Kokkos view containing the local regions.
        using typename Base::region_view_type;
        //! Type of a single Region object.
        using typename Base::region_type;
        //! Array of N rank lists, where N = number of hypercubes for the dimension Dim.
        using neighbor_list = typename FieldLayout_t::neighbor_list;

    public:
        // the maximum number of overlapping ranks
        using locate_type_nd = Kokkos::View<index_t * [1 << Dim], position_memory_space>;

        /*!
         * Proxy class to store all necessary information needed to call getParticleNeighbors
         * in Kokkos parallel regions
         */
        class ParticleNeighborData {
        private:
            friend class ParticleSpatialOverlapLayout;

            ParticleNeighborData(size_type numLocalParticles, Vector<size_type, Dim> cellStrides,
                                 Vector<size_type, Dim> numCells, Vector<T, Dim> cellWidth,
                                 region_type region, hash_type cellStartingIdx, hash_type cellIndex,
                                 hash_type cellParticleCount, hash_type cellPermutationForward,
                                 hash_type cellPermutationBackward)
                : numLocalParticles(numLocalParticles)
                , cellStrides(cellStrides)
                , numCells(numCells)
                , cellWidth(cellWidth)
                , region(region)
                , cellStartingIdx(cellStartingIdx)
                , cellIndex(cellIndex)
                , cellParticleCount(cellParticleCount)
                , cellPermutationForward(cellPermutationForward)
                , cellPermutationBackward(cellPermutationBackward) {}

            size_type numLocalParticles;
            Vector<size_type, Dim> cellStrides;
            Vector<size_type, Dim> numCells;
            Vector<T, Dim> cellWidth;
            region_type region;
            hash_type cellStartingIdx;
            hash_type cellIndex;
            hash_type cellParticleCount;
            hash_type cellPermutationForward;
            hash_type cellPermutationBackward;
        };

    public:
        /*!
         * @param fl Field layout
         * @param mesh Mesh
         * @param rcutoff Overlap of the regions in each dimension
         */
        ParticleSpatialOverlapLayout(FieldLayout<Dim>& fl, Mesh& mesh, const T& rcutoff);

        ParticleSpatialOverlapLayout()
            : ParticleSpatialLayout<T, Dim, PositionProperties...>() {}

        ~ParticleSpatialOverlapLayout() = default;

        void updateLayout(FieldLayout<Dim>&, Mesh&);

        /*!
         * @brief updates particles by exchanging them across ranks according to their positions.
         *         then constructs the particle neighbor list structure
         * @param pc particle container to update
         */
        template <class ParticleContainer>
        void update(ParticleContainer& pc);

        /*!
         * @brief call functor for each combination i, j. make sure to call update first
         * @tparam ExecutionSpace Space in which to generate all indices
         * @tparam Functor type of loop body
         * @param f loop body functor to call for all pair of indices i, j where i are all
         * internal particle indices and j include ghost particles
         */
        template <typename ExecutionSpace, typename Functor>
        void forEachPair(Functor&& f) const;

        /*!
         * @return the proxy of the particle neighbor list data needed to get particle neighbors
         */
        ParticleNeighborData getParticleNeighborData() const;

        /*!
         * @brief Function to get particle neighbors depending on index
         *        (possible inside Kokkos parallel region) make sure to call update first
         * @param particleIndex index of particle to get neighbors for
         * @param particleNeighborData proxy of (own) data required for the calculation
         * @return view of all indices of neighbor particles of particle with given particleIndex
         */
        KOKKOS_FUNCTION static particle_neighbor_list_type getParticleNeighbors(
            index_t particleIndex, const ParticleNeighborData& particleNeighborData);

        /*!
         * @brief Function to get particle neighbors depending on position
         *        (possible inside Kokkos parallel region) make sure to call update first
         * @param pos position of particle to get neighbors for
         * @param particleNeighborData proxy of (own) data required for the calculation
         * @return view of all indices of neighbor particles of particle with given particleIndex
         */
        KOKKOS_FUNCTION static particle_neighbor_list_type getParticleNeighbors(
            const vector_type& pos, const ParticleNeighborData& particleNeighborData);

        /*!
         * @brief utility function to compute how many particles to sent to a given rank
         * @param rank rank to send to
         * @param ranks The view containing which rank each particle belongs to
         * @return number of particles sent to rank
         */
        size_type numberOfSends(int rank, const locate_type& ranks);

        /*!
         * @brief utility function to collect all indices of particles to send to given rank
         * @param rank rank to send to
         * @param ranks The view containing which rank each particle belongs to
         * @param offsets The offsets to determine where the ranks of a particle start in ranks
         * @param hash the view containing all particle indices to send
         */
        void fillHash(int rank, const locate_type& ranks, const locate_type& offsets,
                      hash_type& hash);

        /**
         * @brief This function determines to which rank particles need to be sent after the
         *        iteration step. It starts by first scanning direct rank neighbors, and only does a
         *        global scan if particles are far away from the current rank. It then calculates
         *        how many particles need to be sent to each rank and how many ranks are sent to in
         *        total.
         *
         * @param pc           Particle Container
         * @param ranks        A vector where each value refers to the new rank of the particle
         *                      which rank values correspond to which particles is determined by
         *                      rankOffsets
         * @param rankOffsets  A vector of offsets where rankOffset(i) determines where the ranks of
         *                      particle i in ranks start.
         * @param invalid      A vector marking the particles that need to be sent away, and thus
         *                      locally deleted
         * @param nSends_dview Device view the length of number of ranks, where each value
         *                      determines the number of particles sent to that rank from the
         *                      current rank
         * @param sends_dview  Device view for the number of ranks that are sent to from current
         *                      rank
         *
         * @return tuple with the number of particles sent away and the number of ranks sent to
         */
        template <typename ParticleContainer>
        std::pair<detail::size_type, detail::size_type> locateParticles(
            const ParticleContainer& pc, locate_type& ranks, locate_type& rankOffsets,
            bool_type& invalid, locate_type& nSends_dview, locate_type& sends_dview) const;

        /*!
         * @brief utility function to get a flat view of all neighbor processes
         * @param neighbors FieldLayouts neighbor_list
         * @return view of neighbor ranks
         */
        locate_type getFlatNeighbors(const neighbor_list& neighbors) const;

        /*!
         * @brief utility function to get a view of all non-neighboring ranks
         * @param neighbors_view view of all neighboring ranks
         * @return view of all non-neighboring ranks
         */
        locate_type getNonNeighborRanks(const locate_type& neighbors_view) const;

    protected:
        ///! overlap in each dimension
        const T rcutoff_m;
        ///! number of cells in each dimension
        Vector<size_type, Dim> numCells_m;
        ///! strides to compute cell indices
        Vector<size_type, Dim> cellStrides_m;
        ///! width of cells in each dimension
        Vector<T, Dim> cellWidth_m;
        ///! the number of total cells
        size_type totalCells_m;
        ///! the number of interior cells
        size_type numGhostCells_m;
        ///! the number of ghost cells
        size_type numLocalCells_m;
        ///! the number of local particles (particles in local cells)
        size_type numLocalParticles_m;
        ///! number of ghost cells
        static constexpr size_type numGhostCellsPerDim_m = 1;
        /*!
         * To ensure the interior particles are at indices 0, ..., numLocalParticles_m - 1 the cells
         * need to be permuted such that local cells are at the beginning and ghost cells at the end
         * cellPermutationForward_m at cell index computed from actual position and cellStrides
         * gives permuted index
         */
        hash_type cellPermutationForward_m;
        ///! the inverse of cellPermutationForward_m
        hash_type cellPermutationBackward_m;
        ///! cell i contains particles cellStartingIdx_m(i), ..., cellStartingIdx_m(i + 1) - 1
        hash_type cellStartingIdx_m;
        ///! view storing the cell index of each particle (TODO not needed if getParticleNeighbors
        ///                                                 depending on index is not required)
        hash_type cellIndex_m;
        ///! view of number of particles in each cell
        hash_type cellParticleCount_m;

        using CellIndex_t     = Vector<size_type, Dim>;
        using FlatCellIndex_t = typename CellIndex_t::value_type;

    public:
        /*!
         * @brief initializes all data necessary for the cells
         */
        void initializeCells();

        /*!
         * @brief exchange particles by scanning neighbor ranks first, only scan other ranks if
         *         needed. assumes overlap is smaller than half the smallest region width.
         * @param pc particle container of which to exchange particles
         */
        template <class ParticleContainer>
        void particleExchange(ParticleContainer& pc);

        /*!
         * @brief builds the cell structure, sorts the particles according to the cells and makes
         *         sure only local particles are counted towards pc.getLocalNum()
         * @param pc particle container of which to sort the particles
         */
        template <class ParticleContainer>
        void buildCells(ParticleContainer& pc);

        /*!
         * @brief copies particles close to the boundary and offsets them to their closest periodic
         *        image
         * @param pc particle container of which to construct periodic ghost particles
         */
        template <class ParticleContainer>
        void createPeriodicGhostParticles(ParticleContainer& pc);

    protected:
        /*!
         * @brief determines whether a position is within overlap to the boundary of a region
         * @param pos position to query
         * @param region region of the position
         * @param periodic vector determining which dimensions to consider (as they are periodic)
         * @param overlap distance to consider as close
         */
        template <std::size_t... Idx>
        KOKKOS_INLINE_FUNCTION constexpr static bool isCloseToBoundary(
            const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region,
            Vector<bool, Dim> periodic, T overlap);

        /*!
         * @brief convert a nd-cell-index to flat cell index
         * @param cellIndex to convert
         * @param cellStrides cell strides to flatten the cell index with
         * @param cellPermutationForward the permutation to apply to the flattened index
         */
        KOKKOS_INLINE_FUNCTION constexpr static FlatCellIndex_t toFlatCellIndex(
            const CellIndex_t& cellIndex, const Vector<size_type, Dim>& cellStrides,
            hash_type cellPermutationForward);

        /*!
         * @brieg compute the nd-cell-index from a flattened (non-permuted) index
         * @param nonPermutedIndex the flat index to transform
         * @param numCells in each dimension
         */
        KOKKOS_INLINE_FUNCTION constexpr static CellIndex_t toCellIndex(
            FlatCellIndex_t nonPermutedIndex, const Vector<size_type, Dim>& numCells);

        /*!
         * @brief determines whether cell index is local cell index
         * @param index to test
         * @param numCells in each dimension
         */
        template <std::size_t... Idx>
        KOKKOS_INLINE_FUNCTION constexpr static bool isLocalCellIndex(
            const std::index_sequence<Idx...>&, const CellIndex_t& index,
            const Vector<size_type, Dim>& numCells);

        /*!
         * @brief determines whether a position is in a region including its overlap
         * @param pos position to query
         * @param region region to test
         * @param overlap overlap of the region in every dimension
         */
        template <std::size_t... Idx>
        KOKKOS_INLINE_FUNCTION constexpr static bool positionInRegion(
            const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region,
            T overlap);

        /*!
         * @brief get the nd-cell-index of a position
         * @param pos position to get the cell index for
         * @param region region of the cells
         * @param cellWidth in each dimension
         */
        KOKKOS_INLINE_FUNCTION constexpr static CellIndex_t getCellIndex(
            const vector_type& pos, const region_type& region, const Vector<T, Dim>& cellWidth);

        using cell_particle_neighbor_list_type =
            Kokkos::Array<size_type, detail::countHypercubes(Dim)>;

        /*!
         * @brief get all indices of cell neighbors of a given nd-cell-index
         * @param cellIndex to get the neighbors from
         * @param cellStrides in each dimension
         * @param cellPermutationForward the permutation to apply to all neighbors
         */
        KOKKOS_INLINE_FUNCTION constexpr static cell_particle_neighbor_list_type getCellNeighbors(
            const CellIndex_t& cellIndex, const Vector<size_type, Dim>& cellStrides,
            const hash_type& cellPermutationForward);
    };
}  // namespace ippl

#include "Particle/ParticleSpatialOverlapLayout.hpp"

#endif  // IPPL_PARTICLE_SPATIAL_OVERLAP_LAYOUT_H
