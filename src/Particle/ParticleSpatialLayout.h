
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

#include <vector>

#include "Types/IpplTypes.h"

#include "Communicate/Window.h"
#include "FieldLayout/FieldLayout.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleLayout.h"
#include "Region/RegionLayout.h"

namespace ippl {

    // Controls how the send-count exchange is performed before the particle
    // data transfer. All paths assume GPU-aware MPI (device pointers passed
    // directly to Isend/Irecv/Alltoall).
    enum class CountExchange {
        // One-sided RMA: each sender writes its count directly into the
        // receiver's window.
        RMA,
        // Two-sided point-to-point: per-rank Isend/Irecv over device pointers.
        P2P,
        // Collective Alltoall over device pointers.
        Alltoall
    };

    /*!
     * ParticleSpatialLayout class definition.
     * @tparam T value type
     * @tparam Dim dimension
     * @tparam Mesh type
     */
    template <typename T, unsigned Dim, class Mesh, typename... PositionProperties>
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

        // constructor: this one also takes a Mesh
        ParticleSpatialLayout(FieldLayout<Dim>&, Mesh&, bool fem = false,
                              CountExchange mode = CountExchange::RMA);

        ParticleSpatialLayout()
            : detail::ParticleLayout<T, Dim, PositionProperties...>() {}

        ~ParticleSpatialLayout() = default;

        void updateLayout(FieldLayout<Dim>&, Mesh&);

        template <class ParticleContainer>
        void update(ParticleContainer& pc);

        const RegionLayout_t& getRegionLayout() const { return *rlayout_m; }

    protected:
        //! The RegionLayout which determines where our particles go.
        std::shared_ptr<RegionLayout_t> rlayout_m;

        //! The FieldLayout containing information on nearest neighbors
        FieldLayout_t& flayout_m;

        //! How counts are exchanged
        CountExchange countExchangeMode_;

        //
        // RMA Path
        //

        // Vector keeping track of the recieves from all ranks
        std::vector<size_type> nRecvs_m;

        // MPI RMA window for one-sided communication
        mpi::rma::Window<mpi::rma::Active> window_m;

        //
        // P2P GPU Path
        //
        locate_type recvCounts_d_;  // [nranks]

        //! Type of the Kokkos view containing the local regions.
        using region_view_type = typename RegionLayout_t::view_type;
        //! Type of a single Region object.
        using region_type = typename region_view_type::value_type;
        //! Array of N rank lists, where N = number of hypercubes for the dimension Dim.
        using neighbor_list = typename FieldLayout_t::neighbor_list;

        template <size_t... Idx>
        KOKKOS_INLINE_FUNCTION constexpr static bool positionInRegion(
            const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region);

        template <size_t... Idx>
        KOKKOS_INLINE_FUNCTION constexpr static bool positionInRegionInclusive(
            const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region);

        /*!
         * Evaluates the total number of MPI ranks sharing the spatial nearest neighbors.
         * @param neighbors structure containing, for every spatial direction, a list of
         * MPI ranks IDs corresponding to the nearest neighbors of the current local domain section.
         * @return The total number of the ranks.
         */
        size_type getNeighborSize(const neighbor_list& neighbors) const;

    public:
        /*!
         * For each local particle, determine its destination rank, write the
         * indices of leaving particles into `sendIds_d_`, populate the
         * per-rank send-count and offset buffers, build a compacted list of
         * destination ranks, and mark each particle's "is leaving" status in
         * `leaving_d_` so the destroy pass doesn't have to recompute it.
         * @return number of leaving particles on this rank
         */
        template <typename ParticleContainer>
        size_t locateParticlesPacked(const ParticleContainer& pc);

    private:
        // Fixed-size scratch
        locate_type rankSendCount_d_;  // [nRanks]
        locate_type sendOffsets_d_;    // [nRanks+1]
        hash_type sendIds_d_;          // [capacity >= max nInvalid seen]
        locate_type cursor_d_;         // [nRanks] per-rank insertion cursor
        locate_type destRanks_d_;      // [nRanks] (compacted list)
        bool_type leaving_d_;          // [capacity >= max nLocal seen] mask

        // Single scalar on device to count destinations
        Kokkos::View<size_type, position_memory_space> nDest_d_;

        // Neigbour cache
        locate_type neighbors_d_;          // [neighborSize] cached device neighbors list
        std::vector<int> neighbors_host_;  // flat host copy
        bool neighbors_dirty_      = true;
        size_t neighbors_capacity_ = 0;
        size_type neighbors_used_  = 0;

        // Host mirror buffers
        using host_mem_space   = Kokkos::HostSpace;
        using locate_host_type = typename detail::ViewType<int, 1, host_mem_space>::view_type;

        locate_host_type rankSendCount_h_;  // [nRanks] (mirror)
        locate_host_type sendOffsets_h_;    // [nRanks+1] (mirror)
        locate_host_type destRanks_h_;      // [nRanks] (mirror)

        // Host-side destination list
        std::vector<int> destinationRanks_host_;

        // capacities
        size_t sendIds_capacity_ = 0;
        size_t leaving_capacity_ = 0;
        int nRanks_              = 0;

        void initScratch(int nRanks);
        void ensureSendCapacity(size_t nInvalid);
        void ensureLeavingCapacity(size_t nLocal);
        void ensureNeighborsCached();

        void countExchangeRMA();
        void countExchangeP2P();
        void countExchangeAlltoall();
    };
}  // namespace ippl

#include "Particle/ParticleSpatialLayout.hpp"

#endif
