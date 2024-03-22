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
#include <memory>
#include <numeric>
#include <vector>

#include "Utility/IpplTimings.h"

#include "Communicate/Window.h"

namespace ippl {

    /*!
     * We need this struct since Kokkos parallel_scan only accects
     * one variable of type ReturnType where to perform the reduction operation.
     * For more details, see
     * https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/parallel_scan.html.
     */
    struct increment_type {
        size_t count[2];

        KOKKOS_FUNCTION void init() {
            count[0] = 0;
            count[1] = 0;
        }

        KOKKOS_INLINE_FUNCTION increment_type& operator+=(bool* values) {
            count[0] += values[0];
            count[1] += values[1];
            return *this;
        }

        KOKKOS_INLINE_FUNCTION increment_type& operator+=(increment_type values) {
            count[0] += values.count[0];
            count[1] += values.count[1];
            return *this;
        }
    };

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    ParticleSpatialLayout<T, Dim, Mesh, Properties...>::ParticleSpatialLayout(FieldLayout<Dim>& fl,
                                                                              Mesh& mesh)
        : rlayout_m(fl, mesh)
        , flayout_m(fl) {}

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::updateLayout(FieldLayout<Dim>& fl,
                                                                          Mesh& mesh) {
        rlayout_m.changeDomain(fl, mesh);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::update(ParticleContainer& pc) {
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        this->applyBC(pc.R, rlayout_m.getDomain());
        IpplTimings::stopTimer(ParticleBCTimer);

        static IpplTimings::TimerRef ParticleUpdateTimer = IpplTimings::getTimer("updateParticle");
        IpplTimings::startTimer(ParticleUpdateTimer);
        int nRanks = Comm->size();

        if (nRanks < 2) {
            return;
        }

        /* particle MPI exchange:
         *   1. figure out which particles need to go where
         *   2. fill send buffer and send particles
         *   3. delete invalidated particles
         *   4. receive particles
         */

        static IpplTimings::TimerRef locateTimer = IpplTimings::getTimer("locateParticles");
        IpplTimings::startTimer(locateTimer);
        size_type localnum = pc.getLocalNum();

        // 1st step

        /* the values specify the rank where
         * the particle with that index should go
         */
        locate_type ranks("MPI ranks", localnum);

        /* 0 --> particle valid
         * 1 --> particle invalid
         */
        bool_type invalid("invalid", localnum);

        size_type invalidCount = locateParticles(pc, ranks, invalid);
        IpplTimings::stopTimer(locateTimer);

        // 2nd step

        // figure out how many receives
        static IpplTimings::TimerRef preprocTimer = IpplTimings::getTimer("sendPreprocess");
        IpplTimings::startTimer(preprocTimer);
        mpi::rma::Window<mpi::rma::Active> window;
        std::vector<size_type> nRecvs(nRanks, 0);
        window.create(*Comm, nRecvs.begin(), nRecvs.end());

        std::vector<size_type> nSends(nRanks, 0);

        window.fence(0);

        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == Comm->rank()) {
                // we do not need to send to ourselves
                continue;
            }
            nSends[rank] = numberOfSends(rank, ranks);
            window.put<size_type>(nSends.data() + rank, rank, Comm->rank());
        }
        window.fence(0);
        IpplTimings::stopTimer(preprocTimer);

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        IpplTimings::startTimer(sendTimer);
        // send
        std::vector<MPI_Request> requests(0);

        int tag = Comm->next_tag(mpi::tag::P_SPATIAL_LAYOUT, mpi::tag::P_LAYOUT_CYCLE);

        int sends = 0;
        for (int rank = 0; rank < nRanks; ++rank) {
            if (nSends[rank] > 0) {
                hash_type hash("hash", nSends[rank]);
                fillHash(rank, ranks, hash);

                pc.sendToRank(rank, tag, sends++, requests, hash);
            }
        }
        IpplTimings::stopTimer(sendTimer);

        // 3rd step
        static IpplTimings::TimerRef destroyTimer = IpplTimings::getTimer("particleDestroy");
        IpplTimings::startTimer(destroyTimer);

        pc.internalDestroy(invalid, invalidCount);
        Kokkos::fence();


        IpplTimings::stopTimer(destroyTimer);
        static IpplTimings::TimerRef recvTimer = IpplTimings::getTimer("particleRecv");
        IpplTimings::startTimer(recvTimer);
        // 4th step
        int recvs = 0;
        for (int rank = 0; rank < nRanks; ++rank) {
            if (nRecvs[rank] > 0) {
                pc.recvFromRank(rank, tag, recvs++, nRecvs[rank]);
            }
        }
        IpplTimings::stopTimer(recvTimer);

        IpplTimings::startTimer(sendTimer);

        if (requests.size() > 0) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        IpplTimings::stopTimer(sendTimer);

        IpplTimings::stopTimer(ParticleUpdateTimer);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialLayout<T, Dim, Mesh, Properties...>::positionInRegion(
        const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region) {
        return ((pos[Idx] >= region[Idx].min()) && ...) && ((pos[Idx] <= region[Idx].max()) && ...);
    };

    /* Helper function that evaluates the total number of neighbors for the current rank in Dim dimensions.
    */
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    detail::size_type ParticleSpatialLayout<T, Dim, Mesh, Properties...>::getNeighborSize(
        const neighbor_list& neighbors) const {
        size_type totalSize = 0;

        for (const auto& componentNeighbors : neighbors) {
            totalSize += componentNeighbors.size();
        }

        return totalSize;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <typename ParticleContainer>
    detail::size_type ParticleSpatialLayout<T, Dim, Mesh, Properties...>::locateParticles(
        const ParticleContainer& pc, locate_type& ranks, bool_type& invalid) const {
        auto positions           = pc.R.getView();
        region_view_type Regions = rlayout_m.getdLocalRegions();

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;

        size_type myRank = Comm->rank();

        const auto is = std::make_index_sequence<Dim>{};

        const neighbor_list& neighbors = flayout_m.getNeighbors();

        /// outsideIds: Container of particle IDs that travelled outside of the neighborhood.
        locate_type outsideIds("Particles outside of neighborhood", size_type(pc.getLocalNum()));

        /// outsideCount: Tracks the number of particles that travelled outside of the neighborhood.
        size_type outsideCount       = 0;
        /// invalidCount: Tracks the number of particles that need to be sent to other ranks.
        size_type invalidCount       = 0;

        /// neighborSize: Size of a neighborhood in D dimentions.
        const size_type neighborSize = getNeighborSize(neighbors);

        /// neighbors_view: Kokkos view with the IDs of the neighboring MPI ranks.
        locate_type neighbors_view("Nearest neighbors IDs", neighborSize);

        /* red_val: Used to reduce both the number of invalid particles and the number of particles 
         * outside of the neighborhood (Kokkos::parallel_scan doesn't allow multiple reduction values, so we
         * use the helper class increment_type). First element updates InvalidCount, second
         * one updates outsideCount.
        */
        increment_type red_val;
        red_val.init();

        auto neighbors_mirror =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), neighbors_view);

        size_t k = 0;

        for (const auto& componentNeighbors : neighbors) {
            for (size_t j = 0; j < componentNeighbors.size(); ++j) {
                neighbors_mirror(k) = componentNeighbors[j];
                k++;
            }
        }

        Kokkos::deep_copy(neighbors_view, neighbors_mirror);

        /*! Begin Kokkos loop:
         * Step 1: search in current rank
         * Step 2: search in neighbors
         * Step 3: save information on whether the particle was located
         * Step 4: run additional loop on non-located particles
         */

        Kokkos::parallel_scan(
            "ParticleSpatialLayout::locateParticles()",
            Kokkos::RangePolicy<size_t>(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_type i, increment_type& val, const bool final) {
                /* Step 1
                * inCurr: True if the particle hasn't left the current MPI rank.
                * inNeighbor: True if the particle is found in a neighboring rank.
                * found: True either if inCurr = True or inNeighbor = True.
                * increment: Helper variable to update red_val.
                */
                bool inCurr = false;
                bool inNeighbor = false;
                bool found = false;
                bool increment[2];

                inCurr = positionInRegion(is, positions(i), Regions(myRank)); 

                ranks(i)     = inCurr * myRank;
                invalid(i)   = !inCurr;
                found =  inCurr || found;

                /// Step 2
                for (size_t j = 0; j < neighbors_view.extent(0); ++j) {
                    size_type rank = neighbors_view(j);

                    inNeighbor = positionInRegion(is, positions(i), Regions(rank));

                    ranks(i) = !(inNeighbor) * ranks(i) + inNeighbor * rank;
                    found =  inNeighbor || found;
                }

                /// Step 3
                /* isOut: When the last thread has finished the search, checks whether the particle has been found 
                 * either in the current rank or in a neighboring one.
                 * Used to avoid race conditions when updating outsideIds.
                 */
                bool isOut = (final && !found);

                outsideIds(val.count[1]) = i * isOut;
                increment[0] = invalid(i);
                increment[1] = !found;
                val += increment;

            },
            red_val);

        Kokkos::fence();

        invalidCount  = red_val.count[0];
        outsideCount  = red_val.count[1];

        /// Step 4 
        if (outsideCount > 0){
            static IpplTimings::TimerRef nonNeighboringParticles =
                IpplTimings::getTimer("nonNeighboringParticles");
            IpplTimings::startTimer(nonNeighboringParticles);

            Kokkos::parallel_for(
                "ParticleSpatialLayout::leftParticles()",
                mdrange_type({0, 0}, {outsideCount, Regions.extent(0)}),
                KOKKOS_LAMBDA(const size_t i, const size_type j) {
                    /// pID: (local) ID of the particle that is currently being searched.
                    size_type pId = outsideIds(i);
                    
                    /// inRegion: Checks whether particle pID is inside region j.
                    bool inRegion = positionInRegion(is, positions(pId), Regions(j));
                    ranks(pId) = inRegion * j + !(inRegion) * ranks(pId);
            
                });
            Kokkos::fence();

            IpplTimings::stopTimer(nonNeighboringParticles);
        }

        return invalidCount;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::fillHash(int rank,
                                                                      const locate_type& ranks,
                                                                      hash_type& hash) {
        /* Compute the prefix sum and fill the hash
         */
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::fillHash()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final) {
                    if (rank == ranks(i)) {
                        hash(idx) = i;
                    }
                }

                if (rank == ranks(i)) {
                    idx += 1;
                }
            });
        Kokkos::fence();

    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    size_t ParticleSpatialLayout<T, Dim, Mesh, Properties...>::numberOfSends(
        int rank, const locate_type& ranks) {
        size_t nSends     = 0;
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::numberOfSends()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, size_t& num) { num += size_t(rank == ranks(i)); },
            nSends);
        Kokkos::fence();
        return nSends;
    }

}  // namespace ippl
