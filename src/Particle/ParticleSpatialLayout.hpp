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
     * We need this struct since Kokkos parallel_scan only accepts
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
        , flayout_m(fl)
    {   
        nRecvs_m.resize(Comm->size());
        if (Comm->size() > 1) {
            window_m.create(*Comm, nRecvs_m.begin(), nRecvs_m.end());
        }
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::updateLayout(FieldLayout<Dim>& fl,
                                                                          Mesh& mesh) {
        //flayout_m = fl;
        rlayout_m.changeDomain(fl, mesh);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialLayout<T, Dim, Mesh, Properties...>::update(ParticleContainer& pc) {

        /* Apply Boundary Conditions */
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        this->applyBC(pc.R, rlayout_m.getDomain());
        IpplTimings::stopTimer(ParticleBCTimer);

        /* Update Timer for the rest of the function */
        static IpplTimings::TimerRef ParticleUpdateTimer = IpplTimings::getTimer("updateParticle");
        IpplTimings::startTimer(ParticleUpdateTimer);
        
        int nRanks = Comm->size();
        if (nRanks < 2) {
            return;
        }

        /* particle MPI exchange:
         *   1. figure out which particles need to go where -> locateParticles(...)
         *   2. fill send buffer and send particles
         *   3. delete invalidated particles
         *   4. receive particles
         */

        // 1.  figure out which particles need to go where -> locateParticles(...) ============= //
        
        static IpplTimings::TimerRef locateTimer = IpplTimings::getTimer("locateParticles");
        IpplTimings::startTimer(locateTimer);

        /* Rank-local number of particles */
        size_type localnum = pc.getLocalNum();

        /* The indices correspond to the indices of the local particles,
         * the values correspond to the ranks to which the particles need to be sent
         */        
        locate_type particleRanks("particles' MPI ranks", localnum);

        /* The indices are the indices of the particles,
         * the boolean values describe whether the particle has left the current rank 
         * 0 --> particle valid (inside current rank)
         * 1 --> particle invalid (left rank)
         */
        bool_type invalidParticles("validity of particles", localnum);

        /* The indices are the MPI ranks,
         * the values are the number of particles are sent to that rank from myrank 
         */
        locate_type rankSendCount_dview("rankSendCount Device", nRanks);

        /* The indices have no particluar meaning, 
         * the values are the MPI ranks to which we need to send
         */
        locate_type destinationRanks_dview("destinationRanks Device", nRanks);

        /* nInvalid is the number of invalid particles
         * nDestinationRanks is the number of MPI ranks we need to send to
         */ 
        auto [nInvalid, nDestinationRanks] = 
            locateParticles(pc, 
                particleRanks, invalidParticles, 
                rankSendCount_dview, destinationRanks_dview);

        /* Host space copy of rankSendCount_dview */
        auto rankSendCount_hview = 
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rankSendCount_dview);
        
        /* Host Space copy of destinationRanks_dview */
        auto destinationRanks_hview = 
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), destinationRanks_dview);

        IpplTimings::stopTimer(locateTimer);

        // 2. fill send buffer and send particles =============================================== //

        // 2.1 Remote Memory Access window for one-sided communication
        
        static IpplTimings::TimerRef preprocTimer = IpplTimings::getTimer("sendPreprocess");
        IpplTimings::startTimer(preprocTimer);
       
        std::fill(nRecvs_m.begin(), nRecvs_m.end(), 0); 

        window_m.fence(0);
        
        // Prepare RMA window for the ranks we need to send to  
        for(size_t ridx=0; ridx < nDestinationRanks; ridx++){
            int rank = destinationRanks_hview[ridx];
            if (rank == Comm->rank()){
                // we do not need to send to ourselves
                continue;
            }
            const int* src_ptr = &rankSendCount_hview(rank);
            window_m.put<int>(src_ptr, rank, Comm->rank());
        }
        window_m.fence(0);

        IpplTimings::stopTimer(preprocTimer);

        // 2.2 Particle Sends 

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        IpplTimings::startTimer(sendTimer);
        
        std::vector<MPI_Request> requests(0);

        int tag = Comm->next_tag(mpi::tag::P_SPATIAL_LAYOUT, mpi::tag::P_LAYOUT_CYCLE);

        for(size_t ridx=0; ridx < nDestinationRanks; ridx++){
            int rank = destinationRanks_hview[ridx];
            if(rank == Comm->rank()){
               continue;
            } 
            hash_type hash("hash", rankSendCount_hview(rank));
            fillHash(rank, particleRanks, hash);
            pc.sendToRank(rank, tag, requests, hash);
        }
       
        IpplTimings::stopTimer(sendTimer);


        // 3. Internal destruction of invalid particles ======================================= //
        
        static IpplTimings::TimerRef destroyTimer = IpplTimings::getTimer("particleDestroy");
        IpplTimings::startTimer(destroyTimer);

        pc.internalDestroy(invalidParticles, nInvalid);
        Kokkos::fence();

        IpplTimings::stopTimer(destroyTimer);
        
        // 4. Receive Particles ================================================================ // 
        
        static IpplTimings::TimerRef recvTimer = IpplTimings::getTimer("particleRecv");
        IpplTimings::startTimer(recvTimer);

        for (int rank = 0; rank < nRanks; ++rank) {
            if (nRecvs_m[rank] > 0) {
                pc.recvFromRank(rank, tag, nRecvs_m[rank]);
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
        return ((pos[Idx] > region[Idx].min()) && ...) && ((pos[Idx] <= region[Idx].max()) && ...);
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


    /**
     * @brief This function determines to which rank particles need to be sent after the iteration step.
     *        It starts by first scanning direct rank neighbors, and only does a global scan if there are still 
     *        unfound particles. It then calculates how many particles need to be sent to each rank and how many 
     *        ranks are sent to in total.
     *
     * @param pc           Particle Container
     * @param ranks        A vector the length of the number of particles on the current rank, where each value refers
     *                      to the new rank of the particle
     * @param invalid      A vector marking the particles that need to be sent away, and thus locally deleted
     * @param nSends_dview Device view the length of number of ranks, where each value determines the number 
     *                      of particles sent to that rank from the current rank
     * @param sends_dview  Device view for the number of ranks that are sent to from current rank
     *
     * @return tuple with the number of particles sent away and the number of ranks sent to
     */
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <typename ParticleContainer>
    std::pair<detail::size_type, detail::size_type> ParticleSpatialLayout<T, Dim, Mesh, Properties...>::locateParticles(
        const ParticleContainer& pc, locate_type& ranks, bool_type& invalid, 
        locate_type& nSends_dview, locate_type& sends_dview) const {

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
                //std::cout << "Neighbor: " << neighbors_mirror(k) << std::endl;
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
        static IpplTimings::TimerRef neighborSearch = IpplTimings::getTimer("neighborSearch");
        IpplTimings::startTimer(neighborSearch);

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
                if(final && !found) {
                    outsideIds(val.count[1]) = i;
                }
                //outsideIds(val.count[1]) = i * isOut;
                increment[0] = invalid(i);
                increment[1] = !found;
                val += increment;

            },
            red_val);

        Kokkos::fence();

        invalidCount  = red_val.count[0];
        outsideCount  = red_val.count[1];

        IpplTimings::stopTimer(neighborSearch);

        /// Step 4 
        static IpplTimings::TimerRef nonNeighboringParticles = IpplTimings::getTimer("nonNeighboringParticles");
        IpplTimings::startTimer(nonNeighboringParticles);
        if (outsideCount > 0) {
            Kokkos::parallel_for(
                "ParticleSpatialLayout::leftParticles()",
                mdrange_type({0, 0}, {outsideCount, Regions.extent(0)}),
                KOKKOS_LAMBDA(const size_t i, const size_type j) {
                    /// pID: (local) ID of the particle that is currently being searched.
                    size_type pId = outsideIds(i);
                    
                    /// inRegion: Checks whether particle pID is inside region j.
                    bool inRegion = positionInRegion(is, positions(pId), Regions(j));
                    if(inRegion){
                        ranks(pId) = j;
                    } 
                });
            Kokkos::fence();
        }
        IpplTimings::stopTimer(nonNeighboringParticles);
        
        Kokkos::parallel_for("Calculate nSends",
            Kokkos::RangePolicy<size_t>(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i){
                size_type rank = ranks(i); 
                Kokkos::atomic_fetch_add(&nSends_dview(rank),1);
            }
        ); 
    
        // Number of Ranks we need to send to 
        Kokkos::View<size_type> rankSends("Number of Ranks we need to send to");
        
        Kokkos::parallel_for("Calculate sends",
               Kokkos::RangePolicy<size_t>(0, nSends_dview.extent(0)),
               KOKKOS_LAMBDA(const size_t rank){
                   if(nSends_dview(rank) != 0){
                       size_type index = Kokkos::atomic_fetch_add(&rankSends(), 1);
                       sends_dview(index) = rank;
                   }
               }
        );
        size_type temp;
        Kokkos::deep_copy(temp, rankSends);

        return {invalidCount, temp};
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
