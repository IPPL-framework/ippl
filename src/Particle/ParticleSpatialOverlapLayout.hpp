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
#include <numeric>
#include <vector>

#include "Utility/IpplTimings.h"

#include "../../alpine/ParticleContainer.hpp"
#include "Communicate/Window.h"

namespace ippl {
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::ParticleSpatialOverlapLayout(
        FieldLayout<Dim>& fl, Mesh& mesh, const T& rcutoff)
        : Base(fl, mesh)
        , rcutoff_m(rcutoff)
        , numLocalParticles_m(0) {
        initializeCells();
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::updateLayout(
        FieldLayout<Dim>& fl, Mesh& mesh) {
        Base::updateLayout(fl, mesh);
        initializeCells();
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::initializeCells() {
        const auto rank          = Comm->rank();
        const auto hLocalRegions = this->rlayout_m.gethLocalRegions();
        for (unsigned d = 0; d < Dim; ++d) {
            PAssert(rcutoff_m <= hLocalRegions(rank)[d].length() / 2 &&
                "Cutoff is too big with respect to region. "
                "Particle could be on 3 or more ranks ins one dimension");
        }

        /* precompute information of cell structure. dividing the region into cells of at least
         * rcutoff_m length, the length of the overlap. Use std::floor to make sure the boundary
         * cells are big enough as well.
         */
        totalCells_m    = 1;
        numLocalCells_m = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            const T length              = hLocalRegions(rank)[d].length();
            const size_type nLocalCells = std::floor(length / rcutoff_m);
            // two ghost cells, one in each direction
            numCells_m[d]  = nLocalCells + 2 * numGhostCellsPerDim_m;
            cellWidth_m[d] = length / nLocalCells;
            totalCells_m *= numCells_m[d];
            numLocalCells_m *= nLocalCells;
        }
        numGhostCells_m = totalCells_m - numLocalCells_m;

        /* calculate cell strides to compute flat index from nd-index
         * idx = strides[0] * idx0 + strides[1] * idx1 + ...
         */
        std::exclusive_scan(numCells_m.begin(), numCells_m.end(), cellStrides_m.begin(), 1,
                            std::multiplies());

        cellParticleCount_m = hash_type("cellParticleCount", totalCells_m);
        cellStartingIdx_m   = hash_type("cellStartingIdx", totalCells_m + 1);

        // Compute cell permutation need to place ghost cells at the end to make sure only local
        // particles are at indices 0 to numLocalParticles
        hash_type cellPermutationForward("cell permutation forward", totalCells_m);
        hash_type cellPermutationBackward("cell permutation backward", totalCells_m);

        /* Step 1. compute prefix sum to determine where the local/ghost cells go */
        hash_type localPrefixSum("local prefix sum", totalCells_m);
        hash_type ghostPrefixSum("ghost prefix sum", totalCells_m);
        const auto& numCells = numCells_m;
        constexpr auto is    = std::make_index_sequence<Dim>();

        Kokkos::parallel_scan(
            "scan_local", Kokkos::RangePolicy(0, totalCells_m),
            KOKKOS_LAMBDA(const size_type i, size_type& update, const bool final) {
                const size_type val = isLocalCellIndex(is, toCellIndex(i, numCells), numCells);
                if (final) {
                    localPrefixSum(i) = update;
                }
                update += val;
            });
        Kokkos::parallel_scan(
            "scan_ghost", Kokkos::RangePolicy(0, totalCells_m),
            KOKKOS_LAMBDA(const size_type i, size_type& update, const bool final) {
                const size_type val = !isLocalCellIndex(is, toCellIndex(i, numCells), numCells);
                if (final) {
                    ghostPrefixSum(i) = update;
                }
                update += val;
            });

        /* Step 2. assign the cells at the correct locations of the permutations */
        const auto numLocalCells = numLocalCells_m;
        Kokkos::parallel_for(
            "assign_permutations", Kokkos::RangePolicy(0, totalCells_m),
            KOKKOS_LAMBDA(const size_type i) {
                if (const auto cellIdx = toCellIndex(i, numCells);
                    isLocalCellIndex(is, cellIdx, numCells)) {
                    const size_type localIdx          = localPrefixSum(i);
                    cellPermutationForward(i)         = localIdx;
                    cellPermutationBackward(localIdx) = i;
                } else {
                    const size_type ghostIdx          = numLocalCells + ghostPrefixSum(i);
                    cellPermutationForward(i)         = ghostIdx;
                    cellPermutationBackward(ghostIdx) = i;
                }
            });

        cellPermutationForward_m  = cellPermutationForward;
        cellPermutationBackward_m = cellPermutationBackward;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <std::size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::isCloseToBoundary(
        const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& globalRegion,
        Vector<bool, Dim> periodic, T overlap) {
        return ((periodic[Idx]
                 && (pos[Idx] < globalRegion[Idx].min() + overlap
                     || pos[Idx] > globalRegion[Idx].max() - overlap))
                || ...);
    }

    namespace detail {

        template <typename ParticleContainer, typename index_type>
        inline void copyAttributes(ParticleContainer& pc, const index_type& boundaryIndices) {
            const auto numLoc               = pc.getLocalNum();
            const auto numBoundaryParticles = boundaryIndices.size();
            detail::runForAllSpaces([&]<typename MemorySpace>() {
                size_t numAttributesInSpace = 0;
                pc.template forAllAttributes<MemorySpace>(
                    [&numAttributesInSpace]<typename Attribute>(Attribute&) {
                        ++numAttributesInSpace;
                    });
                if (numAttributesInSpace == 0) {
                    return;
                }

                pc.template forAllAttributes<MemorySpace>(
                    [boundaryIndicesMirror = Kokkos::create_mirror_view_and_copy(
                         MemorySpace(), boundaryIndices)]<typename Attribute>(Attribute& att) {
                        att->internalCopy(boundaryIndicesMirror);
                    });
            });
            Kokkos::fence();
            /* make sure other functions (particleExchange and buildCells) know about the ghost
             * particles. They will not stay pc's particle as their positions are outside the global
             * domain but are needed as ghost particles.
             */
            pc.setLocalNum(numLoc + numBoundaryParticles);
        }
    }  // namespace detail

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::createPeriodicGhostParticles(
        ParticleContainer& pc) {
        static IpplTimings::TimerRef timer = IpplTimings::getTimer("createPeriodicGhostParticles");
        IpplTimings::startTimer(timer);
        /* periodic boundary conditions come in pairs. Thus collect whether each dimension is
         * subject to periodic boundary conditions
         */
        Vector<bool, Dim> periodic;
        for (unsigned d = 0; d < Dim; ++d) {
            periodic[d] = this->getParticleBC()[2 * d];
        }
        // no need to create periodic ghost particles if no dimension is periodic
        if (!std::any_of(periodic.begin(), periodic.end(), [](auto bc) {
                return bc == BC::PERIODIC;
            })) {
            return;
        }

        const auto& globalRegion = this->rlayout_m.getDomain();
        const auto overlap       = rcutoff_m;
        const auto numLoc        = pc.getLocalNum();
        const auto positions     = pc.R.getView();

        constexpr auto is = std::make_index_sequence<Dim>();
        /* Step 1. Determine all particles which are close to the global domain boundary */
        size_type numBoundaryParticles = 0;
        Kokkos::parallel_reduce(
            "count boundary particles", numLoc,
            KOKKOS_LAMBDA(const size_t& i, size_type& sum) {
                if (isCloseToBoundary(is, positions(i), globalRegion, periodic, overlap)) {
                    ++sum;
                }
            },
            Kokkos::Sum<size_type>(numBoundaryParticles));
        if (numBoundaryParticles == 0) {
            return;
        }

        /* Step 2. compute prefix sum to collect boundary particle indices */
        hash_type boundaryIndices("boundaryIndices", numBoundaryParticles);
        Kokkos::parallel_scan(
            "count boundary particles", numLoc,
            KOKKOS_LAMBDA(const size_t& i, size_type& sum, bool final) {
                if (isCloseToBoundary(is, positions(i), globalRegion, periodic, overlap)) {
                    if (final) {
                        boundaryIndices(sum) = i;
                    }
                    ++sum;
                }
            });

        /* Step 3. copy given particles and all its attributes. A separate function is needed as
         * lambdas with captures do not work with nvcc and template default argument ot the layout
         * somehow. NOTE numLoc will change after this call.
         */
        detail::copyAttributes(pc, boundaryIndices);

        /* Step 4. set the position of the copied particles to their periodic image */
        for (unsigned d = 0; d < Dim; ++d) {
            if (!periodic[d]) {
                continue;
            }
            const auto min    = globalRegion[d].min();
            const auto length = globalRegion[d].length();
            const auto middle = min + length / 2;
            Kokkos::parallel_for(
                "correct positions",
                // the copied particles are appended at the end of the attributes
                Kokkos::RangePolicy<position_execution_space>(numLoc,
                                                              numLoc + numBoundaryParticles),
                KOKKOS_LAMBDA(const size_t& i) {
                    positions(i)[d] += positions(i)[d] > middle ? -length : length;
                });
        }

        IpplTimings::stopTimer(timer);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::particleExchange(
        ParticleContainer& pc) {
        /* Apply Boundary Conditions */
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        this->applyBC(pc.R, this->rlayout_m.getDomain());
        createPeriodicGhostParticles(pc);
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

        /* particleRanks are the indices correspond to the indices of the local particles,
         * the values correspond to the ranks to which the particles need to be sent note that the
         * size is not known yet as particles may belong to multiple ranks
         * particleRankOffsets are the offsets of each particle as a particle
         * can be sent to multiple ranks
         */
        locate_type particleRanks("particles' MPI ranks");
        locate_type particleRankOffsets("particles' MPI rank offsets", localnum + 1);

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

        /* The indices have no particular meaning,
         * the values are the MPI ranks to which we need to send
         */
        locate_type destinationRanks_dview("destinationRanks Device", nRanks);

        /* nInvalid is the number of invalid particles
         * nDestinationRanks is the number of MPI ranks we need to send to
         */
        const auto [nInvalid, nDestinationRanks] =
            locateParticles(pc, particleRanks, particleRankOffsets, invalidParticles,
                            rankSendCount_dview, destinationRanks_dview);

        /* Host space copy of rankSendCount_dview */
        auto rankSendCount_hview =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rankSendCount_dview);

        /* Host Space copy of destinationRanks_dview */
        auto destinationRanks_hview =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), destinationRanks_dview);
        Kokkos::fence();

        IpplTimings::stopTimer(locateTimer);

        // 2. fill send buffer and send particles =============================================== //

        // 2.1 Remote Memory Access window for one-sided communication

        static IpplTimings::TimerRef preprocTimer = IpplTimings::getTimer("sendPreprocess");
        IpplTimings::startTimer(preprocTimer);

        std::fill(this->nRecvs_m.begin(), this->nRecvs_m.end(), 0);

        this->window_m.fence(0);

        // Prepare RMA window for the ranks we need to send to
        for (size_t ridx = 0; ridx < nDestinationRanks; ridx++) {
            int rank = destinationRanks_hview[ridx];
            if (rank == Comm->rank()) {
                // we do not need to send to ourselves
                continue;
            }
            const int* src_ptr = &rankSendCount_hview(rank);
            this->window_m.template put<int>(src_ptr, rank, Comm->rank());
        }
        this->window_m.fence(0);

        IpplTimings::stopTimer(preprocTimer);

        // 2.2 Particle Sends

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        IpplTimings::startTimer(sendTimer);

        std::vector<MPI_Request> requests(0);

        int tag = Comm->next_tag(mpi::tag::P_SPATIAL_LAYOUT, mpi::tag::P_LAYOUT_CYCLE);

        for (size_t ridx = 0; ridx < nDestinationRanks; ridx++) {
            int rank = destinationRanks_hview[ridx];
            if (rank == Comm->rank()) {
                continue;
            }
            hash_type hash("hash", rankSendCount_hview(rank));
            fillHash(rank, particleRanks, particleRankOffsets, hash);
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
            if (this->nRecvs_m[rank] > 0) {
                pc.recvFromRank(rank, tag, this->nRecvs_m[rank]);
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
    template <class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::update(ParticleContainer& pc) {
        particleExchange(pc);
        buildCells(pc);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    detail::size_type ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::numberOfSends(
        int rank, const locate_type& ranks) {
        /* the offsets are not required as it is only important how many particles go to each rank
         * and not which particles
         */
        size_t nSends     = 0;
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::numberOfSends()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, size_t& num) {
                num += static_cast<size_t>(rank == ranks(i));
            },
            nSends);
        Kokkos::fence();
        return nSends;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::fillHash(
        int rank, const locate_type& ranks, const locate_type& offsets, hash_type& hash) {
        /* Compute the prefix sum and fill the hash
         */
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::fillHash()", policy_type(0, offsets.extent(0) - 1),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                /* Check if this particle belongs to our target rank. Ranks of particle i are stored
                 * from offsets(i) to offsets(i + 1)
                 */
                bool belongs_to_rank        = false;
                const size_t start_rank_idx = offsets(i);
                const size_t end_rank_idx   = offsets(i + 1);

                for (size_t rank_idx = start_rank_idx; rank_idx < end_rank_idx; ++rank_idx) {
                    if (ranks(rank_idx) == rank) {
                        belongs_to_rank = true;
                        break;  // Found it, no need to continue
                    }
                }

                if (final && belongs_to_rank) {
                    hash(idx) = i;
                }

                if (belongs_to_rank) {
                    idx += 1;
                }
            });
        Kokkos::fence();
    }

    /* Helper function to fill a view with neighbor ranks
     */
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::locate_type
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getFlatNeighbors(
        const neighbor_list& neighbors) const {
        /* collect all neighbors in a set. set instead of vector as with small number of ranks and
         * periodic boundary conditions the same rank can be in the neighbor list multiple times.
         */
        std::set<int> neighborSet;
        for (const auto& componentNeighbors : neighbors) {
            for (const auto& nrank : componentNeighbors) {
                neighborSet.insert(nrank);
            }
        }

        // copy neighbors into view
        locate_type flatNeighbors("Nearest neighbors IDs", neighborSet.size());
        auto hostMirror = Kokkos::create_mirror_view(flatNeighbors);

        size_type i = 0;
        for (const auto& neighbor : neighborSet) {
            hostMirror(i) = neighbor;
            ++i;
        }

        Kokkos::deep_copy(flatNeighbors, hostMirror);
        Kokkos::fence();
        return flatNeighbors;
    }

    /* Helper function to get non-neighboring ranks
     */
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::locate_type
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getNonNeighborRanks(
        const locate_type& neighbors_view) const {
        // Create a view of all non-neighbor ranks. make sure to exclude own its rank
        const auto numNonNeighborRanks = Comm->size() - neighbors_view.extent(0) - 1;
        locate_type nonNeighborRanks("Non Neighbor Ranks", numNonNeighborRanks);
        if (numNonNeighborRanks == 0) {
            return nonNeighborRanks;
        }

        // Step 1. Mark all non-neighbor ranks, by removing all neighbors and self
        const auto total_ranks = Comm->rank();
        bool_type is_remaining("is_remaining", total_ranks);
        Kokkos::deep_copy(is_remaining, true);
        Kokkos::fence();

        const auto myRank = Comm->rank();
        Kokkos::parallel_for(
            "mark_comm_ranks", Kokkos::RangePolicy(myRank, myRank + 1),
            KOKKOS_LAMBDA(const size_t& i) { is_remaining(i) = false; });
        Kokkos::parallel_for(
            "mark_comm_ranks", neighbors_view.extent(0),
            KOKKOS_LAMBDA(const size_t& i) { is_remaining(neighbors_view(i)) = false; });
        Kokkos::fence();

        // Step 2. Fill remaining ranks
        Kokkos::View<size_type> counter("counter");
        Kokkos::parallel_for(
            "fill_remaining", total_ranks, KOKKOS_LAMBDA(const size_t& i) {
                if (is_remaining(i)) {
                    const size_type idx   = Kokkos::atomic_fetch_inc(&counter());
                    nonNeighborRanks(idx) = i;
                }
            });
        return nonNeighborRanks;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <typename ParticleContainer>
    std::pair<detail::size_type, detail::size_type>
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::locateParticles(
        const ParticleContainer& pc, locate_type& ranks, locate_type& rankOffsets,
        bool_type& invalid, locate_type& nSends_dview, locate_type& sends_dview) const {
        const auto positions = pc.R.getView();
        const auto regions   = this->rlayout_m.getdLocalRegions();
        const auto myRank    = Comm->rank();
        const auto localNum  = pc.getLocalNum();
        const T overlap      = rcutoff_m;
        constexpr auto is    = std::make_index_sequence<Dim>();

        /// outsideIds: Container of particle IDs that travelled outside of the neighborhood.
        /// counts: count assignments per particle
        locate_type counts("counts", localNum);
        locate_type outsideIds("Particles outside of neighborhood", localNum);

        /// outsideCount: Tracks the number of particles that travelled outside of the neighborhood.
        size_type outsideCount = 0;
        /// invalidCount: Tracks the number of particles that need to be sent to other ranks.
        size_type invalidCount = 0;

        /// neighbors_view: Kokkos view with the IDs of the neighboring MPI ranks.
        locate_type neighbors_view = getFlatNeighbors(this->flayout_m.getNeighbors());

        /* red_val: Used to reduce both the number of invalid particles and the number of particles
         * outside the neighborhood (Kokkos::parallel_scan doesn't allow multiple reduction
         * values, so we use the helper class increment_type). First element updates InvalidCount,
         * second one updates outsideCount.
         */
        increment_type red_val;
        red_val.init();

        /*! Begin Kokkos loop:
         * Step 1: search in current rank
         * Step 2: search in neighbors
         * Step 3: save information on whether the particle was located
         * Step 4: run additional loop on non-located particles
         */
        static IpplTimings::TimerRef neighborSearch = IpplTimings::getTimer("neighborSearch");
        IpplTimings::startTimer(neighborSearch);

        /* First Pass: count the numbers of neighbor ranks (including self) a particle belongs to */
        Kokkos::parallel_for(
            "ParticleSpatialLayout::locateParticles()", localNum, KOKKOS_LAMBDA(const size_t& i) {
                const bool inCurr = positionInRegion(is, positions(i), regions(myRank), overlap);

                size_type count = inCurr;
                invalid(i)      = !inCurr;
                // Count neighboring regions
                for (size_type j = 0; j < neighbors_view.extent(0); ++j) {
                    const size_type rank = neighbors_view(j);
                    if (positionInRegion(is, positions(i), regions(rank), overlap)) {
                        ++count;
                    }
                }

                counts(i) = count;
            });
        Kokkos::fence();

        /* Second Pass: collect number of particles outside this ranks region and the indices of the
         * respective particles. Note that in comparison to ParticleSpatialLayout::locateParticles()
         * particle can be in multiple ranks. For this reason if a particle is in a neighbor but not
         * here, then we still need to search in all ranks as there is no way to get second
         * neighbors.
         */
        Kokkos::parallel_scan(
            "count_outside", localNum,
            KOKKOS_LAMBDA(const size_type i, increment_type& val, const bool final) {
                const bool inCurr = !invalid(i);
                if (final && !inCurr) {
                    outsideIds(val.count[1]) = i;
                }
                bool increment[2] = {invalid(i), !inCurr};
                val += increment;
            },
            red_val);
        Kokkos::fence();

        invalidCount = red_val.count[0];
        outsideCount = red_val.count[1];

        IpplTimings::stopTimer(neighborSearch);

        /// Step 4
        static IpplTimings::TimerRef nonNeighboringParticles =
            IpplTimings::getTimer("nonNeighboringParticles");
        IpplTimings::startTimer(nonNeighboringParticles);

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;

        /* Count the number of non-neighbor ranks each particle belongs to */
        locate_type outsideCounts("counts of outside neighbors", outsideCount);
        locate_type nonNeighborsView = getNonNeighborRanks(neighbors_view);
        if (outsideCount > 0 && nonNeighborsView.extent(0) > 0) {
            Kokkos::deep_copy(outsideCounts, 0);
            Kokkos::parallel_for(
                "ParticleSpatialLayout::leftParticles()",
                mdrange_type({0, 0}, {outsideCount, nonNeighborsView.extent(0)}),
                KOKKOS_LAMBDA(const size_t i, const size_type j) {
                    /// pID: (local) ID of the particle that is currently being searched.
                    const size_type pId = outsideIds(i);
                    const auto rank     = nonNeighborsView(j);

                    /// inRegion: Checks whether particle pID is inside region j.
                    if (positionInRegion(is, positions(pId), regions(rank), overlap)) {
                        Kokkos::atomic_increment(&outsideCounts(i));
                    }
                });
            Kokkos::fence();
            Kokkos::parallel_for(
                "ParticleSpatialLayout::leftParticles()", outsideCount,
                KOKKOS_LAMBDA(const size_t& i) { counts(outsideIds(i)) += outsideCounts(i); });
            Kokkos::fence();
        }
        IpplTimings::stopTimer(nonNeighboringParticles);

        IpplTimings::startTimer(neighborSearch);
        /* prefix sum for particle rank offsets */
        Kokkos::deep_copy(Kokkos::subview(rankOffsets, 0), 0);
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::locateParticles()", localNum,
            KOKKOS_LAMBDA(const size_t i, size_type& localSum, const bool final) {
                const auto count_i = counts(i);
                if (final) {
                    rankOffsets(i + 1) = localSum + count_i;
                }
                localSum += count_i;
            });
        Kokkos::fence();

        /* Get total number of assignments for allocation from the last entry of offsets */
        auto total_assignments = Kokkos::create_mirror_view(Kokkos::subview(rankOffsets, localNum));
        Kokkos::deep_copy(total_assignments, Kokkos::subview(rankOffsets, localNum));
        Kokkos::fence();
        Kokkos::resize(ranks, total_assignments());

        /* Last Pass: fill the rank data */
        Kokkos::parallel_for(
            "ParticleSpatialLayout::locateParticles()", localNum, KOKKOS_LAMBDA(const size_t& i) {
                const size_t offset   = rankOffsets(i);
                size_type local_count = 0;
                if (positionInRegion(is, positions(i), regions(myRank), overlap)) {
                    ranks(offset) = myRank;
                    local_count++;
                }
                for (size_t j = 0; j < neighbors_view.extent(0); ++j) {
                    const auto nRank = neighbors_view(j);
                    if (positionInRegion(is, positions(i), regions(nRank), overlap)) {
                        ranks(offset + local_count) = nRank;
                        local_count++;
                    }
                }
            });
        Kokkos::fence();

        IpplTimings::stopTimer(neighborSearch);

        /* Last Pass: add the data of the outside particles to the ranks */
        IpplTimings::startTimer(nonNeighboringParticles);
        if (outsideCount > 0) {
            Kokkos::parallel_for(
                "ParticleSpatialLayout::leftParticles()", outsideCount,
                KOKKOS_LAMBDA(const size_t& i) {
                    /// pID: (local) ID of the particle that is currently being searched.
                    const size_type pId    = outsideIds(i);
                    const size_type offset = rankOffsets(pId) + counts(pId);
                    for (size_t local_count = 0, j = 0; j < nonNeighborsView.extent(0); ++j) {
                        const auto rank = nonNeighborsView(j);
                        if (positionInRegion(is, positions(i), regions(rank), overlap)) {
                            ranks(offset + local_count) = rank;
                            local_count++;
                        }
                    }
                });
            Kokkos::fence();
        }
        IpplTimings::stopTimer(nonNeighboringParticles);

        /* compute the number of sends to all ranks */
        Kokkos::deep_copy(nSends_dview, 0);
        Kokkos::parallel_for(
            "Calculate nSends", Kokkos::RangePolicy<size_t>(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i) {
                size_type rank = ranks(i);
                Kokkos::atomic_increment(&nSends_dview(rank));
            });
        Kokkos::fence();

        /* compute the ranks to send to and the number of ranks to send to*/
        Kokkos::View<size_type> rankSends("Number of Ranks we need to send to");
        Kokkos::parallel_for(
            "Calculate sends", Kokkos::RangePolicy<size_t>(0, nSends_dview.extent(0)),
            KOKKOS_LAMBDA(const size_t rank) {
                if (nSends_dview(rank) != 0) {
                    size_type index    = Kokkos::atomic_fetch_inc(&rankSends());
                    sends_dview(index) = rank;
                }
            });
        Kokkos::fence();
        size_type temp;
        Kokkos::deep_copy(temp, rankSends);
        Kokkos::fence();

        return {invalidCount, temp};
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <std::size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::positionInRegion(
        const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region,
        T overlap) {
        return ((pos[Idx] > region[Idx].min() - overlap) && ...)
               && ((pos[Idx] <= region[Idx].max() + overlap) && ...);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::FlatCellIndex_t
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::toFlatCellIndex(
            const CellIndex_t& cellIndex, const Vector<size_type, Dim>& cellStrides,
            hash_type cellPermutationForward) {
        return cellPermutationForward(cellIndex.dot(cellStrides));
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::CellIndex_t
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::toCellIndex(
            FlatCellIndex_t nonPermutedIndex, const Vector<size_type, Dim>& numCells) {
        CellIndex_t ndIndex;
        // #pragma unroll
        for (size_type d = 0; d < Dim; ++d) {
            ndIndex[d] = nonPermutedIndex % numCells[d];
            nonPermutedIndex /= numCells[d];
        }
        return ndIndex;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::CellIndex_t
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getCellIndex(
            const vector_type& pos, const region_type& region, const Vector<T, Dim>& cellWidth) {
        CellIndex_t cellIndex;
        for (unsigned d = 0; d < Dim; ++d) {
            cellIndex[d] = static_cast<size_type>(
                std::floor((pos[d] - region[d].min()) / cellWidth[d]) + numGhostCellsPerDim_m);
        }
        return cellIndex;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <std::size_t... Idx>
    constexpr bool ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::isLocalCellIndex(
        const std::index_sequence<Idx...>&, const CellIndex_t& index,
        const Vector<size_type, Dim>& numCells) {
        return !((index[Idx] == 0 || index[Idx] == numCells[Idx] - 1) || ...);
    }

    namespace detail {
        template <typename ParticleContainer, typename index_type>
        inline void sortParticles(ParticleContainer& pc, const index_type& newIndex) {
            detail::runForAllSpaces([&]<typename MemorySpace>() {
                size_t num_attributes_in_space = 0;
                pc.template forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute&) {
                    ++num_attributes_in_space;
                });
                if (num_attributes_in_space == 0) {
                    return;
                }

                pc.template forAllAttributes<MemorySpace>(
                    [newIndexMirror = Kokkos::create_mirror_view_and_copy(
                         MemorySpace(), newIndex)]<typename Attribute>(Attribute& att) {
                        att->applyPermutation(newIndexMirror);
                    });
            });
            Kokkos::fence();
        }
    }  // namespace detail

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::buildCells(
        ParticleContainer& pc) {
        static IpplTimings::TimerRef cellBuildTimer = IpplTimings::getTimer("cellBuildTimer");
        IpplTimings::startTimer(cellBuildTimer);

        // get local variables of all necessary data as needed for the Kokkos parallel loops
        const auto rank                   = Comm->rank();
        const size_type numLoc            = pc.getLocalNum();
        const auto positions              = pc.R.getView();
        const auto totalCells             = totalCells_m;
        const auto numLocalCells          = numLocalCells_m;
        const auto localRegion            = this->rlayout_m.gethLocalRegions()(rank);
        const auto& cellWidth             = cellWidth_m;
        const auto cellStrides            = cellStrides_m;
        const auto cellPermutationForward = cellPermutationForward_m;

        using int_type = typename particle_neighbor_list_type::value_type;

        // allocate required (temporary) Kokkos views
        hash_type cellIndex("cellIndex", numLoc);
        hash_type cellParticleCount = cellParticleCount_m;
        hash_type cellStartingIdx   = cellStartingIdx_m;
        hash_type cellCurrentIdx("cellCurrentIdx", totalCells + 1);

        using range_policy = Kokkos::RangePolicy<position_execution_space>;

        /* Step 1. calculate cell index for each particle and keep track of how many particles are
         * in each cell
         */
        Kokkos::deep_copy(cellParticleCount, 0);
        Kokkos::parallel_for(
            "CalcCellIndices", range_policy(0, numLoc), KOKKOS_LAMBDA(const size_t& i) {
                const auto locCellIndex = getCellIndex(positions(i), localRegion, cellWidth);
                const auto locCellIndexFlat =
                    toFlatCellIndex(locCellIndex, cellStrides, cellPermutationForward);
                Kokkos::atomic_increment(&cellParticleCount(locCellIndexFlat));
                cellIndex(i) = locCellIndexFlat;
            });
        Kokkos::fence();

        /* Step 2. compute starting indices for each cell from the counts */
        Kokkos::parallel_scan(
            "CalcStartingIndices", range_policy(0, totalCells),
            KOKKOS_LAMBDA(const size_t i, int_type& localSum, bool isFinal) {
                if (isFinal) {
                    cellStartingIdx(i) = localSum;
                }
                localSum += cellParticleCount(i);
            });
        /* set last position of cell staring index to numLoc*/
        Kokkos::deep_copy(
            Kokkos::subview(cellStartingIdx, Kokkos::make_pair(totalCells, totalCells + 1)),
            numLoc);
        Kokkos::fence();

        Kokkos::deep_copy(cellCurrentIdx, cellStartingIdx);
        Kokkos::fence();

        hash_type newIndex("newIndex", numLoc);
        hash_type newCellIndex("cellIndex", numLoc);

        /* Step 3. compute new indices for the particles such that they are sorted according to
         * cellStaringIdx and sort cell indices of the particles in tandem
         */
        Kokkos::parallel_for(
            "Calculate new Indices", range_policy(0, numLoc), KOKKOS_LAMBDA(const size_type& i) {
                const auto locCellIndex = cellIndex(i);
                const size_type newIdx  = Kokkos::atomic_fetch_inc(&cellCurrentIdx(locCellIndex));
                newIndex(i)             = newIdx;
                newCellIndex(newIdx)    = locCellIndex;
            });
        Kokkos::fence();

        /* Step 4. Sort all particles (and all their attributes) according to then new indices
         * (maybe there is a better solution). A separate function is needed as lambdas with
         * captures do not work with nvcc and template default argument ot the layout somehow.
         */
        detail::sortParticles(pc, newIndex);

        /* Step 5. set local number of particles (excluding ghost particles) is the value of
         * cellStartingIdx at index numLocalCells*/
        auto numLocalParticles =
            Kokkos::create_mirror_view(Kokkos::subview(cellStartingIdx, numLocalCells));
        Kokkos::deep_copy(numLocalParticles, Kokkos::subview(cellStartingIdx, numLocalCells));

        numLocalParticles_m = numLocalParticles();
        pc.setLocalNum(numLocalParticles_m);

        /* store the new cell indices */
        cellIndex_m = newCellIndex;

        IpplTimings::stopTimer(cellBuildTimer);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh,
                                              Properties...>::cell_particle_neighbor_list_type
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getCellNeighbors(
            const CellIndex_t& cellIndex, const Vector<size_type, Dim>& cellStrides,
            const hash_type& cellPermutationForward) {
        /* Generate all 3^Dim combinations of offsets (-1, 0, +1) for each dimension by using
         * "base-3" representation of neighbor index. in base-3 each digit is 0, 1, 2 subtracting
         * one leads to the desired offsets -1, 0, +1.
         */
        constexpr size_type numNeighbors = detail::countHypercubes(Dim);
        cell_particle_neighbor_list_type neighborIndices;
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            index_t temp = neighborIdx;

            /* extract the offsets from the base-3 representation of the neighborIdx */
            auto neighborCellIndex = cellIndex;
            for (unsigned d = 0; d < Dim; ++d) {
                neighborCellIndex(d) += (temp % 3) - 1;
                temp /= 3;
            }

            neighborIndices[neighborIdx] =
                toFlatCellIndex(neighborCellIndex, cellStrides, cellPermutationForward);
        }
        return neighborIndices;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::ParticleNeighborData
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getParticleNeighborData() const {
        return ParticleNeighborData(numLocalParticles_m, cellStrides_m, numCells_m, cellWidth_m,
                                    this->rlayout_m.gethLocalRegions()(Comm->rank()),
                                    cellStartingIdx_m, cellIndex_m, cellParticleCount_m,
                                    cellPermutationForward_m, cellPermutationBackward_m);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_FUNCTION
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh,
                                              Properties...>::particle_neighbor_list_type
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getParticleNeighbors(
            const vector_type& pos, const ParticleNeighborData& particleNeighborData) {
        /* get the cell index corresponding to pos */
        const auto locCellIndex =
            getCellIndex(pos, particleNeighborData.region, particleNeighborData.cellStrides,
                         particleNeighborData.cellWidth);
        constexpr size_type numNeighbors = detail::countHypercubes(Dim);

        /* cell neighbors */
        const auto neighbors = getCellNeighbors(locCellIndex, particleNeighborData.cellStrides,
                                                particleNeighborData.cellPermutationForward);

        /* Get sizes of the cell neighbors and total particle neighbors */
        size_type totalParticleInNeighbors = 0;
        size_type maxParticleInNeighbors   = 0;

        Kokkos::Array<typename hash_type::value_type, numNeighbors> neighborSizes;
        Kokkos::Array<typename hash_type::value_type, numNeighbors> neighborOffsets;
        // #pragma unroll
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            auto n = particleNeighborData.cellParticleCount(neighbors[neighborIdx]);
            neighborSizes[neighborIdx]   = n;
            maxParticleInNeighbors       = std::max<size_type>(n, maxParticleInNeighbors);
            neighborOffsets[neighborIdx] = totalParticleInNeighbors;
            totalParticleInNeighbors += n;
        }

        /* Collect the neighbor particles from all cell neighbors */
        particle_neighbor_list_type neighborList("Neighbor list", totalParticleInNeighbors);

        using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        Kokkos::parallel_for(
            "collect neighbors", mdrange_policy({0, 0}, {numNeighbors, maxParticleInNeighbors}),
            KOKKOS_LAMBDA(const size_type& i, const size_type& j) {
                if (j < neighborSizes[i]) {
                    neighborList(neighborOffsets[i] + j) =
                        particleNeighborData.cellStartingIdx(neighbors[i]) + j;
                }
            });
        Kokkos::fence();

        return neighborList;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_FUNCTION
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh,
                                              Properties...>::particle_neighbor_list_type
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getParticleNeighbors(
            index_t particleIndex, const ParticleNeighborData& particleNeighborData) {
        // Get the cell of the particle
        constexpr size_type numNeighbors = detail::countHypercubes(Dim);

        /* get the cell index corresponding to particleIndex */
        const auto locCellIndexFlat = particleNeighborData.cellIndex(particleIndex);
        const auto locCellIndex =
            toCellIndex(particleNeighborData.cellPermutationBackward(locCellIndexFlat),
                        particleNeighborData.numCells);

        /* cell neighbors */
        const auto neighbors = getCellNeighbors(locCellIndex, particleNeighborData.cellStrides,
                                                particleNeighborData.cellPermutationForward);

        /* Get sizes of the cell neighbors and total particle neighbors */
        size_type totalParticleInNeighbors = 0;
        size_type maxParticleInNeighbors   = 0;

        Kokkos::Array<size_type, numNeighbors> neighborSizes;
        Kokkos::Array<size_type, numNeighbors + 1> neighborOffsets;
        // #pragma unroll
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            auto n                 = particleNeighborData.cellParticleCount(neighbors[neighborIdx]);
            maxParticleInNeighbors = std::max<size_type>(n, maxParticleInNeighbors);
            neighborOffsets[neighborIdx] = totalParticleInNeighbors;
            totalParticleInNeighbors += n;
        }
        neighborOffsets[numNeighbors] = totalParticleInNeighbors;

        /* Collect the neighbor particles from all cell neighbors */
        particle_neighbor_list_type neighborList("Neigbor list", totalParticleInNeighbors);

        using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        Kokkos::parallel_for(
            "collect neighbors", mdrange_policy({0, 0}, {numNeighbors, maxParticleInNeighbors}),
            KOKKOS_LAMBDA(const size_type& i, const size_type& j) {
                const auto numParticlesInCell = neighborOffsets[i + 1] - neighborOffsets[i];
                if (j < numParticlesInCell) {
                    neighborList(neighborOffsets[i] + j) =
                        particleNeighborData.cellStartingIdx(neighbors[i]) + j;
                }
            });

        Kokkos::fence();

        return neighborList;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <typename ExecutionSpace, typename Functor>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::forEachPair(Functor&& f) const {
        static IpplTimings::TimerRef interactionTimer = IpplTimings::getTimer("PPInteractionTimer");
        IpplTimings::startTimer(interactionTimer);

        /* get local variables necessary for Kokkos parallel regions */
        const auto cellStartingIdx         = cellStartingIdx_m;
        const auto cellParticleCount       = cellParticleCount_m;
        const auto cellPermutationForward  = cellPermutationForward_m;
        const auto cellPermutationBackward = cellPermutationBackward_m;
        const auto& cellStrides            = cellStrides_m;
        const auto& numCells               = numCells_m;

        constexpr auto numCellNeighbors = detail::countHypercubes(Dim);

        /* Iterate over all local cells (non-ghost cells) and let all particles of this cell
         * interact with all particles from all particles with all cell neighbors
         */
        using team_policy_t = Kokkos::TeamPolicy<ExecutionSpace>;
        using team_t        = typename team_policy_t::member_type;
        Kokkos::parallel_for(
            "ParticleSpatialOverlapLayout::forEachPair()",
            team_policy_t(numLocalCells_m, Kokkos::AUTO()), KOKKOS_LAMBDA(const team_t& team) {
                const size_type cellIdxFlat = team.league_rank();
                if (cellParticleCount(cellIdxFlat) == 0) {
                    return;
                }

                const auto cellParticleOffset = cellStartingIdx(cellIdxFlat);
                const auto numCellParticles   = cellParticleCount(cellIdxFlat);

                /* get nd-cell-index and its neighbors */
                const auto cellIdx = toCellIndex(cellPermutationBackward(cellIdxFlat), numCells);
                const auto cellNeighbors =
                    getCellNeighbors(cellIdx, cellStrides, cellPermutationForward);

                /* iterate over all cell neighbors */
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, numCellNeighbors), [&](const size_t& n) {
                        const auto neighborCellIdx            = cellNeighbors[n];
                        const auto neighborCellParticleOffset = cellStartingIdx(neighborCellIdx);
                        const auto numNeighborCellParticles   = cellParticleCount(neighborCellIdx);

                        /* iterate over all combinations of this cell and neighboring cells
                         * particles
                         */
                        Kokkos::parallel_for(Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(
                                                 team, numCellParticles, numNeighborCellParticles),
                                             [&](const size_t& i, const size_t& j) {
                                                 const auto particleIdx = cellParticleOffset + i;
                                                 const auto neighborIdx =
                                                     neighborCellParticleOffset + j;

                                                 if (neighborIdx == particleIdx) {
                                                     return;
                                                 }

                                                 f(particleIdx, neighborIdx);
                                             });
                    });
            });
        Kokkos::fence();

        IpplTimings::stopTimer(interactionTimer);
    }

}  // namespace ippl
