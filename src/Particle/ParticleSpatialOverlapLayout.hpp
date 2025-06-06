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
#include <memory>
#include <numeric>
#include <vector>

#include "Utility/IpplTimings.h"

#include "Communicate/Window.h"

namespace ippl {
    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::ParticleSpatialOverlapLayout(
        FieldLayout<Dim> &fl, Mesh &mesh, const T &rcutoff)
        : ParticleSpatialLayout<T, Dim, Properties...>(fl, mesh), rcutoff_m(rcutoff) {
        auto rank = Comm->rank();
        auto hLocalRegions = this->rlayout_m.gethLocalRegions();

        totalCells_m = 1;
        numLocalCells_m = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            const T length = hLocalRegions(rank)[d].length();
            const size_type nLocalCells = std::floor(length / rcutoff_m);
            numCells_m[d] = nLocalCells + 2 * numGhostCellsPerDim_m;
            // two ghost cells, one in each direction
            cellWidth_m[d] = length / nLocalCells;
            totalCells_m *= numCells_m[d];
            numLocalCells_m *= nLocalCells;
        }
        numGhostCells_m = totalCells_m - numLocalCells_m;


        cellPermutationForward_m = hash_type("cell permutation forward", totalCells_m);
        cellPermutationBackward_m = hash_type("cell permutation backward", totalCells_m);
        auto hostCellPermutationForward = Kokkos::create_mirror_view(cellPermutationForward_m);
        auto hostCellPermutationBackward = Kokkos::create_mirror_view(cellPermutationBackward_m);

        // TODO think about how to make this parallel without too many stalls
        size_type localIdx = 0, ghostIdx = numLocalCells_m;
        for (size_type i = 0; i < totalCells_m; ++i) {
            if (isLocalCellIndex(i, numCells_m)) {
                hostCellPermutationForward(i) = localIdx;
                hostCellPermutationBackward(localIdx) = i;
                ++localIdx;
            } else {
                hostCellPermutationForward(i) = ghostIdx;
                hostCellPermutationBackward(ghostIdx) = i;
                ++ghostIdx;
            }
        }
        assert(localIdx == numLocalCells_m);
        assert(ghostIdx == totalCells_m);
        Kokkos::deep_copy(cellPermutationForward_m, hostCellPermutationForward);
        Kokkos::deep_copy(cellPermutationBackward_m, hostCellPermutationBackward);

        std::exclusive_scan(numCells_m.begin(), numCells_m.end(), cellStrides_m.begin(), 1, std::multiplies());

        cellParticleCount_m = hash_type("cellParticleCount", totalCells_m);
        cellStartingIdx_m = hash_type("cellStartingIdx", totalCells_m + 1);
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    template<class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::particleExchange1(ParticleContainer &pc) {
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        this->applyBC(pc.R,
                      this->rlayout_m.getDomain()); // TODO if Periodic boundaries are used, this
        // should be considered in the overlap
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
        locate_type_nd ranks("MPI ranks", localnum);

        /* 0 --> particle valid
         * 1 --> particle invalid
         */
        bool_type invalid("invalid", localnum);

        size_type invalidCount = locateParticles(pc, ranks, invalid);

        //
        // TODO <-------------------- got to here, need to decide which method to use
        //

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


    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    template<class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::particleExchange2(ParticleContainer &pc) {
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        // TODO if Periodic boundaries are used, this should be considered in the overlap
        this->applyBC(pc.R, this->rlayout_m.getDomain());
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
        locate_type ranks, offsets; // size_to_be_determined, numLoc + 1

        /* 0 --> particle valid
         * 1 --> particle invalid
         */
        bool_type invalid("invalid", localnum);

        size_type invalidCount = locateParticles(pc, ranks, offsets, invalid);
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
            nSends[rank] = numberOfSends(rank, ranks); // independent of offsets
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
                fillHash(rank, ranks, offsets, hash);

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

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    template<class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::update(ParticleContainer &pc) {
        if (false) {
            // just to check compile errors, TODO remove one again
            particleExchange1(pc);
        } else {
            particleExchange2(pc);
        }
        buildCells(pc);
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    size_t ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::numberOfSends(
        int rank, const locate_type_nd &ranks) {
        size_t nSends = 0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::numberOfSends()", policy_type({0, 0}, {ranks.extent(0), ranks.extent(1)}),
            KOKKOS_LAMBDA(const size_t i, const size_t j, size_t &num) { num += size_t(rank == ranks(i, j)); },
            nSends);
        Kokkos::fence();
        return nSends;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    size_t ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::numberOfSends(
        int rank, const locate_type &ranks) {
        size_t nSends = 0;
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::numberOfSends()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, size_t &num) { num += size_t(rank == ranks(i)); },
            nSends);
        Kokkos::fence();
        return nSends;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::fillHash(int rank,
                                                                             const locate_type_nd &ranks,
                                                                             hash_type &hash) {
        /* Compute the prefix sum and fill the hash
         */
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::fillHash()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, int &idx, const bool final) {
                // Count matches for this particle
                bool belongs_to_rank = false;
                // Check all rank slots for this particle
// #pragma unroll // or reduce this with template param pack reduction
                for (size_t slot = 0; slot < ranks.extent(0); ++slot) {
                    if (ranks(i, slot) == rank) {
                        belongs_to_rank = true;
                        break; // Found it, no need to continue
                    }
                    // Early termination if we hit -1 (assuming ranks are packed)
                    if (ranks(i, slot) == -1) {
                        break;
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

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::fillHash(int rank,
                                                                             const locate_type &ranks,
                                                                             const locate_type &offsets,
                                                                             hash_type &hash) {
        /* Compute the prefix sum and fill the hash
         */
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::fillHash()", policy_type(0, offsets.extent(0) - 1),
            KOKKOS_LAMBDA(const size_t i, int &idx, const bool final) {
                // Check if this particle belongs to our target rank
                bool belongs_to_rank = false;
                const size_t start_rank_idx = offsets(i);
                const size_t end_rank_idx = offsets(i + 1);

                for (size_t rank_idx = start_rank_idx; rank_idx < end_rank_idx; ++rank_idx) {
                    if (ranks(rank_idx) == rank) {
                        belongs_to_rank = true;
                        break; // Found it, no need to continue
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

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    template<typename ParticleContainer>
    detail::size_type ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::locateParticles(
        const ParticleContainer &pc, locate_type &ranks, locate_type &rank_offsets,
        bool_type &invalid) const {
        auto &positions = pc.R.getView();
        typename RegionLayout_t::view_type Regions = this->rlayout_m.getdLocalRegions();
        // Two views: one for data, one for offsets

        size_type numLoc = pc.getLocalNum();

        // First pass: count assignments per particle
        locate_type counts("counts", numLoc);

        const auto overlap = rcutoff_m;

        auto range_policy = Kokkos::RangePolicy<position_execution_space>(0, numLoc);
        Kokkos::parallel_for(
            "count_assignments", range_policy, KOKKOS_LAMBDA(size_t i) {
                int count = 0;
                for (size_t j = 0; j < Regions.extent(0); ++j) {
                    if (positionInRegion(positions(i), Regions(j), overlap)) {
                        count++;
                    }
                }
                counts(i) = count;
            });

        Kokkos::fence();

        // Compute prefix sum for offsets
        // Step 2: Compute offsets using parallel_scan

        locate_type offsets("offets", numLoc + 1);
        Kokkos::deep_copy(Kokkos::subview(offsets, 0), 0);

        // Compute exclusive prefix sum (each offset points to start of particle's data)
        Kokkos::parallel_scan(
            "compute_offsets", range_policy,
            KOKKOS_LAMBDA(const int i, int &localSum, const bool final) {
                const int count_i = counts(i);
                if (final) {
                    offsets(i + 1) = localSum + count_i;
                }
                localSum += count_i;
            });
        Kokkos::fence();

        rank_offsets = offsets;

        // Get total number of assignments for allocation
        auto total_assignments = Kokkos::create_mirror_view(Kokkos::subview(offsets, numLoc));
        Kokkos::deep_copy(total_assignments, Kokkos::subview(offsets, numLoc));

        locate_type rank_data("rank_data", total_assignments()); // TODO does this work?

        const size_type myRank = Comm->rank();

        // Second pass: fill the data
        size_type invalidCount = 0;
        Kokkos::parallel_reduce(
            "fill_assignments", range_policy,
            KOKKOS_LAMBDA(size_t i, size_t &count) {
                size_t offset = rank_offsets(i);
                int local_count = 0;
                for (size_t j = 0; j < Regions.extent(0); ++j) {
                    bool xyz_bool = positionInRegion(positions(i), Regions(j), overlap);
                    if (xyz_bool) {
                        rank_data(offset + local_count) = j;
                        local_count++;
                    }

                    if (j == myRank) {
                        invalid(i) = !xyz_bool;
                        count += !xyz_bool;
                    }
                }
            },
            Kokkos::Sum<size_type>(invalidCount));
        Kokkos::fence();

        ranks = rank_data;

        assert(invalidCount <= numLoc);

        return invalidCount;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::positionInRegion(
        const vector_type &pos, const region_type &region, T overlap) {
        return [&]<size_t ... Idx>(const std::index_sequence<Idx...> &) {
            return ((pos[Idx] > region[Idx].min() - overlap) && ...) && ((pos[Idx] <= region[Idx].max() + overlap) &&
                       ...);
        }(std::make_index_sequence<Dim>());
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getCellIndex(
        const vector_type &pos, const NDRegion_t &region,
        const std::array<size_type, Dim> &strides, const std::array<T, Dim> &cellWidth) {
        return [&]<size_t ... Idx>(const std::index_sequence<Idx...> &) {
            return ((static_cast<size_type>(std::floor((pos[Idx] - region[Idx].min()) / cellWidth[Idx]) +
                                            numGhostCellsPerDim_m)
                     * strides[Idx]) + ...);
        }(std::make_index_sequence<Dim>());
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    constexpr typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::CellIndex_t
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getCellIndex(
        size_type index, const std::array<size_type, Dim> &numCells) {
        CellIndex_t ndIndex;
// #pragma unroll
        for (size_type d = 0; d < Dim; ++d) {
            ndIndex[d] = index % numCells[d];
            index /= numCells[d];
        }
        return ndIndex;
    }


    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    constexpr bool
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::isLocalCellIndex(
        size_type index, const std::array<size_type, Dim> &numCells) {
// #pragma unroll
        for (size_type d = 0; d < Dim; ++d) {
            size_type indexDim = index % numCells[d];
            if (indexDim == 0 || indexDim == numCells[d] - 1) { return false; }
            index /= numCells[d];
        }
        return true;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    template<typename ParticleContainer>
    detail::size_type ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::locateParticles(
        const ParticleContainer &pc, locate_type_nd &ranks, bool_type &invalid) const {
        auto &positions = pc.R.getView();
        typename RegionLayout_t::view_type Regions = this->rlayout_m.getdLocalRegions();

        const size_type myRank = Comm->rank();

        Kokkos::deep_copy(ranks, -1);

        size_type invalidCount = 0;
        const size_type numLoc = pc.getLocalNum();

        const auto overlap = rcutoff_m;

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::locateParticles()",
            mdrange_type({0, 0}, {numLoc, Regions.extent(0)}),
            KOKKOS_LAMBDA(const size_t i, const size_type j, size_type &count) {
                bool xyz_bool = positionInRegion(positions(i), Regions(j), overlap);
                if (xyz_bool) {
                    size_t l = 0;
                    for (; l < ranks.extent(1) && ranks(i, l) >= 0; ++l) {
                    }
                    ranks(i, l) = j;
                }
                if (j == myRank) {
                    invalid(i) = !xyz_bool;
                    count += !xyz_bool;
                }
            },
            Kokkos::Sum<size_type>(invalidCount));
        Kokkos::fence();

        return invalidCount;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    template<class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::buildCells(ParticleContainer &pc) {
        static IpplTimings::TimerRef cellBuildTimer = IpplTimings::getTimer("cellBuildTimer");
        IpplTimings::startTimer(cellBuildTimer);

        size_type nLoc = pc.getLocalNum();
        auto R = pc.R.getView();

        const auto totalCells = totalCells_m;
        const auto numLocalCells = numLocalCells_m;
        // calculate chaining meshwidth and number of mesh cells

        using int_type = typename neighbor_list_type::value_type;

        // allocate required (temporary) Kokkos views
        hash_type cellIndex("cellIndex", nLoc);
        hash_type cellParticleCount = cellParticleCount_m;
        hash_type cellStartingIdx = cellStartingIdx_m;
        hash_type cellCurrentIdx("cellCurrentIdx", totalCells + 1);

        const auto rank = Comm->rank();
        const auto localRegion = this->rlayout_m.getdLocalRegions()(rank);
        const auto &cellWidth = cellWidth_m;

        // calculate cell index for each particle
        using range_policy = Kokkos::RangePolicy<position_execution_space>;
        const auto cellStrides = cellStrides_m;
        const auto cellPermutation = cellPermutationForward_m;

        Kokkos::deep_copy(cellParticleCount, 0);
        Kokkos::parallel_for(
            "CalcCellIndices", range_policy(0, nLoc),
            KOKKOS_LAMBDA(const int i) {
                auto locCellIndex = getCellIndex(R(i), localRegion, cellStrides, cellWidth);
                locCellIndex = cellPermutation(locCellIndex);
                assert(locCellIndex < totalCells && "Invalid Grid Position");

                Kokkos::atomic_increment(&cellParticleCount(locCellIndex));
                cellIndex(i) = locCellIndex;
            });

        Kokkos::fence();

        // compute starting indices for each cell
        Kokkos::parallel_scan(
            range_policy(0, totalCells),
            KOKKOS_LAMBDA(const int i, int_type &localSum, bool isFinal) {
                if (isFinal) {
                    cellStartingIdx(i) = localSum;
                }
                localSum += cellParticleCount(i);
            });
        Kokkos::parallel_for(
            "Set last position", range_policy(totalCells, totalCells + 1),
            KOKKOS_LAMBDA(const int i) { cellStartingIdx(i) = nLoc; });

        Kokkos::fence();

        Kokkos::deep_copy(cellCurrentIdx, cellStartingIdx);

        Kokkos::fence();

        hash_type newIndex("newIndex", nLoc); // TODO this should probably be less

        Kokkos::parallel_for(
            "Calculate new Indices", range_policy(0, nLoc),
            KOKKOS_LAMBDA(const size_type &i) {
                int_type cellNumber = cellIndex(i);
                assert(cellNumber < static_cast<int_type>(totalCells) && "Invalid Cell Number");
                size_type newIdx = Kokkos::atomic_fetch_add(&cellCurrentIdx(cellNumber), 1u);
                assert(newIdx < nLoc && "Invalid Index");
                newIndex(i) = newIdx;
            });

        Kokkos::fence();

        // Move the data around (maybe there is a better solution)

        // TODO consider this
        //  auto filter = [&]<typename MemorySpace>() {
        //      return attributes_m.template get<MemorySpace>().size() > 0;
        //  };
        //  deleteIndex_m.copyToOtherSpaces<memory_space>(filter);
        //  keepIndex_m.copyToOtherSpaces<memory_space>(filter);
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_t num_attributes_in_space = 0;
            pc.template forAllAttributes<MemorySpace>(
                [&]<typename Attribute>(Attribute &) {
                    ++num_attributes_in_space;
                });
            if (num_attributes_in_space > 0) {
                auto newIndexMirror =
                        Kokkos::create_mirror_view_and_copy(MemorySpace(), newIndex);

                pc.template forAllAttributes<MemorySpace>(
                    [&]<typename Attribute>(Attribute &att) {
                        att->applyPermutation(newIndexMirror);
                        // TODO instead could store the permutation but this could lead to a lot of cacheline trashing
                    });
            }
        });

        // set local number of particles (excluding ghost particles)
        size_type numLocalParticles = 0;
        // size_type numMaxParticleInCell = 0;
        Kokkos::parallel_reduce("Comupte nLoc", range_policy(0, numLocalCells/*totalCells*/),
                                KOKKOS_LAMBDA(const size_type &i, size_type &sum/*, size_type &max*/) {
                                    auto n = cellParticleCount(i);
                                    // if (i < numLocalCells) [[likely]] {
                                    sum += n;
                                    // }
                                    // max = std::max<size_type>(n, max);
                                },
                                Kokkos::Sum<size_type>(numLocalParticles)
                                // , Kokkos::Max<size_type>(numMaxParticleInCell)
        );

        cellIndex_m = cellIndex;

        Kokkos::fence();
        pc.setLocalNum(numLocalParticles);

        // this is not needed as they views on the same data
        // cellStartingIdx_m = cellStartingIdx;
        // cellParticleCount_m = cellParticleCount;
        IpplTimings::stopTimer(cellBuildTimer);

    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
    typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::neighbor_info_type
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getNeighborCells(index_t cellIndex,
        const std::array<size_type, Dim> &numCells, const hash_type &cellPermutation) {
        // TODO consider cell permutation
        // Get the base cell coordinates for each dimension
        auto cellIndexNd = getCellIndex(cellIndex, numCells);

        // Generate all 3^Dim combinations of offsets (-1, 0, +1) for each dimension
        constexpr auto is = std::make_index_sequence<Dim>();
        constexpr size_type numNeighbors = detail::countHypercubes(Dim);
        neighbor_info_type neighborIndices{};
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            index_t flatIndex = 0;
            index_t temp = neighborIdx;
            index_t stride = 1;

            // This converts neighborIdx to base-3 representation where each digit is the offset+1
            [&]<size_type ... Idx>(const std::index_sequence<Idx...> &) {
                ((flatIndex += (cellIndexNd[Idx] + (temp % 3) - 1) * stride, temp /= 3, stride *= numCells[Idx]), ...);
            }(is);

            neighborIndices[neighborIdx] = cellPermutation(flatIndex);
        }
        return neighborIndices;
    }


    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    size_t ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getNumCells() const {
        return numLocalCells_m;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::neighbor_list_type
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getParticlesOfCell(size_type cellIndex) const {
        neighbor_list_type cellParticles("cell particles", cellParticleCount_m(cellIndex));
        using range_policy = Kokkos::RangePolicy<position_execution_space>;
        auto offset = cellStartingIdx_m(cellIndex);
        Kokkos::parallel_for("create cell particles", range_policy(0, cellParticleCount_m(cellIndex)),
                             KOKKOS_LAMBDA(const size_type &i) {
                                 cellParticles(i) = offset + i;
                             }
        );
        return cellParticles;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties
        ...>::neighbor_list_type
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getParticlesOfCell(
        const NeighborData &neighborData, size_type cellIndex) {
        neighbor_list_type cellParticles("cell particles", neighborData.cellParticleCount(cellIndex));
        using range_policy = Kokkos::RangePolicy<position_execution_space>;
        auto offset = neighborData.cellStartingIdx(cellIndex);
        Kokkos::parallel_for("create cell particles", range_policy(0, neighborData.cellParticleCount(cellIndex)),
                             KOKKOS_LAMBDA(const size_type &i) {
                                 cellParticles(i) = offset + i;
                             }
        );
        return cellParticles;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    typename ParticleSpatialOverlapLayout<T, Dim, Mesh,
        Properties...>::NeighborData ParticleSpatialOverlapLayout<T, Dim, Mesh,
        Properties...>::getNeighborData() const {
        return {
            .cellStrides = cellStrides_m,
            .numCells = numCells_m,
            .cellWidth = cellWidth_m,
            .region = this->rlayout_m.getdLocalRegions()(Comm->rank()),
            .cellStartingIdx = cellStartingIdx_m,
            .cellIndex = cellIndex_m,
            .cellParticleCount = cellParticleCount_m,
            .cellPermutationForward = cellPermutationForward_m,
            .cellPermutationBackward = cellPermutationBackward_m,
        };
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_FUNCTION typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::neighbor_list_type
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getNeighbors(
        const vector_type &pos, const /*TODO think about the hash again*/ NeighborData &neighborData) {
        // TODO using this to compute PP interaction produces different results!
        // Get the cell of the particle

        auto locCellIndex = getCellIndex(pos, neighborData.region, neighborData.cellStrides,
                                         neighborData.cellWidth);
        auto locCellIndexPermuted = neighborData.cellPermutationForward(locCellIndex);
        // if (locCellIndexPermuted == neighborData.hash) { return; } // TODO this doesnt work yet as a thread cannot have local data yet. caller already has the correct neighbor list

        constexpr size_type numNeighbors = detail::countHypercubes(Dim);

        auto neighbors = getNeighborCells(locCellIndex, neighborData.numCells, neighborData.cellPermutationForward);
        Kokkos::Array<typename hash_type::value_type, numNeighbors> neighborSizes;

        size_type totalParticleInNeighbors = 0;
        size_type maxParticleInNeighbors = 0;

        Kokkos::Array<typename hash_type::value_type, numNeighbors> neighborOffsets;
// #pragma unroll
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            auto n = neighborData.cellParticleCount(neighbors[neighborIdx]);
            neighborSizes[neighborIdx] = n;
            maxParticleInNeighbors = std::max<size_type>(n, maxParticleInNeighbors);
            neighborOffsets[neighborIdx] = totalParticleInNeighbors;
            totalParticleInNeighbors += n;
        }

        neighbor_list_type neighborList("Neigbor list", totalParticleInNeighbors);

        using twod_range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        Kokkos::parallel_for("collect neighbors", twod_range_policy({0, 0}, {numNeighbors, maxParticleInNeighbors}),
                             KOKKOS_LAMBDA(const size_type &i, const size_type &j) {
                                 if (j < neighborSizes[i]) {
                                     neighborList(neighborOffsets[i] + j) =
                                             neighborData.cellStartingIdx(neighbors[i]) + j;
                                 }
                             });


        return neighborList;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_FUNCTION typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::neighbor_list_type
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getNeighbors(
        size_type i, NeighborData &neighborData) {
        // Get the cell of the particle

        auto locCellIndex = neighborData.cellIndex(i);

        constexpr size_type numNeighbors = detail::countHypercubes(Dim);

        auto neighbors = getNeighborCells(locCellIndex, neighborData.numCells, neighborData.cellPermutation);
        Kokkos::Array<typename hash_type::value_type, numNeighbors> neighborSizes;

        size_type totalParticleInNeighbors = 0;
        size_type maxParticleInNeighbors = 0;

        Kokkos::Array<typename hash_type::value_type, numNeighbors> neighborOffsets;

// #pragma unroll
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            auto n = neighborData.cellParticleCount(neighbors[neighborIdx]);
            neighborSizes[neighborIdx] = n;
            maxParticleInNeighbors = std::max<size_type>(n, maxParticleInNeighbors);
            totalParticleInNeighbors += n;
            if (neighborIdx > 0) {
                neighborOffsets[neighborIdx] = neighborOffsets[neighborIdx - 1] + n;
            }
        }

        neighbor_list_type neighborList("Neigbor list", totalParticleInNeighbors);

        using twod_range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        Kokkos::parallel_for("collect neighbors", twod_range_policy({0, 0}, {numNeighbors, maxParticleInNeighbors}),
                             KOKKOS_LAMBDA(const size_type &i, const size_type &j) {
                                 if (j < neighborSizes[i]) {
                                     neighborList(neighborOffsets[i] + j) =
                                             neighborData.cellStartingIdx(neighbors[i]) + j;
                                 }
                             });

        return neighborList;
    }

    template<typename T, unsigned Dim, class Mesh, typename... Properties>
    template<typename ExecutionSpace, typename Functor>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::forEachPair(Functor &&f) const {
        static IpplTimings::TimerRef interactionTimer = IpplTimings::getTimer("PPInteractionTimer");
        IpplTimings::startTimer(interactionTimer);

        const auto cellStartingIdx = cellStartingIdx_m;
        const auto cellParticleCount = cellParticleCount_m;
        const auto cellPermutationForward = cellPermutationForward_m;
        const auto cellPermutationBackward = cellPermutationBackward_m;
        const auto& numCells = numCells_m;

        constexpr auto numCellNeighbors = detail::countHypercubes(Dim);


        using team_policy_t = Kokkos::TeamPolicy<ExecutionSpace>;
        using team_t = typename team_policy_t::member_type;
        Kokkos::parallel_for(
            "ParticleSpatialOverlapLayout::forEachPair()", team_policy_t(numLocalCells_m, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const team_t &team) {
                const size_type cellIdx = team.league_rank();
                if (cellParticleCount(cellIdx) == 0) { return; }

                const auto cellParticleOffset = cellStartingIdx(cellIdx);
                const auto numCellParticles = cellParticleCount(cellIdx);

                const auto cellNeighbors = getNeighborCells(cellPermutationBackward(cellIdx), numCells,
                                                            cellPermutationForward);

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, numCellNeighbors),
                    KOKKOS_LAMBDA(const size_t &n) {
                        const auto neighborCellIdx = cellNeighbors[n];
                        const auto neighborCellParticleOffset = cellStartingIdx(neighborCellIdx);

                        Kokkos::parallel_for(
                            Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(
                                team, numCellParticles, cellParticleCount(neighborCellIdx)),
                            KOKKOS_LAMBDA(const size_t &i, const size_t &j) {
                                const auto particleIdx = cellParticleOffset + i;
                                const auto neighborIdx = neighborCellParticleOffset + j;

                                if (neighborIdx == particleIdx) { return; }

                                f(particleIdx, neighborIdx);
                            });
                    });
            }
        );

        IpplTimings::stopTimer(interactionTimer);
    }
} // namespace ippl
