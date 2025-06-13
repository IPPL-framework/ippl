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
#include <numeric>
#include <vector>

#include "Utility/IpplTimings.h"

#include "Communicate/Window.h"

namespace ippl::fixDefaultTemplateArgument {
    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::ParticleSpatialOverlapLayout(
        FieldLayout<Dim>& fl, Mesh& mesh, const T& rcutoff)
        : Base(fl, mesh)
        , rcutoff_m(rcutoff)
        , numLocalParticles_m(0) {
        // TODO add an assertion that the overlap is small enough (half the region size in every
        //  dimension)?
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

        totalCells_m    = 1;
        numLocalCells_m = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            const T length              = hLocalRegions(rank)[d].length();
            const size_type nLocalCells = std::floor(length / rcutoff_m);
            numCells_m[d]               = nLocalCells + 2 * numGhostCellsPerDim_m;
            // two ghost cells, one in each direction
            cellWidth_m[d] = length / nLocalCells;
            totalCells_m *= numCells_m[d];
            numLocalCells_m *= nLocalCells;
        }
        numGhostCells_m = totalCells_m - numLocalCells_m;

        std::exclusive_scan(numCells_m.begin(), numCells_m.end(), cellStrides_m.begin(), 1,
                            std::multiplies());

        cellParticleCount_m = hash_type("cellParticleCount", totalCells_m);
        cellStartingIdx_m   = hash_type("cellStartingIdx", totalCells_m + 1);

        // Compute cell permutation need to place ghost cells at the end to make sure only local
        // particles are at indices 0 to numLocalParticles
        hash_type cellPermutationForward("cell permutation forward", totalCells_m);
        hash_type cellPermutationBackward("cell permutation backward", totalCells_m);

        // Compute prefix sums using the permutation views to store it
        hash_type localPrefixSum("local prefix sum", totalCells_m);
        hash_type ghostPrefixSum("ghost prefix sum", totalCells_m);
        const auto& numCells     = numCells_m;

        Kokkos::parallel_scan(
            "scan_local", Kokkos::RangePolicy(0, totalCells_m),
            KOKKOS_LAMBDA(const size_type i, size_type& update, const bool final) {
                const size_type val = isLocalCellIndex(toCellIndex(i, numCells), numCells);
                if (final) {
                    localPrefixSum(i) = update;
                }
                update += val;
            });
        Kokkos::parallel_scan(
            "scan_ghost", Kokkos::RangePolicy(0, totalCells_m),
            KOKKOS_LAMBDA(const size_type i, size_type& update, const bool final) {
                const size_type val = !isLocalCellIndex(toCellIndex(i, numCells), numCells);
                if (final) {
                    ghostPrefixSum(i) = update;
                }
                update += val;
            });

        // Final assignment
        const auto numLocalCells = numLocalCells_m;
        Kokkos::parallel_for(
            "assign_permutations", Kokkos::RangePolicy<>(0, totalCells_m),
            KOKKOS_LAMBDA(const size_type i) {
                if (const auto cellIdx = toCellIndex(i, numCells);
                    isLocalCellIndex(cellIdx, numCells)) {
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
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::isCloseToBoundary(
        const vector_type& pos, const region_type& globalRegion, Vector_t<bool, Dim> periodic,
        T overlap) {
        return [&]<std::size_t... Idx>(const std::index_sequence<Idx...>&) {
            return ((periodic[Idx]
                     && (pos[Idx] < globalRegion[Idx].min() + overlap
                         || pos[Idx] > globalRegion[Idx].max() - overlap))
                    || ...);
        }(std::make_index_sequence<Dim>());
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::createPeriodicGhostParticles(
        ParticleContainer& pc) {
        // periodic boundary conditions come in pairs
        Vector_t<bool, Dim> periodic;
        for (unsigned d = 0; d < Dim; ++d) {
            periodic[d] = this->getParticleBC()[2 * d];
        }
        if (!std::any_of(periodic.begin(), periodic.end(), [](auto bc) {
                return bc == BC::PERIODIC;
            })) {
            return;
        }

        const auto& globalRegion = this->rlayout_m.getDomain();
        const auto overlap       = rcutoff_m;

        const auto numLoc = pc.getLocalNum();
        const auto R      = pc.R;

        size_type numBoundaryParticles = 0;
        Kokkos::parallel_reduce(
            "count boundary particles", numLoc,
            KOKKOS_LAMBDA(const size_t& i, size_type& sum) {
                if (isCloseToBoundary(R(i), globalRegion, periodic, overlap)) {
                    ++sum;
                }
            },
            Kokkos::Sum<size_type>(numBoundaryParticles));
        if (numBoundaryParticles == 0) {
            return;
        }

        hash_type boundaryIndices("boundaryIndices", numBoundaryParticles);
        Kokkos::parallel_scan(
            "count boundary particles", numLoc,
            KOKKOS_LAMBDA(const size_t& i, size_type& sum, bool final) {
                if (isCloseToBoundary(R(i), globalRegion, periodic, overlap)) {
                    if (final) {
                        boundaryIndices[sum] = i;
                    }
                    ++sum;
                }
            });

        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_t numAttributesInSpace = 0;
            pc.template forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute&) {
                ++numAttributesInSpace;
            });
            if (numAttributesInSpace == 0) {
                return;
            }

            auto boundaryIndicesMirror =
                Kokkos::create_mirror_view_and_copy(MemorySpace(), boundaryIndices);
            pc.template forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute& att) {
                att->internalCopy(boundaryIndicesMirror);
            });
        });

        // otherwise they will not be considered by exchangeParticles and buildCells
        pc.setLocalNum(numLoc + numBoundaryParticles);

        for (unsigned d = 0; d < Dim; ++d) {
            if (!periodic[d]) {
                continue;
            }
            const auto min    = globalRegion[d].min();
            const auto length = globalRegion[d].length();
            const auto middle = min + length / 2;
            Kokkos::parallel_for(
                "correct positions",
                Kokkos::RangePolicy<position_execution_space>(numLoc,
                                                              numLoc + numBoundaryParticles),
                KOKKOS_LAMBDA(const size_t& i) {
                    R(i)[d] += R(i)[d] > middle ? -length : length;
                });
        }
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::particleExchange(
        ParticleContainer& pc) {
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        this->applyBC(pc.R, this->rlayout_m.getDomain());
        createPeriodicGhostParticles(pc);
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
        locate_type ranks, offsets;  // size_to_be_determined, numLoc + 1

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
            nSends[rank] = numberOfSends(rank, ranks);  // independent of offsets
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

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::update(ParticleContainer& pc) {
        particleExchange(pc);
        buildCells(pc);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    size_t ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::numberOfSends(
        int rank, const locate_type& ranks) {
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
                // Check if this particle belongs to our target rank
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

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <typename ParticleContainer>
    detail::size_type ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::locateParticles(
        const ParticleContainer& pc, locate_type& ranks, locate_type& rank_offsets,
        bool_type& invalid) const {
        auto& positions = pc.R.getView();
        auto regions    = this->rlayout_m.getdLocalRegions();
        // Two views: one for data, one for offsets

        size_type numLoc = pc.getLocalNum();

        // First pass: count assignments per particle
        locate_type counts("counts", numLoc);

        const auto overlap = rcutoff_m;

        auto range_policy = Kokkos::RangePolicy<position_execution_space>(0, numLoc);
        Kokkos::parallel_for(
            "count_assignments", range_policy, KOKKOS_LAMBDA(size_t i) {
                int count = 0;
                for (size_t j = 0; j < regions.extent(0); ++j) {
                    if (positionInRegion(positions(i), regions(j), overlap)) {
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
            KOKKOS_LAMBDA(const size_t i, int& localSum, const bool final) {
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

        locate_type rank_data("rank_data", total_assignments());

        const size_type myRank = Comm->rank();

        // Second pass: fill the data
        size_type invalidCount = 0;
        Kokkos::parallel_reduce(
            "fill_assignments", range_policy,
            KOKKOS_LAMBDA(size_t i, size_t& count) {
                size_t offset   = rank_offsets(i);
                int local_count = 0;
                for (size_t j = 0; j < regions.extent(0); ++j) {
                    bool xyz_bool = positionInRegion(positions(i), regions(j), overlap);
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

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr bool
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::positionInRegion(
        const vector_type& pos, const region_type& region, T overlap) {
        return [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
            return ((pos[Idx] > region[Idx].min() - overlap) && ...)
                   && ((pos[Idx] <= region[Idx].max() + overlap) && ...);
        }(std::make_index_sequence<Dim>());
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::FlatCellIndex_t
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::toFlatCellIndex(
            const CellIndex_t& cellIndex, const Vector_t<size_type, Dim>& cellStrides,
            hash_type cellPermutationForward) {
        return cellPermutationForward(cellIndex.dot(cellStrides));
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::CellIndex_t
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::toCellIndex(
            FlatCellIndex_t nonPermutedIndex, const Vector_t<size_type, Dim>& numCells) {
        CellIndex_t ndIndex;
        // #pragma unroll
        for (size_type d = 0; d < Dim; ++d) {
            ndIndex[d] = nonPermutedIndex % numCells[d];
            nonPermutedIndex /= numCells[d];
        }
        assert(nonPermutedIndex == 0);
        return ndIndex;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::CellIndex_t
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getCellIndex(
            const vector_type& pos, const region_type& region, const Vector_t<T, Dim>& cellWidth) {
        CellIndex_t cellIndex;
        for (unsigned d = 0; d < Dim; ++d) {
            cellIndex[d] = static_cast<size_type>(
                std::floor((pos[d] - region[d].min()) / cellWidth[d]) + numGhostCellsPerDim_m);
        }
        // return [&]<size_t ... Idx>(const std::index_sequence<Idx...> &) {
        //     return CellIndex_t((static_cast<size_type>(std::floor((pos[Idx] - region[Idx].min())
        //     / cellWidth[Idx]) +
        //                                     numGhostCellsPerDim_m), ...));
        // }(std::make_index_sequence<Dim>());
        return cellIndex;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    constexpr bool ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::isLocalCellIndex(
        const CellIndex_t& index, const Vector_t<size_type, Dim>& numCells) {
        return [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
            return !((index[Idx] == 0 || index[Idx] == numCells[Idx] - 1) || ...);
        }(std::make_index_sequence<Dim>());
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    template <class ParticleContainer>
    void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::buildCells(
        ParticleContainer& pc) {
        static IpplTimings::TimerRef cellBuildTimer = IpplTimings::getTimer("cellBuildTimer");
        IpplTimings::startTimer(cellBuildTimer);

        size_type nLoc = pc.getLocalNum();
        auto R         = pc.R.getView();

        const auto totalCells    = totalCells_m;
        const auto numLocalCells = numLocalCells_m;
        // calculate chaining meshwidth and number of mesh cells

        using int_type = typename particle_neighbor_list_type::value_type;

        // allocate required (temporary) Kokkos views
        hash_type cellIndex("cellIndex", nLoc);
        hash_type cellParticleCount = cellParticleCount_m;
        hash_type cellStartingIdx   = cellStartingIdx_m;
        hash_type cellCurrentIdx("cellCurrentIdx", totalCells + 1);

        const auto rank        = Comm->rank();
        const auto localRegion = this->rlayout_m.gethLocalRegions()(rank);
        const auto& cellWidth  = cellWidth_m;

        // calculate cell index for each particle
        using range_policy                = Kokkos::RangePolicy<position_execution_space>;
        const auto cellStrides            = cellStrides_m;
        const auto cellPermutationForward = cellPermutationForward_m;

        Kokkos::deep_copy(cellParticleCount, 0);
        Kokkos::parallel_for(
            "CalcCellIndices", range_policy(0, nLoc), KOKKOS_LAMBDA(const size_t& i) {
                const auto locCellIndex = getCellIndex(R(i), localRegion, cellWidth);
                const auto locCellIndexFlat =
                    toFlatCellIndex(locCellIndex, cellStrides, cellPermutationForward);
                assert(locCellIndexFlat < totalCells && "Invalid Grid Position");

                Kokkos::atomic_increment(&cellParticleCount(locCellIndexFlat));
                cellIndex(i) = locCellIndexFlat;
            });

        Kokkos::fence();

        // compute starting indices for each cell
        Kokkos::parallel_scan(
            range_policy(0, totalCells),
            KOKKOS_LAMBDA(const size_t i, int_type& localSum, bool isFinal) {
                if (isFinal) {
                    cellStartingIdx(i) = localSum;
                }
                localSum += cellParticleCount(i);
            });
        // set last position
        Kokkos::deep_copy(
            Kokkos::subview(cellStartingIdx, Kokkos::make_pair(totalCells, totalCells + 1)), nLoc);

        Kokkos::fence();

        Kokkos::deep_copy(cellCurrentIdx, cellStartingIdx);

        Kokkos::fence();

        hash_type newIndex("newIndex", nLoc);
        hash_type newCellIndex("cellIndex", nLoc);

        Kokkos::parallel_for(
            "Calculate new Indices", range_policy(0, nLoc), KOKKOS_LAMBDA(const size_type& i) {
                auto locCellIndex = cellIndex(i);
                // auto locCellIndex = cellPermutation(getCellIndex(R(i), localRegion, cellStrides,
                // cellWidth));
                assert(locCellIndex < static_cast<int_type>(totalCells) && "Invalid Cell Number");
                size_type newIdx = Kokkos::atomic_fetch_add(&cellCurrentIdx(locCellIndex), 1);
                assert(newIdx < nLoc && "Invalid Index");
                newIndex(i)          = newIdx;
                newCellIndex(newIdx) = locCellIndex;
            });

        Kokkos::fence();

        // Move the data around (maybe there is a better solution)
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_t num_attributes_in_space = 0;
            pc.template forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute&) {
                ++num_attributes_in_space;
            });
            if (num_attributes_in_space == 0) {
                return;
            }
            auto newIndexMirror = Kokkos::create_mirror_view_and_copy(MemorySpace(), newIndex);

            pc.template forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute& att) {
                att->applyPermutation(newIndexMirror);
            });
        });
        Kokkos::fence();

        // set local number of particles (excluding ghost particles)
        size_type numLocalParticles = 0;
        Kokkos::parallel_reduce(
            "Comupte nLoc", range_policy(numLocalCells, numLocalCells + 1),
            KOKKOS_LAMBDA(const size_type& i, size_type& sum) { sum += cellStartingIdx(i); },
            Kokkos::Sum<size_type>(numLocalParticles));

        Kokkos::fence();

        cellIndex_m         = newCellIndex;
        numLocalParticles_m = numLocalParticles;

        pc.setLocalNum(numLocalParticles);

        // this is not needed as they are views on the same underlying memory
        // cellStartingIdx_m = cellStartingIdx;
        // cellParticleCount_m = cellParticleCount;
        IpplTimings::stopTimer(cellBuildTimer);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_INLINE_FUNCTION constexpr
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh,
                                              Properties...>::cell_particle_neighbor_list_type
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getCellNeighbors(
            const CellIndex_t& cellIndex, const Vector_t<size_type, Dim>& cellStrides,
            const hash_type& cellPermutationForward) {
        // Generate all 3^Dim combinations of offsets (-1, 0, +1) for each dimension
        // constexpr auto is = std::make_index_sequence<Dim>();
        constexpr size_type numNeighbors = detail::countHypercubes(Dim);
        cell_particle_neighbor_list_type neighborIndices{};
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            index_t temp = neighborIdx;

            // This converts neighborIdx to base-3 representation where each digit is the offset+1
            auto neighborCellIndex = cellIndex;
            for (unsigned d = 0; d < Dim; ++d) {
                neighborCellIndex(d) += (temp % 3) - 1;
                temp /= 3;
            }
            // [&]<size_type ... Idx>(const std::index_sequence<Idx...> &) {
            //     ((flatIndex += (cellIndex[Idx] + (temp % 3) - 1) * stride, temp /= 3, stride *=
            //     numCells[Idx]), ...);
            // }(is);

            neighborIndices[neighborIdx] =
                toFlatCellIndex(neighborCellIndex, cellStrides, cellPermutationForward);
        }
        return neighborIndices;
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    typename ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::NeighborData
    ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getNeighborData() const {
        return NeighborData(numLocalParticles_m, cellStrides_m, numCells_m, cellWidth_m,
                            this->rlayout_m.gethLocalRegions()(Comm->rank()), cellStartingIdx_m,
                            cellIndex_m, cellParticleCount_m, cellPermutationForward_m,
                            cellPermutationBackward_m);
    }

    template <typename T, unsigned Dim, class Mesh, typename... Properties>
    KOKKOS_FUNCTION
        typename ParticleSpatialOverlapLayout<T, Dim, Mesh,
                                              Properties...>::particle_neighbor_list_type
        ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::getParticleNeighbors(
            const vector_type& pos, const NeighborData& neighborData) {
        // Get the cell of the particle

        const auto locCellIndex = getCellIndex(pos, neighborData.region, neighborData.cellStrides,
                                               neighborData.cellWidth);
        const auto locCellIndexPermuted = neighborData.cellPermutationForward(locCellIndex);

        constexpr size_type numNeighbors = detail::countHypercubes(Dim);

        const auto neighbors = getCellNeighbors(locCellIndex, neighborData.cellStrides,
                                                neighborData.cellPermutationForward);

        size_type totalParticleInNeighbors = 0;
        size_type maxParticleInNeighbors   = 0;

        Kokkos::Array<typename hash_type::value_type, numNeighbors> neighborSizes;
        Kokkos::Array<typename hash_type::value_type, numNeighbors> neighborOffsets;

        // #pragma unroll
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            auto n                       = neighborData.cellParticleCount(neighbors[neighborIdx]);
            neighborSizes[neighborIdx]   = n;
            maxParticleInNeighbors       = std::max<size_type>(n, maxParticleInNeighbors);
            neighborOffsets[neighborIdx] = totalParticleInNeighbors;
            totalParticleInNeighbors += n;
        }

        particle_neighbor_list_type neighborList("Neigbor list", totalParticleInNeighbors);

        using twod_range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        Kokkos::parallel_for(
            "collect neighbors", twod_range_policy({0, 0}, {numNeighbors, maxParticleInNeighbors}),
            KOKKOS_LAMBDA(const size_type& i, const size_type& j) {
                if (j < neighborSizes[i]) {
                    neighborList(neighborOffsets[i] + j) =
                        neighborData.cellStartingIdx(neighbors[i]) + j;
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
            index_t particleIndex, const NeighborData& neighborData) {
        // Get the cell of the particle

        assert(particleIndex < neighborData.numLocalParticles);
        const auto locCellIndexFlat = neighborData.cellIndex(particleIndex);

        constexpr size_type numNeighbors = detail::countHypercubes(Dim);

        const auto locCellIndex = toCellIndex(
            neighborData.cellPermutationBackward(locCellIndexFlat), neighborData.numCells);
        assert(isLocalCellIndex(locCellIndex, neighborData.numCells));
        const auto neighbors = getCellNeighbors(locCellIndex, neighborData.cellStrides,
                                                neighborData.cellPermutationForward);

        size_type totalParticleInNeighbors = 0;
        size_type maxParticleInNeighbors   = 0;

        Kokkos::Array<size_type, numNeighbors> neighborSizes;
        Kokkos::Array<size_type, numNeighbors + 1> neighborOffsets;

        // #pragma unroll
        for (size_type neighborIdx = 0; neighborIdx < numNeighbors; ++neighborIdx) {
            auto n                       = neighborData.cellParticleCount(neighbors[neighborIdx]);
            maxParticleInNeighbors       = std::max<size_type>(n, maxParticleInNeighbors);
            neighborOffsets[neighborIdx] = totalParticleInNeighbors;
            totalParticleInNeighbors += n;
        }
        neighborOffsets[numNeighbors] = totalParticleInNeighbors;

        particle_neighbor_list_type neighborList("Neigbor list", totalParticleInNeighbors);

        using twod_range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;
        Kokkos::parallel_for(
            "collect neighbors", twod_range_policy({0, 0}, {numNeighbors, maxParticleInNeighbors}),
            KOKKOS_LAMBDA(const size_type& i, const size_type& j) {
                const auto numParticlesInCell = neighborOffsets[i + 1] - neighborOffsets[i];
                if (j < numParticlesInCell) {
                    neighborList(neighborOffsets[i] + j) =
                        neighborData.cellStartingIdx(neighbors[i]) + j;
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

        const auto cellStartingIdx         = cellStartingIdx_m;
        const auto cellParticleCount       = cellParticleCount_m;
        const auto cellPermutationForward  = cellPermutationForward_m;
        const auto cellPermutationBackward = cellPermutationBackward_m;
        const auto& cellStrides            = cellStrides_m;
        const auto& numCells               = numCells_m;

        constexpr auto numCellNeighbors = detail::countHypercubes(Dim);

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

                const auto cellIdx = toCellIndex(cellPermutationBackward(cellIdxFlat), numCells);
                const auto cellNeighbors =
                    getCellNeighbors(cellIdx, cellStrides, cellPermutationForward);

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, numCellNeighbors), [&](const size_t& n) {
                        const auto neighborCellIdx            = cellNeighbors[n];
                        const auto neighborCellParticleOffset = cellStartingIdx(neighborCellIdx);
                        const auto numNeigborCellParticles    = cellParticleCount(neighborCellIdx);

                        Kokkos::parallel_for(Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(
                                                 team, numCellParticles, numNeigborCellParticles),
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

    // template<typename T, unsigned Dim, class Mesh, typename... Properties>
    // template<typename ExecutionSpace, typename Functor>
    // void ParticleSpatialOverlapLayout<T, Dim, Mesh, Properties...>::forEachPair(Functor &&f)
    // const {
    //     static IpplTimings::TimerRef interactionTimer =
    //     IpplTimings::getTimer("PPInteractionTimer"); IpplTimings::startTimer(interactionTimer);
    //
    //     const auto cellStartingIdx = cellStartingIdx_m;
    //     const auto cellParticleCount = cellParticleCount_m;
    //     const auto cellPermutationForward = cellPermutationForward_m;
    //     const auto cellPermutationBackward = cellPermutationBackward_m;
    //
    //     const auto numLocalParticles = numLocalParticles_m;
    //     const auto data = getNeighborData();
    //
    //
    //     using team_policy_t = Kokkos::TeamPolicy<ExecutionSpace>;
    //     using team_t = typename team_policy_t::member_type;
    //     // calculate interaction force
    //     Kokkos::parallel_for(
    //         "Particle-Particle", team_policy_t(numLocalParticles, Kokkos::AUTO()),
    //         KOKKOS_LAMBDA(const team_t &team) {
    //             const index_t particleIndex = team.league_rank();
    //
    //             auto neighborList = getParticleNeighbors(particleIndex, data);
    //
    //             Kokkos::parallel_for(
    //                 Kokkos::TeamThreadRange(team, neighborList.extent(0)),
    //                 [&](const size_type &i) {
    //                     const index_t neighborIndex = neighborList(i);
    //                     if (neighborIndex == particleIndex) { return; }
    //
    //                     f(particleIndex, neighborIndex);
    //                 }
    //             );
    //         });
    //
    //     Kokkos::fence();
    //
    //     IpplTimings::stopTimer(interactionTimer);
    // }
}  // namespace ippl::fixDefaultTemplateArgument
