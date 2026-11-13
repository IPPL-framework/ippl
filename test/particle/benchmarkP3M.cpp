#include "Ippl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "Utility/Timer.h"

#include "Interaction/TruncatedGreenParticleInteraction.h"
#include "Particle/ParticleSpatialOverlapLayout.h"

namespace {
    using Vector     = ippl::Vector<double, 3>;
    using Mesh       = ippl::UniformCartesian<double, 3>;
    using BaseLayout = ippl::ParticleSpatialOverlapLayout<double, 3, Mesh>;
    using Space      = Kokkos::DefaultExecutionSpace;

    struct Layout : BaseLayout {
        using BaseLayout::BaseLayout;
        using BaseLayout::cellParticleCount_m;
        using BaseLayout::cellPermutationBackward_m;
        using BaseLayout::cellPermutationForward_m;
        using BaseLayout::cellStartingIdx_m;
        using BaseLayout::cellStrides_m;
        using BaseLayout::getCellNeighbors;
        using BaseLayout::numCells_m;
        using BaseLayout::numLocalCells_m;
        using BaseLayout::toCellIndex;
    };

    struct Particles : ippl::ParticleBase<Layout> {
        explicit Particles(Layout& layout)
            : ippl::ParticleBase<Layout>(layout) {
            this->addAttribute(charge);
            this->addAttribute(field);
            this->addAttribute(momentum);
            this->setParticleBC(ippl::BC::NO);
        }
        ippl::ParticleAttrib<double> charge;
        ippl::ParticleAttrib<Vector> field, momentum;
    };

    std::atomic<std::uint64_t> allocations{0}, allocationBytes{0};
    void countAllocation(Kokkos::Tools::SpaceHandle, const char*, const void*,
                         std::uint64_t bytes) {
        ++allocations;
        allocationBytes += bytes;
    }

    KOKKOS_FUNCTION double uniform(std::uint64_t seed) {
        seed += 0x9e3779b97f4a7c15ULL;
        seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
        seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
        seed ^= seed >> 31;
        return (static_cast<double>(seed >> 11) + 0.5) / 9007199254740992.0;
    }

    double maxRank(double value) {
        double result;
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return result;
    }
    std::uint64_t sumRanks(std::uint64_t value) {
        std::uint64_t result;
        MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        return result;
    }
    double median(std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const auto n = values.size();
        return (values[n / 2] + values[(n - 1) / 2]) / 2;
    }
}  // namespace

/** @brief Fixed-position P3M microbenchmark with deterministic particles across MPI counts.
 *
 *  Times the three phases of ParticleSpatialOverlapLayout::update() (updateLayout, particle
 *  exchange, cell build) and the particle-particle solve per step, as the median over the steps
 *  of the maximum over ranks. A separate update afterwards counts the Kokkos allocations.
 *  @note Arguments: global particles, cutoff, box side, steps, distribution (uniform/clustered).
 *        Timings exclude diagnostics and three warm-up steps.
 */
int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: benchmarkP3M N cutoff side steps uniform|clustered\n";
        return 1;
    }
    const auto total  = std::stoull(argv[1]);
    const auto cutoff = std::stod(argv[2]);
    const auto side   = std::stod(argv[3]);
    const auto steps  = std::stoi(argv[4]);
    const std::string distribution(argv[5]);
    if (total == 0 || !(cutoff > 0) || !(side > 0) || steps < 1
        || (distribution != "uniform" && distribution != "clustered"))
        return 1;
    ippl::initialize(argc, argv);
    {
        const auto rank   = ippl::Comm->rank();
        const auto nRanks = ippl::Comm->size();
        if (cutoff > side / (2 * nRanks)) {
            if (rank == 0)
                std::cerr << "Cutoff exceeds half the rank-local z extent\n";
            ippl::finalize();
            return 1;
        }
        Timer::enableFences = false;
        const int nz        = 32 * nRanks;
        ippl::NDIndex<3> domain(ippl::Index(32), ippl::Index(32), ippl::Index(nz));
        ippl::FieldLayout<3> fields(MPI_COMM_WORLD, domain, {false, false, true});
        Mesh mesh(domain, Vector{side / 32, side / 32, side / nz}, Vector(0));
        Layout layout(fields, mesh, cutoff);
        Particles particles(layout);
        const std::size_t begin = total * rank / nRanks;
        const std::size_t end   = total * (rank + 1) / nRanks;
        particles.create(end - begin);
        const auto r         = particles.R.getView();
        const auto q         = particles.charge.getView();
        const auto p         = particles.momentum.getView();
        const bool clustered = distribution == "clustered";
        Kokkos::parallel_for(
            "P3M benchmark initial particles", end - begin, KOKKOS_LAMBDA(const std::size_t i) {
                for (unsigned d = 0; d < 3; ++d) {
                    double x = uniform(3 * (begin + i) + d);
                    if (clustered)
                        x = 0.35 + 0.3 * x;
                    r(i)[d] = side * x;
                    p(i)[d] = 0;
                }
                q(i) = 1;
            });
        ippl::ParameterList parameters;
        parameters.add("rcut", cutoff);
        parameters.add("alpha", 2 / cutoff);
        parameters.add("force_constant", 1.0);
        using Interaction =
            ippl::TruncatedGreenParticleInteraction<Particles, decltype(particles.field),
                                                    decltype(particles.charge)>;
        Interaction interaction(particles, particles.field, particles.R, particles.charge,
                                parameters);
        std::vector<double> topologyTimes, exchangeTimes, cellTimes, updateTimes, forceTimes,
            stepTimes;
        for (int step = -3; step < steps; ++step) {
            ippl::Comm->barrier();
            Kokkos::Timer timer;
            layout.updateLayout(fields, mesh);
            Kokkos::fence();
            const double topologyTime = timer.seconds();
            Kokkos::Timer exchangeTimer;
            layout.particleExchange(particles);
            Kokkos::fence();
            const double exchangeTime = exchangeTimer.seconds();
            Kokkos::Timer cellTimer;
            layout.buildCells(particles);
            Kokkos::fence();
            const double cellTime   = cellTimer.seconds();
            const double updateTime = timer.seconds();
            particles.field         = Vector(0);
            Kokkos::fence();
            Kokkos::Timer forceTimer;
            interaction.solve();
            Kokkos::fence();
            const double forceTime = forceTimer.seconds();
            const double stepTime  = timer.seconds();
            if (step >= 0) {
                topologyTimes.push_back(maxRank(topologyTime));
                exchangeTimes.push_back(maxRank(exchangeTime));
                cellTimes.push_back(maxRank(cellTime));
                updateTimes.push_back(maxRank(updateTime));
                forceTimes.push_back(maxRank(forceTime));
                stepTimes.push_back(maxRank(stepTime));
            }
        }

        // Collect allocation traffic in a separate update so callbacks do not affect timings.
        Kokkos::Tools::Experimental::set_allocate_data_callback(countAllocation);
        layout.updateLayout(fields, mesh);
        particles.update();
        Kokkos::fence();
        Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
        const auto counts        = layout.cellParticleCount_m;
        const auto starts        = layout.cellStartingIdx_m;
        const auto backward      = layout.cellPermutationBackward_m;
        const auto forward       = layout.cellPermutationForward_m;
        const auto strides       = layout.cellStrides_m;
        const auto cells         = layout.numCells_m;
        std::uint64_t candidates = 0, occupied = 0;
        int maximumOccupancy = 0;
        Kokkos::parallel_reduce(
            "P3M candidate count", layout.numLocalCells_m,
            KOKKOS_LAMBDA(const std::size_t cell, std::uint64_t& sum) {
                if (counts(cell) == 0)
                    return;
                const auto neighbors = Layout::getCellNeighbors(
                    Layout::toCellIndex(backward(cell), cells), strides, forward);
                std::uint64_t surrounding = 0;
                for (unsigned n = 0; n < 27; ++n)
                    surrounding += counts(neighbors[n]);
                sum += static_cast<std::uint64_t>(counts(cell)) * (surrounding - 1);
            },
            candidates);
        Kokkos::parallel_reduce(
            "P3M occupied cells", layout.numLocalCells_m,
            KOKKOS_LAMBDA(const std::size_t cell, std::uint64_t& sum) { sum += counts(cell) != 0; },
            occupied);
        Kokkos::parallel_reduce(
            "P3M max occupancy", layout.numLocalCells_m,
            KOKKOS_LAMBDA(const std::size_t cell, int& maximum) {
                maximum = Kokkos::max(maximum, counts(cell));
            },
            Kokkos::Max<int>(maximumOccupancy));
        const auto hostStarts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), starts);
        Kokkos::View<std::uint64_t*, Space::memory_space> accepted("accepted", 1);
        const auto positions = particles.R;
        layout.template forEachPair<Space>(KOKKOS_LAMBDA(const std::size_t i, const std::size_t j) {
            const Vector dist = positions(i) - positions(j);
            const double r2   = dist.dot(dist);
            if (r2 > 0 && r2 < cutoff * cutoff)
                Kokkos::atomic_inc(&accepted(0));
        });
        const auto hostAccepted =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), accepted);
        const auto globalCandidates = sumRanks(candidates);
        const auto globalAccepted   = sumRanks(hostAccepted(0));
        const auto globalCells      = sumRanks(counts.extent(0));
        const auto globalOccupied   = sumRanks(occupied);
        const auto globalOwned      = sumRanks(particles.getLocalNum());
        const auto globalGhosts =
            sumRanks(hostStarts(starts.extent(0) - 1) - particles.getLocalNum());
        const auto globalAllocations = sumRanks(allocations.load());
        const auto globalBytes       = sumRanks(allocationBytes.load());
        const auto globalMaximum     = maxRank(maximumOccupancy);
        if (rank == 0) {
            std::cout << "ranks,particles,distribution,cutoff,side,update_s,force_s,step_s,"
                         "candidates,accepted,cells,occupied,max_occupancy,ghosts,allocations,"
                         "allocated_bytes,topology_s,exchange_s,cells_s\n"
                      << nRanks << ',' << globalOwned << ',' << distribution << ',' << cutoff << ','
                      << side << ',' << median(updateTimes) << ',' << median(forceTimes) << ','
                      << median(stepTimes) << ',' << globalCandidates << ',' << globalAccepted
                      << ',' << globalCells << ',' << globalOccupied << ',' << globalMaximum << ','
                      << globalGhosts << ',' << globalAllocations << ',' << globalBytes << ','
                      << median(topologyTimes) << ',' << median(exchangeTimes) << ','
                      << median(cellTimes) << '\n';
        }
    }
    ippl::finalize();
}
