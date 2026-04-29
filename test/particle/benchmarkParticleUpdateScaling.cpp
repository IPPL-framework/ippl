//
// benchmarkParticleUpdateScaling.cpp
//
// Scaling benchmark for particle update. Times only ParticleSpatialLayout::update()
// (with barrier before/after), reports per-step max wall time across ranks, and
// appends a CSV row per run.
//
// Usage:
//   srun ./benchmarkParticleUpdateScaling Nx Ny Nz nParticles nSteps
//       [--warmup N] [--exchange rma|p2p|alltoall]
//       [--ranks-per-node R] [--overallocate F] [--info N]
//
// --ranks-per-node defaults to the number of MPI ranks (single-node). On a
// multi-node run, pass the actual GPUs/CPUs per node so the CSV reports it.
//

#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "BenchParticles.h"
#include "Utility/IpplTimings.h"

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

int main(int argc, char* argv[]) {
    using namespace bench;
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);

        if (argc < 6) {
            if (ippl::Comm->rank() == 0)
                std::cerr << "Usage: " << argv[0]
                          << " Nx Ny Nz nParticles nSteps"
                             " [--warmup N] [--exchange rma|p2p|alltoall]"
                             " [--ranks-per-node R] [--overallocate F] [--info N]\n";
            ippl::finalize();
            return 1;
        }

        ippl::Vector<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};
        const unsigned int totalP = std::atoi(argv[4]);
        const unsigned int nt     = std::atoi(argv[5]);

        int warmupSteps         = 5;
        std::string exchangeStr = "alltoall";
        int infoEvery           = 0;
        int ranksPerNode        = 0;  // 0 → fill in from comm size below

        for (int i = 6; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--warmup" && i + 1 < argc)
                warmupSteps = std::stoi(argv[++i]);
            else if (flag == "--exchange" && i + 1 < argc)
                exchangeStr = argv[++i];
            else if (flag == "--info" && i + 1 < argc)
                infoEvery = std::stoi(argv[++i]);
            else if (flag == "--ranks-per-node" && i + 1 < argc)
                ranksPerNode = std::stoi(argv[++i]);
        }

        const ippl::CountExchange exchangeMode = parseMode(exchangeStr);

        msg << "benchmarkParticleUpdateScaling\n"
            << "nt=" << nt << " Np=" << totalP << " grid=" << nr << " exchange=" << exchangeStr
            << " warmup=" << warmupSteps << endl;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; ++i)
            domain[i] = ippl::Index(nr[i]);

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        Vector_t rmin(0.0), rmax(1.0);
        double dx       = rmax[0] / double(nr[0]);
        double dy       = rmax[1] / double(nr[1]);
        double dz       = rmax[2] / double(nr[2]);
        Vector_t hr     = {dx, dy, dz};
        Vector_t origin = {rmin[0], rmin[1], rmin[2]};
        double hr_min   = std::min({dx, dy, dz});
        const double dt = 1.0;

        Mesh_t mesh(domain, hr, origin);
        FieldLayout_t FL(MPI_COMM_WORLD, domain, isParallel);
        PLayout_t PL(FL, mesh, /*fem=*/false, exchangeMode);

        using bunch_type = BenchParticles<PLayout_t>;
        auto P           = std::make_unique<bunch_type>(PL, isParallel);

        unsigned long int nloc = totalP / ippl::Comm->size();
        P->create(nloc);

        Kokkos::Random_XorShift64_Pool<> initPool(
            static_cast<uint64_t>(42 + ippl::Comm->rank() * nloc));

        auto R_view = P->R.getView();
        Kokkos::parallel_for(
            "init_positions", Kokkos::RangePolicy<>(0, (int)nloc), KOKKOS_LAMBDA(const int i) {
                auto gen = initPool.get_state();
                for (int d = 0; d < 3; ++d)
                    R_view(i)[d] = gen.drand();
                initPool.free_state(gen);
            });
        Kokkos::fence();
        P->qm = 1.0 / totalP;
        P->E  = 0.0;

        ippl::Comm->barrier();
        P->update();
        ippl::Comm->barrier();

        if (ippl::Comm->rank() == 0)
            std::cout << "Running " << warmupSteps << " warmup step(s)...\n";

        Kokkos::Random_XorShift64_Pool<> pool(static_cast<uint64_t>(42 + 100 * ippl::Comm->rank()));

        for (int w = 0; w < warmupSteps; ++w) {
            auto P_view = P->P.getView();
            Kokkos::parallel_for(
                "warmup_randP", Kokkos::RangePolicy<>(0, (int)P->getLocalNum()),
                KOKKOS_LAMBDA(const int i) {
                    auto gen = pool.get_state();
                    for (int d = 0; d < 3; ++d)
                        P_view(i)[d] = gen.drand() * hr_min;
                    pool.free_state(gen);
                });
            Kokkos::fence();
            P->R = P->R + dt * P->P;
            P->update();
            P->P = P->P + dt * P->qm * P->E;
        }

        if (ippl::Comm->rank() == 0)
            std::cout << "Warmup done. Starting timed run.\n";

        std::vector<double> stepTimes;
        stepTimes.reserve(nt);

        for (unsigned int it = 0; it < nt; ++it) {
            {
                auto P_view = P->P.getView();
                Kokkos::parallel_for(
                    "randP", Kokkos::RangePolicy<>(0, (int)P->getLocalNum()),
                    KOKKOS_LAMBDA(const int i) {
                        auto gen = pool.get_state();
                        for (int d = 0; d < 3; ++d)
                            P_view(i)[d] = gen.drand() * hr_min;
                        pool.free_state(gen);
                    });
                Kokkos::fence();
            }

            P->R = P->R + dt * P->P;

            ippl::Comm->barrier();
            auto t0 = Clock::now();

            P->update();

            Kokkos::fence();
            ippl::Comm->barrier();
            double elapsed = Sec(Clock::now() - t0).count();
            stepTimes.push_back(elapsed);

            P->P = P->P + dt * P->qm * P->E;

            if (infoEvery > 0 && (it + 1) % infoEvery == 0 && ippl::Comm->rank() == 0)
                std::cout << "  step " << (it + 1) << "/" << nt
                          << "  local particles: " << P->getLocalNum() << "\n";
        }

        std::vector<double> maxTimes(nt);
        MPI_Reduce(stepTimes.data(), maxTimes.data(), (int)nt, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);

        if (ippl::Comm->rank() == 0) {
            double total  = std::accumulate(maxTimes.begin(), maxTimes.end(), 0.0);
            double minVal = *std::min_element(maxTimes.begin(), maxTimes.end());
            double maxVal = *std::max_element(maxTimes.begin(), maxTimes.end());
            double mean   = total / (double)nt;

            std::cout << "\nParticleUpdate stats (max across ranks per step):\n"
                      << "  total=" << total << " s\n"
                      << "  mean=" << mean << " s\n"
                      << "  min=" << minVal << " s\n"
                      << "  max=" << maxVal << " s\n";

            const std::string csvPath = "particle_scaling_results.csv";
            const bool fileExists     = std::ifstream(csvPath).good();
            std::ofstream csv(csvPath, std::ios::app);
            if (!csv) {
                std::cerr << "Could not open " << csvPath << " for writing\n";
            } else {
                if (!fileExists)
                    csv << "exchange_mode,num_ranks,ranks_per_node,num_nodes,"
                           "update_total_s,update_mean_s,"
                           "update_min_s,update_max_s,nsteps\n";

                const int nRanks = ippl::Comm->size();
                const int rpn    = (ranksPerNode > 0) ? ranksPerNode : nRanks;
                const int nNodes = (nRanks + rpn - 1) / rpn;
                csv << std::fixed << std::setprecision(6) << exchangeStr << "," << nRanks << ","
                    << rpn << "," << nNodes << "," << total << "," << mean << "," << minVal << ","
                    << maxVal << "," << nt << "\n";
                std::cout << "Appended results to " << csvPath << "\n";
            }
        }
    }
    ippl::finalize();
    return 0;
}
