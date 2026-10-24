//
// benchmarkLargeMessageMigration.cpp
//
// Times ParticleSpatialLayout::update() when single rank pairs exchange messages of a chosen
// size, up to and beyond the 2 GiB MPI int-count limit, and records the device memory the
// exchange needs. The particles carry the attributes of an OPALX ParticleContainer (123 bytes
// per particle), so message sizes translate directly to OPALX runs.
//
// The domain is split into one slab per rank along x. Before each timed update, the first
// fraction*N local particles of every rank are moved into other slabs:
//   shift     all to rank (r + 1) % P: one large message per rank pair
//   alltoall  spread evenly over the P - 1 other ranks
// Each rank keeps N particles, so repeated updates are statistically identical.
//
// Usage:
//   srun ./benchmarkLargeMessageMigration --particles-per-rank N [--fraction f]
//       [--pattern shift|alltoall] [--repeats R] [--warmup W] [--sample-us U]
//       [--label name] [--csv file] [--timings]
//
// The memory columns are the device bytes in use minus the bytes in use before the first
// exchange, maximised over ranks: "after" once update() returned (buffers kept in the pool
// and grown particle storage), "peak" sampled every U microseconds by a second host thread
// during one extra, untimed update.
//

#include "Ippl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "Utility/IpplException.h"

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#else
#include <unistd.h>
#endif

// Named namespace: CUDA extended lambdas need enclosing functions with external linkage.
namespace migration_bench {
    constexpr unsigned Dim = 3;
    using Mesh_t           = ippl::UniformCartesian<double, Dim>;
    using PLayout_t        = ippl::ParticleSpatialLayout<double, Dim, Mesh_t>;
    using FieldLayout_t    = ippl::FieldLayout<Dim>;
    using Vector_t         = ippl::Vector<double, Dim>;
    using Clock            = std::chrono::steady_clock;

    // Mirrors the default OPALX ParticleContainer: R, ID, dt, Phi, Bin, P, E, B, InvalidMask.
    class OpalxLikeBunch
        : public ippl::ParticleBase<PLayout_t, Kokkos::DefaultExecutionSpace::memory_space> {
        using Base = ippl::ParticleBase<PLayout_t, Kokkos::DefaultExecutionSpace::memory_space>;

    public:
        ippl::ParticleAttrib<double> dt;
        ippl::ParticleAttrib<double> Phi;
        ippl::ParticleAttrib<short int> Bin;
        typename Base::particle_position_type P;
        typename Base::particle_position_type E;
        typename Base::particle_position_type B;
        ippl::ParticleAttrib<bool> InvalidMask;

        explicit OpalxLikeBunch(PLayout_t& layout)
            : Base(layout) {
            this->addAttribute(dt);
            this->addAttribute(Phi);
            this->addAttribute(Bin);
            this->addAttribute(P);
            this->addAttribute(E);
            this->addAttribute(B);
            this->addAttribute(InvalidMask);
        }

        static constexpr std::size_t bytesPerParticle = 4 * sizeof(Vector_t) + sizeof(std::int64_t)
                                                        + 2 * sizeof(double) + sizeof(short int)
                                                        + sizeof(bool);
    };

    std::size_t deviceBytesInUse() {
#if defined(KOKKOS_ENABLE_CUDA)
        std::size_t freeBytes = 0, totalBytes = 0;
        cudaMemGetInfo(&freeBytes, &totalBytes);
        return totalBytes - freeBytes;
#else
        // Resident set size of this process (Linux), the closest host-side analogue.
        std::ifstream statm("/proc/self/statm");
        std::size_t pages = 0, resident = 0;
        if (statm >> pages >> resident) {
            return resident * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        }
        return 0;
#endif
    }

    std::size_t deviceBytesTotal() {
#if defined(KOKKOS_ENABLE_CUDA)
        std::size_t freeBytes = 0, totalBytes = 0;
        cudaMemGetInfo(&freeBytes, &totalBytes);
        return totalBytes;
#else
        return 0;
#endif
    }

    //! Records the maximum of deviceBytesInUse() while it is alive.
    class PeakSampler {
    public:
        explicit PeakSampler(int intervalUs)
            : peak_m(deviceBytesInUse()) {
#if defined(KOKKOS_ENABLE_CUDA)
            int device = 0;
            cudaGetDevice(&device);
#endif
            thread_m = std::thread([this, intervalUs
#if defined(KOKKOS_ENABLE_CUDA)
                                    ,
                                    device
#endif
            ] {
#if defined(KOKKOS_ENABLE_CUDA)
                cudaSetDevice(device);
#endif
                while (!stop_m.load(std::memory_order_relaxed)) {
                    peak_m = std::max(peak_m.load(), deviceBytesInUse());
                    std::this_thread::sleep_for(std::chrono::microseconds(intervalUs));
                }
            });
        }

        std::size_t stop() {
            stop_m = true;
            thread_m.join();
            return std::max(peak_m.load(), deviceBytesInUse());
        }

    private:
        std::atomic<bool> stop_m{false};
        std::atomic<std::size_t> peak_m;
        std::thread thread_m;
    };

    // The 606 branch has no message cap; report -1 there.
    template <typename Comm>
    long long maxMessageBytes(const Comm& comm) {
        if constexpr (requires { comm.getMaxMessageBytes(); }) {
            return static_cast<long long>(comm.getMaxMessageBytes());
        } else {
            return -1;
        }
    }

    // Moves the first nMove local particles into the slabs of other ranks (x in [r, r + 1)) and
    // copies the new x into P[0], so a later check can tell whether attributes stayed together.
    void displace(OpalxLikeBunch& bunch, std::size_t nMove, bool alltoall) {
        const int rank   = ippl::Comm->rank();
        const int nRanks = ippl::Comm->size();
        auto R           = bunch.R.getView();
        auto P           = bunch.P.getView();
        Kokkos::parallel_for(
            "displace", Kokkos::RangePolicy<>(0, bunch.getLocalNum()),
            KOKKOS_LAMBDA(const std::size_t i) {
                int dest = rank;
                if (i < nMove) {
                    dest = alltoall ? (rank + 1 + static_cast<int>(i % (nRanks - 1))) % nRanks
                                    : (rank + 1) % nRanks;
                }
                R(i)[0] = dest + 0.25 + 0.5 * ((i % 1024) / 1024.0);
                R(i)[1] = 0.5;
                R(i)[2] = 0.5;
                P(i)[0] = R(i)[0];
            });
        Kokkos::fence();
    }

    // Local particles outside this rank's slab or with P[0] != R[0] after an update.
    std::size_t countMisplaced(OpalxLikeBunch& bunch) {
        const double lo = ippl::Comm->rank();
        auto R          = bunch.R.getView();
        auto P          = bunch.P.getView();
        std::size_t bad = 0;
        Kokkos::parallel_reduce(
            "countMisplaced", Kokkos::RangePolicy<>(0, bunch.getLocalNum()),
            KOKKOS_LAMBDA(const std::size_t i, std::size_t& sum) {
                if (R(i)[0] < lo || R(i)[0] > lo + 1.0 || P(i)[0] != R(i)[0]) {
                    ++sum;
                }
            },
            bad);
        return bad;
    }

    double timedUpdate(OpalxLikeBunch& bunch) {
        Kokkos::fence();
        ippl::Comm->barrier();
        const auto t0 = Clock::now();
        bunch.update();
        Kokkos::fence();
        ippl::Comm->barrier();
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }

    double median(std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const std::size_t n = values.size();
        return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    }
}  // namespace migration_bench

using namespace migration_bench;

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    // Report allocation failures with their message, so the sweep can tell them apart.
    const auto fail = [](const std::string& what) {
        std::cerr << "[benchmarkLargeMessageMigration] rank " << ippl::Comm->rank()
                  << " failed: " << what << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    };
    try {
        std::size_t particlesPerRank = 0;
        double fraction              = 1.0;
        std::string pattern          = "shift";
        int repeats                  = 5;
        int warmup                   = 1;
        int sampleUs                 = 200;
        std::string label            = "unnamed";
        std::string csvPath          = "large_message_migration.csv";
        bool printTimings            = false;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            const bool hasValue    = i + 1 < argc;
            if (flag == "--particles-per-rank" && hasValue)
                particlesPerRank = std::stoull(argv[++i]);
            else if (flag == "--fraction" && hasValue)
                fraction = std::stod(argv[++i]);
            else if (flag == "--pattern" && hasValue)
                pattern = argv[++i];
            else if (flag == "--repeats" && hasValue)
                repeats = std::stoi(argv[++i]);
            else if (flag == "--warmup" && hasValue)
                warmup = std::stoi(argv[++i]);
            else if (flag == "--sample-us" && hasValue)
                sampleUs = std::stoi(argv[++i]);
            else if (flag == "--label" && hasValue)
                label = argv[++i];
            else if (flag == "--csv" && hasValue)
                csvPath = argv[++i];
            else if (flag == "--timings")
                printTimings = true;
        }

        const int rank   = ippl::Comm->rank();
        const int nRanks = ippl::Comm->size();
        if (particlesPerRank == 0 || nRanks < 2 || (pattern != "shift" && pattern != "alltoall")
            || fraction <= 0.0 || fraction > 1.0 || repeats < 1) {
            if (rank == 0) {
                std::cerr << "Usage: " << argv[0]
                          << " --particles-per-rank N [--fraction f] [--pattern shift|alltoall]"
                             " [--repeats R] [--warmup W] [--sample-us U] [--label name]"
                             " [--csv file]  (needs at least 2 ranks)\n";
            }
            ippl::finalize();
            return 1;
        }
        const bool alltoall = pattern == "alltoall";

        // One slab of 4 cells per rank along x; region of rank r is x in [r, r + 1].
        ippl::NDIndex<Dim> domain;
        domain[0]                        = ippl::Index(4 * nRanks);
        domain[1]                        = ippl::Index(4);
        domain[2]                        = ippl::Index(4);
        std::array<bool, Dim> isParallel = {true, false, false};
        Vector_t hr                      = {0.25, 0.25, 0.25};
        Vector_t origin                  = {0.0, 0.0, 0.0};
        Mesh_t mesh(domain, hr, origin);
        FieldLayout_t fieldLayout(MPI_COMM_WORLD, domain, isParallel);
        PLayout_t particleLayout(fieldLayout, mesh);

        OpalxLikeBunch bunch(particleLayout);
        bunch.create(particlesPerRank);
        bunch.dt          = 1.0;
        bunch.Phi         = 0.0;
        bunch.Bin         = static_cast<short int>(0);
        bunch.P           = 1.0;
        bunch.E           = 0.0;
        bunch.B           = 0.0;
        bunch.InvalidMask = false;
        displace(bunch, 0, alltoall);
        bunch.update();

        const std::size_t nMove       = static_cast<std::size_t>(fraction * particlesPerRank);
        const std::size_t peers       = alltoall ? nRanks - 1 : 1;
        const std::size_t bytesPerRun = nMove * OpalxLikeBunch::bytesPerParticle;
        const std::size_t messageSize = bytesPerRun / peers;

        Kokkos::fence();
        ippl::Comm->barrier();
        const std::size_t baseline = deviceBytesInUse();

        std::vector<double> times;
        std::size_t afterMax = 0;
        for (int it = 0; it < warmup + repeats; ++it) {
            displace(bunch, nMove, alltoall);
            const double t = timedUpdate(bunch);
            afterMax       = std::max(afterMax, deviceBytesInUse());
            if (it >= warmup) {
                times.push_back(t);
            }
        }

        displace(bunch, nMove, alltoall);
        Kokkos::fence();
        ippl::Comm->barrier();
        PeakSampler sampler(sampleUs);
        bunch.update();
        Kokkos::fence();
        const std::size_t peak = sampler.stop();
        afterMax               = std::max(afterMax, deviceBytesInUse());

        // Every rank keeps its particle count, and every particle must sit in its owner's slab
        // with its attributes intact.
        unsigned long long localNum = bunch.getLocalNum(), minNum = 0, maxNum = 0;
        unsigned long long misplaced = countMisplaced(bunch), misplacedSum = 0;
        MPI_Allreduce(&localNum, &minNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localNum, &maxNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&misplaced, &misplacedSum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        const bool valid =
            minNum == particlesPerRank && maxNum == particlesPerRank && misplacedSum == 0;

        // Times: maximum over ranks per repetition (all ranks barrier, so they agree closely).
        std::vector<double> maxTimes(times.size());
        MPI_Allreduce(times.data(), maxTimes.data(), static_cast<int>(times.size()), MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
        long long overhead[3] = {
            static_cast<long long>(afterMax) - static_cast<long long>(baseline),
            static_cast<long long>(peak) - static_cast<long long>(baseline),
            static_cast<long long>(baseline)};
        long long overheadMax[3];
        MPI_Allreduce(overhead, overheadMax, 3, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            const double tMedian = median(maxTimes);
            const double tMin    = *std::min_element(maxTimes.begin(), maxTimes.end());
            const double tMax    = *std::max_element(maxTimes.begin(), maxTimes.end());
            const char* nodes    = std::getenv("SLURM_NNODES");

            std::ifstream existing(csvPath);
            const bool writeHeader = !existing.good() || existing.peek() == EOF;
            existing.close();
            std::ofstream csv(csvPath, std::ios::app);
            if (writeHeader) {
                csv << "label,max_message_bytes,ranks,nodes,pattern,particles_per_rank,fraction,"
                       "bytes_per_particle,message_bytes,bytes_sent_per_rank,repeats,"
                       "t_median_s,t_min_s,t_max_s,mem_after_bytes,mem_peak_bytes,"
                       "mem_baseline_bytes,mem_total_bytes,valid\n";
            }
            csv << label << ',' << maxMessageBytes(*ippl::Comm) << ',' << nRanks << ','
                << (nodes ? nodes : "1") << ',' << pattern << ',' << particlesPerRank << ','
                << fraction << ',' << OpalxLikeBunch::bytesPerParticle << ',' << messageSize << ','
                << bytesPerRun << ',' << repeats << ',' << tMedian << ',' << tMin << ',' << tMax
                << ',' << overheadMax[0] << ',' << overheadMax[1] << ',' << overheadMax[2] << ','
                << deviceBytesTotal() << ',' << (valid ? 1 : 0) << '\n';

            std::cout << "[benchmarkLargeMessageMigration] " << label << " pattern=" << pattern
                      << " message=" << messageSize / 1e9 << " GB t_median=" << tMedian
                      << " s mem_after=" << overheadMax[0] / 1e9
                      << " GB mem_peak=" << overheadMax[1] / 1e9 << " GB valid=" << valid
                      << std::endl;
        }
        if (printTimings) {
            IpplTimings::print();
        }
    } catch (const IpplException& e) {
        fail(e.where() + ": " + e.what());
    } catch (const std::exception& e) {
        fail(e.what());
    }
    ippl::finalize();
    return 0;
}
