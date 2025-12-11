#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "FFT/NUFFT/ESKernel.h"
#include "FFT/NUFFT/NUFFTUtilities.h"
#include "Interpolation/ScatterConfig.h"

namespace ippl::benchmark {

    // ============================================================================
    // Timing and Statistics
    // ============================================================================

    struct TimingStats {
        double min_ms;
        double max_ms;
        double mean_ms;
        double stddev_ms;
        double median_ms;
        int num_samples;
    };

    inline TimingStats compute_stats(std::vector<double>& times_ms) {
        TimingStats stats{};
        if (times_ms.empty()) {
            return stats;
        }

        stats.num_samples = static_cast<int>(times_ms.size());

        std::sort(times_ms.begin(), times_ms.end());
        stats.min_ms    = times_ms.front();
        stats.max_ms    = times_ms.back();
        stats.median_ms = times_ms[times_ms.size() / 2];

        double sum    = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        stats.mean_ms = sum / stats.num_samples;

        double sq_sum = 0.0;
        for (double t : times_ms) {
            sq_sum += (t - stats.mean_ms) * (t - stats.mean_ms);
        }
        stats.stddev_ms =
            (stats.num_samples > 1) ? std::sqrt(sq_sum / (stats.num_samples - 1)) : 0.0;

        return stats;
    }

    class Timer {
    public:
        void start() {
            Kokkos::fence();
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        double stop() {
            Kokkos::fence();
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time_);
            return duration.count();
        }

    private:
        std::chrono::high_resolution_clock::time_point start_time_;
    };

    // ============================================================================
    // Benchmark Configuration
    // ============================================================================

    struct BenchmarkParams {
        std::size_t n_particles = 100000;
        std::size_t n_grid      = 64;
        int warmup_runs         = 5;
        int benchmark_runs      = 20;
        double kernel_tol       = 1e-6;
        bool verbose            = false;
    };

    inline void print_header(const std::string& title) {
        int rank = ippl::Comm->rank();
        if (rank == 0) {
            std::cout << "\n" << std::string(70, '=') << "\n";
            std::cout << " " << title << "\n";
            std::cout << std::string(70, '=') << "\n\n";
        }
    }

    inline void print_subheader(const std::string& subtitle) {
        int rank = ippl::Comm->rank();
        if (rank == 0) {
            std::cout << "\n--- " << subtitle << " ---\n";
        }
    }

    inline void print_stats_row(const std::string& label, const TimingStats& stats) {
        int rank = ippl::Comm->rank();
        if (rank == 0) {
            std::cout << std::left << std::setw(25) << label << std::right << std::fixed
                      << std::setprecision(3) << std::setw(10) << stats.mean_ms << " ms"
                      << " ± " << std::setw(8) << stats.stddev_ms << " ms"
                      << "  [min: " << std::setw(8) << stats.min_ms << ", max: " << std::setw(8)
                      << stats.max_ms << ", med: " << std::setw(8) << stats.median_ms << "]\n";
        }
    }

    inline void print_params(const BenchmarkParams& params) {
        int rank   = ippl::Comm->rank();
        int nRanks = ippl::Comm->size();
        if (rank == 0) {
            std::cout << "Configuration:\n";
            std::cout << "  MPI ranks:       " << nRanks << "\n";
            std::cout << "  Particles:       " << params.n_particles << "\n";
            std::cout << "  Grid:            " << params.n_grid << "^3\n";
            std::cout << "  Warmup runs:     " << params.warmup_runs << "\n";
            std::cout << "  Benchmark runs:  " << params.benchmark_runs << "\n";
            std::cout << "  Kernel tol:      " << params.kernel_tol << "\n";
            std::cout << "  Execution space: " << Kokkos::DefaultExecutionSpace::name() << "\n";
            std::cout << "\n";
        }
    }

    // ============================================================================
    // Particle Bunch Template
    // ============================================================================

    template <class PLayout>
    struct BenchmarkBunch : public ippl::ParticleBase<PLayout> {
        using base_type    = ippl::ParticleBase<PLayout>;
        using real_type    = double;
        using complex_type = Kokkos::complex<real_type>;

        explicit BenchmarkBunch(PLayout& playout)
            : base_type(playout) {
            this->addAttribute(Q);
            this->addAttribute(Q_result);
        }

        ippl::ParticleAttrib<complex_type> Q;
        ippl::ParticleAttrib<complex_type> Q_result;
    };

    // ============================================================================
    // Setup Utilities
    // ============================================================================

    template <typename ExecSpace, unsigned Dim, typename real_type, typename Mesh_t,
              typename FieldLayout_t, typename PLayout_t, typename Bunch_t, typename Field_t>
    void setup_benchmark_data(const BenchmarkParams& params,
                              const ippl::NUFFT::ESKernel<real_type>& kernel, Mesh_t& mesh,
                              FieldLayout_t& layout, PLayout_t& playout, Bunch_t& bunch,
                              Field_t& grid_data, int nghost) {
        using complex_type = Kokkos::complex<real_type>;
        using RandPoolType = Kokkos::Random_XorShift64_Pool<ExecSpace>;

        int rank   = ippl::Comm->rank();
        int nRanks = ippl::Comm->size();

        // Get mesh parameters
        const auto& origin = mesh.getOrigin();
        const auto& hx     = mesh.getMeshSpacing();
        const auto& lDom   = layout.getLocalNDIndex();

        // Compute local bounds (no longer used for particle positions,
        // but kept here in case other code relies on them)
        ippl::Vector<real_type, Dim> local_min, local_max;
        for (unsigned d = 0; d < Dim; ++d) {
            local_min[d] = origin[d] + lDom[d].first() * hx[d];
            local_max[d] = origin[d] + (lDom[d].last() + 1) * hx[d];
        }

        // Create particles
        std::size_t Ntotal = params.n_particles;
        if (Ntotal % nRanks != 0) {
            if (rank == 0) {
                std::cerr << "Warning: Ntotal adjusted to be divisible by nRanks\n";
            }
            Ntotal = ((Ntotal / nRanks) + 1) * nRanks;
        }
        std::size_t nLoc = Ntotal / nRanks;
        bunch.create(nLoc);

        // Initialize particles on device
        RandPoolType rand_pool(42 + rank * 12345);

        auto R_view  = bunch.R.getView();
        auto Q_view  = bunch.Q.getView();
        auto Qr_view = bunch.Q_result.getView();

        // --- NEW: global uniform sampling in [0, 2*pi] for each coordinate ---
        const real_type two_pi = static_cast<real_type>(2.0 * M_PI);
        // ---------------------------------------------------------------------

        Kokkos::parallel_for(
            "InitParticles", Kokkos::RangePolicy<ExecSpace>(0, nLoc),
            KOKKOS_LAMBDA(const std::size_t i) {
                auto gen = rand_pool.get_state();

                ippl::Vector<real_type, Dim> r;
                // positions ~ U(0, 2*pi) in each dimension
                for (unsigned d = 0; d < Dim; ++d) {
                    r[d] = gen.drand(0.0, two_pi);
                }
                R_view(i) = r;

                // Box-Muller for complex Gaussian
                real_type u1 = gen.drand(1e-10, 1.0);
                real_type u2 = gen.drand(0.0, 1.0);
                real_type u3 = gen.drand(1e-10, 1.0);
                real_type u4 = gen.drand(0.0, 1.0);

                real_type re = Kokkos::sqrt(-2.0 * Kokkos::log(u1)) * Kokkos::cos(2.0 * M_PI * u2);
                real_type im = Kokkos::sqrt(-2.0 * Kokkos::log(u3)) * Kokkos::cos(2.0 * M_PI * u4);
                Q_view(i)    = complex_type(re, im);
                Qr_view(i)   = complex_type(0.0, 0.0);

                rand_pool.free_state(gen);
            });
        Kokkos::fence();

        // --- unchanged grid init below ---

        auto grid_view    = grid_data.getView();
        const int i_start = lDom[0].first();
        const int j_start = lDom[1].first();
        const int k_start = lDom[2].first();
        const int i_end   = lDom[0].last() + 1;
        const int j_end   = lDom[1].last() + 1;
        const int k_end   = lDom[2].last() + 1;

        const std::size_t ng0 = params.n_grid;
        const std::size_t ng1 = params.n_grid;
        const int ng          = nghost;

        using mdrange_policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;

        Kokkos::parallel_for(
            "InitGrid", mdrange_policy({k_start, j_start, i_start}, {k_end, j_end, i_end}),
            KOKKOS_LAMBDA(const int k_global, const int j_global, const int i_global) {
                std::size_t global_idx = k_global * ng0 * ng1 + j_global * ng0 + i_global;
                Kokkos::Random_XorShift64<ExecSpace> gen(42 + global_idx * 2);

                real_type re = gen.drand(-1.0, 1.0);
                real_type im = gen.drand(-1.0, 1.0);

                int i_local = i_global - i_start;
                int j_local = j_global - j_start;
                int k_local = k_global - k_start;

                grid_view(i_local + ng, j_local + ng, k_local + ng) = complex_type(re, im);
            });
        Kokkos::fence();

        grid_data.fillHalo();
        bunch.update();
    }
    // ============================================================================
    // Configuration String Helpers
    // ============================================================================

    inline std::string scatter_method_name(ippl::Interpolation::ScatterMethod method) {
        switch (method) {
            case ippl::Interpolation::ScatterMethod::Atomic:
                return "Atomic";
            case ippl::Interpolation::ScatterMethod::Tiled:
                return "Tiled";
            case ippl::Interpolation::ScatterMethod::OutputFocused:
                return "OutputFocused";
            default:
                return "Unknown";
        }
    }

    inline std::string gather_method_name(ippl::Interpolation::GatherMethod method) {
        switch (method) {
            case ippl::Interpolation::GatherMethod::Atomic:
                return "Atomic";
            case ippl::Interpolation::GatherMethod::Tiled:
                return "Tiled";
            case ippl::Interpolation::GatherMethod::Native:
                return "OutputFocused";
            case ippl::Interpolation::GatherMethod::AtomicSort:
                return "AtomicSort";
            default:
                return "Unknown";
        }
    }

    inline std::string config_label(const ippl::Interpolation::ScatterConfig& cfg) {
        std::string label = scatter_method_name(cfg.method);
        if (cfg.method == ippl::Interpolation::ScatterMethod::Tiled
            || cfg.method == ippl::Interpolation::ScatterMethod::OutputFocused) {
            label += " (tile=" + std::to_string(cfg.tile_size_3d) + ")";
        }
        return label;
    }

    inline std::string config_label(const ippl::Interpolation::GatherConfig& cfg) {
        std::string label = gather_method_name(cfg.method);
        if (cfg.method == ippl::Interpolation::GatherMethod::AtomicSort) {
            label += " [sorted]";
        }
        return label;
    }

    // ============================================================================
    // Command Line Parsing
    // ============================================================================

    inline BenchmarkParams parse_args(int argc, char* argv[]) {
        BenchmarkParams params;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--particles" && i + 1 < argc) {
                params.n_particles = std::stoull(argv[++i]);
            } else if (arg == "--grid" && i + 1 < argc) {
                params.n_grid = std::stoull(argv[++i]);
            } else if (arg == "--warmup" && i + 1 < argc) {
                params.warmup_runs = std::stoi(argv[++i]);
            } else if (arg == "--runs" && i + 1 < argc) {
                params.benchmark_runs = std::stoi(argv[++i]);
            } else if (arg == "--tol" && i + 1 < argc) {
                params.kernel_tol = std::stod(argv[++i]);
            } else if (arg == "-v" || arg == "--verbose") {
                params.verbose = true;
            } else if (arg == "--help" || arg == "-h") {
                if (ippl::Comm->rank() == 0) {
                    std::cout << "Usage: " << argv[0] << " [options]\n"
                              << "Options:\n"
                              << "  --particles N   Number of particles (default: 100000)\n"
                              << "  --grid N        Grid size per dimension (default: 64)\n"
                              << "  --warmup N      Number of warmup runs (default: 5)\n"
                              << "  --runs N        Number of benchmark runs (default: 20)\n"
                              << "  --tol T         Kernel tolerance (default: 1e-6)\n"
                              << "  -v, --verbose   Verbose output\n"
                              << "  -h, --help      Show this help\n";
                }
                ippl::finalize();
                std::exit(0);
            }
        }

        return params;
    }

}  // namespace ippl::benchmark

#endif  // BENCHMARK_UTILS_H