/**
 * @file BenchmarkTileSweep.cpp
 * @brief Performance benchmark sweeping tile size and kernel width
 *
 * Generates data for:
 *   Plot 1: Performance vs tile size (fixed kernel width)
 *   Plot 2: Performance vs kernel width (fixed tile size)
 *   Plot 3: Heatmap of performance across tile size × kernel width
 *
 * Usage: ./BenchmarkTileSweep [options]
 *   --grid N           Grid size per dimension (default: 256)
 *   --rho R            Particles per grid point (default: 10)
 *   --warmup N         Number of warmup runs (default: 3)
 *   --runs N           Number of benchmark runs (default: 10)
 *   --output FILE      Output CSV file prefix (default: tile_sweep)
 *   --min-tile T       Minimum tile size (default: 1)
 *   --max-tile T       Maximum tile size (default: 8)
 *   --min-width W      Minimum kernel width (default: 2)
 *   --max-width W      Maximum kernel width (default: 12)
 *   --dist D           Particle distribution: uniform, clustered (default: uniform)
 *   --ncu-mode         Single run mode for Nsight Compute profiling
 *   -v, --verbose      Verbose output
 */

#include "Ippl.h"
#include <Kokkos_Random.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <algorithm>
#include <vector>

using namespace ippl;

// ============================================================================
// Manual Timer Class
// ============================================================================

class ManualTimer {
public:
    using clock_type = std::chrono::high_resolution_clock;
    using time_point = clock_type::time_point;

    void start() {
        Kokkos::fence();
        start_time_ = clock_type::now();
    }

    double stop() {
        Kokkos::fence();
        auto end_time = clock_type::now();
        auto duration = std::chrono::duration<double>(end_time - start_time_);
        return duration.count();
    }

private:
    time_point start_time_;
};

// ============================================================================
// Benchmark Parameters
// ============================================================================

struct BenchParams {
    int n_grid = 128;
    double rho = 1.0;
    int warmup_runs = 3;
    int benchmark_runs = 5;
    std::string output_prefix = "tile_sweep";
    std::string distribution = "uniform";
    bool verbose = false;
    bool ncu_mode = false;

    // Sweep ranges
    int min_tile_size = 1;
    int max_tile_size = 8;
    int min_kernel_width = 2;
    int max_kernel_width = 8;

    size_t n_particles() const {
        return static_cast<size_t>(rho * n_grid * n_grid * n_grid);
    }
};

BenchParams parse_bench_args(int argc, char* argv[]) {
    BenchParams params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--grid" || arg == "-n") && i + 1 < argc) {
            params.n_grid = std::atoi(argv[++i]);
        } else if (arg == "--rho" && i + 1 < argc) {
            params.rho = std::atof(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            params.warmup_runs = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            params.benchmark_runs = std::atoi(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            params.output_prefix = argv[++i];
        } else if (arg == "--dist" && i + 1 < argc) {
            params.distribution = argv[++i];
        } else if (arg == "--min-tile" && i + 1 < argc) {
            params.min_tile_size = std::atoi(argv[++i]);
        } else if (arg == "--max-tile" && i + 1 < argc) {
            params.max_tile_size = std::atoi(argv[++i]);
        } else if (arg == "--min-width" && i + 1 < argc) {
            params.min_kernel_width = std::atoi(argv[++i]);
        } else if (arg == "--max-width" && i + 1 < argc) {
            params.max_kernel_width = std::atoi(argv[++i]);
        } else if (arg == "--ncu-mode") {
            params.ncu_mode = true;
            params.warmup_runs = 1;
            params.benchmark_runs = 1;
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        }
    }
    return params;
}

// ============================================================================
// Statistics Helper
// ============================================================================

struct TimingStats {
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    double median_ms;
    size_t count;
};

TimingStats compute_stats(const std::vector<double>& times_sec) {
    TimingStats stats{};
    stats.count = times_sec.size();

    if (stats.count == 0) {
        return stats;
    }

    std::vector<double> times_ms(stats.count);
    for (size_t i = 0; i < stats.count; ++i) {
        times_ms[i] = times_sec[i] * 1000.0;
    }

    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    stats.mean_ms = sum / stats.count;

    double sq_sum = 0.0;
    for (double t : times_ms) {
        sq_sum += (t - stats.mean_ms) * (t - stats.mean_ms);
    }
    stats.stddev_ms = (stats.count > 1) ? std::sqrt(sq_sum / (stats.count - 1)) : 0.0;

    stats.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    stats.max_ms = *std::max_element(times_ms.begin(), times_ms.end());

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    if (stats.count % 2 == 0) {
        stats.median_ms = (sorted[stats.count/2 - 1] + sorted[stats.count/2]) / 2.0;
    } else {
        stats.median_ms = sorted[stats.count/2];
    }

    return stats;
}

// ============================================================================
// Benchmark Result
// ============================================================================

struct BenchmarkResult {
    std::string method;             // "Tiled" or "OutputFocused"
    std::string distribution;
    int tile_size;
    int kernel_width;
    size_t n_particles;
    size_t n_grid;
    double rho;

    TimingStats stats;
    std::vector<double> times_sec;

    double throughput_Mpts_per_sec() const {
        return (n_particles / (stats.mean_ms * 1e-3)) / 1e6;
    }

    double time_per_point_ns() const {
        return (stats.mean_ms * 1e6) / n_particles;
    }
};

// ============================================================================
// Tile Sweep Benchmark Implementation
// ============================================================================

template <typename ExecSpace>
class TileSweepBenchmark {
public:
    static constexpr unsigned Dim = 3;
    using real_type = double;
    using complex_type = Kokkos::complex<real_type>;
    using MemSpace = typename ExecSpace::memory_space;

    using Mesh_t = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t = typename Mesh_t::DefaultCentering;
    using Field_t = ippl::Field<complex_type, Dim, Mesh_t, Centering_t>;
    using PLayout_t = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t = ippl::ParticleBase<PLayout_t>;

    TileSweepBenchmark(const BenchParams& params)
        : params_(params) {}

    void run() {
        print_header();

        std::vector<BenchmarkResult> results;

        // Generate kernel widths to test
        std::vector<int> widths;
        for (int w = params_.min_kernel_width; w <= params_.max_kernel_width; ++w) {
            widths.push_back(w);
        }

        // Generate tile sizes to test
        std::vector<int> tile_sizes;
        for (int t = params_.min_tile_size; t <= params_.max_tile_size; ++t) {
            tile_sizes.push_back(t);
        }

        int total_configs = widths.size() * tile_sizes.size() * 2;  // 2 methods
        int current_config = 0;

        // Sweep over kernel widths
        for (int width : widths) {
            // Create kernel with specific width
            // Width is approximately -log10(tol) + 1, so tol ≈ 10^(-(width-1))
            double tol = std::pow(10.0, -(width - 1));
            ippl::NUFFT::ESKernel<real_type> kernel(tol);

            // Verify we got the expected width
            int actual_width = kernel.width();
            if (actual_width != width && ippl::Comm->rank() == 0) {
                std::cout << "Note: Requested width " << width
                          << ", got " << actual_width << " (tol=" << tol << ")\n";
            }

            int nghost = actual_width / 2 + 1;

            // Setup domain and initialize particles (once per width)
            setup_domain(nghost);
            initialize(kernel, nghost);

            size_t n_particles = bunch_->getLocalNum();

            // Sweep over tile sizes
            for (int tile_size : tile_sizes) {
                ++current_config;

                if (ippl::Comm->rank() == 0) {
                    std::cout << "\r[" << current_config << "/" << total_configs << "] "
                              << "width=" << actual_width << ", tile=" << tile_size
                              << "          " << std::flush;
                }

                // Benchmark Tiled scatter
                {
                    auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
                    cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
                    cfg.tile_size_3d = tile_size;

                    auto result = benchmark_scatter("Tiled", cfg, kernel, nghost, n_particles, tile_size);
                    results.push_back(result);
                }

                ++current_config;

                // Benchmark OutputFocused scatter
                {
                    auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
                    cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
                    cfg.tile_size_3d = tile_size;

                    auto result = benchmark_scatter("OutputFocused", cfg, kernel, nghost, n_particles, tile_size);
                    results.push_back(result);
                }
            }

            cleanup();
        }

        if (ippl::Comm->rank() == 0) {
            std::cout << "\n";
        }

        // Output results
        write_full_csv(results);
        write_heatmap_csv(results, "Tiled");
        write_heatmap_csv(results, "OutputFocused");
        write_optimal_csv(results);

        print_summary(results);
    }

    void print_header() {
        if (ippl::Comm->rank() != 0) return;

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "     Tile Size × Kernel Width Sweep Benchmark\n";
        std::cout << "================================================================\n";
        std::cout << "Grid size:       " << params_.n_grid << "^3 = "
                  << (params_.n_grid * params_.n_grid * params_.n_grid) << " points\n";
        std::cout << "Particles/grid:  " << params_.rho << "\n";
        std::cout << "Total particles: " << params_.n_particles() << "\n";
        std::cout << "Distribution:    " << params_.distribution << "\n";
        std::cout << "Tile sizes:      " << params_.min_tile_size << " - " << params_.max_tile_size << "\n";
        std::cout << "Kernel widths:   " << params_.min_kernel_width << " - " << params_.max_kernel_width << "\n";
        std::cout << "Warmup runs:     " << params_.warmup_runs << "\n";
        std::cout << "Benchmark runs:  " << params_.benchmark_runs << "\n";
        std::cout << "================================================================\n\n";
    }

    BenchmarkResult benchmark_scatter(const std::string& method,
                                       const ippl::Interpolation::ScatterConfig& cfg,
                                       const ippl::NUFFT::ESKernel<real_type>& kernel,
                                       int nghost,
                                       size_t n_particles,
                                       int tile_size) {
        // Build result structure (filled with NaN on failure)
        BenchmarkResult r;
        r.method = method;
        r.distribution = params_.distribution;
        r.tile_size = tile_size;
        r.kernel_width = kernel.width();
        r.n_particles = n_particles;
        r.n_grid = params_.n_grid;
        r.rho = params_.rho;

        try {
            ManualTimer timer;

            // Warmup runs
            for (int i = 0; i < params_.warmup_runs; ++i) {
                *grid_ = complex_type(0.0, 0.0);
                Q_.scatter_kernel(*grid_, R_, kernel, cfg);
                grid_->accumulateHalo();
            }
            Kokkos::fence();

            // Benchmark runs
            std::vector<double> times;
            times.reserve(params_.benchmark_runs);

            for (int i = 0; i < params_.benchmark_runs; ++i) {
                *grid_ = complex_type(0.0, 0.0);
                Kokkos::fence();

                timer.start();
                Q_.scatter_kernel(*grid_, R_, kernel, cfg);
                grid_->accumulateHalo();
                double elapsed = timer.stop();

                times.push_back(elapsed);
            }

            r.times_sec = times;
            r.stats = compute_stats(times);

        } catch (const std::runtime_error& e) {
            // Handle insufficient shared memory or other runtime errors
            if (ippl::Comm->rank() == 0 && params_.verbose) {
                std::cout << "\n    [SKIP] " << method << " tile=" << tile_size
                          << " width=" << kernel.width() << ": " << e.what() << "\n";
            }

            // Fill with NaN to indicate failure
            r.times_sec.clear();
            r.stats.mean_ms = std::numeric_limits<double>::quiet_NaN();
            r.stats.stddev_ms = std::numeric_limits<double>::quiet_NaN();
            r.stats.min_ms = std::numeric_limits<double>::quiet_NaN();
            r.stats.max_ms = std::numeric_limits<double>::quiet_NaN();
            r.stats.median_ms = std::numeric_limits<double>::quiet_NaN();
            r.stats.count = 0;
        }

        return r;
    }

    void setup_domain(int nghost) {
        for (unsigned d = 0; d < Dim; ++d) {
            n_grid_[d] = params_.n_grid;
        }

        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d) {
            domain[d] = ippl::Index(n_grid_[d]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        layout_ = std::make_unique<ippl::FieldLayout<Dim>>(
            MPI_COMM_WORLD, domain, isParallel, true);

        for (unsigned d = 0; d < Dim; ++d) {
            origin_[d] = 0.0;
            hx_[d] = 2.0 * M_PI / static_cast<real_type>(n_grid_[d]);
        }

        mesh_ = std::make_unique<Mesh_t>(domain, hx_, origin_);
    }

    void initialize(const ippl::NUFFT::ESKernel<real_type>& kernel, int nghost) {
        grid_ = std::make_unique<Field_t>(*mesh_, *layout_, nghost);
        playout_ = std::make_unique<PLayout_t>(*layout_, *mesh_);
        bunch_ = std::make_unique<Bunch_t>(*playout_);

        bunch_->addAttribute(R_);
        bunch_->addAttribute(Q_);

        bunch_->setParticleBC(ippl::BC::PERIODIC);

        size_t n_local = params_.n_particles() / ippl::Comm->size();
        bunch_->create(n_local);

        auto R_view = R_.getView();
        Kokkos::Random_XorShift64_Pool<> rand_pool(42 + ippl::Comm->rank());

        if (params_.distribution == "uniform") {
            Kokkos::parallel_for("init_uniform", n_local,
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        R_view(i)[d] = gen.drand() * 2.0 * M_PI;
                    }
                    rand_pool.free_state(gen);
                });
        } else if (params_.distribution == "clustered") {
            Kokkos::parallel_for("init_clustered", n_local,
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        double u1 = gen.drand();
                        double u2 = gen.drand();
                        double z = Kokkos::sqrt(-2.0 * Kokkos::log(u1 + 1e-10))
                                   * Kokkos::cos(2.0 * M_PI * u2);
                        R_view(i)[d] = M_PI + 0.3 * z;
                        while (R_view(i)[d] < 0) R_view(i)[d] += 2.0 * M_PI;
                        while (R_view(i)[d] >= 2.0 * M_PI) R_view(i)[d] -= 2.0 * M_PI;
                    }
                    rand_pool.free_state(gen);
                });
        }

        auto Q_view = Q_.getView();
        Kokkos::parallel_for("init_values", n_local,
            KOKKOS_LAMBDA(const size_t i) {
                Q_view(i) = complex_type(1.0, 0.0);
            });

        Kokkos::fence();
    }

    void cleanup() {
        bunch_.reset();
        playout_.reset();
        grid_.reset();
        mesh_.reset();
        layout_.reset();
    }

    void write_full_csv(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_full.csv";
        std::ofstream out(filename);

        out << "method,distribution,tile_size,kernel_width,n_particles,n_grid,rho,"
            << "mean_ms,stddev_ms,min_ms,max_ms,median_ms,"
            << "throughput_Mpts_s,time_per_pt_ns,status\n";

        for (const auto& r : results) {
            out << r.method << ","
                << r.distribution << ","
                << r.tile_size << ","
                << r.kernel_width << ","
                << r.n_particles << ","
                << r.n_grid << ","
                << std::fixed << std::setprecision(1) << r.rho << ",";

            if (std::isnan(r.stats.mean_ms)) {
                out << "nan,nan,nan,nan,nan,nan,nan,failed\n";
            } else {
                out << std::setprecision(4) << r.stats.mean_ms << ","
                    << r.stats.stddev_ms << ","
                    << r.stats.min_ms << ","
                    << r.stats.max_ms << ","
                    << r.stats.median_ms << ","
                    << std::setprecision(2) << r.throughput_Mpts_per_sec() << ","
                    << r.time_per_point_ns() << ",ok\n";
            }
        }

        out.close();
        std::cout << "Wrote full results to: " << filename << "\n";
    }

    void write_heatmap_csv(const std::vector<BenchmarkResult>& results,
                           const std::string& method) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_heatmap_" + method + ".csv";
        std::ofstream out(filename);

        // Collect unique widths and tile sizes
        std::vector<int> widths, tiles;
        for (const auto& r : results) {
            if (r.method == method) {
                if (std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end()) {
                    widths.push_back(r.kernel_width);
                }
                if (std::find(tiles.begin(), tiles.end(), r.tile_size) == tiles.end()) {
                    tiles.push_back(r.tile_size);
                }
            }
        }
        std::sort(widths.begin(), widths.end());
        std::sort(tiles.begin(), tiles.end());

        // Header: tile_size, width_2, width_3, ...
        out << "tile_size";
        for (int w : widths) {
            out << ",width_" << w;
        }
        out << "\n";

        // Data rows
        for (int t : tiles) {
            out << t;
            for (int w : widths) {
                bool found = false;
                for (const auto& r : results) {
                    if (r.method == method && r.tile_size == t && r.kernel_width == w) {
                        if (std::isnan(r.stats.mean_ms)) {
                            out << ",nan";
                        } else {
                            out << "," << std::fixed << std::setprecision(2)
                                << r.throughput_Mpts_per_sec();
                        }
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    out << ",nan";
                }
            }
            out << "\n";
        }

        out.close();
        std::cout << "Wrote heatmap for " << method << " to: " << filename << "\n";
    }

    void write_optimal_csv(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_optimal.csv";
        std::ofstream out(filename);

        out << "method,kernel_width,optimal_tile_size,throughput_Mpts_s,time_ms\n";

        // Find optimal tile size for each (method, width) combination
        std::vector<std::string> methods = {"Tiled", "OutputFocused"};
        std::vector<int> widths;
        for (const auto& r : results) {
            if (std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end()) {
                widths.push_back(r.kernel_width);
            }
        }
        std::sort(widths.begin(), widths.end());

        for (const auto& method : methods) {
            for (int w : widths) {
                const BenchmarkResult* best = nullptr;
                double best_throughput = 0.0;

                for (const auto& r : results) {
                    if (r.method == method && r.kernel_width == w) {
                        // Skip NaN results
                        if (std::isnan(r.stats.mean_ms)) continue;

                        double tp = r.throughput_Mpts_per_sec();
                        if (tp > best_throughput) {
                            best_throughput = tp;
                            best = &r;
                        }
                    }
                }

                if (best) {
                    out << method << ","
                        << w << ","
                        << best->tile_size << ","
                        << std::fixed << std::setprecision(2) << best->throughput_Mpts_per_sec() << ","
                        << std::setprecision(4) << best->stats.mean_ms << "\n";
                } else {
                    // All configurations failed for this method/width
                    out << method << ","
                        << w << ","
                        << "nan,nan,nan\n";
                }
            }
        }

        out.close();
        std::cout << "Wrote optimal configurations to: " << filename << "\n";
    }

    void print_summary(const std::vector<BenchmarkResult>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "                    Results Summary\n";
        std::cout << "================================================================\n";

        // Count failed configurations
        int failed_count = 0;
        for (const auto& r : results) {
            if (std::isnan(r.stats.mean_ms)) {
                ++failed_count;
            }
        }
        if (failed_count > 0) {
            std::cout << "\nNote: " << failed_count << " configuration(s) failed "
                      << "(likely insufficient shared memory)\n";
        }

        // Find best configurations
        std::vector<std::string> methods = {"Tiled", "OutputFocused"};

        for (const auto& method : methods) {
            std::cout << "\n" << method << " - Optimal tile sizes by kernel width:\n";
            std::cout << std::string(60, '-') << "\n";
            std::cout << std::left << std::setw(8) << "Width"
                      << std::right << std::setw(12) << "Best Tile"
                      << std::setw(14) << "Mpts/s"
                      << std::setw(12) << "Time (ms)" << "\n";
            std::cout << std::string(60, '-') << "\n";

            std::vector<int> widths;
            for (const auto& r : results) {
                if (r.method == method) {
                    if (std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end()) {
                        widths.push_back(r.kernel_width);
                    }
                }
            }
            std::sort(widths.begin(), widths.end());

            for (int w : widths) {
                const BenchmarkResult* best = nullptr;
                double best_throughput = 0.0;

                for (const auto& r : results) {
                    if (r.method == method && r.kernel_width == w) {
                        // Skip NaN results
                        if (std::isnan(r.stats.mean_ms)) continue;

                        double tp = r.throughput_Mpts_per_sec();
                        if (tp > best_throughput) {
                            best_throughput = tp;
                            best = &r;
                        }
                    }
                }

                if (best) {
                    std::cout << std::left << std::setw(8) << w
                              << std::right << std::setw(12) << best->tile_size
                              << std::fixed << std::setprecision(1)
                              << std::setw(14) << best->throughput_Mpts_per_sec()
                              << std::setprecision(3)
                              << std::setw(12) << best->stats.mean_ms << "\n";
                } else {
                    std::cout << std::left << std::setw(8) << w
                              << std::right << std::setw(12) << "N/A"
                              << std::setw(14) << "N/A"
                              << std::setw(12) << "N/A"
                              << "  (all failed)\n";
                }
            }
        }

        // Compare methods at their optimal configurations
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "          Method Comparison (at optimal tile sizes)\n";
        std::cout << "================================================================\n";
        std::cout << std::left << std::setw(8) << "Width"
                  << std::setw(20) << "Tiled (Mpts/s)"
                  << std::setw(20) << "OutputFocused (Mpts/s)"
                  << std::setw(12) << "Ratio" << "\n";
        std::cout << std::string(60, '-') << "\n";

        std::vector<int> widths;
        for (const auto& r : results) {
            if (std::find(widths.begin(), widths.end(), r.kernel_width) == widths.end()) {
                widths.push_back(r.kernel_width);
            }
        }
        std::sort(widths.begin(), widths.end());

        for (int w : widths) {
            double tiled_tp = 0.0, output_tp = 0.0;

            for (const auto& r : results) {
                if (r.kernel_width == w && !std::isnan(r.stats.mean_ms)) {
                    double tp = r.throughput_Mpts_per_sec();
                    if (r.method == "Tiled" && tp > tiled_tp) {
                        tiled_tp = tp;
                    } else if (r.method == "OutputFocused" && tp > output_tp) {
                        output_tp = tp;
                    }
                }
            }

            std::cout << std::left << std::setw(8) << w;

            if (tiled_tp > 0) {
                std::cout << std::right << std::fixed << std::setprecision(1)
                          << std::setw(20) << tiled_tp;
            } else {
                std::cout << std::right << std::setw(20) << "N/A";
            }

            if (output_tp > 0) {
                std::cout << std::fixed << std::setprecision(1)
                          << std::setw(20) << output_tp;
            } else {
                std::cout << std::setw(20) << "N/A";
            }

            if (tiled_tp > 0 && output_tp > 0) {
                double ratio = tiled_tp / output_tp;
                std::string winner = (tiled_tp > output_tp) ? "T" : "O";
                std::cout << std::setprecision(2)
                          << std::setw(10) << ratio
                          << "  " << winner;
            } else {
                std::cout << std::setw(10) << "N/A" << "  -";
            }
            std::cout << "\n";
        }

        std::cout << "\n";
    }

private:
    BenchParams params_;

    ippl::Vector<std::size_t, Dim> n_grid_;
    ippl::Vector<real_type, Dim> origin_;
    ippl::Vector<real_type, Dim> hx_;

    std::unique_ptr<ippl::FieldLayout<Dim>> layout_;
    std::unique_ptr<Mesh_t> mesh_;
    std::unique_ptr<Field_t> grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t> bunch_;

    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R_;
    ippl::ParticleAttrib<complex_type> Q_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        auto params = parse_bench_args(argc, argv);

        TileSweepBenchmark<Kokkos::DefaultExecutionSpace> benchmark(params);
        benchmark.run();
    }

    ippl::finalize();
    return EXIT_SUCCESS;
}
