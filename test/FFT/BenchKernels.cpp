/**
 * @file BenchmarkRoofline.cpp
 * @brief Throughput benchmark for scatter/gather kernels
 *
 * Generates data for:
 *   Plot 1: Kernel variant comparison at fixed parameters
 *   Plot 2: Throughput vs kernel width w (accuracy sensitivity)
 *
 * For roofline analysis, use Nsight Compute with --set roofline:
 *   ncu --set roofline ./BenchmarkRoofline --ncu-mode
 *
 * Usage: ./BenchmarkRoofline [options]
 *   --grid N        Grid size per dimension (default: 256)
 *   --rho R         Particles per grid point (default: 10)
 *   --tol T         Kernel tolerance (default: 1e-6, gives w~7)
 *   --warmup N      Number of warmup runs (default: 5)
 *   --runs N        Number of benchmark runs (default: 20)
 *   --output FILE   Output CSV file prefix (default: benchmark)
 *   --ncu-mode      Single run mode for Nsight Compute profiling
 *   --dist D        Particle distribution: uniform, clustered (default: uniform)
 *   -v, --verbose   Verbose output
 */

#include "Ippl.h"
#include <Kokkos_Random.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
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
        Kokkos::fence();  // Ensure all previous GPU work is complete
        start_time_ = clock_type::now();
    }

    double stop() {
        Kokkos::fence();  // Ensure all GPU work from this section is complete
        auto end_time = clock_type::now();
        auto duration = std::chrono::duration<double>(end_time - start_time_);
        return duration.count();  // Returns seconds
    }

private:
    time_point start_time_;
};

// ============================================================================
// Benchmark Parameters
// ============================================================================

struct BenchParams {
    int n_grid = 128;
    double rho = 10.0;              // particles per grid point
    double kernel_tol = 1e-6;       // determines kernel width
    int warmup_runs = 3;
    int benchmark_runs = 5;
    std::string output_prefix = "benchmark";
    std::string distribution = "uniform";  // or "clustered"
    bool verbose = false;
    bool ncu_mode = false;          // single-run mode for Nsight Compute

    // Derived
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
        } else if ((arg == "--tol" || arg == "-t") && i + 1 < argc) {
            params.kernel_tol = std::atof(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            params.warmup_runs = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            params.benchmark_runs = std::atoi(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            params.output_prefix = argv[++i];
        } else if (arg == "--dist" && i + 1 < argc) {
            params.distribution = argv[++i];
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

    // Convert to milliseconds
    std::vector<double> times_ms(stats.count);
    for (size_t i = 0; i < stats.count; ++i) {
        times_ms[i] = times_sec[i] * 1000.0;
    }

    // Mean
    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    stats.mean_ms = sum / stats.count;

    // Stddev
    double sq_sum = 0.0;
    for (double t : times_ms) {
        sq_sum += (t - stats.mean_ms) * (t - stats.mean_ms);
    }
    stats.stddev_ms = (stats.count > 1) ? std::sqrt(sq_sum / (stats.count - 1)) : 0.0;

    // Min/Max
    stats.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    stats.max_ms = *std::max_element(times_ms.begin(), times_ms.end());

    // Median
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
// Throughput Metrics
// ============================================================================

struct ThroughputMetrics {
    std::string kernel_name;
    std::string operation;          // "scatter" or "gather"
    std::string distribution;
    int kernel_width;
    double tolerance;
    size_t n_particles;
    size_t n_grid;
    double rho;

    // Timing results
    TimingStats total_stats;        // Total time (including sort if applicable)
    TimingStats kernel_only_stats;  // Kernel time (excluding sort)
    TimingStats sort_stats;         // Sort time only (if applicable)

    // Raw measurements for CSV export
    std::vector<double> total_times_sec;
    std::vector<double> kernel_times_sec;
    std::vector<double> sort_times_sec;

    // Derived throughput metrics (using total time)
    double throughput_Mpts_per_sec() const {
        return (n_particles / (total_stats.mean_ms * 1e-3)) / 1e6;
    }

    double time_per_point_ns() const {
        return (total_stats.mean_ms * 1e6) / n_particles;
    }
};

// ============================================================================
// Throughput Benchmark Implementation
// ============================================================================

template <typename ExecSpace>
class ThroughputBenchmark {
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

    ThroughputBenchmark(const BenchParams& params)
        : params_(params)
        , kernel_(params.kernel_tol) {}

    void run() {
        print_header();

        std::vector<ThroughputMetrics> results;

        // Run benchmark at the specified tolerance
        run_all_kernels(kernel_, results);

        // Additionally sweep over tolerances for throughput-vs-accuracy plot
        if (!params_.ncu_mode) {
            std::vector<double> tolerances = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9};
            for (double tol : tolerances) {
                if (true || std::abs(tol - params_.kernel_tol) > 1e-15) {  // skip duplicate
                    ippl::NUFFT::ESKernel<real_type> sweep_kernel(tol);
                    run_all_kernels(sweep_kernel, results);
                }
            }
        }

        // Output results
        write_csv(results);
        write_raw_csv(results);

        print_summary(results);
    }

    void print_header() {
        if (ippl::Comm->rank() != 0) return;

        int w = kernel_.width();
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "     Throughput Benchmark for Scatter/Gather Kernels\n";
        std::cout << "================================================================\n";
        std::cout << "Grid size:       " << params_.n_grid << "^3 = "
                  << (params_.n_grid * params_.n_grid * params_.n_grid) << " points\n";
        std::cout << "Particles/grid:  " << params_.rho << "\n";
        std::cout << "Total particles: " << params_.n_particles() << "\n";
        std::cout << "Distribution:    " << params_.distribution << "\n";
        std::cout << "Tolerance:       " << params_.kernel_tol << "\n";
        std::cout << "Kernel width:    " << w << "\n";
        std::cout << "Warmup runs:     " << params_.warmup_runs << "\n";
        std::cout << "Benchmark runs:  " << params_.benchmark_runs << "\n";
        if (params_.ncu_mode) {
            std::cout << "Mode:            NCU profiling (single run)\n";
        }
        std::cout << "================================================================\n\n";
    }

    void run_all_kernels(const ippl::NUFFT::ESKernel<real_type>& kernel,
                         std::vector<ThroughputMetrics>& results) {
        int w = kernel.width();
        int nghost = w / 2 + 1;

        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "Running benchmarks with w=" << w << "\n";
        }

        // Setup domain and data structures
        setup_domain(nghost);
        initialize(kernel, nghost);

        size_t n_particles = bunch_->getLocalNum();

        // ===== SCATTER BENCHMARKS =====

        // Atomic scatter (unsorted) - baseline
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
            auto metrics = benchmark_scatter("Atomic", cfg, kernel, nghost, n_particles);
            results.push_back(metrics);
        }

        // Tiled scatter (Sorted Spread with Shared Memory)
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
            cfg.tile_size_3d = 4;
            auto metrics = benchmark_scatter("Tiled", cfg, kernel, nghost, n_particles);
            results.push_back(metrics);
        }

        // OutputFocused scatter (Grid-Parallel)
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
            int tile_size = 2;
            std::array<int, 10> tile_sizes = {
                1,1,4,3,3,2,4,2,2
            };
            if (w < 10) {
                tile_size = tile_sizes[w];
            }
            cfg.tile_size_3d = tile_size;
            auto metrics = benchmark_scatter("GridParallel", cfg, kernel, nghost, n_particles);
            results.push_back(metrics);
        }

        // ===== GATHER BENCHMARKS =====

        // Direct gather (unsorted)
        {
            auto cfg = ippl::Interpolation::GatherConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::Atomic;
            cfg.sort = false;
            auto metrics = benchmark_gather("Direct", cfg, kernel, nghost, n_particles);
            results.push_back(metrics);
        }

        // Sorted gather
        {
            auto cfg = ippl::Interpolation::GatherConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::AtomicSort;
            auto metrics = benchmark_gather("Sorted", cfg, kernel, nghost, n_particles);
            results.push_back(metrics);
        }

        // Team-Parallel gather (Tiled)
        {
            auto cfg = ippl::Interpolation::GatherConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::Tiled;
            cfg.tile_size_3d = 3;
            auto metrics = benchmark_gather("TeamParallel", cfg, kernel, nghost, n_particles);
            results.push_back(metrics);
        }

        // Team-Parallel gather (Native)
        {
            auto cfg = ippl::Interpolation::GatherConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::Native;
            cfg.tile_size_3d = 3;
            auto metrics = benchmark_gather("Native", cfg, kernel, nghost, n_particles);
            results.push_back(metrics);
        }

        // Cleanup
        cleanup();
    }

    ThroughputMetrics benchmark_scatter(const std::string& name,
                                         const ippl::Interpolation::ScatterConfig& cfg,
                                         const ippl::NUFFT::ESKernel<real_type>& kernel,
                                         int nghost,
                                         size_t n_particles) {
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "  Benchmarking scatter: " << name << "\n";
        }

        ManualTimer timer;

        // Warmup runs (not timed)
        for (int i = 0; i < params_.warmup_runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            Q_.scatter_kernel(*grid_, R_, kernel, cfg);
            grid_->accumulateHalo();
        }
        Kokkos::fence();  // Ensure warmup is complete

        // Benchmark runs with manual timing
        std::vector<double> total_times;
        total_times.reserve(params_.benchmark_runs);

        for (int i = 0; i < params_.benchmark_runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            Kokkos::fence();  // Ensure grid reset is complete

            timer.start();
            Q_.scatter_kernel(*grid_, R_, kernel, cfg);
            grid_->accumulateHalo();
            double elapsed = timer.stop();

            total_times.push_back(elapsed);
        }

        // Build metrics
        ThroughputMetrics m;
        m.kernel_name = name;
        m.operation = "scatter";
        m.distribution = params_.distribution;
        m.kernel_width = kernel.width();
        m.tolerance = std::pow(10.0, -(kernel.width() - 1));
        m.n_particles = n_particles;
        m.n_grid = params_.n_grid;
        m.rho = params_.rho;

        m.total_times_sec = total_times;
        m.total_stats = compute_stats(total_times);
        m.kernel_only_stats = m.total_stats;  // No separate sort timing for scatter
        m.sort_stats = TimingStats{};  // Empty

        return m;
    }

    ThroughputMetrics benchmark_gather(const std::string& name,
                                        const ippl::Interpolation::GatherConfig& cfg,
                                        const ippl::NUFFT::ESKernel<real_type>& kernel,
                                        int nghost,
                                        size_t n_particles) {
        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "  Benchmarking gather: " << name << "\n";
        }

        ManualTimer timer;

        // Warmup runs (not timed)
        for (int i = 0; i < params_.warmup_runs; ++i) {
            Q_result_ = complex_type(0.0, 0.0);
            Q_result_.gather(*grid_, R_, kernel, false, cfg);
        }
        Kokkos::fence();  // Ensure warmup is complete

        // Benchmark runs with manual timing
        std::vector<double> total_times;
        total_times.reserve(params_.benchmark_runs);

        for (int i = 0; i < params_.benchmark_runs; ++i) {
            Q_result_ = complex_type(0.0, 0.0);
            Kokkos::fence();  // Ensure reset is complete

            timer.start();
            Q_result_.gather(*grid_, R_, kernel, false, cfg);
            double elapsed = timer.stop();

            total_times.push_back(elapsed);
        }

        // Build metrics
        ThroughputMetrics m;
        m.kernel_name = name;
        m.operation = "gather";
        m.distribution = params_.distribution;
        m.kernel_width = kernel.width();
        m.tolerance = std::pow(10.0, -(kernel.width() - 1));
        m.n_particles = n_particles;
        m.n_grid = params_.n_grid;
        m.rho = params_.rho;

        m.total_times_sec = total_times;
        m.total_stats = compute_stats(total_times);
        m.kernel_only_stats = m.total_stats;  // No separate sort timing available
        m.sort_stats = TimingStats{};  // Empty

        return m;
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

        // Register particle attributes
        bunch_->addAttribute(R_);
        bunch_->addAttribute(Q_);
        bunch_->addAttribute(Q_result_);

        bunch_->setParticleBC(ippl::BC::PERIODIC);

        // Create particles
        size_t n_local = params_.n_particles() / ippl::Comm->size();
        bunch_->create(n_local);

        // Initialize positions based on distribution
        auto R_view = R_.getView();
        Kokkos::Random_XorShift64_Pool<> rand_pool(42 + ippl::Comm->rank());

        if (params_.distribution == "uniform") {
            // Uniform random distribution in [0, 2*pi]^3
            Kokkos::parallel_for("init_uniform", n_local,
                KOKKOS_LAMBDA(const size_t i) {
                    auto gen = rand_pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        R_view(i)[d] = gen.drand() * 2.0 * M_PI;
                    }
                    rand_pool.free_state(gen);
                });
        } else if (params_.distribution == "clustered") {
            // Clustered distribution: particles concentrated in small region
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

        // Initialize particle values
        auto Q_view = Q_.getView();
        Kokkos::parallel_for("init_values", n_local,
            KOKKOS_LAMBDA(const size_t i) {
                Q_view(i) = complex_type(1.0, 0.0);
            });

        // Initialize grid for gather
        *grid_ = complex_type(1.0, 0.0);

        Kokkos::fence();
    }

    void cleanup() {
        bunch_.reset();
        playout_.reset();
        grid_.reset();
        mesh_.reset();
        layout_.reset();
    }

    void write_csv(const std::vector<ThroughputMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_throughput.csv";
        std::ofstream out(filename);

        // CSV header
        out << "kernel,operation,distribution,width,tolerance,n_particles,n_grid,rho,"
            << "total_mean_ms,total_stddev_ms,total_min_ms,total_max_ms,total_median_ms,"
            << "kernel_mean_ms,kernel_stddev_ms,"
            << "sort_mean_ms,sort_stddev_ms,"
            << "throughput_Mpts_per_s,time_per_pt_ns\n";

        for (const auto& m : results) {
            out << m.kernel_name << ","
                << m.operation << ","
                << m.distribution << ","
                << m.kernel_width << ","
                << std::scientific << std::setprecision(1) << m.tolerance << ","
                << m.n_particles << ","
                << m.n_grid << ","
                << std::fixed << std::setprecision(1) << m.rho << ","
                << std::setprecision(4) << m.total_stats.mean_ms << ","
                << m.total_stats.stddev_ms << ","
                << m.total_stats.min_ms << ","
                << m.total_stats.max_ms << ","
                << m.total_stats.median_ms << ","
                << m.kernel_only_stats.mean_ms << ","
                << m.kernel_only_stats.stddev_ms << ","
                << m.sort_stats.mean_ms << ","
                << m.sort_stats.stddev_ms << ","
                << std::setprecision(2) << m.throughput_Mpts_per_sec() << ","
                << std::setprecision(2) << m.time_per_point_ns() << "\n";
        }

        out.close();
        std::cout << "\nWrote results to: " << filename << "\n";
    }

    void write_raw_csv(const std::vector<ThroughputMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_raw.csv";
        std::ofstream out(filename);

        // CSV header for raw measurements
        out << "kernel,operation,distribution,width,run,time_sec\n";

        for (const auto& m : results) {
            for (size_t i = 0; i < m.total_times_sec.size(); ++i) {
                out << m.kernel_name << ","
                    << m.operation << ","
                    << m.distribution << ","
                    << m.kernel_width << ","
                    << i << ","
                    << std::scientific << std::setprecision(9) << m.total_times_sec[i] << "\n";
            }
        }

        out.close();
        std::cout << "Wrote raw measurements to: " << filename << "\n";
    }

    void print_summary(const std::vector<ThroughputMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        int target_w = kernel_.width();

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "        Results Summary (w=" << target_w << ", "
                  << params_.distribution << " distribution)\n";
        std::cout << "================================================================\n";

        // Print scatter results
        std::cout << "\nSCATTER (type-1 spreading):\n";
        std::cout << std::left << std::setw(16) << "Kernel"
                  << std::right << std::setw(12) << "Total (ms)"
                  << std::setw(12) << "Stddev"
                  << std::setw(14) << "Mpts/s"
                  << std::setw(12) << "ns/pt" << "\n";
        std::cout << std::string(66, '-') << "\n";

        double scatter_baseline = 0.0;
        for (const auto& m : results) {
            if (m.operation == "scatter" && m.kernel_width == target_w) {
                if (m.kernel_name == "Atomic") scatter_baseline = m.total_stats.mean_ms;
                double speedup = (scatter_baseline > 0) ? scatter_baseline / m.total_stats.mean_ms : 1.0;
                std::cout << std::left << std::setw(16) << m.kernel_name
                          << std::right << std::fixed
                          << std::setw(12) << std::setprecision(3) << m.total_stats.mean_ms
                          << std::setw(12) << std::setprecision(3) << m.total_stats.stddev_ms
                          << std::setw(14) << std::setprecision(1) << m.throughput_Mpts_per_sec()
                          << std::setw(12) << std::setprecision(2) << m.time_per_point_ns()
                          << " (" << std::setprecision(2) << speedup << "x)\n";
            }
        }

        // Print gather results
        std::cout << "\nGATHER (type-2 interpolation):\n";
        std::cout << std::left << std::setw(16) << "Kernel"
                  << std::right << std::setw(12) << "Total (ms)"
                  << std::setw(12) << "Stddev"
                  << std::setw(14) << "Mpts/s"
                  << std::setw(12) << "ns/pt" << "\n";
        std::cout << std::string(66, '-') << "\n";

        double gather_baseline = 0.0;
        for (const auto& m : results) {
            if (m.operation == "gather" && m.kernel_width == target_w) {
                if (m.kernel_name == "Direct") gather_baseline = m.total_stats.mean_ms;
                double speedup = (gather_baseline > 0) ? gather_baseline / m.total_stats.mean_ms : 1.0;
                std::cout << std::left << std::setw(16) << m.kernel_name
                          << std::right << std::fixed
                          << std::setw(12) << std::setprecision(3) << m.total_stats.mean_ms
                          << std::setw(12) << std::setprecision(3) << m.total_stats.stddev_ms
                          << std::setw(14) << std::setprecision(1) << m.throughput_Mpts_per_sec()
                          << std::setw(12) << std::setprecision(2) << m.time_per_point_ns()
                          << " (" << std::setprecision(2) << speedup << "x)\n";
            }
        }

        std::cout << "\n";

        // Print Nsight Compute hint
        if (!params_.ncu_mode) {
            std::cout << "For roofline analysis, run with Nsight Compute:\n";
            std::cout << "  ncu --set roofline ./BenchmarkRoofline --ncu-mode\n\n";
        }
    }

    BenchParams params_;
    ippl::NUFFT::ESKernel<real_type> kernel_;

    ippl::Vector<std::size_t, Dim> n_grid_;
    ippl::Vector<real_type, Dim> origin_;
    ippl::Vector<real_type, Dim> hx_;

    std::unique_ptr<ippl::FieldLayout<Dim>> layout_;
    std::unique_ptr<Mesh_t> mesh_;
    std::unique_ptr<Field_t> grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t> bunch_;

    // Particle attributes (stored separately for direct access)
    ippl::ParticleAttrib<ippl::Vector<real_type, Dim>> R_;
    ippl::ParticleAttrib<complex_type> Q_;
    ippl::ParticleAttrib<complex_type> Q_result_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        auto params = parse_bench_args(argc, argv);

        ThroughputBenchmark<Kokkos::DefaultExecutionSpace> benchmark(params);
        benchmark.run();
    }

    ippl::finalize();
    return EXIT_SUCCESS;
}
