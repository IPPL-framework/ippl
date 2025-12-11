/**
 * @file BenchmarkRoofline.cpp
 * @brief Roofline analysis benchmark for scatter/gather kernels
 *
 * Generates data for:
 *   Plot 1: Roofline (all kernel variants at fixed w)
 *   Plot 2: Throughput vs kernel width w
 *
 * Usage: ./BenchmarkRoofline [options]
 *   --grid N        Grid size per dimension (default: 256)
 *   --rho R         Particles per grid point (default: 10)
 *   --width W       Kernel width for roofline plot (default: 6)
 *   --warmup N      Number of warmup runs (default: 5)
 *   --runs N        Number of benchmark runs (default: 20)
 *   --output FILE   Output CSV file prefix (default: roofline)
 *   -v, --verbose   Verbose output
 */

#include "BenchmarkUtils.h"

#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace ippl;

// ============================================================================
// Roofline Analysis Parameters
// ============================================================================

struct RooflineParams {
    int n_grid = 16;
    double rho = 10.0;  // particles per grid point
    int kernel_width = 6;
    int warmup_runs = 5;
    int benchmark_runs = 20;
    std::string output_prefix = "roofline";
    bool verbose = false;

    // Derived
    size_t n_particles() const {
        return static_cast<size_t>(rho * n_grid * n_grid * n_grid);
    }
};

RooflineParams parse_roofline_args(int argc, char* argv[]) {
    RooflineParams params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--grid" || arg == "-n") && i + 1 < argc) {
            params.n_grid = std::atoi(argv[++i]);
        } else if (arg == "--rho" && i + 1 < argc) {
            params.rho = std::atof(argv[++i]);
        } else if ((arg == "--width" || arg == "-w") && i + 1 < argc) {
            params.kernel_width = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            params.warmup_runs = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            params.benchmark_runs = std::atoi(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            params.output_prefix = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        }
    }
    return params;
}

// ============================================================================
// Roofline Metrics Computation
// ============================================================================

struct RooflineMetrics {
    std::string kernel_name;
    std::string operation;  // "scatter" or "gather"
    int kernel_width;
    double time_ms;
    double throughput_particles_per_sec;
    double achieved_bandwidth_gb_s;
    double arithmetic_intensity;
    double achieved_flops;
};

// Compute bytes moved for scatter operation
// Read: positions (3 doubles), values (1 complex = 2 doubles)
// Write: grid values (w^3 atomics, each complex = 2 doubles)
double compute_scatter_bytes(size_t n_particles, int w) {
    constexpr size_t sizeof_double = 8;
    constexpr size_t sizeof_complex = 16;

    size_t bytes_read = n_particles * (3 * sizeof_double + sizeof_complex);
    size_t bytes_written = n_particles * w * w * w * sizeof_complex;
    return static_cast<double>(bytes_read + bytes_written);
}

// Compute bytes moved for gather operation
// Read: positions (3 doubles), grid values (w^3 complex)
// Write: output values (1 complex per particle)
double compute_gather_bytes(size_t n_particles, int w) {
    constexpr size_t sizeof_double = 8;
    constexpr size_t sizeof_complex = 16;

    size_t bytes_read = n_particles * (3 * sizeof_double + w * w * w * sizeof_complex);
    size_t bytes_written = n_particles * sizeof_complex;
    return static_cast<double>(bytes_read + bytes_written);
}

// Compute FLOPs for scatter/gather
// Per particle: 3*w kernel evaluations (ES kernel ~20 FLOPs each)
//               w^3 multiply-adds (4 FLOPs each for complex)
double compute_flops(size_t n_particles, int w) {
    constexpr int flops_per_kernel_eval = 20;  // exp, sqrt, etc.
    constexpr int flops_per_grid_point = 4;    // complex multiply-add

    double kernel_flops = 3.0 * w * flops_per_kernel_eval;
    double grid_flops = w * w * w * flops_per_grid_point;
    return n_particles * (kernel_flops + grid_flops);
}

// ============================================================================
// Roofline Benchmark Implementation
// ============================================================================

template <typename ExecSpace>
class RooflineBenchmark {
public:
    static constexpr unsigned Dim = 3;
    using real_type = double;
    using complex_type = Kokkos::complex<real_type>;
    using MemSpace = typename ExecSpace::memory_space;

    using Mesh_t = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t = typename Mesh_t::DefaultCentering;
    using Field_t = ippl::Field<complex_type, Dim, Mesh_t, Centering_t>;
    using PLayout_t = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t = benchmark::BenchmarkBunch<PLayout_t>;

    RooflineBenchmark(const RooflineParams& params)
        : params_(params) {}

    void run() {
        print_header();

        // Run roofline benchmark at fixed w
        std::vector<RooflineMetrics> roofline_results;
        run_roofline_benchmark(params_.kernel_width, roofline_results);

        // Run throughput vs w benchmark
        std::vector<RooflineMetrics> throughput_results;
        for (int w : {4, 6, 8, 10, 12}) {
            run_roofline_benchmark(w, throughput_results);
        }

        // Output results
        write_roofline_csv(roofline_results);
        write_throughput_csv(throughput_results);

        print_summary(roofline_results);
    }

    void print_header() {
        if (ippl::Comm->rank() != 0) return;

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "          Roofline Analysis Benchmark\n";
        std::cout << "================================================================\n";
        std::cout << "Grid size:       " << params_.n_grid << "^3\n";
        std::cout << "Particles/grid:  " << params_.rho << "\n";
        std::cout << "Total particles: " << params_.n_particles() << "\n";
        std::cout << "Kernel width:    " << params_.kernel_width << " (for roofline plot)\n";
        std::cout << "Warmup runs:     " << params_.warmup_runs << "\n";
        std::cout << "Benchmark runs:  " << params_.benchmark_runs << "\n";
        std::cout << "================================================================\n\n";
    }

    void run_roofline_benchmark(int kernel_width, std::vector<RooflineMetrics>& results) {
        // Compute tolerance from kernel width (approximate inverse of ES kernel width selection)
        double tol = std::pow(10.0, -kernel_width);

        ippl::NUFFT::ESKernel<real_type> kernel(tol);
        int actual_width = kernel.width();
        int nghost = actual_width / 2 + 1;

        if (ippl::Comm->rank() == 0 && params_.verbose) {
            std::cout << "Running benchmarks with w=" << actual_width
                      << " (tol=" << tol << ")\n";
        }

        // Setup domain and data structures
        setup_domain(nghost);
        initialize(kernel, nghost);

        size_t n_particles = bunch_->getLocalNum();

        // ===== SCATTER BENCHMARKS =====

        // Atomic scatter (no sort)
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
            cfg.sort = false;
            double time_ms = benchmark_scatter(cfg, kernel, nghost);
            results.push_back(compute_metrics("Atomic", "scatter", actual_width,
                                               n_particles, time_ms));
        }

        // Atomic scatter (sorted)
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
            cfg.sort = true;
            double time_ms = benchmark_scatter(cfg, kernel, nghost);
            results.push_back(compute_metrics("Atomic-Sorted", "scatter", actual_width,
                                               n_particles, time_ms));
        }

        // Tiled scatter (best tile size)
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
            cfg.tile_size_3d = 3;
            double time_ms = benchmark_scatter(cfg, kernel, nghost);
            results.push_back(compute_metrics("Tiled", "scatter", actual_width,
                                               n_particles, time_ms));
        }

        // OutputFocused scatter (Grid-Parallel)
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
            cfg.tile_size_3d = 3;
            cfg.sort = true;
            double time_ms = benchmark_scatter(cfg, kernel, nghost);
            results.push_back(compute_metrics("Grid-Parallel", "scatter", actual_width,
                                               n_particles, time_ms));
        }

        // ===== GATHER BENCHMARKS =====

        // Atomic gather (no sort) - Direct Gather
        {
            auto cfg = ippl::Interpolation::GatherConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::Atomic;
            cfg.sort = false;
            double time_ms = benchmark_gather(cfg, kernel, nghost);
            results.push_back(compute_metrics("Direct", "gather", actual_width,
                                               n_particles, time_ms));
        }

        // Atomic gather (sorted) - Sorted Gather
        {
            auto cfg = ippl::Interpolation::GatherConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::Atomic;
            cfg.sort = true;
            double time_ms = benchmark_gather(cfg, kernel, nghost);
            results.push_back(compute_metrics("Sorted", "gather", actual_width,
                                               n_particles, time_ms));
        }

        // Tiled gather - Team-Parallel Gather
        {
            auto cfg = ippl::Interpolation::GatherConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::GatherMethod::Tiled;
            cfg.tile_size_3d = 3;
            cfg.sort = true;
            double time_ms = benchmark_gather(cfg, kernel, nghost);
            results.push_back(compute_metrics("Team-Parallel", "gather", actual_width,
                                               n_particles, time_ms));
        }

        // Cleanup
        cleanup();
    }

    RooflineMetrics compute_metrics(const std::string& name, const std::string& op,
                                     int w, size_t n_particles, double time_ms) {
        RooflineMetrics m;
        m.kernel_name = name;
        m.operation = op;
        m.kernel_width = w;
        m.time_ms = time_ms;

        double time_s = time_ms / 1000.0;
        m.throughput_particles_per_sec = n_particles / time_s;

        double bytes = (op == "scatter") ? compute_scatter_bytes(n_particles, w)
                                          : compute_gather_bytes(n_particles, w);
        m.achieved_bandwidth_gb_s = bytes / (time_s * 1e9);

        double flops = compute_flops(n_particles, w);
        m.achieved_flops = flops / time_s;
        m.arithmetic_intensity = flops / bytes;

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
        bunch_->setParticleBC(ippl::BC::PERIODIC);

        // Create particles
        size_t n_local = params_.n_particles() / ippl::Comm->size();
        bunch_->create(n_local);

        // Initialize random positions in [0, 2*pi]^3
        auto R = bunch_->R.getView();
        Kokkos::Random_XorShift64_Pool<> rand_pool(42 + ippl::Comm->rank());

        Kokkos::parallel_for("init_positions", n_local,
            KOKKOS_LAMBDA(const size_t i) {
                auto gen = rand_pool.get_state();
                for (unsigned d = 0; d < Dim; ++d) {
                    R(i)[d] = gen.drand() * 2.0 * M_PI;
                }
                rand_pool.free_state(gen);
            });

        // Initialize particle values
        auto Q = bunch_->Q.getView();
        Kokkos::parallel_for("init_values", n_local,
            KOKKOS_LAMBDA(const size_t i) {
                Q(i) = complex_type(1.0, 0.0);
            });

        // Initialize grid
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

    double benchmark_scatter(const ippl::Interpolation::ScatterConfig& cfg,
                              const ippl::NUFFT::ESKernel<real_type>& kernel,
                              int nghost) {
        benchmark::Timer timer;
        std::vector<double> times;

        // Warmup
        for (int i = 0; i < params_.warmup_runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);
            bunch_->Q.scatter_kernel(*grid_, bunch_->R, kernel, cfg);
            grid_->accumulateHalo();
        }

        // Benchmark
        for (int i = 0; i < params_.benchmark_runs; ++i) {
            *grid_ = complex_type(0.0, 0.0);

            timer.start();
            bunch_->Q.scatter_kernel(*grid_, bunch_->R, kernel, cfg);
            grid_->accumulateHalo();
            double elapsed = timer.stop();

            times.push_back(elapsed);
        }

        auto stats = benchmark::compute_stats(times);
        return stats.mean_ms;
    }

    double benchmark_gather(const ippl::Interpolation::GatherConfig& cfg,
                             const ippl::NUFFT::ESKernel<real_type>& kernel,
                             int nghost) {
        benchmark::Timer timer;
        std::vector<double> times;

        // Warmup
        for (int i = 0; i < params_.warmup_runs; ++i) {
            bunch_->Q_result = complex_type(0.0, 0.0);
            bunch_->Q_result.gather(*grid_, bunch_->R, kernel, false, cfg);
        }

        // Benchmark
        for (int i = 0; i < params_.benchmark_runs; ++i) {
            bunch_->Q_result = complex_type(0.0, 0.0);

            timer.start();
            bunch_->Q_result.gather(*grid_, bunch_->R, kernel, false, cfg);
            double elapsed = timer.stop();

            times.push_back(elapsed);
        }

        auto stats = benchmark::compute_stats(times);
        return stats.mean_ms;
    }

    void write_roofline_csv(const std::vector<RooflineMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_roofline.csv";
        std::ofstream out(filename);

        out << "kernel,operation,width,time_ms,throughput_particles_per_s,"
            << "bandwidth_gb_s,arithmetic_intensity,gflops\n";

        for (const auto& m : results) {
            if (m.kernel_width == params_.kernel_width) {
                out << m.kernel_name << ","
                    << m.operation << ","
                    << m.kernel_width << ","
                    << std::fixed << std::setprecision(4) << m.time_ms << ","
                    << std::scientific << std::setprecision(4) << m.throughput_particles_per_sec << ","
                    << std::fixed << std::setprecision(2) << m.achieved_bandwidth_gb_s << ","
                    << std::setprecision(4) << m.arithmetic_intensity << ","
                    << std::setprecision(2) << m.achieved_flops / 1e9 << "\n";
            }
        }

        out.close();
        std::cout << "Wrote roofline data to: " << filename << "\n";
    }

    void write_throughput_csv(const std::vector<RooflineMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::string filename = params_.output_prefix + "_throughput.csv";
        std::ofstream out(filename);

        out << "kernel,operation,width,time_ms,throughput_particles_per_s,"
            << "bandwidth_gb_s,arithmetic_intensity,gflops\n";

        for (const auto& m : results) {
            out << m.kernel_name << ","
                << m.operation << ","
                << m.kernel_width << ","
                << std::fixed << std::setprecision(4) << m.time_ms << ","
                << std::scientific << std::setprecision(4) << m.throughput_particles_per_sec << ","
                << std::fixed << std::setprecision(2) << m.achieved_bandwidth_gb_s << ","
                << std::setprecision(4) << m.arithmetic_intensity << ","
                << std::setprecision(2) << m.achieved_flops / 1e9 << "\n";
        }

        out.close();
        std::cout << "Wrote throughput data to: " << filename << "\n";
    }

    void print_summary(const std::vector<RooflineMetrics>& results) {
        if (ippl::Comm->rank() != 0) return;

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "                    Results Summary (w=" << params_.kernel_width << ")\n";
        std::cout << "================================================================\n";

        std::cout << std::left << std::setw(20) << "Kernel"
                  << std::setw(10) << "Op"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(14) << "BW (GB/s)"
                  << std::setw(12) << "AI"
                  << std::setw(14) << "GFLOP/s" << "\n";
        std::cout << std::string(82, '-') << "\n";

        for (const auto& m : results) {
            if (m.kernel_width == params_.kernel_width) {
                std::cout << std::left << std::setw(20) << m.kernel_name
                          << std::setw(10) << m.operation
                          << std::right << std::fixed
                          << std::setw(12) << std::setprecision(3) << m.time_ms
                          << std::setw(14) << std::setprecision(1) << m.achieved_bandwidth_gb_s
                          << std::setw(12) << std::setprecision(3) << m.arithmetic_intensity
                          << std::setw(14) << std::setprecision(1) << m.achieved_flops / 1e9
                          << "\n";
            }
        }

        std::cout << "\n";
    }

    RooflineParams params_;

    ippl::Vector<std::size_t, Dim> n_grid_;
    ippl::Vector<real_type, Dim> origin_;
    ippl::Vector<real_type, Dim> hx_;

    std::unique_ptr<ippl::FieldLayout<Dim>> layout_;
    std::unique_ptr<Mesh_t> mesh_;
    std::unique_ptr<Field_t> grid_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t> bunch_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        auto params = parse_roofline_args(argc, argv);

        RooflineBenchmark<Kokkos::DefaultExecutionSpace> benchmark(params);
        benchmark.run();
    }

    ippl::finalize();
    return EXIT_SUCCESS;
}