/**
 * @file BenchmarkScatter.cpp
 * @brief Benchmark for scatter kernel implementations (Atomic, Tiled, OutputFocused)
 *
 * Usage: ./BenchmarkScatter [options]
 *   --particles N   Number of particles (default: 100000)
 *   --grid N        Grid size per dimension (default: 64)
 *   --warmup N      Number of warmup runs (default: 5)
 *   --runs N        Number of benchmark runs (default: 20)
 *   --tol T         Kernel tolerance (default: 1e-6)
 *   -v, --verbose   Verbose output
 */

#include "BenchmarkUtils.h"

#include <cmath>
#include <complex>
#include <vector>

using namespace ippl;

// ============================================================================
// Scatter Benchmark Implementation
// ============================================================================

template <typename ExecSpace>
class ScatterBenchmark {
public:
    static constexpr unsigned Dim = 3;
    using real_type    = double;
    using complex_type = Kokkos::complex<real_type>;
    using MemSpace     = typename ExecSpace::memory_space;

    using Mesh_t       = ippl::UniformCartesian<real_type, Dim>;
    using Centering_t  = typename Mesh_t::DefaultCentering;
    using Field_t      = ippl::Field<complex_type, Dim, Mesh_t, Centering_t>;
    using PLayout_t    = ippl::ParticleSpatialLayout<real_type, Dim>;
    using Bunch_t      = benchmark::BenchmarkBunch<PLayout_t>;

    ScatterBenchmark(const benchmark::BenchmarkParams& params)
        : params_(params)
        , kernel_(params.kernel_tol)
        , nghost_((kernel_.width()) / 2 + 1)
    {
        setup_domain();
    }

    void run_all_benchmarks() {
        benchmark::print_header("Scatter Kernel Benchmark");
        benchmark::print_params(params_);

        // Initialize data structures
        initialize();

        // Define configurations to benchmark
        std::vector<ippl::Interpolation::ScatterConfig> configs;

        // Atomic (baseline)
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
            cfg.sort = false;
            configs.push_back(cfg);
        }
        {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Atomic;
            cfg.sort = true;
            configs.push_back(cfg);
        }

        // Tiled with different tile sizes
        for (int tile_size : {8, 12, 16, 20, 24}) {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::Tiled;
            cfg.tile_size_3d = tile_size;
            cfg.sort = false;
            configs.push_back(cfg);

            cfg.sort = true;
            configs.push_back(cfg);
        }

        // OutputFocused with different tile sizes
        for (int tile_size : {8, 12, 16, 20, 24}) {
            auto cfg = ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
            cfg.method = ippl::Interpolation::ScatterMethod::OutputFocused;
            cfg.tile_size_3d = tile_size;
            cfg.sort = false;
            configs.push_back(cfg);

            cfg.sort = true;
            configs.push_back(cfg);
        }

        // Run benchmarks and collect results
        benchmark::print_subheader("Results");

        if (ippl::Comm->rank() == 0) {
            std::cout << std::left << std::setw(25) << "Configuration"
                      << std::right << std::setw(14) << "Mean"
                      << std::setw(14) << "Stddev"
                      << std::setw(14) << "Min"
                      << std::setw(14) << "Max"
                      << std::setw(14) << "Median" << "\n";
            std::cout << std::string(95, '-') << "\n";
        }

        std::vector<benchmark::TimingStats> results;
        for (const auto& cfg : configs) {
            auto stats = benchmark_config(cfg);
            results.push_back(stats);
            benchmark::print_stats_row(benchmark::config_label(cfg), stats);
        }

        // Summary table
        print_summary(configs, results);
    }

private:
    void setup_domain() {
        // Grid setup
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

    void initialize() {
        // Create field and particle layout
        grid_output_ = std::make_unique<Field_t>(*mesh_, *layout_, nghost_);

        playout_ = std::make_unique<PLayout_t>(*layout_, *mesh_);
        bunch_ = std::make_unique<Bunch_t>(*playout_);
        bunch_->setParticleBC(ippl::BC::PERIODIC);

        // Setup benchmark data
        benchmark::setup_benchmark_data<ExecSpace, Dim, real_type>(
            params_, kernel_, *mesh_, *layout_, *playout_, *bunch_, *grid_output_, nghost_);
    }

    benchmark::TimingStats benchmark_config(const ippl::Interpolation::ScatterConfig& cfg) {
        benchmark::Timer timer;
        std::vector<double> times;

        // Warmup runs
        for (int i = 0; i < params_.warmup_runs; ++i) {
            *grid_output_ = complex_type(0.0, 0.0);
            bunch_->Q.scatter_kernel(*grid_output_, bunch_->R, kernel_, cfg);
            grid_output_->accumulateHalo();
        }

        // Benchmark runs
        for (int i = 0; i < params_.benchmark_runs; ++i) {
            *grid_output_ = complex_type(0.0, 0.0);

            timer.start();
            bunch_->Q.scatter_kernel(*grid_output_, bunch_->R, kernel_, cfg);
            grid_output_->accumulateHalo();
            double elapsed = timer.stop();

            times.push_back(elapsed);

            if (params_.verbose && ippl::Comm->rank() == 0) {
                std::cout << "  Run " << (i + 1) << ": " << elapsed << " ms\n";
            }
        }

        return benchmark::compute_stats(times);
    }

    void print_summary(const std::vector<ippl::Interpolation::ScatterConfig>& configs,
                       const std::vector<benchmark::TimingStats>& results) {
        if (ippl::Comm->rank() != 0) return;

        benchmark::print_subheader("Performance Summary by Method");

        // Find best times for each method category
        struct MethodBest {
            std::string name;
            double best_time = std::numeric_limits<double>::max();
            std::string best_config;
        };

        std::vector<MethodBest> methods = {
            {"Atomic", std::numeric_limits<double>::max(), ""},
            {"Tiled", std::numeric_limits<double>::max(), ""},
            {"OutputFocused", std::numeric_limits<double>::max(), ""}
        };

        for (size_t i = 0; i < configs.size(); ++i) {
            const auto& cfg = configs[i];
            const auto& stats = results[i];
            int idx = static_cast<int>(cfg.method);
            if (stats.mean_ms < methods[idx].best_time) {
                methods[idx].best_time = stats.mean_ms;
                methods[idx].best_config = benchmark::config_label(cfg);
            }
        }

        std::cout << "\nBest configuration per method:\n";
        double baseline = methods[0].best_time;  // Atomic as baseline
        for (const auto& m : methods) {
            if (m.best_time < std::numeric_limits<double>::max()) {
                double speedup = baseline / m.best_time;
                std::cout << "  " << std::left << std::setw(15) << m.name
                          << ": " << std::fixed << std::setprecision(3)
                          << std::setw(10) << m.best_time << " ms"
                          << " (speedup: " << std::setprecision(2) << speedup << "x)"
                          << " - " << m.best_config << "\n";
            }
        }
    }

    benchmark::BenchmarkParams params_;
    ippl::NUFFT::ESKernel<real_type> kernel_;
    int nghost_;

    ippl::Vector<std::size_t, Dim> n_grid_;
    ippl::Vector<real_type, Dim> origin_;
    ippl::Vector<real_type, Dim> hx_;

    std::unique_ptr<ippl::FieldLayout<Dim>> layout_;
    std::unique_ptr<Mesh_t> mesh_;
    std::unique_ptr<Field_t> grid_output_;
    std::unique_ptr<PLayout_t> playout_;
    std::unique_ptr<Bunch_t> bunch_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        auto params = benchmark::parse_args(argc, argv);

        // Run benchmark with default execution space
        ScatterBenchmark<Kokkos::DefaultExecutionSpace> benchmark(params);
        benchmark.run_all_benchmarks();
    }

    ippl::finalize();
    return EXIT_SUCCESS;
}