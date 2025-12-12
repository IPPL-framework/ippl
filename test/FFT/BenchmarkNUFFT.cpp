#include "Ippl.h"
#include "Utility/IpplTimings.h"
#include "Utility/ParameterList.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

// ============================================================================
// Data Structures
// ============================================================================

struct BenchmarkResult {
    int num_ranks;
    size_t num_particles;
    int grid_size;
    double mean_time;
    double min_time;
    double max_time;
    double stddev;
    double throughput_mpts;  // Million points per second
    std::vector<double> all_times;
};

struct MemoryInfo {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
};

// ============================================================================
// Particle Bunch Definition
// ============================================================================

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    using charge_container_type = ippl::ParticleAttrib<double>;
    charge_container_type Q;

    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }

    ~Bunch() = default;
};

// ============================================================================
// Random Generation Functors
// ============================================================================

template <typename T, class GeneratorPool, unsigned Dim>
struct GenerateRandomParticlesWithCharges {
    using view_type        = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type       = typename T::value_type;
    using view_type_scalar = typename ippl::detail::ViewType<value_type, 1>::view_type;

    view_type x;
    view_type_scalar Q;
    GeneratorPool rand_pool;
    T minU, maxU;

    GenerateRandomParticlesWithCharges(view_type x_, view_type_scalar Q_,
                                       GeneratorPool rand_pool_, T& minU_, T& maxU_)
        : x(x_)
        , Q(Q_)
        , rand_pool(rand_pool_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
        }
        Q(i) = rand_gen.drand(0.0, 1.0);

        rand_pool.free_state(rand_gen);
    }
};

template <typename T, class GeneratorPool, unsigned Dim>
struct GenerateRandomField {
    using view_type = typename ippl::detail::ViewType<T, Dim>::view_type;
    view_type f;
    GeneratorPool rand_pool;

    GenerateRandomField(view_type f_, GeneratorPool rand_pool_)
        : f(f_)
        , rand_pool(rand_pool_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i, const size_t j, const size_t k) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        f(i, j, k).real() = rand_gen.drand(0.0, 1.0);
        f(i, j, k).imag() = rand_gen.drand(0.0, 1.0);

        rand_pool.free_state(rand_gen);
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

MemoryInfo getCudaMemoryInfo() {
    MemoryInfo info = {0, 0, 0};
#ifdef KOKKOS_ENABLE_CUDA
    cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
    info.used_bytes = info.total_bytes - info.free_bytes;
#endif
    return info;
}

void printMemoryUsage(const std::string& label) {
    if (ippl::Comm->rank() == 0) {
#ifdef KOKKOS_ENABLE_CUDA
        MemoryInfo info = getCudaMemoryInfo();
        std::cout << "[Memory] " << label << ": "
                  << "Used: " << (info.used_bytes / (1024.0 * 1024.0)) << " MB, "
                  << "Free: " << (info.free_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
#endif
    }
}

auto computeStats(const std::vector<double>& times) {
    double sum = 0.0, sum_sq = 0.0;
    double min_t = times[0], max_t = times[0];
    for (double t : times) {
        sum += t;
        sum_sq += t * t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    double mean   = sum / times.size();
    double stddev = std::sqrt(sum_sq / times.size() - mean * mean);
    return std::make_tuple(mean, stddev, min_t, max_t);
}

std::vector<double> gatherMaxTimes(const std::vector<double>& local_times) {
    int num_runs = local_times.size();
    std::vector<double> global_max_times(num_runs);

    MPI_Reduce(local_times.data(), global_max_times.data(), num_runs, MPI_DOUBLE, MPI_MAX, 0,
               ippl::Comm->getCommunicator());

    return global_max_times;
}

void writeTimingsCSV(const std::string& filename, int num_ranks, int grid_size,
                     size_t num_particles, const BenchmarkResult& type1_result,
                     const BenchmarkResult& type2_result, bool append = false) {
    if (ippl::Comm->rank() != 0)
        return;

    std::ofstream file;
    if (append) {
        file.open(filename, std::ios::app);
    } else {
        file.open(filename);
        file << "num_ranks,grid_size,num_particles,transform_type,run_index,time_ms\n";
    }

    // Write Type 1 times
    for (size_t i = 0; i < type1_result.all_times.size(); ++i) {
        file << num_ranks << "," << grid_size << "," << num_particles << ",type1," << i << ","
             << std::fixed << std::setprecision(6) << type1_result.all_times[i] << "\n";
    }

    // Write Type 2 times
    for (size_t i = 0; i < type2_result.all_times.size(); ++i) {
        file << num_ranks << "," << grid_size << "," << num_particles << ",type2," << i << ","
             << std::fixed << std::setprecision(6) << type2_result.all_times[i] << "\n";
    }

    file.close();
    std::cout << "[CSV] Timings written to " << filename << std::endl;
}

void writeComponentTimingsCSV(const std::string& filename, int num_ranks, int grid_size,
                              size_t num_particles) {
    if (ippl::Comm->rank() != 0)
        return;

    std::ofstream file(filename);
    file << "num_ranks,grid_size,num_particles,timer,run,time_s\n";

    // List of timers to export
    std::vector<std::string> timers = {
        "scatterTimerNUFFT1",
        "accumulateHaloNUFFT1",
        "FFTNUFFT1",
        "deconvolutionNUFFT1",
        "PrecorrectionNUFFT2",
        "FFTNUFFT2",
        "FillHaloNUFFT2",
        "GatherNUFFT2",
        "NativeNUFFT1",
        "NativeNUFFT2"
    };

    for (const auto& timer_name : timers) {
        const auto& measurements = IpplTimings::getMeasurements(timer_name);
        for (size_t i = 0; i < measurements.size(); ++i) {
            file << num_ranks << "," << grid_size << "," << num_particles << ","
                 << timer_name << "," << i << ","
                 << std::fixed << std::setprecision(9) << measurements[i] << "\n";
        }
    }

    file.close();
    std::cout << "[CSV] Component timings written to " << filename << std::endl;
}

void printComponentTimings() {
    if (ippl::Comm->rank() != 0)
        return;

    auto printTimerSummary = [](const std::string& name, const std::string& label) {
        const auto& measurements = IpplTimings::getMeasurements(name);
        if (!measurements.empty()) {
            double sum = 0.0;
            for (double m : measurements) sum += m;
            double mean_ms = (sum / measurements.size()) * 1000.0;
            std::cout << std::setw(25) << std::left << label
                      << std::right << std::setw(10) << std::fixed
                      << std::setprecision(2) << mean_ms << " ms\n";
        }
    };

    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "COMPONENT TIMING BREAKDOWN\n";
    std::cout << std::string(50, '=') << "\n";

    std::cout << "\nType-1 (spreading):\n";
    std::cout << std::string(40, '-') << "\n";
    printTimerSummary("scatterTimerNUFFT1", "Scatter (spreading)");
    printTimerSummary("accumulateHaloNUFFT1", "Halo accumulate");
    printTimerSummary("FFTNUFFT1", "FFT (forward)");
    printTimerSummary("deconvolutionNUFFT1", "Deconvolution");

    std::cout << "\nType-2 (interpolation):\n";
    std::cout << std::string(40, '-') << "\n";
    printTimerSummary("PrecorrectionNUFFT2", "Precorrection");
    printTimerSummary("FFTNUFFT2", "FFT (backward)");
    printTimerSummary("FillHaloNUFFT2", "Halo fill");
    printTimerSummary("GatherNUFFT2", "Gather (interpolation)");

    std::cout << "\nTotal:\n";
    std::cout << std::string(40, '-') << "\n";
    printTimerSummary("NativeNUFFT1", "Complete Type-1");
    printTimerSummary("NativeNUFFT2", "Complete Type-2");
}

void printResult(const std::string& label, const BenchmarkResult& result) {
    if (ippl::Comm->rank() != 0)
        return;

    std::cout << "\n" << label << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Ranks:       " << result.num_ranks << std::endl;
    std::cout << "  Grid:        " << result.grid_size << "^3" << std::endl;
    std::cout << "  Particles:   " << result.num_particles << std::endl;
    std::cout << "  Mean time:   " << result.mean_time << " ms" << std::endl;
    std::cout << "  Min time:    " << result.min_time << " ms" << std::endl;
    std::cout << "  Max time:    " << result.max_time << " ms" << std::endl;
    std::cout << "  Stddev:      " << result.stddev << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::setprecision(2) << result.throughput_mpts << " Mpts/s"
              << std::endl;
}

// ============================================================================
// Benchmark Functions
// ============================================================================

BenchmarkResult benchmarkNUFFTType1(int grid_size, int particles_per_point, double tolerance,
                                    const std::string& spread_method, int warmup_runs,
                                    int benchmark_runs) {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;
    using size_type            = ippl::detail::size_type;
    using Vector_t             = ippl::Vector<double, dim>;
    using playout_type         = ippl::ParticleSpatialLayout<double, dim>;
    using bunch_type           = Bunch<playout_type>;
    using field_type = typename ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type;
    using real_field_type = typename ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type;
    using FFT_type        = ippl::FFT<ippl::NUFFTransform, real_field_type>;

    const double pi = std::acos(-1.0);

    // Setup domain
    ippl::Vector<int, dim> pt = {grid_size, grid_size, grid_size};
    ippl::Index I(pt[0]);
    ippl::Index J(pt[1]);
    ippl::Index K(pt[2]);
    ippl::NDIndex<dim> owned(I, J, K);

    std::array<bool, dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

    Vector_t minU = {0, 0, 0};
    Vector_t maxU = {2 * pi, 2 * pi, 2 * pi};

    std::array<double, dim> dx = {
        (maxU[0] - minU[0]) / double(pt[0]),
        (maxU[1] - minU[1]) / double(pt[1]),
        (maxU[2] - minU[2]) / double(pt[2]),
    };

    Vector_t hx     = {dx[0], dx[1], dx[2]};
    Vector_t origin = {minU[0], minU[1], minU[2]};
    Mesh_t mesh(owned, hx, origin);

    playout_type pl(layout, mesh);

    size_type Np   = static_cast<size_type>(std::pow(grid_size, 3)) * particles_per_point;
    size_type nloc = Np / ippl::Comm->size();

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n=== Benchmarking NUFFT Type 1 ===" << std::endl;
        std::cout << "Grid size:       " << grid_size << "^3" << std::endl;
        std::cout << "Total particles: " << Np << std::endl;
        std::cout << "Local particles: " << nloc << std::endl;
        std::cout << "Spread method:   " << spread_method << std::endl;
        std::cout << "Tolerance:       " << tolerance << std::endl;
        std::cout << "Warmup runs:     " << warmup_runs << std::endl;
        std::cout << "Benchmark runs:  " << benchmark_runs << std::endl;
    }

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("Before allocation");

    // Create bunch and field
    bunch_type bunch(pl);
    bunch.setParticleBC(ippl::BC::PERIODIC);
    bunch.create(nloc);

    field_type field(mesh, layout);

    // Generate random particles with charges
    Kokkos::Random_XorShift64_Pool<> rand_pool64(42 + ippl::Comm->rank());
    Kokkos::parallel_for(
        nloc, GenerateRandomParticlesWithCharges<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                  bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));
    Kokkos::fence();

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After particle generation");

    // Setup FFT parameters
    ippl::ParameterList fftParams;
    fftParams.add("tolerance", tolerance);
    fftParams.add("use_finufft_defaults", false);
    fftParams.add("use_kokkos_nufft", false);
    fftParams.add("spread_method", spread_method);
    fftParams.add("tile_size_3d", 3);
    fftParams.add("z_tiles", 1);
    fftParams.add("team_size", 2);
    fftParams.add("sort", true);

    auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
    bunch.update();

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After FFT plan creation");

    std::vector<double> times(benchmark_runs);

    // Warmup
    if (ippl::Comm->rank() == 0) {
        std::cout << "Running warmup..." << std::endl;
    }

    for (int i = 0; i < warmup_runs; ++i) {
        fft->transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
    }

    MPI_Barrier(ippl::Comm->getCommunicator());

    // Reset timers after warmup
    IpplTimings::resetAllTimers();

    // Benchmark
    if (ippl::Comm->rank() == 0) {
        std::cout << "Running benchmark..." << std::endl;
    }

    for (int run = 0; run < benchmark_runs; ++run) {
        MPI_Barrier(ippl::Comm->getCommunicator());
        Kokkos::fence();

        auto start = std::chrono::high_resolution_clock::now();
        fft->transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());

        auto end    = std::chrono::high_resolution_clock::now();
        times[run] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Gather statistics
    std::vector<double> global_times = gatherMaxTimes(times);

    auto [mean, stddev, min_t, max_t] = computeStats(times);

    double global_mean, global_min, global_max;
    MPI_Allreduce(&mean, &global_mean, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(&min_t, &global_min, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_t, &global_max, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());

    double throughput = (Np / global_mean * 1000.0) / 1e6;  // Mpts/s

    BenchmarkResult result;
    result.num_ranks      = ippl::Comm->size();
    result.num_particles  = Np;
    result.grid_size      = grid_size;
    result.mean_time      = global_mean;
    result.min_time       = global_min;
    result.max_time       = global_max;
    result.stddev         = stddev;
    result.throughput_mpts = throughput;
    result.all_times      = global_times;

    return result;
}

BenchmarkResult benchmarkNUFFTType2(int grid_size, int particles_per_point, double tolerance,
                                    const std::string& gather_method, int warmup_runs,
                                    int benchmark_runs) {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;
    using size_type            = ippl::detail::size_type;
    using Vector_t             = ippl::Vector<double, dim>;
    using playout_type         = ippl::ParticleSpatialLayout<double, dim>;
    using bunch_type           = Bunch<playout_type>;
    using field_type = typename ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type;
    using real_field_type = typename ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type;
    using FFT_type        = ippl::FFT<ippl::NUFFTransform, real_field_type>;

    const double pi = std::acos(-1.0);

    // Setup domain
    ippl::Vector<int, dim> pt = {grid_size, grid_size, grid_size};
    ippl::Index I(pt[0]);
    ippl::Index J(pt[1]);
    ippl::Index K(pt[2]);
    ippl::NDIndex<dim> owned(I, J, K);

    std::array<bool, dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

    Vector_t minU = {0, 0, 0};
    Vector_t maxU = {2 * pi,2 * pi,2 * pi};

    std::array<double, dim> dx = {
        (maxU[0] - minU[0]) / double(pt[0]),
        (maxU[1] - minU[1]) / double(pt[1]),
        (maxU[2] - minU[2]) / double(pt[2]),
    };

    Vector_t hx     = {dx[0], dx[1], dx[2]};
    Vector_t origin = {minU[0], minU[1], minU[2]};
    Mesh_t mesh(owned, hx, origin);

    playout_type pl(layout, mesh);

    size_type Np   = static_cast<size_type>(std::pow(grid_size, 3)) * particles_per_point;
    size_type nloc = Np / ippl::Comm->size();

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n=== Benchmarking NUFFT Type 2 ===" << std::endl;
        std::cout << "Grid size:       " << grid_size << "^3" << std::endl;
        std::cout << "Total particles: " << Np << std::endl;
        std::cout << "Local particles: " << nloc << std::endl;
        std::cout << "Gather method:   " << gather_method << std::endl;
        std::cout << "Tolerance:       " << tolerance << std::endl;
        std::cout << "Warmup runs:     " << warmup_runs << std::endl;
        std::cout << "Benchmark runs:  " << benchmark_runs << std::endl;
    }

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("Before allocation");

    // Create bunch and field
    bunch_type bunch(pl);
    bunch.setParticleBC(ippl::BC::PERIODIC);
    bunch.create(nloc);

    field_type field(mesh, layout);

    // Generate random particles (positions only for Type 2)
    Kokkos::Random_XorShift64_Pool<> rand_pool64(42 + ippl::Comm->rank());

    auto R_view = bunch.R.getView();
    Kokkos::parallel_for(
        "GeneratePositions", nloc, KOKKOS_LAMBDA(const size_t i) {
            auto rand_gen = rand_pool64.get_state();
            for (unsigned d = 0; d < dim; ++d) {
                R_view(i)[d] = rand_gen.drand(minU[d], maxU[d]);
            }
            rand_pool64.free_state(rand_gen);
        });
    Kokkos::fence();

    bunch.update();

    // Generate random field
    const int nghost = field.getNghost();
    auto fview       = field.getView();
    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "GenerateRandomField",
        mdrange_type({nghost, nghost, nghost},
                     {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
        GenerateRandomField<Kokkos::complex<double>, Kokkos::Random_XorShift64_Pool<>, dim>(
            fview, rand_pool64));
    Kokkos::fence();

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After data generation");

    // Setup FFT parameters
    ippl::ParameterList fftParams;
    fftParams.add("tolerance", tolerance);
    fftParams.add("use_finufft_defaults", false);
    fftParams.add("use_kokkos_nufft", false);
    fftParams.add("gather_method", gather_method);
    fftParams.add("sort", true);

    auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After FFT plan creation");

    std::vector<double> times(benchmark_runs);

    // Warmup
    if (ippl::Comm->rank() == 0) {
        std::cout << "Running warmup..." << std::endl;
    }

    for (int i = 0; i < warmup_runs; ++i) {
        fft->transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
    }

    MPI_Barrier(ippl::Comm->getCommunicator());

    // Reset timers after warmup (only if Type 1 wasn't run, otherwise keep accumulated)
    // Note: If running both types, Type 1 already reset timers, so we don't reset again here
    // IpplTimings::resetAllTimers();

    // Benchmark
    if (ippl::Comm->rank() == 0) {
        std::cout << "Running benchmark..." << std::endl;
    }

    for (int run = 0; run < benchmark_runs; ++run) {
        MPI_Barrier(ippl::Comm->getCommunicator());
        Kokkos::fence();

        auto start = std::chrono::high_resolution_clock::now();
        fft->transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());

        auto end    = std::chrono::high_resolution_clock::now();
        times[run] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Gather statistics
    std::vector<double> global_times = gatherMaxTimes(times);

    auto [mean, stddev, min_t, max_t] = computeStats(times);

    double global_mean, global_min, global_max;
    MPI_Allreduce(&mean, &global_mean, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(&min_t, &global_min, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_t, &global_max, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());

    double throughput = (Np / global_mean * 1000.0) / 1e6;  // Mpts/s

    BenchmarkResult result;
    result.num_ranks      = ippl::Comm->size();
    result.num_particles  = Np;
    result.grid_size      = grid_size;
    result.mean_time      = global_mean;
    result.min_time       = global_min;
    result.max_time       = global_max;
    result.stddev         = stddev;
    result.throughput_mpts = throughput;
    result.all_times      = global_times;

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Default parameters
        int grid_size           = 8;
        int particles_per_point = 1;
        double tolerance        = 1e-4;
        int warmup_runs         = 3;
        int benchmark_runs      = 10;
        std::string spread_method = "output_focused";
        std::string gather_method = "atomic_sort";
        std::string csv_filename  = "nufft_scaling.csv";
        std::string component_csv = "nufft_components.csv";
        bool run_type1            = true;
        bool run_type2            = true;
        bool dump_components      = true;

        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--grid" && i + 1 < argc) {
                grid_size = std::atoi(argv[++i]);
            } else if (arg == "--ppp" && i + 1 < argc) {
                particles_per_point = std::atoi(argv[++i]);
            } else if (arg == "--tol" && i + 1 < argc) {
                tolerance = std::atof(argv[++i]);
            } else if (arg == "--warmup" && i + 1 < argc) {
                warmup_runs = std::atoi(argv[++i]);
            } else if (arg == "--runs" && i + 1 < argc) {
                benchmark_runs = std::atoi(argv[++i]);
            } else if (arg == "--spread" && i + 1 < argc) {
                spread_method = argv[++i];
            } else if (arg == "--gather" && i + 1 < argc) {
                gather_method = argv[++i];
            } else if (arg == "--csv" && i + 1 < argc) {
                csv_filename = argv[++i];
            } else if (arg == "--component-csv" && i + 1 < argc) {
                component_csv = argv[++i];
            } else if (arg == "--type1-only") {
                run_type1 = true;
                run_type2 = false;
            } else if (arg == "--type2-only") {
                run_type1 = false;
                run_type2 = true;
            } else if (arg == "--no-components") {
                dump_components = false;
            } else if (arg == "--help") {
                if (ippl::Comm->rank() == 0) {
                    std::cout << "NUFFT Scaling Benchmark\n"
                              << "Usage: " << argv[0] << " [options]\n\n"
                              << "Options:\n"
                              << "  --grid N          Grid size (default: 256)\n"
                              << "  --ppp N           Particles per grid point (default: 10)\n"
                              << "  --tol T           Tolerance (default: 1e-4)\n"
                              << "  --warmup N        Warmup runs (default: 3)\n"
                              << "  --runs N          Benchmark runs (default: 10)\n"
                              << "  --spread S        Spread method: output_focused, tiled, atomic (default: output_focused)\n"
                              << "  --gather S        Gather method: tiled, atomic, atomic_sort, native (default: atomic_sort)\n"
                              << "  --csv FILE        Output CSV filename (default: nufft_scaling.csv)\n"
                              << "  --component-csv F Component timings CSV (default: nufft_components.csv)\n"
                              << "  --type1-only      Run only Type 1 benchmark\n"
                              << "  --type2-only      Run only Type 2 benchmark\n"
                              << "  --no-components   Don't dump component timings\n"
                              << "  --help            Show this help message\n";
                }
                ippl::finalize();
                return 0;
            }
        }

        int num_ranks = ippl::Comm->size();
        size_t num_particles = static_cast<size_t>(std::pow(grid_size, 3)) * particles_per_point;

        if (ippl::Comm->rank() == 0) {
            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "NUFFT Scaling Benchmark" << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            std::cout << "Number of ranks: " << num_ranks << std::endl;
            std::cout << "Grid size:       " << grid_size << "^3" << std::endl;
            std::cout << "Particles:       " << num_particles << " (" << particles_per_point << " per point)" << std::endl;
            std::cout << "Tolerance:       " << tolerance << std::endl;
            std::cout << "CSV output:      " << csv_filename << std::endl;
            std::cout << "Component CSV:   " << component_csv << std::endl;
            printMemoryUsage("Initial state");
        }

        BenchmarkResult type1_result, type2_result;

        // Run Type 1 benchmark
        if (run_type1) {
            type1_result = benchmarkNUFFTType1(grid_size, particles_per_point, tolerance,
                                               spread_method, warmup_runs, benchmark_runs);
            printResult("NUFFT Type 1 Results", type1_result);
        }

        // Run Type 2 benchmark
        if (run_type2) {
            type2_result = benchmarkNUFFTType2(grid_size, particles_per_point, tolerance,
                                               gather_method, warmup_runs, benchmark_runs);
            printResult("NUFFT Type 2 Results", type2_result);
        }

        // Write results to CSV
        if (run_type1 && run_type2) {
            writeTimingsCSV(csv_filename, num_ranks, grid_size, num_particles,
                            type1_result, type2_result);
        }

        // Print and dump component timings
        if (dump_components) {
            printComponentTimings();
            writeComponentTimingsCSV(component_csv, num_ranks, grid_size, num_particles);
        }

        // Print summary
        if (ippl::Comm->rank() == 0) {
            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "SUMMARY" << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            std::cout << std::fixed << std::setprecision(3);

            if (run_type1) {
                std::cout << "Type 1: " << type1_result.mean_time << " ms ("
                          << std::setprecision(2) << type1_result.throughput_mpts << " Mpts/s)" << std::endl;
            }
            if (run_type2) {
                std::cout << "Type 2: " << type2_result.mean_time << " ms ("
                          << std::setprecision(2) << type2_result.throughput_mpts << " Mpts/s)" << std::endl;
            }
            std::cout << std::string(80, '=') << std::endl;
        }
    }
    ippl::finalize();
    return 0;
}