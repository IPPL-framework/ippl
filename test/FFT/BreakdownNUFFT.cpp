
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
#include <string>
#include <vector>

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }

    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random_particles_with_charges {
    using view_type        = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type       = typename T::value_type;
    using view_type_scalar = typename ippl::detail::ViewType<value_type, 1>::view_type;

    view_type x;
    view_type_scalar Q;
    GeneratorPool rand_pool;
    T minU, maxU;

    generate_random_particles_with_charges(view_type x_, view_type_scalar Q_,
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
struct generate_random_field {
    using view_type = typename ippl::detail::ViewType<T, Dim>::view_type;
    view_type f;
    GeneratorPool rand_pool;

    generate_random_field(view_type f_, GeneratorPool rand_pool_)
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

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t      = ippl::UniformCartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        using size_type   = ippl::detail::size_type;

        const double pi = std::acos(-1.0);

        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;
        typedef ippl::Vector<double, 3> Vector_t;
        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type field_type;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type real_field_type;
        typedef ippl::FFT<ippl::NUFFTransform, real_field_type> FFT_type;

        // Parse command line arguments
        int grid_size = 256;
        int rho = 10;
        double tol = 1e-4;
        int warmup_runs = 5;
        int benchmark_runs = 20;

        if (argc > 1) grid_size = std::atoi(argv[1]);
        if (argc > 2) rho = std::atoi(argv[2]);
        if (argc > 3) tol = std::stod(argv[3]);
        if (argc > 4) warmup_runs = std::atoi(argv[4]);
        if (argc > 5) benchmark_runs = std::atoi(argv[5]);

        if (ippl::Comm->rank() == 0) {
            std::cout << "========================================\n";
            std::cout << "NUFFT Breakdown Benchmark\n";
            std::cout << "========================================\n";
            std::cout << "Grid:      " << grid_size << "^3\n";
            std::cout << "Rho:       " << rho << " particles/cell\n";
            std::cout << "Tolerance: " << tol << "\n";
            std::cout << "Warmup:    " << warmup_runs << "\n";
            std::cout << "Runs:      " << benchmark_runs << "\n";
            std::cout << "========================================\n\n";
        }

        // Setup mesh and layout
        ippl::Vector<int, dim> pt = {grid_size, grid_size, grid_size};
        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::array<bool, dim> isParallel;
        isParallel.fill(false);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        Vector_t minU = {-pi, -pi, -pi};
        Vector_t maxU = {pi, pi, pi};

        std::array<double, dim> dx = {
            (maxU[0] - minU[0]) / double(pt[0]),
            (maxU[1] - minU[1]) / double(pt[1]),
            (maxU[2] - minU[2]) / double(pt[2]),
        };

        Vector_t hx     = {dx[0], dx[1], dx[2]};
        Vector_t origin = {minU[0], minU[1], minU[2]};
        ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

        playout_type pl(layout, mesh);

        size_type Np = std::pow(grid_size, 3) * rho;
        size_type nloc = Np / ippl::Comm->size();

        // Create particles and field
        bunch_type bunch(pl);
        bunch.setParticleBC(ippl::BC::PERIODIC);
        bunch.create(nloc);

        field_type field(mesh, layout);

        // Generate random particles with charges
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42));
        Kokkos::parallel_for(nloc,
            generate_random_particles_with_charges<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));
        Kokkos::fence();

        // Generate random field for type-2
        const int nghost = field.getNghost();
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto fview = field.getView();
        Kokkos::parallel_for(
            mdrange_type({nghost, nghost, nghost},
                        {fview.extent(0) - nghost, fview.extent(1) - nghost,
                         fview.extent(2) - nghost}),
            generate_random_field<Kokkos::complex<double>, Kokkos::Random_XorShift64_Pool<>, dim>(
                field.getView(), rand_pool64));
        Kokkos::fence();

        // Create NUFFT with best implementations:
        // - Grid-Parallel (output_focused) for spreading
        // - Sorted (atomic_sort) for interpolation
        ippl::ParameterList fftParams;
        fftParams.add("tolerance", tol);
        fftParams.add("use_finufft_defaults", false);
        fftParams.add("use_kokkos_nufft", false);
        fftParams.add("spread_method", "output_focused");  // Grid-Parallel
        fftParams.add("gather_method", "atomic_sort");     // Sorted
        fftParams.add("tile_size_3d", 4);

        // Create FFT objects for type-1 and type-2
        auto fft_type1 = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
        auto fft_type2 = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);

        // ============================================================
        // Warmup runs
        // ============================================================
        if (ippl::Comm->rank() == 0) {
            std::cout << "Running warmup (" << warmup_runs << " iterations)..." << std::flush;
        }

        for (int i = 0; i < warmup_runs; ++i) {
            // Type-1: particles -> field
            fft_type1->transform(bunch.R, bunch.Q, field);
            Kokkos::fence();

            // Type-2: field -> particles
            fft_type2->transform(bunch.R, bunch.Q, field);
            Kokkos::fence();
        }

        if (ippl::Comm->rank() == 0) {
            std::cout << " done\n";
        }

        // ============================================================
        // Reset timers after warmup
        // ============================================================
        IpplTimings::resetAllTimers();

        // ============================================================
        // Benchmark runs
        // ============================================================
        if (ippl::Comm->rank() == 0) {
            std::cout << "Running benchmark (" << benchmark_runs << " iterations)..." << std::flush;
        }

        // Total timer for complete type-1 + type-2 cycle
        static IpplTimings::TimerRef totalTimer = IpplTimings::getTimer("Total_NUFFT_Cycle");

        for (int i = 0; i < benchmark_runs; ++i) {
            IpplTimings::startTimer(totalTimer);

            // Type-1: particles -> field (spreading)
            // Internal timers: scatterTimerNUFFT1, accumulateHaloNUFFT1,
            //                  FFTNUFFT1, deconvolutionNUFFT1
            fft_type1->transform(bunch.R, bunch.Q, field);
            Kokkos::fence();

            // Type-2: field -> particles (interpolation)
            // Internal timers: PrecorrectionNUFFT2, FFTNUFFT2,
            //                  FillHaloNUFFT2, GatherNUFFT2
            fft_type2->transform(bunch.R, bunch.Q, field);
            Kokkos::fence();

            IpplTimings::stopTimer(totalTimer);
        }

        if (ippl::Comm->rank() == 0) {
            std::cout << " done\n\n";
        }

        // ============================================================
        // Print timing results
        // ============================================================
        IpplTimings::print();

        // ============================================================
        // Export to CSV for plotting
        // ============================================================
        std::string csv_filename = "nufft_breakdown_" + std::to_string(grid_size) +
                                   "_rho" + std::to_string(rho) + ".csv";
        IpplTimings::dumpToCSV(csv_filename);

        if (ippl::Comm->rank() == 0) {
            std::cout << "\nResults exported to: " << csv_filename << "\n";

            // Print summary
            std::cout << "\n========================================\n";
            std::cout << "SUMMARY\n";
            std::cout << "========================================\n";

            auto printTimerSummary = [&](const std::string& name, const std::string& label) {
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
            printTimerSummary("Total_NUFFT_Cycle", "Full cycle (T1+T2)");

            // Throughput
            const auto& totalMeasurements = IpplTimings::getMeasurements("Total_NUFFT_Cycle");
            if (!totalMeasurements.empty()) {
                double sum = 0.0;
                for (double m : totalMeasurements) sum += m;
                double mean_s = sum / totalMeasurements.size();
                double throughput = Np / mean_s / 1e6;
                std::cout << "\nThroughput: " << std::fixed << std::setprecision(1)
                          << throughput << " Mpts/s\n";
            }
        }
    }
    ippl::finalize();
    return 0;
}