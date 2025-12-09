#include "Ippl.h"
#include "Utility/ParameterList.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <iostream>
#include <random>
#include <typeinfo>
#include <chrono>

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
struct generate_random_particles {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type = typename T::value_type;
    // Output View for the random numbers
    view_type x;

    // The GeneratorPool
    GeneratorPool rand_pool;

    T minU, maxU;

    // Initialize all members
    generate_random_particles(view_type x_, GeneratorPool rand_pool_, T& minU_, T& maxU_)
        : x(x_)
        , rand_pool(rand_pool_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
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

    // The GeneratorPool
    GeneratorPool rand_pool;

    // Initialize all members
    generate_random_field(view_type f_, GeneratorPool rand_pool_)
        : f(f_)
        , rand_pool(rand_pool_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i, const size_t j, const size_t k) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        f(i, j, k).real() = rand_gen.drand(0.0, 1.0);
        f(i, j, k).imag() = rand_gen.drand(0.0, 1.0);

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

void printHeader(const std::string& test_name) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << test_name << "\n";
    std::cout << std::string(80, '=') << "\n";
}

void printResult(const std::string& method, double time_ms, size_t n_particles,
                 int grid_size, const std::string& type) {
    std::cout << std::setw(20) << method
              << " | Type " << type
              << " | Grid: " << std::setw(3) << grid_size << "^3"
              << " | Particles: " << std::setw(10) << n_particles
              << " | Time: " << std::setw(10) << std::fixed << std::setprecision(3)
              << time_ms << " ms"
              << " | Throughput: " << std::setw(8) << std::fixed << std::setprecision(2)
              << (n_particles / time_ms * 1000.0 / 1e6) << " Mpts/s"
              << std::endl;
}

template<typename FFT_type, typename Field, typename Bunch>
double benchmarkType1(FFT_type& fft, Field& field, Bunch& bunch,
                      const std::string& method, int warmup_runs = 2, int benchmark_runs = 5) {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / benchmark_runs;
}

template<typename FFT_type, typename Field, typename Bunch>
double benchmarkType2(FFT_type& fft, Field& field, Bunch& bunch,
                      const std::string& method, int warmup_runs = 2, int benchmark_runs = 5) {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
        Kokkos::fence();
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        fft.transform(bunch.R, bunch.Q, field);
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / benchmark_runs;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;
        using size_type            = ippl::detail::size_type;

        const double pi = std::acos(-1.0);

        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;
        typedef ippl::Vector<double, 3> Vector_t;
        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type field_type;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type real_field_type;
        typedef ippl::FFT<ippl::NUFFTransform, real_field_type> FFT_type;

        // Test configurations: grid size and particles per grid point
        std::vector<int> grid_sizes = {32, 64};
        std::vector<int> particles_per_point = {1, 10};

        for (int grid_size : grid_sizes) {
            for (int ppp : particles_per_point) {

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

                size_type Np = std::pow(grid_size, 3) * ppp;
                size_type nloc = Np / ippl::Comm->size();

                printHeader("Configuration: Grid=" + std::to_string(grid_size) + "^3, " +
                           "Particles=" + std::to_string(Np) + " (" + std::to_string(ppp) + " per point)");

                // ============================================================
                // TYPE 1 NUFFT BENCHMARKS
                // ============================================================

                {
                    bunch_type bunch(pl);
                    bunch.setParticleBC(ippl::BC::PERIODIC);
                    bunch.create(nloc);

                    field_type field(mesh, layout);

                    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42));
                    Kokkos::parallel_for(nloc,
                        generate_random_particles_with_charges<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                            bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));

                    // OutputFocused method
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
#ifdef ENABLE_GPU_NUFFT
                        fftParams.add("gpu_method", 1);
                        fftParams.add("gpu_sort", 0);
                        fftParams.add("gpu_kerevalmeth", 1);
#else
                        fftParams.add("spread_kerevalmeth", 1);
                        fftParams.add("spread_sort", 2);
                        fftParams.add("nthreads", 0);
#endif
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", false);
                        fftParams.add("spread_method", "output_focused");
                        fftParams.add("tile_size_3d", 3);
                        fftParams.add("z_tiles", 1);
                        fftParams.add("team_size", 32);
                        fftParams.add("sort", true);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                        double time_ms = benchmarkType1(*fft, field, bunch, "OutputFocused");
                        printResult("IPPL OutputFocused", time_ms, Np, grid_size, "1");
                    }

                    // Tiled method
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
#ifdef ENABLE_GPU_NUFFT
                        fftParams.add("gpu_method", 1);
                        fftParams.add("gpu_sort", 0);
                        fftParams.add("gpu_kerevalmeth", 1);
#else
                        fftParams.add("spread_kerevalmeth", 1);
                        fftParams.add("spread_sort", 2);
                        fftParams.add("nthreads", 0);
#endif
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", false);
                        fftParams.add("spread_method", "tiled");
                        fftParams.add("tile_size_3d", 3);
                        fftParams.add("z_tiles", 1);
                        fftParams.add("team_size", 32);
                        fftParams.add("sort", true);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                        double time_ms = benchmarkType1(*fft, field, bunch, "Tiled");
                        printResult("IPPL Tiled", time_ms, Np, grid_size, "1");
                    }

                    // Atomic method
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
#ifdef ENABLE_GPU_NUFFT
                        fftParams.add("gpu_method", 1);
                        fftParams.add("gpu_sort", 0);
                        fftParams.add("gpu_kerevalmeth", 1);
#else
                        fftParams.add("spread_kerevalmeth", 1);
                        fftParams.add("spread_sort", 2);
                        fftParams.add("nthreads", 0);
#endif
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", false);
                        fftParams.add("spread_method", "atomic");

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                        double time_ms = benchmarkType1(*fft, field, bunch, "Atomic");
                        printResult("IPPL Atomic", time_ms, Np, grid_size, "1");
                    }

#ifdef KOKKOS_NUFFT_AVAILABLE
                    // kokkos_nufft reference
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", true);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                        double time_ms = benchmarkType1(*fft, field, bunch, "kokkos_nufft");
                        printResult("kokkos_nufft", time_ms, Np, grid_size, "1");
                    }
#endif

#ifdef ENABLE_FINUFFT
                    // FINUFFT/cuFINUFFT reference
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
                        fftParams.add("use_finufft_defaults", true);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 1, fftParams);
                        double time_ms = benchmarkType1(*fft, field, bunch, "FINUFFT");
#ifdef ENABLE_GPU_NUFFT
                        printResult("cuFINUFFT", time_ms, Np, grid_size, "1");
#else
                        printResult("FINUFFT", time_ms, Np, grid_size, "1");
#endif
                    }
#endif
                }

                // ============================================================
                // TYPE 2 NUFFT BENCHMARKS
                // ============================================================

                {
                    bunch_type bunch(pl);
                    bunch.setParticleBC(ippl::BC::PERIODIC);
                    bunch.create(nloc);

                    field_type field(mesh, layout);

                    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42));

                    // Generate random particles
                    Kokkos::parallel_for(nloc,
                        generate_random_particles<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                            bunch.R.getView(), rand_pool64, minU, maxU));

                    // Generate random field
                    const int nghost = field.getNghost();
                    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
                    auto fview = field.getView();
                    Kokkos::parallel_for(
                        mdrange_type({nghost, nghost, nghost},
                                    {fview.extent(0) - nghost, fview.extent(1) - nghost,
                                     fview.extent(2) - nghost}),
                        generate_random_field<Kokkos::complex<double>, Kokkos::Random_XorShift64_Pool<>, dim>(
                            field.getView(), rand_pool64));

                    // Tiled method
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
#ifdef ENABLE_GPU_NUFFT
                        fftParams.add("gpu_method", 1);
                        fftParams.add("gpu_sort", 0);
                        fftParams.add("gpu_kerevalmeth", 1);
#else
                        fftParams.add("spread_kerevalmeth", 1);
                        fftParams.add("spread_sort", 2);
                        fftParams.add("nthreads", 0);
#endif
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", false);
                        fftParams.add("gather_method", "tiled");
                        fftParams.add("sort", true);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                        double time_ms = benchmarkType2(*fft, field, bunch, "Tiled");
                        printResult("IPPL Tiled", time_ms, Np, grid_size, "2");
                    }

                    // Atomic method
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
#ifdef ENABLE_GPU_NUFFT
                        fftParams.add("gpu_method", 1);
                        fftParams.add("gpu_sort", 0);
                        fftParams.add("gpu_kerevalmeth", 1);
#else
                        fftParams.add("spread_kerevalmeth", 1);
                        fftParams.add("spread_sort", 2);
                        fftParams.add("nthreads", 0);
#endif
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", false);
                        fftParams.add("gather_method", "atomic");

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                        double time_ms = benchmarkType2(*fft, field, bunch, "Atomic");
                        printResult("IPPL Atomic", time_ms, Np, grid_size, "2");
                    }

                    // Atomic method
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
#ifdef ENABLE_GPU_NUFFT
                        fftParams.add("gpu_method", 1);
                        fftParams.add("gpu_sort", 0);
                        fftParams.add("gpu_kerevalmeth", 1);
#else
                        fftParams.add("spread_kerevalmeth", 1);
                        fftParams.add("spread_sort", 2);
                        fftParams.add("nthreads", 0);
#endif
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", false);
                        fftParams.add("gather_method", "native");

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                        double time_ms = benchmarkType2(*fft, field, bunch, "Atomic");
                        printResult("IPPL Native", time_ms, Np, grid_size, "2");
                    }

#ifdef KOKKOS_NUFFT_AVAILABLE
                    // kokkos_nufft reference
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
                        fftParams.add("use_finufft_defaults", false);
                        fftParams.add("use_kokkos_nufft", true);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                        double time_ms = benchmarkType2(*fft, field, bunch, "kokkos_nufft");
                        printResult("kokkos_nufft", time_ms, Np, grid_size, "2");
                    }
#endif

#ifdef ENABLE_FINUFFT
                    // FINUFFT/cuFINUFFT reference
                    {
                        ippl::ParameterList fftParams;
                        fftParams.add("tolerance", 1e-10);
                        fftParams.add("use_finufft_defaults", true);

                        auto fft = std::make_unique<FFT_type>(layout, nloc, 2, fftParams);
                        double time_ms = benchmarkType2(*fft, field, bunch, "FINUFFT");
#ifdef ENABLE_GPU_NUFFT
                        printResult("cuFINUFFT", time_ms, Np, grid_size, "2");
#else
                        printResult("FINUFFT", time_ms, Np, grid_size, "2");
#endif
                    }
#endif
                }

                std::cout << std::endl;
            }
        }
    }
    ippl::finalize();
    return 0;
}
