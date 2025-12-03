#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <iostream>
#include <random>
#include <typeinfo>

#include "Utility/ParameterList.h"

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

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

template <unsigned Dim>
ippl::Vector<int, Dim> centeredToCornerDC(const ippl::Vector<int, Dim>& kVec,
                                          const ippl::Vector<int, Dim>& n_modes) {
    ippl::Vector<int, Dim> cornerIdx;
    for (unsigned d = 0; d < Dim; ++d) {
        if (kVec[d] >= 0) {
            cornerIdx[d] = kVec[d];
        } else {
            cornerIdx[d] = 2 * n_modes[d] + kVec[d];
        }
    }
    return cornerIdx;
}

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random_field {
    using view_type = typename ippl::detail::ViewType<T, Dim>::view_type;
    ippl::Vector<int, Dim> n_modes;
    view_type f;

    // The GeneratorPool
    GeneratorPool rand_pool;
    int nghost;
    ippl::Vector<int, Dim> tile_start;

    // Initialize all members
    generate_random_field(view_type f_, ippl::Vector<int, Dim> n_modes_, GeneratorPool rand_pool_,
                          int nghost_, ippl::Vector<int, Dim> tile_start_)
        : f(f_)
        , n_modes(n_modes_)
        , rand_pool(rand_pool_)
        , nghost(nghost_)
        , tile_start(tile_start_) {}

    KOKKOS_INLINE_FUNCTION void operator()(int i, int j, int k) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        ippl::Vector<int, 3> kVec{i, j, k};

        auto i_rescaled = centeredToCornerDC(kVec, n_modes);

        f(i_rescaled[0], i_rescaled[1], i_rescaled[2]).real() = rand_gen.drand(0.0, 1.0);
        f(i_rescaled[0], i_rescaled[1], i_rescaled[2]).imag() = rand_gen.drand(0.0, 1.0);

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    // sleep(10);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        const double pi = std::acos(-1.0);

        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;

        int myRank = ippl::Comm->rank();
        int nRanks = ippl::Comm->size();

        // Number of modes (output size before upsampling)
        ippl::Vector<int, dim> n_modes = {32, 32, 32};

        ippl::Index I(n_modes[0]);
        ippl::Index J(n_modes[1]);
        ippl::Index K(n_modes[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        typedef ippl::Vector<double, 3> Vector_t;
        Vector_t minU = {0, 0, 0};
        Vector_t maxU = {2 * pi, 2 * pi, 2 * pi};

        std::array<double, dim> dx = {
            (maxU[0] - minU[0]) / double(n_modes[0]),
            (maxU[1] - minU[1]) / double(n_modes[1]),
            (maxU[2] - minU[2]) / double(n_modes[2]),
        };
        Vector_t hx     = {dx[0], dx[1], dx[2]};
        Vector_t origin = {minU[0], minU[1], minU[2]};
        ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

        playout_type pl(layout, mesh);

        bunch_type bunch(pl);
        bunch.setParticleBC(ippl::BC::PERIODIC);

        using size_type = ippl::detail::size_type;

        size_type Np = std::pow(32, 3) * 20;

        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type
            field_type;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type real_field_type;

        ippl::ParameterList fftParams;

        fftParams.add("tolerance", 1e-6);
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
        fftParams.add("tile_size_3d", 6);
        fftParams.add("z_tiles", 1);
        fftParams.add("team_size", 32);
        fftParams.add("sort", true);
        fftParams.add("use_upsampled_inputs", true);

        typedef ippl::FFT<ippl::NUFFTransform, real_field_type> FFT_type;

        if (myRank == 0) {
            std::cout << "Width "
                      << static_cast<int>(
                             std::ceil(std::log10((1.0) / fftParams.get<double>("tolerance"))))
                             + 1
                      << std::endl;
        }

        int type       = 2;
        size_type nloc = Np / nRanks;

        bunch.create(nloc);

        // Create FFT object first to get upsampled grid size
        std::unique_ptr<FFT_type> fft = std::make_unique<FFT_type>(layout, nloc, type, fftParams);

        // Get the upsampled grid size from the NUFFT
        double sigma     = 2.0;  // Default upsampling factor
        double tolerance = fftParams.get<double>("tolerance");
        int kernel_width = static_cast<int>(std::ceil(std::log10(1.0 / tolerance))) + 1;

        ippl::Vector<int, dim> n_grid;
        for (unsigned d = 0; d < dim; ++d) {
            size_t upsampled = std::max<size_t>(sigma * n_modes[d], 2 * kernel_width);
            n_grid[d]        = (upsampled);
        }

        if (myRank == 0) {
            std::cout << "Upsampled grid size: " << n_grid[0] << " x " << n_grid[1] << " x "
                      << n_grid[2] << std::endl;
        }

        // Create upsampled grid layout and mesh
        ippl::Index I_up(n_grid[0]);
        ippl::Index J_up(n_grid[1]);
        ippl::Index K_up(n_grid[2]);
        ippl::NDIndex<dim> owned_upsampled(I_up, J_up, K_up);

        ippl::FieldLayout<dim> layout_upsampled(MPI_COMM_WORLD, owned_upsampled, isParallel);

        Vector_t hx_up, origin_up;
        for (unsigned d = 0; d < dim; ++d) {
            origin_up[d] = 0;
            hx_up[d]     = (2.0 * pi) / n_grid[d];
        }
        ippl::UniformCartesian<double, 3> mesh_upsampled(owned_upsampled, hx_up, origin_up);

        // Create input field on upsampled grid
        const int nghost = kernel_width / 2 + 1;
        field_type field_upsampled(mesh_upsampled, layout_upsampled, nghost);

        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + myRank));
        Kokkos::parallel_for(
            nloc, generate_random_particles<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                      bunch.R.getView(), rand_pool64, minU, maxU));

        bunch.update();

        // Initialize field with random values on upsampled grid
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto fview         = field_upsampled.getView();

        const auto& lDom_up = layout_upsampled.getLocalNDIndex();
        ippl::Vector<int, 3> tile_start;
        for (int d = 0; d < 3; ++d) {
            tile_start[d] = lDom_up[d].first();
        }

        Kokkos::parallel_for(
            mdrange_type({-n_modes[0] / 2, -n_modes[1] / 2, -n_modes[2] / 2},
                         {n_modes[0] / 2, n_modes[1] / 2, n_modes[2] / 2}),
            generate_random_field<Kokkos::complex<double>, Kokkos::Random_XorShift64_Pool<>, dim>(
                field_upsampled.getView(), n_modes, rand_pool64, nghost, tile_start));

        field_upsampled.fillHalo();
        fft->transform(bunch.R, bunch.Q, field_upsampled);

        auto Q_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bunch.Q.getView());

        // Pick some target point to check
        size_type nloc_actual = bunch.getLocalNum();
        int idx               = nloc_actual / 2;

        Kokkos::complex<double> reducedValue(0.0, 0.0);

        auto Rview = bunch.R.getView();

        Kokkos::complex<double> imag = {0.0, 1.0};

        auto lDom = field_upsampled.getLayout().getLocalNDIndex();
        ippl::Vector<int, 3> local_first, local_last;
        for (unsigned d = 0; d < 3; ++d) {
            local_first[d] = lDom[d].first();
            local_last[d]  = lDom[d].last();
        }

        // Type 2 NUDFT: sum over Fourier modes
        Kokkos::parallel_reduce(
            "NUDFT type2",
            mdrange_type({local_first[0], local_first[1], local_first[2]},
                         {local_last[0] + 1, local_last[1] + 1, local_last[2] + 1}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, Kokkos::complex<double>& valL) {
                double arg = 0.0;

                auto in_bounds = [&](double g, double n_modes) {
                    return (g >= 0 && g < n_modes / 2)
                           || (g >= n_modes + n_modes / 2 && g < 2 * n_modes);
                };

                int li_in = i - local_first[0] + nghost;
                int lj_in = j - local_first[1] + nghost;
                int lk_in = k - local_first[2] + nghost;

                if (in_bounds(i, n_modes[0]) && in_bounds(j, n_modes[1])
                    && in_bounds(k, n_modes[2])) {
                    auto rescale = [&](int in, int n) {
                        if (in < n) {
                            return in;
                        } else {
                            return in - n;
                        }
                    };
                    ippl::Vector<int, 3> iVec = {rescale(i, n_modes[0]), rescale(j, n_modes[1]),
                                                 rescale(k, n_modes[2])};

                    for (size_t d = 0; d < dim; ++d) {
                        arg += (2 * pi / (hx[d] * n_modes[d])) * (iVec[d] - (n_modes[d] / 2))
                               * Rview(idx)[d];
                    }

                    valL += (Kokkos::cos(arg) + imag * Kokkos::sin(arg))
                            * fview(i + nghost - local_first[0], j + nghost - local_first[1],
                                    k + nghost - local_first[2]);
                }
            },
            Kokkos::Sum<Kokkos::complex<double>>(reducedValue));

        std::cout << "Redvalue: " << reducedValue << std::endl;

        // Global reduce for multi-rank
        Kokkos::complex<double> reducedValue_global(0.0, 0.0);
        double send_buf[2] = {reducedValue.real(), reducedValue.imag()};
        double recv_buf[2] = {0.0, 0.0};
        MPI_Reduce(send_buf, recv_buf, 2, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
        reducedValue_global = Kokkos::complex<double>(recv_buf[0], recv_buf[1]);

        if (myRank == 0) {
            double abs_error_real = std::fabs(reducedValue_global.real() - Q_result(idx));
            double rel_error_real = std::fabs(reducedValue_global.real() - Q_result(idx))
                                    / std::fabs(reducedValue_global.real());

            std::cout << "DFT reference: " << std::setprecision(16) << reducedValue_global.real()
                      << " + " << reducedValue_global.imag() << "i" << std::endl;
            std::cout << "NUFFT result:  " << std::setprecision(16) << Q_result(idx) << std::endl;
            std::cout << "Abs Error in real part: " << std::setprecision(16) << abs_error_real
                      << " Rel. error in real part: " << std::setprecision(16) << rel_error_real
                      << std::endl;

            // Check if error is within tolerance
            bool passed = (rel_error_real < tolerance * 100);
            std::cout << "Test " << (passed ? "PASSED" : "FAILED") << std::endl;
        }
        ippl::Comm->barrier();
    }
    ippl::finalize();
    return 0;
}