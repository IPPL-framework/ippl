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
struct generate_random {
    using view_type        = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type       = typename T::value_type;
    using view_type_scalar = typename ippl::detail::ViewType<value_type, 1>::view_type;
    // Output View for the random numbers
    view_type x;

    view_type_scalar Q;

    // The GeneratorPool
    GeneratorPool rand_pool;

    T minU, maxU;

    // Initialize all members
    generate_random(view_type x_, view_type_scalar Q_, GeneratorPool rand_pool_, T& minU_, T& maxU_)
        : x(x_)
        , Q(Q_)
        , rand_pool(rand_pool_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
        }
        Q(i) = rand_gen.drand(0.0, 1.0);

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

/**
 * @brief Check if a global index is owned by this rank's local domain
 */
template <unsigned Dim>
bool isOwnedLocally(const ippl::NDIndex<Dim>& lDom, const ippl::Vector<int, Dim>& globalIdx) {
    for (unsigned d = 0; d < Dim; ++d) {
        if (globalIdx[d] < lDom[d].first() || globalIdx[d] > lDom[d].last()) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Convert global index to local index (with ghost offset)
 */
template <unsigned Dim>
ippl::Vector<int, Dim> globalToLocal(const ippl::NDIndex<Dim>& lDom,
                                     const ippl::Vector<int, Dim>& globalIdx, int nghost) {
    ippl::Vector<int, Dim> localIdx;
    std::cout << "nghost=" << nghost << std::endl;
    for (unsigned d = 0; d < Dim; ++d) {
        localIdx[d] = globalIdx[d] - lDom[d].first() + nghost;
    }
    return localIdx;
}

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

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    sleep(10);
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
        ippl::Vector<int, dim> n_modes = {16, 16, 16};

        // ippl::Vector<int, dim> n_modes = {8, 8, 8};

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

        size_type Np = std::pow(16, 3);

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

        int type       = 1;
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
            // Round up to next power of 2
            n_grid[d] = std::bit_ceil(upsampled);
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

        // Create output field on upsampled grid
        const int nghost = 1;
        field_type field_upsampled(mesh_upsampled, layout_upsampled, nghost);

        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + myRank));
        Kokkos::parallel_for(nloc,
                             generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                                 bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));

        bunch.update();
        field_upsampled = Kokkos::complex(0.0);
        // field_upsampled.fillHalo();

        fft->transform(bunch.R, bunch.Q, field_upsampled);

        // Get local domain info for upsampled grid
        const auto& lDom_up    = layout_upsampled.getLocalNDIndex();
        const int nghost_field = field_upsampled.getNghost();

        ippl::Vector<int, 3> kVec;
        kVec[0] = (int)(0.37 * n_modes[0]);  // Positive frequency
        kVec[1] = (int)(0.16 * n_modes[1]);
        kVec[2] = (int)(0.23 * n_modes[2]);

        // Convert frequency indices to corner-DC format
        ippl::Vector<int, 3> globalIdx = centeredToCornerDC<dim>(kVec, n_modes);

        if (myRank == 0) {
            std::cout << "Testing frequency k = (" << kVec[0] << ", " << kVec[1] << ", " << kVec[2]
                      << ")" << std::endl;
            std::cout << "Corner-DC global index = (" << globalIdx[0] << ", " << globalIdx[1]
                      << ", " << globalIdx[2] << ")" << std::endl;
        }

        // Check if this rank owns the mode we want to check
        bool iOwnMode = isOwnedLocally<dim>(lDom_up, globalIdx);

        // Extract the NUFFT result from the rank that owns it
        Kokkos::complex<double> nufft_result(0.0, 0.0);

        if (iOwnMode) {
            auto field_host = field_upsampled.getHostMirror();
            Kokkos::deep_copy(field_host, field_upsampled.getView());

            auto localIdx = globalToLocal<dim>(lDom_up, globalIdx, nghost_field);
            nufft_result  = field_host(localIdx[0], localIdx[1], localIdx[2]);

            std::cout << "Rank " << myRank << " owns mode, local index = (" << localIdx[0] << ", "
                      << localIdx[1] << ", " << localIdx[2] << ")" << std::endl;

            std::cout << "Observes result " << nufft_result << std::endl;
        }

        // Reduce to rank 0 (only one rank has non-zero value)
        Kokkos::complex<double> nufft_result_global(0.0, 0.0);
        double send_buf[2] = {nufft_result.real(), nufft_result.imag()};
        double recv_buf[2] = {0.0, 0.0};
        MPI_Reduce(send_buf, recv_buf, 2, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
        nufft_result_global = Kokkos::complex<double>(recv_buf[0], recv_buf[1]);

        if (myRank == 0) {
            std::cout << "Nufft result " << nufft_result_global << std::endl;
        }

        // Compute DFT reference on all particles
        // DFT: f_k = sum_j c_j * exp(-i * k * x_j)
        // where k is the frequency vector: k[d] = 2*pi/L * kVec[d]
        Kokkos::complex<double> dft_local(0.0, 0.0);
        Kokkos::complex<double> imag = {0.0, 1.0};

        auto Rview = bunch.R.getView();
        auto Qview = bunch.Q.getView();

        // Get local particle count after update
        size_type nloc_actual = bunch.getLocalNum();

        // The frequency in physical units
        // For domain [0, 2*pi), the frequencies are just kVec[d]
        Kokkos::parallel_reduce(
            "NUDFT type1 local", nloc_actual,
            KOKKOS_LAMBDA(const size_t idx, Kokkos::complex<double>& valL) {
                double arg = 0.0;
                for (size_t d = 0; d < dim; ++d) {
                    arg += (2 * pi / (hx[d] * n_modes[d])) * kVec[d] * Rview(idx)[d];
                }
                // Type 1 NUFFT: sum of c_j * exp(-i * k * x_j)
                valL += (Kokkos::cos(arg) - imag * Kokkos::sin(arg)) * Qview(idx);
            },
            Kokkos::Sum<Kokkos::complex<double>>(dft_local));

        // Global reduce of DFT result
        Kokkos::complex<double> dft_global(0.0, 0.0);
        double dft_send[2] = {dft_local.real(), dft_local.imag()};
        double dft_recv[2] = {0.0, 0.0};
        MPI_Reduce(dft_send, dft_recv, 2, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
        dft_global = Kokkos::complex<double>(dft_recv[0], dft_recv[1]);

        // Compute and print errors on rank 0
        if (myRank == 0) {
            double abs_error_real = std::fabs(dft_global.real() - nufft_result_global.real());
            double rel_error_real = std::fabs(dft_global.real() - nufft_result_global.real())
                                    / std::fabs(dft_global.real());
            double abs_error_imag = std::fabs(dft_global.imag() - nufft_result_global.imag());
            double rel_error_imag = std::fabs(dft_global.imag() - nufft_result_global.imag())
                                    / std::fabs(dft_global.imag());

            std::cout << "DFT reference: " << std::setprecision(16) << dft_global.real() << " + "
                      << dft_global.imag() << "i" << std::endl;
            std::cout << "NUFFT result:  " << std::setprecision(16) << nufft_result_global.real()
                      << " + " << nufft_result_global.imag() << "i" << std::endl;
            std::cout << "Abs Error in real part: " << std::setprecision(16) << abs_error_real
                      << " Rel. error in real part: " << std::setprecision(16) << rel_error_real
                      << std::endl;
            std::cout << "Abs Error in imag part: " << std::setprecision(16) << abs_error_imag
                      << " Rel. error in imag part: " << std::setprecision(16) << rel_error_imag
                      << std::endl;

            // Check if error is within tolerance
            bool passed = (rel_error_real < tolerance * 100) && (rel_error_imag < tolerance * 100);
            std::cout << "Test " << (passed ? "PASSED" : "FAILED") << std::endl;
        }

        // Optional: Debug output - check multiple modes
        if (myRank == 0 && false) {  // Set to true for debugging
            std::cout << "\nChecking additional modes:" << std::endl;
        }

        ippl::Comm->barrier();
    }
    ippl::finalize();
    return 0;
}