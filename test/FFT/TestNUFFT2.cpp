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

    using charge_container_type = ippl::ParticleAttrib<double>;
    charge_container_type Q;
};

// Random particles in [minU,maxU], random charges in [0,1]
template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random_particles {
    using view_type        = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type       = typename T::value_type;
    using view_type_scalar = typename ippl::detail::ViewType<double, 1>::view_type;

    view_type x;
    view_type_scalar Q;

    GeneratorPool rand_pool;
    T minU, maxU;

    generate_random_particles(view_type x_, view_type_scalar Q_, GeneratorPool rand_pool_,
                              T& minU_, T& maxU_)
        : x(x_)
        , Q(Q_)
        , rand_pool(rand_pool_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
        }
        Q(i) = rand_gen.drand(0.0, 1.0);

        rand_pool.free_state(rand_gen);
    }
};

// Random complex Fourier modes on the *interior* of the grid
template <typename ComplexT, class GeneratorPool, unsigned Dim>
struct generate_random_field {
    using view_type = typename ippl::detail::ViewType<ComplexT, Dim>::view_type;

    view_type f;
    GeneratorPool rand_pool;
    int nghost;

    generate_random_field(view_type f_, GeneratorPool rand_pool_, int nghost_)
        : f(f_)
        , rand_pool(rand_pool_)
        , nghost(nghost_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        auto& v = f(i + nghost, j + nghost, k + nghost);
        v.real() = rand_gen.drand(0.0, 1.0);
        v.imag() = rand_gen.drand(0.0, 1.0);

        rand_pool.free_state(rand_gen);
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t      = ippl::UniformCartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;

        const double pi = std::acos(-1.0);

        using playout_type = ippl::ParticleSpatialLayout<double, 3>;
        using bunch_type   = Bunch<playout_type>;

        int myRank = ippl::Comm->rank();
        int nRanks = ippl::Comm->size();

        // Mode grid (Fourier mode counts)
        ippl::Vector<int, dim> pt = {16, 16, 16};
        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        using Vector_t = ippl::Vector<double, 3>;
        Vector_t minU  = {0.0, 0.0, 0.0};
        Vector_t maxU  = {2 * pi, 2 * pi, 2 * pi};

        std::array<double, dim> dx = {
            (maxU[0] - minU[0]) / double(pt[0]),
            (maxU[1] - minU[1]) / double(pt[1]),
            (maxU[2] - minU[2]) / double(pt[2]),
        };

        Vector_t hx     = {dx[0], dx[1], dx[2]};
        Vector_t origin = {minU[0], minU[1], minU[2]};
        ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

        playout_type pl(layout, mesh);

        bunch_type bunch(pl);
        bunch.setParticleBC(ippl::BC::PERIODIC);

        using size_type = ippl::detail::size_type;

        // Number of particles
        size_type Np = std::pow(16, 3);

        using field_type =
            ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t, Kokkos::DefaultExecutionSpace>::uniform_type;
        using real_field_type =
            ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type;

        // Just 1 ghost layer around the Fourier mode field
        const int nghost = 1;
        field_type field(mesh, layout, nghost);

        ippl::ParameterList fftParams;

        fftParams.add("tolerance", 1e-7);
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
        fftParams.add("use_upsampled_inputs", false);
        fftParams.add("use_kokkos_nufft", false);

        // Type-2 uses gather, so set gather method
        fftParams.add("spread_method", "tiled");
        fftParams.add("gather_method", "atomic_sort");
        fftParams.add("sort", true);

        using FFT_type = ippl::FFT<ippl::NUFFTransform, real_field_type>;

        double tolerance = fftParams.get<double>("tolerance");

        if (myRank == 0) {
            std::cout << "Testing Type-2 NUFFT (use_upsampled_inputs = false)" << std::endl;
            std::cout << "Grid size: " << pt[0] << " x " << pt[1] << " x " << pt[2] << std::endl;
            std::cout << "Number of ranks: " << nRanks << std::endl;
            std::cout << "Kernel width: "
                      << static_cast<int>(std::ceil(std::log10(1.0 / tolerance))) + 1
                      << std::endl;
        }

        std::unique_ptr<FFT_type> fft;

        int type = 2;  // Type-2 NUFFT

        size_type nloc = Np / nRanks;

        bunch.create(nloc);
        fft = std::make_unique<FFT_type>(layout, nloc, type, fftParams);

        // RNG - use rank-specific seed for particles
        Kokkos::Random_XorShift64_Pool<> rand_pool_particles((size_type)(42 + myRank));
        // Use same seed across ranks for field to ensure consistent global field
        Kokkos::Random_XorShift64_Pool<> rand_pool_field((size_type)(123));

        // Random particles and charges
        Kokkos::parallel_for(
            nloc,
            generate_random_particles<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                bunch.R.getView(), bunch.Q.getView(), rand_pool_particles, minU, maxU));

        // Get local domain info for the field
        const auto& lDom = layout.getLocalNDIndex();
        int local_i_start = lDom[0].first();
        int local_j_start = lDom[1].first();
        int local_k_start = lDom[2].first();
        int local_i_end = lDom[0].last() + 1;
        int local_j_end = lDom[1].last() + 1;
        int local_k_end = lDom[2].last() + 1;

        // Random Fourier modes on the local portion of the grid
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto fview = field.getView();

        int local_ni = local_i_end - local_i_start;
        int local_nj = local_j_end - local_j_start;
        int local_nk = local_k_end - local_k_start;

        Kokkos::parallel_for(
            "fill_modes",
            mdrange_type({0, 0, 0}, {local_ni, local_nj, local_nk}),
            generate_random_field<Kokkos::complex<double>,
                                  Kokkos::Random_XorShift64_Pool<>, dim>(
                field.getView(), rand_pool_field, nghost));

        bunch.update();
        field.fillHalo();

        // Get local particle count after update
        size_type nloc_actual = bunch.getLocalNum();

        if (myRank == 0) {
            std::cout << "Total particles: " << Np << std::endl;
        }

        // We'll test with a specific particle. Pick the first particle on rank 0.
        // We need to broadcast this particle's position to all ranks for the reference calculation.

        // Get test particle position from rank 0
        Vector_t test_pos;
        if (nloc_actual > 0) {
            auto R_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bunch.R.getView());
            test_pos[0] = R_host(0)[0];
            test_pos[1] = R_host(0)[1];
            test_pos[2] = R_host(0)[2];
        }

        // Find which rank has particles and use the first particle from that rank
        int rank_with_particles = -1;
        int has_particles = (nloc_actual > 0) ? 1 : 0;

        // Allgather to find first rank with particles
        std::vector<int> all_has_particles(nRanks);
        MPI_Allgather(&has_particles, 1, MPI_INT, all_has_particles.data(), 1, MPI_INT,
                      ippl::Comm->getCommunicator());

        for (int r = 0; r < nRanks; ++r) {
            if (all_has_particles[r]) {
                rank_with_particles = r;
                break;
            }
        }

        // Broadcast test particle position from the rank that has it
        double pos_buf[3] = {test_pos[0], test_pos[1], test_pos[2]};
        MPI_Bcast(pos_buf, 3, MPI_DOUBLE, rank_with_particles, ippl::Comm->getCommunicator());
        test_pos[0] = pos_buf[0];
        test_pos[1] = pos_buf[1];
        test_pos[2] = pos_buf[2];

        if (myRank == 0) {
            std::cout << "Test particle position: (" << test_pos[0] << ", "
                      << test_pos[1] << ", " << test_pos[2] << ")" << std::endl;
        }

        // Reference NUDFT type-2:
        // q(x_j) = sum_k f_k * exp(+i k · x_j)
        // Each rank computes partial sum over its local modes, then reduce
        Kokkos::complex<double> dft_local(0.0, 0.0);
        Kokkos::complex<double> imag = {0.0, 1.0};

        // Create device copies of test position and grid info
        auto test_pos_d = test_pos;
        auto pt_d = pt;
        auto local_i_start_d = local_i_start;
        auto local_j_start_d = local_j_start;
        auto local_k_start_d = local_k_start;

        Kokkos::parallel_reduce(
            "NUDFT type2 local",
            mdrange_type({0, 0, 0}, {local_ni, local_nj, local_nk}),
            KOKKOS_LAMBDA(const int li, const int lj, const int lk,
                          Kokkos::complex<double>& valL) {
                // Convert local index to global index
                int i = li + local_i_start_d;
                int j = lj + local_j_start_d;
                int k = lk + local_k_start_d;

                // Corner indexing -> centered integer frequency k_c
                int kc0 = (i < pt_d[0] / 2 ? i : i - pt_d[0]);
                int kc1 = (j < pt_d[1] / 2 ? j : j - pt_d[1]);
                int kc2 = (k < pt_d[2] / 2 ? k : k - pt_d[2]);

                // Domain length is 2*pi, so factor 2*pi/L = 1
                double arg = 0.0;
                arg += kc0 * test_pos_d[0];
                arg += kc1 * test_pos_d[1];
                arg += kc2 * test_pos_d[2];

                auto fk = fview(li + nghost, lj + nghost, lk + nghost);

                valL += (Kokkos::cos(arg) + imag * Kokkos::sin(arg)) * fk;
            },
            Kokkos::Sum<Kokkos::complex<double>>(dft_local));

        // Global reduce of DFT result
        Kokkos::complex<double> dft_global(0.0, 0.0);
        double dft_send[2] = {dft_local.real(), dft_local.imag()};
        std::cout << "rank " << ippl::Comm->rank() << " ovserves " << dft_global << std::endl;
        double dft_recv[2] = {0.0, 0.0};
        MPI_Allreduce(dft_send, dft_recv, 2, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
        dft_global = Kokkos::complex<double>(dft_recv[0], dft_recv[1]);

        double ref_real = dft_global.real();

        // Type-2 NUFFT: grid (field) -> values at particle positions (bunch.Q)
        fft->transform(bunch.R, bunch.Q, field);

        // Get the NUFFT result from the rank that has the test particle
        double nufft_val = 0.0;
        if (myRank == rank_with_particles && nloc_actual > 0) {
            auto Q_result = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), bunch.Q.getView());
            nufft_val = Q_result(0);
        }

        // Broadcast the NUFFT result to all ranks (for comparison)
        MPI_Bcast(&nufft_val, 1, MPI_DOUBLE, rank_with_particles, ippl::Comm->getCommunicator());

        // Compare real parts
        double abs_error_real = std::fabs(ref_real - nufft_val);
        double rel_error_real = abs_error_real / std::fabs(ref_real);

        if (myRank == 0) {
            std::cout << "\n=== Results (Type-2 NUFFT, use_upsampled_inputs = false) ===" << std::endl;
            std::cout << "NUFFT (type-2) value at test particle: "
                      << std::setprecision(16) << nufft_val << std::endl;
            std::cout << "Reference NUDFT value (real part): "
                      << std::setprecision(16) << ref_real << std::endl;
            std::cout << "Abs Error in real part: " << std::setprecision(16)
                      << abs_error_real
                      << "  Rel. error in real part: " << std::setprecision(16)
                      << rel_error_real << std::endl;

            // Check if error is within tolerance
            bool passed = (rel_error_real < tolerance * 100);
            std::cout << "Test " << (passed ? "PASSED" : "FAILED") << std::endl;
        }

        ippl::Comm->barrier();
    }
    ippl::finalize();
    return 0;
}