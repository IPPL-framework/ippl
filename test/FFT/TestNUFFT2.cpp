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

        std::cout << "Width "
                  << static_cast<int>(std::ceil(
                         std::log10((1.0) / fftParams.get<double>("tolerance"))))
                         + 1
                  << std::endl;

        std::unique_ptr<FFT_type> fft;

        int type = 2;  // Type-2 NUFFT

        size_type nloc = Np / ippl::Comm->size();

        bunch.create(nloc);
        fft = std::make_unique<FFT_type>(layout, nloc, type, fftParams);

        // RNG
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42));

        // Random particles and charges
        Kokkos::parallel_for(
            nloc,
            generate_random_particles<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));

        // Random Fourier modes on the base grid (interior only)
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto fview         = field.getView();

        Kokkos::parallel_for(
            "fill_modes",
            mdrange_type({0, 0, 0}, {pt[0], pt[1], pt[2]}),
            generate_random_field<Kokkos::complex<double>,
                                  Kokkos::Random_XorShift64_Pool<>, dim>(
                field.getView(), rand_pool64, nghost));

        bunch.update();
        field.fillHalo();

        // Pick one particle to check (e.g. middle local particle)
        size_type nloc_actual = bunch.getLocalNum();
        size_type idx         = nloc_actual / 2;

        auto Rview = bunch.R.getView();
        Kokkos::complex<double> imag = {0.0, 1.0};

        // Reference NUDFT type-2:
        // q(x_j) = sum_k f_k * exp(+i k · x_j)
        Kokkos::complex<double> dft_value(0.0, 0.0);

        Kokkos::parallel_reduce(
            "NUDFT type2",
            mdrange_type({0, 0, 0}, {pt[0], pt[1], pt[2]}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          Kokkos::complex<double>& valL) {
                // Corner indexing -> centered integer frequency k_c
                // Using same convention as type-1:
                // index in [0, N-1] maps to integer k_c in
                // [0, N/2] ∪ [-N/2+1, -1]
                int kc0 = (i < pt[0] / 2 ? i : i - pt[0]);
                int kc1 = (j < pt[1] / 2 ? j : j - pt[1]);
                int kc2 = (k < pt[2] / 2 ? k : k - pt[2]);

                // Domain length is 2*pi, so factor 2*pi/L = 1
                double arg = 0.0;
                arg += kc0 * Rview(idx)[0];
                arg += kc1 * Rview(idx)[1];
                arg += kc2 * Rview(idx)[2];

                auto fk = fview(i + nghost, j + nghost, k + nghost);

                valL += (Kokkos::cos(arg) + imag * Kokkos::sin(arg)) * fk;
            },
            Kokkos::Sum<Kokkos::complex<double>>(dft_value));
        double ref_real  = dft_value.real();

        // Type-2 NUFFT: grid (field) -> values at particle positions (bunch.Q)
        fft->transform(bunch.R, bunch.Q, field);

        // Copy NUFFT result (real-valued Q) to host
        auto Q_result = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bunch.Q.getView());



        // Compare real parts: NUFFT stores result in real Q
        double nufft_val = Q_result(idx);

        double abs_error_real = std::fabs(ref_real - nufft_val);
        double rel_error_real = abs_error_real / std::fabs(ref_real);

        std::cout << "NUFFT (type-2) value at particle " << idx << " : "
                  << std::setprecision(16) << nufft_val << std::endl;
        std::cout << "Reference NUDFT value (real part): "
                  << std::setprecision(16) << ref_real << std::endl;

        std::cout << "Abs Error in real part: " << std::setprecision(16)
                  << abs_error_real
                  << "  Rel. error in real part: " << std::setprecision(16)
                  << rel_error_real << std::endl;
    }
    ippl::finalize();
    return 0;
}
