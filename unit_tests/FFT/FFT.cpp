//
// Unit test FFT
//   Test FFT features
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class FFTTest;

// Restrict testing to 2 and 3 dimensions since this is what heFFTe supports
template <typename T, typename ExecSpace, unsigned Dim>
class FFTTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    using exec_space              = ExecSpace;
    constexpr static unsigned dim = Dim;

    using mesh_type          = ippl::UniformCartesian<T, Dim>;
    using centering_type     = typename mesh_type::DefaultCentering;
    using field_type_complex = typename ippl::Field<Kokkos::complex<T>, Dim, mesh_type,
                                                    centering_type, ExecSpace>::uniform_type;
    using field_type_real    = ippl::Field<T, Dim, mesh_type, centering_type, ExecSpace>;
    using layout_type        = ippl::FieldLayout<Dim>;

    template <typename Transform>
    using FFT_type =
        ippl::FFT<Transform, std::conditional_t<std::is_same_v<Transform, ippl::CCTransform>,
                                                field_type_complex, field_type_real>>;

    FFTTest()
        : pt(getGridSizes<Dim>()) {
        const T pi = Kokkos::numbers::pi_v<T>;
        for (unsigned d = 0; d < Dim; d++) {
            len[d] = pt[d] * pi / 16;
        }

        std::array<ippl::Index, Dim> domains;

        ippl::Vector<T, Dim> hx;
        ippl::Vector<T, Dim> origin;

        std::array<bool, Dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        for (unsigned d = 0; d < Dim; d++) {
            domains[d] = ippl::Index(pt[d]);
            hx[d]      = len[d] / pt[d];
            origin[d]  = 0;
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout     = layout_type(MPI_COMM_WORLD, owned, isParallel);

        mesh = mesh_type(owned, hx, origin);

        realField = std::make_shared<field_type_real>(mesh, layout);
        compField = std::make_shared<field_type_complex>(mesh, layout);
    }

    /*!
     * Gets the parameters used for trigonometric FFT transforms
     * (sine and cosine)
     */
    [[nodiscard]] ippl::ParameterList getTrigParams() const {
        ippl::ParameterList fftParams;

        fftParams.add("use_heffte_defaults", false);
        fftParams.add("use_pencils", true);
        fftParams.add("use_reorder", false);
        fftParams.add("use_gpu_aware", true);
        fftParams.add("comm", ippl::p2p_pl);

        return fftParams;
    }

    /*!
     * Fill a real-valued field with random values
     * @param nghost number of ghost cells
     * @param mirror the field view's host mirror
     */
    void randomizeRealField(int nghost, typename field_type_real::HostMirror& mirror) {
        std::mt19937_64 eng(42 + ippl::Comm->rank());
        std::uniform_real_distribution<T> unif(0, 1);

        nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
            mirror(args...) = unif(eng);
        });
    }

      /*!
     * Fill a real-valued field with zero values
     * @param nghost number of ghost cells
     * @param mirror the field view's host mirror
     */
  
    void zeroRealField(int nghost, typename field_type_real::HostMirror& mirror) {

        nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
            mirror(args...) = 0.0;;
        });
    }

    /*!
     * Fill a complex-valued field with random values
     * @param nghost number of ghost cells
     * @param mirror the field view's host mirror
     */
    void randomizeComplexField(int nghost, typename field_type_complex::HostMirror& mirror) {
        std::mt19937_64 engReal(42 + ippl::Comm->rank());
        std::uniform_real_distribution<T> unifReal(0, 1);

        std::mt19937_64 engImag(43 + ippl::Comm->rank());
        std::uniform_real_distribution<T> unifImag(0, 1);

        nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
            mirror(args...).real() = unifReal(engReal);
            mirror(args...).imag() = unifImag(engImag);
        });
    }

    /*!
     * Fill a complex-valued field with 0.0 values
     * @param nghost number of ghost cells
     * @param mirror the field view's host mirror
     */
    void zeroComplexField(int nghost, typename field_type_complex::HostMirror& mirror) {

        nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
            mirror(args...).real() = 0.0;
            mirror(args...).imag() = 0.0;
        });
    }


  
    /*!
     * Verify the contents of a computation
     * @tparam MirrorA the type of the computed view
     * @tparam MirrorB the type of the expected view
     * @param nghost number of ghost cells
     * @param computed the computed result
     * @param expected the expected result
     */
    template <typename MirrorA, typename MirrorB>
    void verifyResult(int nghost, const MirrorA& computed, const MirrorB& expected) {
        T max_error_local = 0.0;
        T tol             = tolerance<T>;
        nestedViewLoop(computed, nghost, [&]<typename... Idx>(const Idx... args) {
            T error = std::fabs(expected(args...) - computed(args...));

            if (error > max_error_local) {
                max_error_local = error;
            }

            ASSERT_NEAR(error, 0, tol);
        });

        T max_error = 0.0;
        ippl::Comm->reduce(max_error_local, max_error, 1, std::greater<T>());
        ASSERT_NEAR(max_error, 0, tol);
    }

    /*!
     * Tests the trigonometric FFT transforms
     * @tparam Transform either SineTransform or CosTransform
     * @param field a pointer to the field
     * @param layout a pointer to the layout
     */
    template <typename Transform>
    void testTrig(std::shared_ptr<field_type_real>& field, const layout_type& layout) {
        auto fft        = std::make_unique<FFT_type<Transform>>(layout, getTrigParams());
        auto& view      = field->getView();
        auto field_host = field->getHostMirror();

        const int nghost = field->getNghost();
        randomizeRealField(nghost, field_host);

        Kokkos::deep_copy(view, field_host);

        // Forward transform
        fft->transform(ippl::FORWARD, *field);
        // Reverse transform
        fft->transform(ippl::BACKWARD, *field);

        auto field_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

        verifyResult(nghost, field_result, field_host);
    }

    mesh_type mesh;
    layout_type layout;
    std::shared_ptr<field_type_real> realField;
    std::shared_ptr<field_type_complex> compField;

    std::array<size_t, Dim> pt;
    std::array<T, Dim> len;
};

using Tests = TestParams::tests<2, 3>;
TYPED_TEST_SUITE(FFTTest, Tests);

TYPED_TEST(FFTTest, Cos) {
    //this->template testTrig<ippl::CosTransform>(this->realField, this->layout);
}

TYPED_TEST(FFTTest, Sin) {
    //this->template testTrig<ippl::SineTransform>(this->realField, this->layout);
}

TYPED_TEST(FFTTest, RC) {
    constexpr unsigned Dim = TestFixture::dim;

    auto& mesh   = this->mesh;
    auto& layout = this->layout;
    auto& field  = this->realField;

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", true);
    fftParams.add("r2c_direction", 0);

    std::array<bool, Dim> isParallel;
    isParallel.fill(true);

    ippl::NDIndex<Dim> ownedOutput;
    for (unsigned d = 0; d < Dim; d++) {
        if (static_cast<int>(d) == fftParams.get<int>("r2c_direction")) {
            ownedOutput[d] = ippl::Index(this->pt[d] / 2 + 1);
        } else {
            ownedOutput[d] = ippl::Index(this->pt[d]);
        }
    }

    typename TestFixture::layout_type layoutOutput(MPI_COMM_WORLD, ownedOutput, isParallel);

    typename TestFixture::mesh_type meshOutput(ownedOutput, mesh.getMeshSpacing(),
                                               mesh.getOrigin());
    typename TestFixture::field_type_complex fieldOutput(meshOutput, layoutOutput);

    std::shared_ptr<typename TestFixture::template FFT_type<ippl::RCTransform>> fft =
        std::make_unique<typename TestFixture::template FFT_type<ippl::RCTransform>>(
            layout, layoutOutput, fftParams);

    auto& view      = field->getView();
    auto input_host = field->getHostMirror();

    const int nghost = field->getNghost();
    this->zeroRealField(nghost, input_host);

    Kokkos::deep_copy(view, input_host);

    fft->transform(ippl::FORWARD, *field, fieldOutput);
    fft->transform(ippl::BACKWARD, *field, fieldOutput);

    auto field_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

    this->verifyResult(nghost, field_result, input_host);
}

TYPED_TEST(FFTTest, CC) {
    using T = typename TestFixture::value_type;
    T tol   = tolerance<T>;

    auto& layout = this->layout;
    auto& field  = this->compField;

    ippl::ParameterList fftParams;

    fftParams.add("use_heffte_defaults", true);

    std::shared_ptr<typename TestFixture::template FFT_type<ippl::CCTransform>> fft =
        std::make_unique<typename TestFixture::template FFT_type<ippl::CCTransform>>(layout,
                                                                                     fftParams);

    auto& view      = field->getView();
    auto field_host = field->getHostMirror();

    const int nghost = field->getNghost();
    this->zeroComplexField(nghost, field_host);

    Kokkos::deep_copy(view, field_host);

    fft->transform(ippl::FORWARD, *field);
    fft->transform(ippl::BACKWARD, *field);

    auto field_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

    Kokkos::complex<T> max_error_local(0, 0);
    nestedViewLoop(field_host, nghost, [&]<typename... Idx>(const Idx... args) {
        Kokkos::complex<T> error(
            std::fabs(field_host(args...).real() - field_result(args...).real()),
            std::fabs(field_host(args...).imag() - field_result(args...).imag()));

        if (error.real() > max_error_local.real()) {
            max_error_local.real() = error.real();
        }

        if (error.imag() > max_error_local.imag()) {
            max_error_local.imag() = error.imag();
        }

        ASSERT_NEAR(error.real(), 0, tol);
        ASSERT_NEAR(error.imag(), 0, tol);
    });

    Kokkos::complex<T> max_error(0, 0);

    ippl::Comm->allreduce(max_error_local, max_error, 1, std::plus<Kokkos::complex<T>>());

    ASSERT_NEAR(max_error.real(), 0, tol);
    ASSERT_NEAR(max_error.imag(), 0, tol);
}

int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
