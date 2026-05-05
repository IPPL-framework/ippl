//
// Unit test NUFFT
//   Type-1 (particles -> grid) and Type-2 (grid -> particles).
//   Parameterised over precision (double, float) and dimension (2, 3).
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_Random.hpp>
#include <random>

#include "NUFFTTestUtils.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

namespace {

    template <unsigned Dim>
    ippl::Vector<int, Dim> makeStandardTestMode(int N) {
        // 0.18*N, 0.13*N, 0.10*N — well inside the centered band for any N >= 16.
        ippl::Vector<int, Dim> k;
        if constexpr (Dim >= 1)
            k[0] = std::max(1, int(0.18 * N));
        if constexpr (Dim >= 2)
            k[1] = std::max(1, int(0.13 * N));
        if constexpr (Dim >= 3)
            k[2] = std::max(1, int(0.10 * N));
        return k;
    }

}  // namespace

//=============================================================================
// NUFFT Type-1 (Particles -> Grid)
//=============================================================================

template <typename>
class NUFFT1Test;

template <typename T, typename ExecSpace, unsigned Dim>
class NUFFT1Test<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    using exec_space              = ExecSpace;
    constexpr static unsigned dim = Dim;

    using mesh_type       = ippl::UniformCartesian<T, Dim>;
    using centering_type  = typename mesh_type::DefaultCentering;
    using field_type      = typename ippl::Field<Kokkos::complex<T>, Dim, mesh_type, centering_type,
                                                 ExecSpace>::uniform_type;
    using real_field_type = ippl::Field<T, Dim, mesh_type, centering_type, ExecSpace>;
    using layout_type     = ippl::FieldLayout<Dim>;
    using playout_type    = ippl::ParticleSpatialLayout<T, Dim>;
    using bunch_type      = ippl::test::Bunch<T, playout_type>;
    using FFT_type        = ippl::FFT<ippl::NUFFTransform, real_field_type>;

    NUFFT1Test() {
        const T pi = Kokkos::numbers::pi_v<T>;
        for (unsigned d = 0; d < Dim; d++) {
            minU[d] = 0;
            maxU[d] = 2 * pi;
        }
    }

    void setupGrid(size_t gridSize) {
        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<T, Dim> hx, origin;
        for (unsigned d = 0; d < Dim; d++) {
            domains[d] = ippl::Index(gridSize);
            hx[d]      = (maxU[d] - minU[d]) / gridSize;
            origin[d]  = 0;
            nModes[d]  = gridSize;
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout     = std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel);
        mesh       = std::make_shared<mesh_type>(owned, hx, origin);
    }

    void setupParticles(size_t numParticles) {
        playout = std::make_shared<playout_type>(*layout, *mesh);
        bunch   = std::make_shared<bunch_type>(*playout);
        bunch->setParticleBC(ippl::BC::PERIODIC);
        bunch->create(numParticles / ippl::Comm->size());
    }

    void generateRandomParticles(unsigned seed = 42) {
        std::mt19937_64 eng(seed + ippl::Comm->rank());
        std::uniform_real_distribution<T> unifPos(0, 1);
        std::uniform_real_distribution<T> unifCharge(0, 1);

        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();

        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            for (unsigned d = 0; d < Dim; ++d) {
                R_host(i)[d] = minU[d] + unifPos(eng) * (maxU[d] - minU[d]);
            }
            Q_host(i) = unifCharge(eng);
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
        Kokkos::fence();
        bunch->update();
    }

    void runStandardType1(const ippl::ParameterList& params, const ippl::Vector<int, Dim>& testMode,
                          double tolerance) {
        const bool useUpsampling = params.get<bool>("use_upsampled_inputs");
        const int nghost         = 1;

        auto fft = std::make_unique<FFT_type>(*layout, bunch->getLocalNum(), 1, params);

        field_type field;
        std::shared_ptr<layout_type> layoutUp;
        std::shared_ptr<mesh_type> meshUp;

        if (useUpsampling) {
            constexpr T sigma = T(2);
            std::array<ippl::Index, Dim> domains;
            std::array<bool, Dim> isParallel;
            isParallel.fill(true);
            ippl::Vector<T, Dim> hxUp, originUp;
            for (unsigned d = 0; d < Dim; ++d) {
                domains[d]  = ippl::Index(static_cast<int>(sigma * nModes[d]));
                hxUp[d]     = (maxU[d] - minU[d]) / static_cast<int>(sigma * nModes[d]);
                originUp[d] = 0;
            }
            auto ownedUp = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
            layoutUp     = std::make_shared<layout_type>(MPI_COMM_WORLD, ownedUp, isParallel);
            meshUp       = std::make_shared<mesh_type>(ownedUp, hxUp, originUp);
            field.initialize(*meshUp, *layoutUp, nghost);
        } else {
            field.initialize(*mesh, *layout, nghost);
        }

        field = Kokkos::complex<T>(0);
        fft->transform(bunch->R, bunch->Q, field);

        auto globalIdx =
            ippl::test::IndexUtils<Dim>::centeredToCornerDC(testMode, nModes, useUpsampling);
        const auto& lDom = field.getLayout().getLocalNDIndex();

        Kokkos::complex<T> nufftResult(0, 0);
        if (ippl::test::IndexUtils<Dim>::isOwnedLocally(lDom, globalIdx)) {
            auto fieldHost = field.getHostMirror();
            Kokkos::deep_copy(fieldHost, field.getView());
            auto localIdx = ippl::test::IndexUtils<Dim>::globalToLocal(lDom, globalIdx, nghost);
            nufftResult   = ippl::test::readFieldAt<decltype(fieldHost), Dim>(fieldHost, localIdx);
        }

        T sendBuf[2] = {nufftResult.real(), nufftResult.imag()};
        T recvBuf[2] = {0, 0};
        MPI_Allreduce(sendBuf, recvBuf, 2, ippl::test::mpiDatatypeFor<T>(), MPI_SUM,
                      ippl::Comm->getCommunicator());
        nufftResult = Kokkos::complex<T>(recvBuf[0], recvBuf[1]);

        auto dftResult = ippl::test::DFTReference<T, Dim>::computeType1Mode(
            bunch->R.getView(), bunch->Q.getView(), testMode, mesh->getMeshSpacing(), nModes,
            bunch->getLocalNum());

        auto err = ippl::test::ErrorMetrics<T>::compute(dftResult, nufftResult);
        if (ippl::Comm->rank() == 0) {
            EXPECT_NEAR(err.relError, 0.0, tolerance * 100);
        }
    }

    std::shared_ptr<layout_type> layout;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<playout_type> playout;
    std::shared_ptr<bunch_type> bunch;

    ippl::Vector<T, Dim> minU, maxU;
    ippl::Vector<int, Dim> nModes;
};

template <typename T, unsigned Dim>
using SpaceParam = Parameters<T, Kokkos::DefaultExecutionSpace, Rank<Dim>>;
using NUFFTTypes = ::testing::Types<SpaceParam<double, 3>, SpaceParam<float, 3>,
                                    SpaceParam<double, 2>, SpaceParam<float, 2>>;

TYPED_TEST_SUITE(NUFFT1Test, NUFFTTypes);

namespace {
    // Tolerance is precision-dependent: float can't reliably hit 1e-7, so we
    // pick a per-precision floor.
    template <typename T>
    constexpr double smallTol() {
        return std::is_same_v<T, float> ? 1e-4 : 1e-7;
    }
    template <typename T>
    constexpr double mediumTol() {
        return std::is_same_v<T, float> ? 1e-3 : 1e-4;
    }
}  // namespace

TYPED_TEST(NUFFT1Test, BasicCorrectness_NoUpsampling) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(512);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createNativeParams<T>(smallTol<T>(), false);
    this->runStandardType1(params, makeStandardTestMode<Dim>(16), smallTol<T>());
}

TYPED_TEST(NUFFT1Test, BasicCorrectness_WithUpsampling) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(512);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createNativeParams<T>(smallTol<T>(), true);
    this->runStandardType1(params, makeStandardTestMode<Dim>(16), smallTol<T>());
}

TYPED_TEST(NUFFT1Test, MediumGrid_NoUpsampling) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createNativeParams<T>(mediumTol<T>(), false);
    this->runStandardType1(params, makeStandardTestMode<Dim>(16), mediumTol<T>());
}

TYPED_TEST(NUFFT1Test, SpreadMethod_Atomic) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createNativeParams<T>(mediumTol<T>(), false, "atomic");
    params.update("sort", false);
    this->runStandardType1(params, makeStandardTestMode<Dim>(16), mediumTol<T>());
}

TYPED_TEST(NUFFT1Test, SpreadMethod_Tiled) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createNativeParams<T>(mediumTol<T>(), false, "tiled");
    this->runStandardType1(params, makeStandardTestMode<Dim>(16), mediumTol<T>());
}

TYPED_TEST(NUFFT1Test, SpreadMethod_OutputFocused) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params =
        ippl::test::NUFFTParams::createNativeParams<T>(mediumTol<T>(), false, "output_focused");
    this->runStandardType1(params, makeStandardTestMode<Dim>(16), mediumTol<T>());
}

TYPED_TEST(NUFFT1Test, ToleranceSweep) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();

    const std::vector<double> tols = std::is_same_v<T, float>
                                         ? std::vector<double>{1e-2, 1e-3, 1e-4}
                                         : std::vector<double>{1e-4, 1e-7, 1e-10};

    for (double tol : tols) {
        auto params = ippl::test::NUFFTParams::createNativeParams<T>(static_cast<T>(tol), false);
        this->runStandardType1(params, makeStandardTestMode<Dim>(16), tol);
    }
}

#ifdef ENABLE_FINUFFT
TYPED_TEST(NUFFT1Test, FINUFFT_NoUpsampling) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createFinufftParams<T>(smallTol<T>(), false);
    this->runStandardType1(params, makeStandardTestMode<Dim>(16), smallTol<T>());
}
#endif

//=============================================================================
// NUFFT Type-2 (Grid -> Particles)
//=============================================================================

template <typename>
class NUFFT2Test;

template <typename T, typename ExecSpace, unsigned Dim>
class NUFFT2Test<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    using exec_space              = ExecSpace;
    constexpr static unsigned dim = Dim;

    using mesh_type       = ippl::UniformCartesian<T, Dim>;
    using centering_type  = typename mesh_type::DefaultCentering;
    using field_type      = typename ippl::Field<Kokkos::complex<T>, Dim, mesh_type, centering_type,
                                                 ExecSpace>::uniform_type;
    using real_field_type = ippl::Field<T, Dim, mesh_type, centering_type, ExecSpace>;
    using layout_type     = ippl::FieldLayout<Dim>;
    using playout_type    = ippl::ParticleSpatialLayout<T, Dim>;
    using bunch_type      = ippl::test::Bunch<T, playout_type>;
    using FFT_type        = ippl::FFT<ippl::NUFFTransform, real_field_type>;

    NUFFT2Test() {
        const T pi = Kokkos::numbers::pi_v<T>;
        for (unsigned d = 0; d < Dim; d++) {
            minU[d] = 0;
            maxU[d] = 2 * pi;
        }
    }

    void setupGrid(size_t gridSize) {
        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        ippl::Vector<T, Dim> hx, origin;
        for (unsigned d = 0; d < Dim; d++) {
            domains[d] = ippl::Index(gridSize);
            hx[d]      = (maxU[d] - minU[d]) / gridSize;
            origin[d]  = 0;
            nModes[d]  = gridSize;
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout     = std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel);
        mesh       = std::make_shared<mesh_type>(owned, hx, origin);
    }

    void setupParticles(size_t numParticles) {
        playout = std::make_shared<playout_type>(*layout, *mesh);
        bunch   = std::make_shared<bunch_type>(*playout);
        bunch->setParticleBC(ippl::BC::PERIODIC);
        bunch->create(numParticles / ippl::Comm->size());
    }

    void generateRandomParticles(unsigned seed = 42) {
        std::mt19937_64 eng(seed + ippl::Comm->rank());
        std::uniform_real_distribution<T> unifPos(0, 1);
        std::uniform_real_distribution<T> unifCharge(0, 1);

        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            for (unsigned d = 0; d < Dim; ++d)
                R_host(i)[d] = minU[d] + unifPos(eng) * (maxU[d] - minU[d]);
            Q_host(i) = unifCharge(eng);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
        Kokkos::fence();
        bunch->update();
    }

    // Random complex field
    void generateRandomField(field_type& field, unsigned seed = 123) {
        using gen_pool = Kokkos::Random_XorShift64_Pool<exec_space>;
        gen_pool randPool(seed);

        auto fieldView = field.getView();
        int nghost     = field.getNghost();

        const auto& lDom = field.getLayout().getLocalNDIndex();
        ippl::Vector<int, Dim> ext;
        for (unsigned d = 0; d < Dim; ++d)
            ext[d] = lDom[d].length();

        if constexpr (Dim == 3) {
            using mdr = Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec_space>;
            Kokkos::parallel_for(
                "fill_random_field", mdr({0, 0, 0}, {ext[0], ext[1], ext[2]}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    typename gen_pool::generator_type g = randPool.get_state();
                    fieldView(i + nghost, j + nghost, k + nghost) =
                        Kokkos::complex<T>(g.drand(0.0, 1.0), g.drand(0.0, 1.0));
                    randPool.free_state(g);
                });
        } else if constexpr (Dim == 2) {
            using mdr = Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>;
            Kokkos::parallel_for(
                "fill_random_field", mdr({0, 0}, {ext[0], ext[1]}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    typename gen_pool::generator_type g = randPool.get_state();
                    fieldView(i + nghost, j + nghost) =
                        Kokkos::complex<T>(g.drand(0.0, 1.0), g.drand(0.0, 1.0));
                    randPool.free_state(g);
                });
        }
        Kokkos::fence();
        field.fillHalo();
    }

    // For upsampling, NUFFT type-2 only treats the corner-DC band as actual modes
    // and zeros the rest in pre-correction; mirror that on the test input so the
    // reference DFT (which iterates the full grid) sees the same data.
    void zeroNonCornerDCBand(field_type& field) {
        auto fieldView   = field.getView();
        const int nghost = field.getNghost();
        const auto& lDom = field.getLayout().getLocalNDIndex();

        ippl::Vector<int, Dim> localFirst, localLength;
        for (unsigned d = 0; d < Dim; ++d) {
            localFirst[d]  = lDom[d].first();
            localLength[d] = lDom[d].length();
        }
        ippl::Vector<int, Dim> nModesD = nModes;

        auto in_band = KOKKOS_LAMBDA(int g, int n) {
            return (g >= 0 && g < n / 2) || (g >= n + n / 2 && g < 2 * n);
        };

        if constexpr (Dim == 3) {
            using mdr = Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec_space>;
            Kokkos::parallel_for(
                "zero_outside_corner_dc",
                mdr({0, 0, 0}, {localLength[0], localLength[1], localLength[2]}),
                KOKKOS_LAMBDA(const int li, const int lj, const int lk) {
                    int gi = li + localFirst[0];
                    int gj = lj + localFirst[1];
                    int gk = lk + localFirst[2];
                    if (!(in_band(gi, nModesD[0]) && in_band(gj, nModesD[1])
                          && in_band(gk, nModesD[2]))) {
                        fieldView(li + nghost, lj + nghost, lk + nghost) = Kokkos::complex<T>(0, 0);
                    }
                });
        } else if constexpr (Dim == 2) {
            using mdr = Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>;
            Kokkos::parallel_for(
                "zero_outside_corner_dc", mdr({0, 0}, {localLength[0], localLength[1]}),
                KOKKOS_LAMBDA(const int li, const int lj) {
                    int gi = li + localFirst[0];
                    int gj = lj + localFirst[1];
                    if (!(in_band(gi, nModesD[0]) && in_band(gj, nModesD[1]))) {
                        fieldView(li + nghost, lj + nghost) = Kokkos::complex<T>(0, 0);
                    }
                });
        }
        Kokkos::fence();
        field.fillHalo();
    }

    ippl::Vector<T, Dim> getTestParticlePosition() {
        ippl::Vector<T, Dim> testPos;
        size_t nloc = bunch->getLocalNum();

        if (nloc > 0) {
            auto RHost =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bunch->R.getView());
            for (unsigned d = 0; d < Dim; ++d)
                testPos[d] = RHost(0)[d];
        }

        int rankWithParticles = -1;
        int hasParticles      = (nloc > 0) ? 1 : 0;
        std::vector<int> all(ippl::Comm->size());
        MPI_Allgather(&hasParticles, 1, MPI_INT, all.data(), 1, MPI_INT,
                      ippl::Comm->getCommunicator());
        for (int r = 0; r < ippl::Comm->size(); ++r) {
            if (all[r]) {
                rankWithParticles = r;
                break;
            }
        }

        T posBuf[Dim];
        for (unsigned d = 0; d < Dim; ++d)
            posBuf[d] = testPos[d];
        MPI_Bcast(posBuf, static_cast<int>(Dim), ippl::test::mpiDatatypeFor<T>(), rankWithParticles,
                  ippl::Comm->getCommunicator());
        for (unsigned d = 0; d < Dim; ++d)
            testPos[d] = posBuf[d];

        return testPos;
    }

    T extractNUFFTResultAtTestParticle() {
        size_t nloc           = bunch->getLocalNum();
        T nufftVal            = 0;
        int rankWithParticles = -1;
        int hasParticles      = (nloc > 0) ? 1 : 0;
        std::vector<int> all(ippl::Comm->size());
        MPI_Allgather(&hasParticles, 1, MPI_INT, all.data(), 1, MPI_INT,
                      ippl::Comm->getCommunicator());
        for (int r = 0; r < ippl::Comm->size(); ++r) {
            if (all[r]) {
                rankWithParticles = r;
                break;
            }
        }
        if (ippl::Comm->rank() == rankWithParticles && nloc > 0) {
            auto QResult =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bunch->Q.getView());
            nufftVal = QResult(0);
        }
        MPI_Bcast(&nufftVal, 1, ippl::test::mpiDatatypeFor<T>(), rankWithParticles,
                  ippl::Comm->getCommunicator());
        return nufftVal;
    }

    void runStandardType2(const ippl::ParameterList& params, double tolerance) {
        const bool useUpsampling = params.get<bool>("use_upsampled_inputs");
        const int nghost         = 1;

        auto fft = std::make_unique<FFT_type>(*layout, bunch->getLocalNum(), 2, params);

        field_type field;
        std::shared_ptr<layout_type> layoutUp;
        std::shared_ptr<mesh_type> meshUp;

        if (useUpsampling) {
            constexpr T sigma = T(2);
            std::array<ippl::Index, Dim> domains;
            std::array<bool, Dim> isParallel;
            isParallel.fill(true);
            ippl::Vector<T, Dim> hxUp, originUp;
            for (unsigned d = 0; d < Dim; ++d) {
                domains[d]  = ippl::Index(static_cast<int>(sigma * nModes[d]));
                hxUp[d]     = (maxU[d] - minU[d]) / static_cast<int>(sigma * nModes[d]);
                originUp[d] = 0;
            }
            auto ownedUp = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
            layoutUp     = std::make_shared<layout_type>(MPI_COMM_WORLD, ownedUp, isParallel);
            meshUp       = std::make_shared<mesh_type>(ownedUp, hxUp, originUp);
            field.initialize(*meshUp, *layoutUp, nghost);
        } else {
            field.initialize(*mesh, *layout, nghost);
        }

        generateRandomField(field);
        if (useUpsampling) {
            zeroNonCornerDCBand(field);
        }

        auto testPos            = getTestParticlePosition();
        const auto& fieldLayout = field.getLayout();

        ippl::Vector<T, Dim> hxField;
        ippl::Vector<int, Dim> nModesField;
        if (useUpsampling) {
            for (unsigned d = 0; d < Dim; ++d) {
                nModesField[d] = 2 * nModes[d];
                hxField[d]     = (maxU[d] - minU[d]) / nModesField[d];
            }
        } else {
            nModesField = nModes;
            hxField     = mesh->getMeshSpacing();
        }

        auto dftResult = ippl::test::DFTReference<T, Dim>::computeType2Value(
            field.getView(), testPos, fieldLayout.getLocalNDIndex(), hxField, nModesField, nghost);
        T refReal = dftResult.real();

        // L-infinity scale of input field for magnitude-aware tolerance
        T localFieldMax = 0;
        {
            const auto& lDom = field.getLayout().getLocalNDIndex();
            ippl::Vector<int, Dim> ext;
            for (unsigned d = 0; d < Dim; ++d)
                ext[d] = lDom[d].length();
            auto fview = field.getView();
            T tmp      = 0;
            if constexpr (Dim == 3) {
                using mdr = Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec_space>;
                Kokkos::parallel_reduce(
                    "field_max_abs", mdr({0, 0, 0}, {ext[0], ext[1], ext[2]}),
                    KOKKOS_LAMBDA(const int i, const int j, const int k, T& m) {
                        T v = Kokkos::abs(fview(i + nghost, j + nghost, k + nghost));
                        if (v > m)
                            m = v;
                    },
                    Kokkos::Max<T>(tmp));
            } else if constexpr (Dim == 2) {
                using mdr = Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>;
                Kokkos::parallel_reduce(
                    "field_max_abs", mdr({0, 0}, {ext[0], ext[1]}),
                    KOKKOS_LAMBDA(const int i, const int j, T& m) {
                        T v = Kokkos::abs(fview(i + nghost, j + nghost));
                        if (v > m)
                            m = v;
                    },
                    Kokkos::Max<T>(tmp));
            }
            localFieldMax = tmp;
        }
        T globalFieldMax = 0;
        MPI_Allreduce(&localFieldMax, &globalFieldMax, 1, ippl::test::mpiDatatypeFor<T>(), MPI_MAX,
                      ippl::Comm->getCommunicator());

        auto QView = bunch->Q.getView();
        Kokkos::parallel_for(
            "zero_Q", bunch->getLocalNum(), KOKKOS_LAMBDA(const size_t i) { QView(i) = 0.0; });
        Kokkos::fence();

        fft->transform(bunch->R, bunch->Q, field);

        T nufftVal = extractNUFFTResultAtTestParticle();

        size_t Ntot = 1;
        for (unsigned d = 0; d < Dim; ++d)
            Ntot *= static_cast<size_t>(nModesField[d]);

        T absError = std::fabs(refReal - nufftVal);
        T scale    = globalFieldMax * static_cast<T>(Ntot);

        if (ippl::Comm->rank() == 0) {
            EXPECT_LT(absError, tolerance * 100 * scale)
                << "absError=" << absError << " scale=" << scale << " refReal=" << refReal
                << " nufftVal=" << nufftVal;
        }
    }

    std::shared_ptr<layout_type> layout;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<playout_type> playout;
    std::shared_ptr<bunch_type> bunch;

    ippl::Vector<T, Dim> minU, maxU;
    ippl::Vector<int, Dim> nModes;
};

TYPED_TEST_SUITE(NUFFT2Test, NUFFTTypes);

TYPED_TEST(NUFFT2Test, BasicCorrectness_NoUpsampling) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(512);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createNativeParams<T>(smallTol<T>(), false, "tiled",
                                                                 "atomic_sort");
    this->runStandardType2(params, smallTol<T>());
}

TYPED_TEST(NUFFT2Test, BasicCorrectness_WithUpsampling) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(512);
    this->generateRandomParticles();
    auto params =
        ippl::test::NUFFTParams::createNativeParams<T>(smallTol<T>(), true, "tiled", "atomic_sort");
    this->runStandardType2(params, smallTol<T>());
}

TYPED_TEST(NUFFT2Test, MediumGrid_NoUpsampling) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createNativeParams<T>(smallTol<T>(), false, "tiled",
                                                                 "atomic_sort");
    this->runStandardType2(params, smallTol<T>());
}

TYPED_TEST(NUFFT2Test, GatherMethod_Atomic) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params =
        ippl::test::NUFFTParams::createNativeParams<T>(smallTol<T>(), false, "tiled", "atomic");
    this->runStandardType2(params, smallTol<T>());
}

TYPED_TEST(NUFFT2Test, GatherMethod_AtomicSort) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createNativeParams<T>(smallTol<T>(), false, "tiled",
                                                                 "atomic_sort");
    this->runStandardType2(params, smallTol<T>());
}

TYPED_TEST(NUFFT2Test, ToleranceSweep) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();

    const std::vector<double> tols = std::is_same_v<T, float>
                                         ? std::vector<double>{1e-2, 1e-3, 1e-4}
                                         : std::vector<double>{1e-4, 1e-7, 1e-10};

    for (double tol : tols) {
        auto params = ippl::test::NUFFTParams::createNativeParams<T>(static_cast<T>(tol), false,
                                                                     "tiled", "atomic_sort");
        this->runStandardType2(params, tol);
    }
}

#ifdef ENABLE_FINUFFT
TYPED_TEST(NUFFT2Test, FINUFFT_NoUpsampling) {
    using T                = typename TestFixture::value_type;
    [[maybe_unused]] constexpr unsigned Dim = TestFixture::dim;
    this->setupGrid(16);
    this->setupParticles(4096);
    this->generateRandomParticles();
    auto params = ippl::test::NUFFTParams::createFinufftParams<T>(smallTol<T>(), false);
    this->runStandardType2(params, smallTol<T>());
}
#endif

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
