//
// Unit test NUFFT
//   Test NUFFT features (Type-1 and Type-2)
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_Random.hpp>
#include <iomanip>
#include <random>

#include "NUFFTTestUtils.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

//=============================================================================
// NUFFT Type-1 Test Fixture (Particles -> Grid)
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

    using playout_type = ippl::ParticleSpatialLayout<T, Dim>;
    using bunch_type   = ippl::test::Bunch<T, playout_type>;
    using FFT_type     = ippl::FFT<ippl::NUFFTransform, real_field_type>;

    NUFFT1Test() {
        const T pi = Kokkos::numbers::pi_v<T>;
        for (unsigned d = 0; d < Dim; d++) {
            minU[d] = 0;
            maxU[d] = 2 * pi;
        }
    }

    /**
     * @brief Setup grid and mesh for given size
     */
    void setupGrid(const std::array<size_t, Dim>& gridSize) {
        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<T, Dim> hx, origin;
        for (unsigned d = 0; d < Dim; d++) {
            domains[d] = ippl::Index(gridSize[d]);
            hx[d]      = (maxU[d] - minU[d]) / gridSize[d];
            origin[d]  = 0;
            nModes[d]  = gridSize[d];
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout     = std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel);
        mesh       = std::make_shared<mesh_type>(owned, hx, origin);
    }

    /**
     * @brief Setup particles
     */
    void setupParticles(size_t numParticles) {
        playout = std::make_shared<playout_type>(*layout, *mesh);
        bunch   = std::make_shared<bunch_type>(*playout);
        bunch->setParticleBC(ippl::BC::PERIODIC);

        size_t nloc = numParticles / ippl::Comm->size();
        bunch->create(nloc);
    }

    /**
     * @brief Generate random particles using host mirror pattern
     */
    void generateRandomParticles(unsigned seed = 42) {
        std::mt19937_64 eng(seed + ippl::Comm->rank());
        std::uniform_real_distribution<T> unifPos(0, 1);
        std::uniform_real_distribution<T> unifCharge(0, 1);

        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();

        size_t nloc = bunch->getLocalNum();
        for (size_t i = 0; i < nloc; ++i) {
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

    /**
     * @brief Generate constant particles using host mirror pattern (for debugging)
     */
    void generateConstantParticles(T posValue = 0.5, T chargeValue = 0.5) {
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();

        size_t nloc = bunch->getLocalNum();
        for (size_t i = 0; i < nloc; ++i) {
            for (unsigned d = 0; d < Dim; ++d) {
                R_host(i)[d] = minU[d] + posValue * (maxU[d] - minU[d]);
            }
            Q_host(i) = chargeValue;
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
        Kokkos::fence();
        bunch->update();
    }

    /**
     * @brief Compute DFT reference for single mode
     */
    Kokkos::complex<T> computeDFTReference(const ippl::Vector<int, Dim>& kVec) {
        return ippl::test::DFTReference<T, Dim>::computeType1Mode(
            bunch->R.getView(), bunch->Q.getView(), kVec, mesh->getMeshSpacing(), nModes,
            bunch->getLocalNum());
    }

    /**
     * @brief Extract NUFFT result at mode
     */
    Kokkos::complex<T> extractNUFFTResult(const field_type& field,
                                          const ippl::Vector<int, Dim>& kVec) {
        using namespace ippl::test;

        auto globalIdx   = IndexUtils<Dim>::centeredToCornerDC(kVec, nModes);
        const auto& lDom = layout->getLocalNDIndex();

        Kokkos::complex<T> result(0, 0);
        if (IndexUtils<Dim>::isOwnedLocally(lDom, globalIdx)) {
            auto fieldHost = field.getHostMirror();
            Kokkos::deep_copy(fieldHost, field.getView());

            auto localIdx = IndexUtils<Dim>::globalToLocal(lDom, globalIdx, field.getNghost());
            if constexpr (Dim == 3) {
                result = fieldHost(localIdx[0], localIdx[1], localIdx[2]);
            } else if constexpr (Dim == 2) {
                result = fieldHost(localIdx[0], localIdx[1]);
            } else if constexpr (Dim == 1) {
                result = fieldHost(localIdx[0]);
            }
        }

        // MPI Allreduce to broadcast result
        T sendBuf[2] = {result.real(), result.imag()};
        T recvBuf[2] = {0, 0};
        MPI_Allreduce(sendBuf, recvBuf, 2, std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE,
                      MPI_SUM, ippl::Comm->getCommunicator());
        return Kokkos::complex<T>(recvBuf[0], recvBuf[1]);
    }

    /**
     * @brief Run Type-1 transform and validate
     */
    void runType1Test(const ippl::ParameterList& params, const ippl::Vector<int, Dim>& testMode,
                      double tolerance) {
        bool useUpsampling = params.get<bool>("use_upsampled_inputs");

        // Create FFT
        auto fft = std::make_unique<FFT_type>(*layout, bunch->getLocalNum(), 1, params);

        // Create output field - upsampled if needed
        const int nghost = 1;
        field_type field;
        std::shared_ptr<layout_type> layoutUp;
        std::shared_ptr<mesh_type> meshUp;

        if (useUpsampling) {
            // Create upsampled grid (2x in each dimension)
            T sigma = 2.0;
            ippl::Vector<int, Dim> nGrid;
            for (unsigned d = 0; d < Dim; ++d) {
                nGrid[d] = static_cast<int>(sigma * nModes[d]);
            }

            // Create upsampled layout and mesh with proper lifetime
            std::array<ippl::Index, Dim> domains;
            std::array<bool, Dim> isParallel;
            isParallel.fill(true);

            ippl::Vector<T, Dim> hxUp, originUp;
            for (unsigned d = 0; d < Dim; ++d) {
                domains[d]  = ippl::Index(nGrid[d]);
                hxUp[d]     = (maxU[d] - minU[d]) / nGrid[d];
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

        // Execute transform
        fft->transform(bunch->R, bunch->Q, field);

        // Extract result at test mode
        auto globalIdx =
            useUpsampling ? ippl::test::IndexUtils<Dim>::centeredToCornerDC(testMode, nModes, true)
                          : ippl::test::IndexUtils<Dim>::centeredToCornerDC(testMode, nModes);

        const auto& lDom = field.getLayout().getLocalNDIndex();
        Kokkos::complex<T> nufftResult(0, 0);

        if (ippl::test::IndexUtils<Dim>::isOwnedLocally(lDom, globalIdx)) {
            auto fieldHost = field.getHostMirror();
            Kokkos::deep_copy(fieldHost, field.getView());
            auto localIdx = ippl::test::IndexUtils<Dim>::globalToLocal(lDom, globalIdx, nghost);

            if constexpr (Dim == 3) {
                nufftResult = fieldHost(localIdx[0], localIdx[1], localIdx[2]);
            } else if constexpr (Dim == 2) {
                nufftResult = fieldHost(localIdx[0], localIdx[1]);
            } else if constexpr (Dim == 1) {
                nufftResult = fieldHost(localIdx[0]);
            }
        }

        // MPI Allreduce to broadcast result
        T sendBuf[2] = {nufftResult.real(), nufftResult.imag()};
        T recvBuf[2] = {0, 0};
        MPI_Allreduce(sendBuf, recvBuf, 2, std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE,
                      MPI_SUM, ippl::Comm->getCommunicator());
        nufftResult = Kokkos::complex<T>(recvBuf[0], recvBuf[1]);

        // Compute DFT reference
        auto dftResult = computeDFTReference(testMode);

        // Validate
        auto error = ippl::test::ErrorMetrics<T>::compute(dftResult, nufftResult);

        if (ippl::Comm->rank() == 0) {
            std::cout << "DFT result: " << dftResult << std::endl;
            std::cout << "NUFFT result: " << nufftResult << std::endl;
            std::cout << "Absolute error: " << error.absError << std::endl;
            std::cout << "Relative error: " << error.relError << std::endl;
            EXPECT_NEAR(error.relError, 0.0, tolerance * 100);
        }
    }

    std::shared_ptr<layout_type> layout;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<playout_type> playout;
    std::shared_ptr<bunch_type> bunch;

    ippl::Vector<T, Dim> minU, maxU;
    ippl::Vector<int, Dim> nModes;
};

using PrecisionTypes = TestParams::Precisions;
template <typename T>
using DefaultSpaceParam = Parameters<T, Kokkos::DefaultExecutionSpace, Rank<3>>;
using Tests = ::testing::Types<DefaultSpaceParam<double> /*, DefaultSpaceParam<float>*/>;
TYPED_TEST_SUITE(NUFFT1Test, Tests);

//=============================================================================
// NUFFT Type-1 Tests (Particles → Grid)
//=============================================================================

TYPED_TEST(NUFFT1Test, BasicCorrectness_SmallGrid_NoUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(512);
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(1e-7, false);

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-7);
}

TYPED_TEST(NUFFT1Test, BasicCorrectness_SmallGrid_WithUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(512);
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(1e-7, true);

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-7);
}

TYPED_TEST(NUFFT1Test, BasicCorrectness_MediumGrid_NoUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(1e-4, false);

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = (int)(0.37 * gridSize[0]);
    testMode[1] = (int)(0.16 * gridSize[1]);
    testMode[2] = (int)(0.23 * gridSize[2]);

    this->runType1Test(params, testMode, 1e-4);
}

TYPED_TEST(NUFFT1Test, BasicCorrectness_MediumGrid_WithUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(64);  // Smaller for upsampled test
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(1e-7, true);

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = (int)(0.37 * gridSize[0]);
    testMode[1] = (int)(0.16 * gridSize[1]);
    testMode[2] = (int)(0.23 * gridSize[2]);

    this->runType1Test(params, testMode, 1e-7);
}

TYPED_TEST(NUFFT1Test, SpreadMethod_Atomictile_x) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-4, false, "atomic");
    params.update("sort", false);
    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-4);
}

TYPED_TEST(NUFFT1Test, SpreadMethod_Tiled) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-4, false, "tiled");

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-4);
}

TYPED_TEST(NUFFT1Test, SpreadMethod_OutputFocused) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-4, false, "output_focused");

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-4);
}

TYPED_TEST(NUFFT1Test, Tolerance_1e4) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(1e-4, false);

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-4);
}

TYPED_TEST(NUFFT1Test, Tolerance_1e7) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(1e-7, false);

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-7);
}

TYPED_TEST(NUFFT1Test, Tolerance_1e10) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(1e-10, false);

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-10);
}

#ifdef ENABLE_FINUFFT
TYPED_TEST(NUFFT1Test, FINUFFT_NoUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createFinufftParams<typename TestFixture::value_type>(1e-7, false);

    ippl::Vector<int, TestFixture::dim> testMode;
    testMode[0] = 3;
    testMode[1] = 2;
    testMode[2] = 1;

    this->runType1Test(params, testMode, 1e-7);
}
#endif

//=============================================================================
// NUFFT Type-2 Test Fixture (Grid → Particles)
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

    using playout_type = ippl::ParticleSpatialLayout<T, Dim>;
    using bunch_type   = ippl::test::Bunch<T, playout_type>;
    using FFT_type     = ippl::FFT<ippl::NUFFTransform, real_field_type>;

    NUFFT2Test() {
        const T pi = Kokkos::numbers::pi_v<T>;
        for (unsigned d = 0; d < Dim; d++) {
            minU[d] = 0;
            maxU[d] = 2 * pi;
        }
    }

    void setupGrid(const std::array<size_t, Dim>& gridSize) {
        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<T, Dim> hx, origin;
        for (unsigned d = 0; d < Dim; d++) {
            domains[d] = ippl::Index(gridSize[d]);
            hx[d]      = (maxU[d] - minU[d]) / gridSize[d];
            origin[d]  = 0;
            nModes[d]  = gridSize[d];
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout     = std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel);
        mesh       = std::make_shared<mesh_type>(owned, hx, origin);
    }

    void setupParticles(size_t numParticles) {
        playout = std::make_shared<playout_type>(*layout, *mesh);
        bunch   = std::make_shared<bunch_type>(*playout);
        bunch->setParticleBC(ippl::BC::PERIODIC);

        size_t nloc = numParticles / ippl::Comm->size();
        bunch->create(nloc);
    }

    void createSingleParticleAtOrigin() {
        auto RView = bunch->R.getView();
        auto QView = bunch->Q.getView();

        if (ippl::Comm->rank() == 0 && bunch->getLocalNum() > 0) {
            auto RHost = Kokkos::create_mirror_view(RView);
            auto QHost = Kokkos::create_mirror_view(QView);

            // Place particle at origin
            RHost(0)[0] = 0.0;
            RHost(0)[1] = 0.0;
            RHost(0)[2] = 0.0;
            QHost(0)    = 0.0;

            Kokkos::deep_copy(RView, RHost);
            Kokkos::deep_copy(QView, QHost);
        }
        Kokkos::fence();
        bunch->update();
    }

    void generateRandomParticles(unsigned seed = 42) {
        std::mt19937_64 eng(seed + ippl::Comm->rank());
        std::uniform_real_distribution<T> unifPos(0, 1);
        std::uniform_real_distribution<T> unifCharge(0, 1);

        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();

        size_t nloc = bunch->getLocalNum();
        for (size_t i = 0; i < nloc; ++i) {
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

    void generateRandomField(field_type& field, unsigned seed = 123) {
        // Use Kokkos random pool like the original test
        using generator_pool = Kokkos::Random_XorShift64_Pool<exec_space>;
        generator_pool randPool(seed);

        auto fieldView = field.getView();
        int nghost     = field.getNghost();

        // Get local domain dimensions
        const auto& lDom = field.getLayout().getLocalNDIndex();
        int localNi      = lDom[0].length();
        int localNj      = lDom[1].length();
        int localNk      = lDom[2].length();

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec_space>;
        Kokkos::parallel_for(
            "fill_random_field", mdrange_type({0, 0, 0}, {localNi, localNj, localNk}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                typename generator_pool::generator_type randGen = randPool.get_state();
                auto& v  = fieldView(i + nghost, j + nghost, k + nghost);
                v.real() = randGen.drand(0.0, 1.0);
                v.imag() = randGen.drand(0.0, 1.0);
                randPool.free_state(randGen);
            });
        Kokkos::fence();
        field.fillHalo();
    }

    // Zero everything in `field` (sized to the upsampled grid) outside the
    // corner-DC band that NUFFT treats as actual modes.
    void zeroNonCornerDCBand(field_type& field, const ippl::Vector<int, Dim>& nModes) {
        auto fieldView = field.getView();
        const int nghost = field.getNghost();
        const auto& lDom = field.getLayout().getLocalNDIndex();

        ippl::Vector<int, Dim> localFirst, localLength;
        for (unsigned d = 0; d < Dim; ++d) {
            localFirst[d]  = lDom[d].first();
            localLength[d] = lDom[d].length();
        }

        // 2 * nModes is the upsampled grid size in each dim; the band is
        // [0, n/2) U [n + n/2, 2n).
        ippl::Vector<int, Dim> nModesD = nModes;

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec_space>;
        Kokkos::parallel_for(
            "zero_outside_corner_dc",
            mdrange_type({0, 0, 0}, {localLength[0], localLength[1], localLength[2]}),
            KOKKOS_LAMBDA(const int li, const int lj, const int lk) {
                auto in_band = [](int g, int n) {
                    return (g >= 0 && g < n / 2) || (g >= n + n / 2 && g < 2 * n);
                };
                const int gi = li + localFirst[0];
                const int gj = lj + localFirst[1];
                const int gk = lk + localFirst[2];
                if (!(in_band(gi, nModesD[0]) && in_band(gj, nModesD[1])
                      && in_band(gk, nModesD[2]))) {
                    fieldView(li + nghost, lj + nghost, lk + nghost) = Kokkos::complex<T>(0, 0);
                }
            });
        Kokkos::fence();
        field.fillHalo();
    }

    void generateConstantField(field_type& field, T realVal = 1.0, T imagVal = 0.0) {
        auto field_host = field.getHostMirror();
        int nghost      = field.getNghost();

        nestedViewLoop(field_host, nghost, [&]<typename... Idx>(const Idx... args) {
            field_host(args...) = Kokkos::complex<T>(realVal, imagVal);
        });

        Kokkos::deep_copy(field.getView(), field_host);
        field.fillHalo();
    }

    ippl::Vector<T, Dim> getTestParticlePosition() {
        ippl::Vector<T, Dim> testPos;
        size_t nloc = bunch->getLocalNum();

        if (nloc > 0) {
            auto RHost =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bunch->R.getView());
            testPos[0] = RHost(0)[0];
            testPos[1] = RHost(0)[1];
            testPos[2] = RHost(0)[2];
        }

        // Find which rank has particles and broadcast
        int rankWithParticles = -1;
        int hasParticles      = (nloc > 0) ? 1 : 0;

        std::vector<int> allHasParticles(ippl::Comm->size());
        MPI_Allgather(&hasParticles, 1, MPI_INT, allHasParticles.data(), 1, MPI_INT,
                      ippl::Comm->getCommunicator());

        for (int r = 0; r < ippl::Comm->size(); ++r) {
            if (allHasParticles[r]) {
                rankWithParticles = r;
                break;
            }
        }

        T posBuf[3] = {testPos[0], testPos[1], testPos[2]};
        MPI_Bcast(posBuf, 3, std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, rankWithParticles,
                  ippl::Comm->getCommunicator());
        testPos[0] = posBuf[0];
        testPos[1] = posBuf[1];
        testPos[2] = posBuf[2];

        return testPos;
    }

    T extractNUFFTResultAtTestParticle() {
        size_t nloc           = bunch->getLocalNum();
        T nufftVal            = 0.0;
        int rankWithParticles = -1;
        int hasParticles      = (nloc > 0) ? 1 : 0;

        std::vector<int> allHasParticles(ippl::Comm->size());
        MPI_Allgather(&hasParticles, 1, MPI_INT, allHasParticles.data(), 1, MPI_INT,
                      ippl::Comm->getCommunicator());

        for (int r = 0; r < ippl::Comm->size(); ++r) {
            if (allHasParticles[r]) {
                rankWithParticles = r;
                break;
            }
        }

        if (ippl::Comm->rank() == rankWithParticles && nloc > 0) {
            auto QResult =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bunch->Q.getView());
            nufftVal = QResult(0);
        }

        MPI_Bcast(&nufftVal, 1, std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE,
                  rankWithParticles, ippl::Comm->getCommunicator());

        return nufftVal;
    }

    void runType2Test(const ippl::ParameterList& params, double tolerance) {
        bool useUpsampling = params.get<bool>("use_upsampled_inputs");

        // Create FFT
        auto fft = std::make_unique<FFT_type>(*layout, bunch->getLocalNum(), 2, params);

        // Create input field - upsampled if needed
        const int nghost = 1;
        field_type field;
        std::shared_ptr<layout_type> layoutUp;
        std::shared_ptr<mesh_type> meshUp;

        if (useUpsampling) {
            // Create upsampled grid (2x in each dimension)
            T sigma = 2.0;
            ippl::Vector<int, Dim> nGrid;
            for (unsigned d = 0; d < Dim; ++d) {
                nGrid[d] = static_cast<int>(sigma * nModes[d]);
            }

            // Create upsampled layout and mesh with proper lifetime
            std::array<ippl::Index, Dim> domains;
            std::array<bool, Dim> isParallel;
            isParallel.fill(true);

            ippl::Vector<T, Dim> hxUp, originUp;
            for (unsigned d = 0; d < Dim; ++d) {
                domains[d]  = ippl::Index(nGrid[d]);
                hxUp[d]     = (maxU[d] - minU[d]) / nGrid[d];
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

        // When the user-supplied "modes" field lives on the upsampled grid,
        // NUFFT type-2 only treats the corner-DC band as real modes and zeros
        // everything else during the pre-correction step. Mirror that here so
        // the DFT reference (which iterates the full grid) and NUFFT see the
        // same input data.
        if (useUpsampling) {
            zeroNonCornerDCBand(field, nModes);
        }

        // Get test particle position before transform
        auto testPos = getTestParticlePosition();

        // IMPORTANT: Compute DFT reference BEFORE transform (transform modifies the field!)
        const auto& fieldLayout = field.getLayout();

        // Compute mesh spacing based on grid size
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

        // Now execute the transform (after computing DFT reference)
        auto QView = bunch->Q.getView();
        Kokkos::parallel_for(
            "zero_Q", bunch->getLocalNum(), KOKKOS_LAMBDA(const size_t i) { QView(i) = 0.0; });
        Kokkos::fence();

        fft->transform(bunch->R, bunch->Q, field);

        // Get NUFFT result
        T nufftVal = extractNUFFTResultAtTestParticle();

        // Validate (compare real parts)
        T absError = std::fabs(refReal - nufftVal);
        T relError = std::fabs(absError / refReal);

        if (ippl::Comm->rank() == 0) {
            EXPECT_NEAR(relError, 0.0, tolerance * 100);
        }
    }

    std::shared_ptr<layout_type> layout;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<playout_type> playout;
    std::shared_ptr<bunch_type> bunch;

    ippl::Vector<T, Dim> minU, maxU;
    ippl::Vector<int, Dim> nModes;
};

TYPED_TEST_SUITE(NUFFT2Test, Tests);

//=============================================================================
// NUFFT Type-2 Tests (Grid → Particles)
//=============================================================================

TYPED_TEST(NUFFT2Test, BasicCorrectness_SmallGrid_NoUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(512);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-7, false, "tiled", "atomic_sort");
    this->runType2Test(params, 1e-7);
}

TYPED_TEST(NUFFT2Test, BasicCorrectness_SmallGrid_WithUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(512);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-7, true, "tiled", "atomic_sort");
    this->runType2Test(params, 1e-7);
}

TYPED_TEST(NUFFT2Test, BasicCorrectness_MediumGrid_NoUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-7, false, "tiled", "atomic_sort");
    this->runType2Test(params, 1e-7);
}

TYPED_TEST(NUFFT2Test, BasicCorrectness_MediumGrid_WithUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-7, true, "tiled", "atomic_sort");
    this->runType2Test(params, 1e-7);
}

TYPED_TEST(NUFFT2Test, GatherMethod_Atomic) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-7, false, "tiled", "atomic");
    this->runType2Test(params, 1e-7);
}

TYPED_TEST(NUFFT2Test, GatherMethod_AtomicSort) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-7, false, "tiled", "atomic_sort");
    this->runType2Test(params, 1e-7);
}

TYPED_TEST(NUFFT2Test, GatherMethod_Tiled) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-7, false, "tiled", "tiled");
    this->runType2Test(params, 1e-7);
}

TYPED_TEST(NUFFT2Test, Tolerance_1e4) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-4, false, "tiled", "atomic_sort");
    this->runType2Test(params, 1e-4);
}

TYPED_TEST(NUFFT2Test, Tolerance_1e7) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-7, false, "tiled", "atomic_sort");
    this->runType2Test(params, 1e-7);
}

TYPED_TEST(NUFFT2Test, Tolerance_1e10) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params = ippl::test::NUFFTParams::createNativeParams<typename TestFixture::value_type>(
        1e-10, false, "tiled", "atomic_sort");
    this->runType2Test(params, 1e-10);
}

#ifdef ENABLE_FINUFFT
TYPED_TEST(NUFFT2Test, FINUFFT_NoUpsampling) {
    std::array<size_t, TestFixture::dim> gridSize;
    gridSize.fill(16);

    this->setupGrid(gridSize);
    this->setupParticles(4096);
    this->generateRandomParticles();

    auto params =
        ippl::test::NUFFTParams::createFinufftParams<typename TestFixture::value_type>(1e-7, false);
    this->runType2Test(params, 1e-7);
}

#endif

//=============================================================================
// Main
//=============================================================================

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
