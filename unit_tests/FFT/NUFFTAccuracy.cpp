//
// NUFFT Accuracy vs Tolerance Test
//
// Sweeps through tolerances and reports relative error using L-infinity norm:
//   error = max|NUFFT - DFT| / max|DFT|
//
// This metric avoids ill-conditioning from modes with small magnitudes.
// Compares native IPPL NUFFT against FINUFFT reference (if enabled).
// Output is CSV-friendly for easy plotting.
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <iomanip>
#include <random>
#include <vector>

#include "NUFFTTestUtils.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename T, unsigned Dim>
class NUFFTAccuracyTest : public ::testing::Test {
public:
    using value_type     = T;
    using exec_space     = Kokkos::DefaultExecutionSpace;
    constexpr static unsigned dim = Dim;

    using mesh_type       = ippl::UniformCartesian<T, Dim>;
    using centering_type  = typename mesh_type::DefaultCentering;
    using field_type      = typename ippl::Field<Kokkos::complex<T>, Dim, mesh_type,
                                                 centering_type, exec_space>::uniform_type;
    using real_field_type = ippl::Field<T, Dim, mesh_type, centering_type, exec_space>;
    using layout_type     = ippl::FieldLayout<Dim>;
    using playout_type    = ippl::ParticleSpatialLayout<T, Dim>;
    using bunch_type      = ippl::test::Bunch<T, playout_type>;
    using FFT_type        = ippl::FFT<ippl::NUFFTransform, real_field_type>;

    NUFFTAccuracyTest() {
        const T pi = Kokkos::numbers::pi_v<T>;
        for (unsigned d = 0; d < Dim; d++) {
            rmin_m[d] = 0;
            rmax_m[d] = 2 * pi;
        }
    }

    void setup(size_t gridSize, size_t numParticles) {
        gridSize_m = gridSize;

        // Setup grid
        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<T, Dim> hx, origin;
        for (unsigned d = 0; d < Dim; d++) {
            domains[d]  = ippl::Index(gridSize);
            hx[d]       = (rmax_m[d] - rmin_m[d]) / gridSize;
            origin[d]   = 0;
            nModes_m[d] = gridSize;
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout_m   = std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel);
        mesh_m     = std::make_shared<mesh_type>(owned, hx, origin);

        // Setup particles
        playout_m = std::make_shared<playout_type>(*layout_m, *mesh_m);
        bunch_m   = std::make_shared<bunch_type>(*playout_m);
        bunch_m->setParticleBC(ippl::BC::PERIODIC);

        size_t nloc = numParticles / ippl::Comm->size();
        bunch_m->create(nloc);

        // Generate random particles (same seed for reproducibility)
        std::mt19937_64 eng(42 + ippl::Comm->rank());
        std::uniform_real_distribution<T> unifPos(0, 1);
        std::uniform_real_distribution<T> unifQ(-1, 1);

        auto R_host = bunch_m->R.getHostMirror();
        auto Q_host = bunch_m->Q.getHostMirror();

        for (size_t i = 0; i < nloc; ++i) {
            for (unsigned d = 0; d < Dim; ++d) {
                R_host(i)[d] = rmin_m[d] + unifPos(eng) * (rmax_m[d] - rmin_m[d]);
            }
            Q_host(i) = unifQ(eng);
        }

        Kokkos::deep_copy(bunch_m->R.getView(), R_host);
        Kokkos::deep_copy(bunch_m->Q.getView(), Q_host);
        Kokkos::fence();
        bunch_m->update();
    }

    // Compute DFT reference for a single mode
    Kokkos::complex<T> computeDFTReference(const ippl::Vector<int, Dim>& kVec) {
        return ippl::test::DFTReference<T, Dim>::computeType1Mode(
            bunch_m->R.getView(), bunch_m->Q.getView(), kVec,
            mesh_m->getMeshSpacing(), nModes_m, bunch_m->getLocalNum());
    }

    // Run Type-1 transform and compute relative error using L-infinity norm
    // Error = max|NUFFT - DFT| / max|DFT|
    T runType1AndGetMaxError(T tolerance, bool useFinufft = false) {
        ippl::ParameterList params;
        if (useFinufft) {
#ifdef ENABLE_FINUFFT
            params = ippl::test::NUFFTParams::createFinufftParams<T>(tolerance, false);
#else
            // FINUFFT not enabled, return -1 to indicate skipped
            return T(-1);
#endif
        } else {
            params = ippl::test::NUFFTParams::createNativeParams<T>(tolerance, false);
        }

        auto fft = std::make_unique<FFT_type>(*layout_m, bunch_m->getLocalNum(), 1, params);

        const int nghost = 1;
        field_type field;
        field.initialize(*mesh_m, *layout_m, nghost);
        field = Kokkos::complex<T>(0);

        fft->transform(bunch_m->R, bunch_m->Q, field);

        // Copy field to host ONCE
        auto fieldHost = field.getHostMirror();
        Kokkos::deep_copy(fieldHost, field.getView());

        const auto& lDom = layout_m->getLocalNDIndex();

        // Compute max absolute error and max DFT magnitude across all modes
        T localMaxAbsError = 0;
        T localMaxDFTMag = 0;

        int N = static_cast<int>(gridSize_m);
        int halfN = N / 2;

        // Iterate over all modes in centered frequency space: k in [-N/2, N/2-1].
        // computeDFTReference does an MPI_Allreduce internally, so every rank
        // must call it on every mode in the same order to stay in lockstep.
        // The owned-mode check below only gates the error update, never the
        // allreduce.
        for (int kx = -halfN; kx < halfN; ++kx) {
            for (int ky = -halfN; ky < halfN; ++ky) {
                for (int kz = -halfN; kz < halfN; ++kz) {
                    ippl::Vector<int, Dim> kVec;
                    kVec[0] = kx;
                    kVec[1] = ky;
                    kVec[2] = kz;

                    auto globalIdx = ippl::test::IndexUtils<Dim>::centeredToCornerDC(kVec, nModes_m);
                    auto dftResult = computeDFTReference(kVec);

                    if (!ippl::test::IndexUtils<Dim>::isOwnedLocally(lDom, globalIdx)) {
                        continue;
                    }

                    auto localIdx = ippl::test::IndexUtils<Dim>::globalToLocal(lDom, globalIdx, nghost);
                    Kokkos::complex<T> nufftResult = fieldHost(localIdx[0], localIdx[1], localIdx[2]);

                    T absError = Kokkos::abs(nufftResult - dftResult);
                    T dftMag = Kokkos::abs(dftResult);

                    if (absError > localMaxAbsError) {
                        localMaxAbsError = absError;
                    }
                    if (dftMag > localMaxDFTMag) {
                        localMaxDFTMag = dftMag;
                    }
                }
            }
        }

        // MPI reduce to get global max absolute error and max DFT magnitude
        T globalMaxAbsError = 0;
        T globalMaxDFTMag = 0;
        MPI_Allreduce(&localMaxAbsError, &globalMaxAbsError, 1,
                      std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE,
                      MPI_MAX, ippl::Comm->getCommunicator());
        MPI_Allreduce(&localMaxDFTMag, &globalMaxDFTMag, 1,
                      std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE,
                      MPI_MAX, ippl::Comm->getCommunicator());

        // Relative error = max|error| / max|DFT|
        return (globalMaxDFTMag > 0) ? (globalMaxAbsError / globalMaxDFTMag) : globalMaxAbsError;
    }

    std::shared_ptr<layout_type> layout_m;
    std::shared_ptr<mesh_type> mesh_m;
    std::shared_ptr<playout_type> playout_m;
    std::shared_ptr<bunch_type> bunch_m;

    ippl::Vector<T, Dim> rmin_m, rmax_m;
    ippl::Vector<int, Dim> nModes_m;
    size_t gridSize_m;
};

// Test for double precision, 3D
using AccuracyTest3D = NUFFTAccuracyTest<double, 3>;

TEST_F(AccuracyTest3D, ToleranceSweep) {
    const size_t gridSize     = 16;
    const size_t numParticles = 10240;

    setup(gridSize, numParticles);

    // Coarse tolerance sweep: one point per pair of decades is enough to
    // verify accuracy/tolerance correlation; the previous 21-point sweep was
    // for plotting smoothness, not coverage.
    const std::vector<double> tolerances{1e-2, 1e-4, 1e-6, 1e-8, 1e-10};

    // Store results: tolerance, native_error, finufft_error
    struct ResultData {
        double tol;
        double nativeErr;
        double finufftErr;
    };
    std::vector<ResultData> results;

#ifdef ENABLE_FINUFFT
    bool hasFinufft = true;
#else
    bool hasFinufft = false;
#endif

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n";
        std::cout << "========================================================================\n";
        std::cout << "NUFFT Type-1: Relative Error vs Tolerance\n";
        std::cout << "Error metric: max|NUFFT - DFT| / max|DFT| (L-infinity relative error)\n";
        std::cout << "========================================================================\n";
        std::cout << "Grid size:     " << gridSize << "^3 ("
                  << gridSize * gridSize * gridSize << " modes)\n";
        std::cout << "Num particles: " << numParticles << "\n";
        std::cout << "FINUFFT:       " << (hasFinufft ? "enabled" : "disabled") << "\n";
        std::cout << "========================================================================\n";

        std::cout << std::setw(14) << "Tolerance" << std::setw(16) << "Native Err";
        if (hasFinufft)
            std::cout << std::setw(16) << "FINUFFT Err";
        std::cout << std::setw(12) << "Err/Tol" << "\n";
        std::cout << "------------------------------------------------------------------------\n";
    }

    for (double tol : tolerances) {
        double nativeError  = runType1AndGetMaxError(tol, false);
        double finufftError = hasFinufft ? runType1AndGetMaxError(tol, true) : -1.0;
        double ratio        = nativeError / tol;

        results.push_back({tol, nativeError, finufftError});

        if (ippl::Comm->rank() == 0) {
            std::cout << std::scientific << std::setprecision(3) << std::setw(14) << tol
                      << std::setw(16) << nativeError;
            if (hasFinufft)
                std::cout << std::setw(16) << finufftError;
            std::cout << std::fixed << std::setprecision(2) << std::setw(12) << ratio << "\n";
        }

        // Soft expectation: max error should be within 100x of tolerance
        EXPECT_LT(nativeError, tol * 100)
            << "Native: Tolerance " << tol << " achieved max error " << nativeError;

        if (hasFinufft && finufftError >= 0) {
            EXPECT_LT(finufftError, tol * 100)
                << "FINUFFT: Tolerance " << tol << " achieved max error " << finufftError;
        }
    }

    // Print CSV output for plotting
    if (ippl::Comm->rank() == 0) {
        std::cout << "========================================================================\n";
        std::cout << "\n# CSV Output (paste into file for plotting):\n";
        if (hasFinufft) {
            std::cout << "# tolerance,native_error,finufft_error\n";
            for (const auto& res : results) {
                std::cout << std::scientific << std::setprecision(8)
                          << res.tol << "," << res.nativeErr << "," << res.finufftErr << "\n";
            }
        } else {
            std::cout << "# tolerance,native_error\n";
            for (const auto& res : results) {
                std::cout << std::scientific << std::setprecision(8)
                          << res.tol << "," << res.nativeErr << "\n";
            }
        }
        std::cout << "\n";
    }
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