#include "Ippl.h"

#include <cstddef>
#include <string>
#include <vector>

#include "Nodes1D/Nodes1D.h"
#include "Nodes1DOracleData.h"
#include "gtest/gtest.h"

using namespace ippl::nodes1d;
using namespace nodes1d_oracle;

namespace {

constexpr double kTol = 1e-12;  // n=64 Newton Chebyshev weights need ~1.5e-13 headroom

constexpr RootFinderMethod kAllMethods[] = {
    RootFinderMethod::GolubWelsch,
    RootFinderMethod::DenseGolubWelsch,
    RootFinderMethod::Newton,
};

const char* methodName(RootFinderMethod m) {
    switch (m) {
        case RootFinderMethod::GolubWelsch:
            return "GolubWelsch";
        case RootFinderMethod::DenseGolubWelsch:
            return "DenseGolubWelsch";
        case RootFinderMethod::Newton:
            return "Newton";
    }
    return "Unknown";
}

void expectArraysNear(const double* got, const double* ref, std::size_t n, double tol,
                      const char* what) {
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(got[i], ref[i], tol) << what << " i=" << i << " n=" << n;
    }
}

std::string tag(const char* family, RootFinderMethod m, const char* kind, std::size_t n) {
    return std::string(family) + "/" + methodName(m) + " " + kind + " n=" + std::to_string(n);
}

void expectGLOracle(RootFinderMethod method, const OracleSample& s) {
    std::vector<double> x(s.n), w(s.n);
    computeGaussLegendre(s.n, x.data(), w.data(), 40, 1, InitialGuessType::Asymptotic, method);
    expectArraysNear(x.data(), s.nodes, s.n, kTol, tag("GL", method, "nodes", s.n).c_str());
    expectArraysNear(w.data(), s.weights, s.n, kTol, tag("GL", method, "weights", s.n).c_str());
}

void expectGLLOracle(RootFinderMethod method, const OracleSample& s) {
    std::vector<double> x(s.n), w(s.n);
    computeGaussLobatto(s.n, x.data(), w.data(), 40, 1, method);
    expectArraysNear(x.data(), s.nodes, s.n, kTol, tag("GLL", method, "nodes", s.n).c_str());
    expectArraysNear(w.data(), s.weights, s.n, kTol, tag("GLL", method, "weights", s.n).c_str());
}

void expectJacobiOracle(RootFinderMethod method, double alpha, double beta, const OracleSample& s,
                        const char* family) {
    std::vector<double> x(s.n), w(s.n);
    computeGaussJacobi(s.n, alpha, beta, x.data(), w.data(), 40, 1, InitialGuessType::Asymptotic,
                       method);
    expectArraysNear(x.data(), s.nodes, s.n, kTol, tag(family, method, "nodes", s.n).c_str());
    expectArraysNear(w.data(), s.weights, s.n, kTol, tag(family, method, "weights", s.n).c_str());
}

}  // namespace

// Gauss–Legendre nodes/weights on [-1,1] vs SciPy (GL_SAMPLES) for every
// RootFinderMethod (GolubWelsch, DenseGolubWelsch, Newton).
TEST(Nodes1DLegendre, OracleSamplesAllMethods) {
    for (RootFinderMethod method : kAllMethods) {
        for (const OracleSample& s : GL_SAMPLES) {
            expectGLOracle(method, s);
        }
    }
}

// GL quadrature is exact for the constant 1: weights sum to length( [-1,1] ) = 2,
// and nodes are strictly increasing. Independent of the SciPy tables.
TEST(Nodes1DLegendre, WeightSumAndOrderingAllMethods) {
    for (RootFinderMethod method : kAllMethods) {
        for (const OracleSample& s : GL_SAMPLES) {
            std::vector<double> x(s.n), w(s.n);
            computeGaussLegendre(s.n, x.data(), w.data(), 40, 1, InitialGuessType::Asymptotic,
                                 method);
            double sum = 0.0;
            for (double wi : w) {
                sum += wi;
            }
            EXPECT_NEAR(sum, 2.0, kTol) << methodName(method) << " n=" << s.n;
            for (std::size_t i = 1; i < s.n; ++i) {
                EXPECT_LT(x[i - 1], x[i]) << methodName(method) << " n=" << s.n;
            }
        }
    }
}

// Kokkos View overload (host fill + deep_copy) matches the pointer API.
// NUFFT Correction.h uses this overload.
TEST(Nodes1DLegendre, KokkosViewOverloadMatchesPointer) {
    using exec_space = Kokkos::DefaultExecutionSpace;
    for (const OracleSample& s : GL_SAMPLES) {
        std::vector<double> xref(s.n), wref(s.n);
        computeGaussLegendre(s.n, xref.data(), wref.data());

        Kokkos::View<double*, typename exec_space::memory_space> nodes("n", s.n);
        Kokkos::View<double*, typename exec_space::memory_space> weights("w", s.n);
        computeGaussLegendre<exec_space>(nodes, weights);

        auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nodes);
        auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weights);
        for (std::size_t i = 0; i < s.n; ++i) {
            EXPECT_NEAR(hn(i), xref[i], kTol) << "n=" << s.n;
            EXPECT_NEAR(hw(i), wref[i], kTol) << "n=" << s.n;
        }
    }
}

// Gauss–Lobatto (GLL) nodes/weights on [-1,1] vs SciPy (GLL_SAMPLES) for every
// RootFinderMethod.
TEST(Nodes1DLobatto, OracleSamplesAllMethods) {
    for (RootFinderMethod method : kAllMethods) {
        for (const OracleSample& s : GLL_SAMPLES) {
            expectGLLOracle(method, s);
        }
    }
}

// GLL always includes the endpoints ±1, weights sum to 2, nodes strictly
// increasing — for every RootFinderMethod and every n in GLL_SAMPLES.
TEST(Nodes1DLobatto, EndpointsWeightSumOrderingAllMethods) {
    for (RootFinderMethod method : kAllMethods) {
        for (const OracleSample& s : GLL_SAMPLES) {
            std::vector<double> x(s.n), w(s.n);
            computeGaussLobatto(s.n, x.data(), w.data(), 40, 1, method);
            EXPECT_DOUBLE_EQ(x.front(), -1.0) << methodName(method) << " n=" << s.n;
            EXPECT_DOUBLE_EQ(x.back(), 1.0) << methodName(method) << " n=" << s.n;
            double sum = 0.0;
            for (double wi : w) {
                sum += wi;
            }
            EXPECT_NEAR(sum, 2.0, kTol) << methodName(method) << " n=" << s.n;
            for (std::size_t i = 1; i < s.n; ++i) {
                EXPECT_LT(x[i - 1], x[i]) << methodName(method) << " n=" << s.n;
            }
        }
    }
}

// Gauss–Jacobi(α=β=-1/2) ≡ Chebyshev–Gauss vs SciPy (JAC_CHEB_SAMPLES) for
// every RootFinderMethod.
TEST(Nodes1DJacobi, ChebyshevOracleSamplesAllMethods) {
    for (RootFinderMethod method : kAllMethods) {
        for (const OracleSample& s : JAC_CHEB_SAMPLES) {
            expectJacobiOracle(method, -0.5, -0.5, s, "JAC_CHEB");
        }
    }
}

// Gauss–Jacobi(α=β=1) vs SciPy (JAC_11_SAMPLES) for every RootFinderMethod.
// These interiors also feed the GLL Golub–Welsch path.
TEST(Nodes1DJacobi, Jacobi11OracleSamplesAllMethods) {
    for (RootFinderMethod method : kAllMethods) {
        for (const OracleSample& s : JAC_11_SAMPLES) {
            expectJacobiOracle(method, 1.0, 1.0, s, "JAC_11");
        }
    }
}

// Jacobi(α=β=0) is Gauss–Legendre: computeGaussJacobi and computeGaussLegendre
// must agree node-for-node for every method and every n in GL_SAMPLES.
TEST(Nodes1DJacobi, AlphaBetaZeroMatchesLegendreAllMethods) {
    for (RootFinderMethod method : kAllMethods) {
        for (const OracleSample& s : GL_SAMPLES) {
            std::vector<double> xj(s.n), wj(s.n), xl(s.n), wl(s.n);
            computeGaussJacobi(s.n, 0.0, 0.0, xj.data(), wj.data(), 40, 1,
                               InitialGuessType::Asymptotic, method);
            computeGaussLegendre(s.n, xl.data(), wl.data(), 40, 1, InitialGuessType::Asymptotic,
                                 method);
            for (std::size_t i = 0; i < s.n; ++i) {
                EXPECT_NEAR(xj[i], xl[i], kTol) << methodName(method) << " n=" << s.n;
                EXPECT_NEAR(wj[i], wl[i], kTol) << methodName(method) << " n=" << s.n;
            }
        }
    }
}

// Newton with Chebyshev (not Asymptotic) initial guesses: Jacobi(1,1) n=10
// must still converge (via the retry/Brent ladder) and match GolubWelsch.
// Oracle samples only exercise Asymptotic starts.
TEST(Nodes1DRootFinder, ChebyshevFirstNewtonJacobi11StillWorksViaLadder) {
    constexpr std::size_t n = 10;
    std::vector<double> x(n), w(n), xref(n), wref(n);
    computeGaussJacobi(n, 1.0, 1.0, xref.data(), wref.data(), 40, 1, InitialGuessType::Asymptotic,
                       RootFinderMethod::GolubWelsch);
    EXPECT_NO_THROW(computeGaussJacobi(n, 1.0, 1.0, x.data(), w.data(), 40, 1,
                                       InitialGuessType::Chebyshev, RootFinderMethod::Newton));
    expectArraysNear(x.data(), xref.data(), n, kTol, "JAC_11 Chebyshev-first Newton vs GW");
}

// affineMapPoint/Weight maps GL from [-1,1] onto [0,1] (x'=(x+1)/2, w'=w/2)
// and matches SciPy GL_AFFINE_01_SAMPLES; mapped weights sum to 1.
TEST(Nodes1DAffine, OracleGLMappedToUnitInterval) {
    for (const OracleSample& s : GL_AFFINE_01_SAMPLES) {
        std::vector<double> x(s.n), w(s.n);
        computeGaussLegendre(s.n, x.data(), w.data());
        for (std::size_t i = 0; i < s.n; ++i) {
            x[i] = affineMapPoint(x[i], -1.0, 1.0, 0.0, 1.0);
            w[i] = affineMapWeight(w[i], -1.0, 1.0, 0.0, 1.0);
        }
        expectArraysNear(x.data(), s.nodes, s.n, kTol, "affine nodes");
        expectArraysNear(w.data(), s.weights, s.n, kTol, "affine weights");
        double sum = 0.0;
        for (double wi : w) {
            sum += wi;
        }
        EXPECT_NEAR(sum, 1.0, kTol) << "n=" << s.n;
    }
}

// Mapping GL [-1,1] → [2,5] and back recovers the original nodes/weights;
// weights on [2,5] sum to 3. Also checks the Vector overload
// affineMapNodesWeights round-trip [-1,1] ↔ [0,1] for n=3.
TEST(Nodes1DAffine, RoundTripAndVectorHelper) {
    for (const OracleSample& s : GL_SAMPLES) {
        const std::size_t n = s.n;
        std::vector<double> x(n), w(n);
        computeGaussLegendre(n, x.data(), w.data());
        const auto x0 = x;
        const auto w0 = w;

        for (std::size_t i = 0; i < n; ++i) {
            x[i] = affineMapPoint(x[i], -1.0, 1.0, 2.0, 5.0);
            w[i] = affineMapWeight(w[i], -1.0, 1.0, 2.0, 5.0);
        }
        double sum = 0.0;
        for (double wi : w) {
            sum += wi;
        }
        EXPECT_NEAR(sum, 3.0, kTol) << "n=" << n;

        for (std::size_t i = 0; i < n; ++i) {
            x[i] = affineMapPoint(x[i], 2.0, 5.0, -1.0, 1.0);
            w[i] = affineMapWeight(w[i], 2.0, 5.0, -1.0, 1.0);
            EXPECT_NEAR(x[i], x0[i], kTol);
            EXPECT_NEAR(w[i], w0[i], kTol);
        }
    }

    ippl::Vector<double, 3> nodes, weights;
    computeGaussLegendre(nodes, weights);
    const auto nodes0   = nodes;
    const auto weights0 = weights;
    affineMapNodesWeights(nodes, weights, -1.0, 1.0, 0.0, 1.0);
    affineMapNodesWeights(nodes, weights, 0.0, 1.0, -1.0, 1.0);
    for (unsigned i = 0; i < 3; ++i) {
        EXPECT_NEAR(nodes[i], nodes0[i], kTol);
        EXPECT_NEAR(weights[i], weights0[i], kTol);
    }
}

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
