#include "Ippl.h"

#include <vector>

#include "Nodes1D/Nodes1D.h"
#include "gtest/gtest.h"

// FEM GaussChebyshevQuadrature (n=7) matches nodes1d::computeGaussJacobi(-0.5,-0.5)
// with the default GolubWelsch backend.
TEST(GaussChebyshevQuadrature, MatchesNodes1D) {
    using ElementType = ippl::EdgeElement<double>;
    const ElementType ref;
    constexpr unsigned N = 7;
    constexpr double kTol = 1e-12;
    const ippl::GaussChebyshevQuadrature<double, N, ElementType> fem(ref, 10, 1);
    std::vector<double> x(N), w(N);
    ippl::nodes1d::computeGaussJacobi(N, -0.5, -0.5, x.data(), w.data());
    const auto& q  = fem.getIntegrationNodes1D(-1.0, 1.0);
    const auto& ww = fem.getWeights1D(-1.0, 1.0);
    for (unsigned i = 0; i < N; ++i) {
        EXPECT_NEAR(x[i], q[i], kTol);
        EXPECT_NEAR(w[i], ww[i], kTol);
    }
}

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
