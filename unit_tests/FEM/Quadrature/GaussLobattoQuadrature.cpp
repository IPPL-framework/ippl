#include "Ippl.h"

#include "Nodes1D/Nodes1D.h"
#include "gtest/gtest.h"

// FEM GaussLobattoQuadrature (n=7, default GolubWelsch) must match nodes1d::computeGaussLobatto.
TEST(GaussLobattoQuadrature, MatchesNodes1D) {
    using ElementType = ippl::EdgeElement<double>;
    const ElementType ref;
    constexpr unsigned N = 7;
    const ippl::GaussLobattoQuadrature<double, N, ElementType> fem(ref);
    double nodes[N], weights[N];
    ippl::nodes1d::computeGaussLobatto(N, nodes, weights);
    const auto& q = fem.getIntegrationNodes1D(-1.0, 1.0);
    const auto& w = fem.getWeights1D(-1.0, 1.0);
    for (unsigned i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(nodes[i], q[i]);
        EXPECT_DOUBLE_EQ(weights[i], w[i]);
    }
    EXPECT_DOUBLE_EQ(-1.0, q[0]);
    EXPECT_DOUBLE_EQ(1.0, q[N - 1]);
}

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
