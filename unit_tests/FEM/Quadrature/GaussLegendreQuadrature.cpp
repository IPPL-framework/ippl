#include "Ippl.h"

#include "Nodes1D/Nodes1D.h"
#include "gtest/gtest.h"

// FEM GaussLegendreQuadrature (n=7, default GolubWelsch) must be bit-identical
// to nodes1d::computeGaussLegendre — same Nodes1D path, no extra rounding.
TEST(GaussLegendreQuadrature, MatchesNodes1D) {
    using ElementType = ippl::EdgeElement<double>;
    const ElementType ref;
    constexpr unsigned N = 7;
    const ippl::GaussLegendreQuadrature<double, N, ElementType> fem(ref, 10, 1);
    double nodes[N], weights[N];
    ippl::nodes1d::computeGaussLegendre(N, nodes, weights);
    const auto& q = fem.getIntegrationNodes1D(-1.0, 1.0);
    const auto& w = fem.getWeights1D(-1.0, 1.0);
    for (unsigned i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(nodes[i], q[i]);
        EXPECT_DOUBLE_EQ(weights[i], w[i]);
    }
}

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
