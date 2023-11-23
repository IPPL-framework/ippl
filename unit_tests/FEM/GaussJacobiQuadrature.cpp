
#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class GaussJacobiQuadratureTest;

template <typename T, typename ExecSpace, unsigned NumNodes1D>
class GaussJacobiQuadratureTest<Parameters<T, ExecSpace, Rank<NumNodes1D>>>
    : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t = T;

    using ElementType = ippl::EdgeElement<T>;

    GaussJacobiQuadratureTest()
        : ref_element()
        , gaussLegendreQuadrature(ref_element, 10)
        , chebychevGaussQuadrature(ref_element, 10) {}

    const ElementType ref_element;

    const ippl::GaussLegendreQuadrature<T, NumNodes1D, ElementType> gaussLegendreQuadrature;
    const ippl::ChebyshevGaussQuadrature<T, NumNodes1D, ElementType> chebychevGaussQuadrature;
};

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using NumNodes   = TestParams::Ranks<1, 2, 3, 4, 5, 6, 7>;
using Combos     = CreateCombinations<Precisions, Spaces, NumNodes>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(GaussJacobiQuadratureTest, Tests);

TYPED_TEST(GaussJacobiQuadratureTest, GaussLegendreQuadrature) {
    using T = typename TestFixture::value_t;

    const auto& gaussLegendreQuadrature = this->gaussLegendreQuadrature;
    const std::size_t& numNodes1D       = this->gaussLegendreQuadrature.numNodes1D;

    const T& tol = std::numeric_limits<T>::epsilon() * 10;

    // Gauss-Legendre Quadrature
    const auto& q = gaussLegendreQuadrature.getIntegrationNodes1D(-1.0, 1.0);
    const auto& w = gaussLegendreQuadrature.getWeights1D(-1.0, 1.0);

    if (numNodes1D == 1) {
        EXPECT_NEAR(q[0], 0.0, tol);

        EXPECT_NEAR(w[0], 2.0, tol);
    } else if (numNodes1D == 2) {
        EXPECT_NEAR(q[0], -0.5773502691896258, tol);
        EXPECT_NEAR(q[1], 0.5773502691896258, tol);

        EXPECT_NEAR(w[0], 1.0, tol);
        EXPECT_NEAR(w[1], 1.0, tol);
    } else if (numNodes1D == 3) {
        EXPECT_NEAR(q[0], -0.7745966692414834, tol);
        EXPECT_NEAR(q[1], 0.0, tol);
        EXPECT_NEAR(q[2], 0.7745966692414834, tol);

        EXPECT_NEAR(w[0], 0.5555555555555556, tol);
        EXPECT_NEAR(w[1], 0.8888888888888888, tol);
        EXPECT_NEAR(w[2], 0.5555555555555556, tol);
    } else if (numNodes1D == 4) {
        EXPECT_NEAR(q[0], -0.8611363115940526, tol);
        EXPECT_NEAR(q[1], -0.3399810435848563, tol);
        EXPECT_NEAR(q[2], 0.3399810435848563, tol);
        EXPECT_NEAR(q[3], 0.8611363115940526, tol);

        EXPECT_NEAR(w[0], 0.3478548451374538, tol);
        EXPECT_NEAR(w[1], 0.6521451548625461, tol);
        EXPECT_NEAR(w[2], 0.6521451548625461, tol);
        EXPECT_NEAR(w[3], 0.3478548451374538, tol);
    } else if (numNodes1D == 5) {
        EXPECT_NEAR(q[0], -0.9061798459386640, tol);
        EXPECT_NEAR(q[1], -0.5384693101056831, tol);
        EXPECT_NEAR(q[2], 0.0, tol);
        EXPECT_NEAR(q[3], 0.5384693101056831, tol);
        EXPECT_NEAR(q[4], 0.9061798459386640, tol);

        EXPECT_NEAR(w[0], 0.2369268850561891, tol);
        EXPECT_NEAR(w[1], 0.4786286704993665, tol);
        EXPECT_NEAR(w[2], 0.5688888888888889, tol);
        EXPECT_NEAR(w[3], 0.4786286704993665, tol);
        EXPECT_NEAR(w[4], 0.2369268850561891, tol);
    } else if (numNodes1D == 6) {
        EXPECT_NEAR(q[0], -0.9324695142031521, tol);
        EXPECT_NEAR(q[1], -0.6612093864662645, tol);
        EXPECT_NEAR(q[2], -0.2386191860831969, tol);
        EXPECT_NEAR(q[3], 0.2386191860831969, tol);
        EXPECT_NEAR(q[4], 0.6612093864662645, tol);
        EXPECT_NEAR(q[5], 0.9324695142031521, tol);

        EXPECT_NEAR(w[0], 0.1713244923791704, tol);
        EXPECT_NEAR(w[1], 0.3607615730481386, tol);
        EXPECT_NEAR(w[2], 0.4679139345726910, tol);
        EXPECT_NEAR(w[3], 0.4679139345726910, tol);
        EXPECT_NEAR(w[4], 0.3607615730481386, tol);
        EXPECT_NEAR(w[5], 0.1713244923791704, tol);
    } else if (numNodes1D == 7) {
        EXPECT_NEAR(q[0], -0.9491079123427585, tol);
        EXPECT_NEAR(q[1], -0.7415311855993945, tol);
        EXPECT_NEAR(q[2], -0.4058451513773972, tol);
        EXPECT_NEAR(q[3], 0.0, tol);
        EXPECT_NEAR(q[4], 0.4058451513773972, tol);
        EXPECT_NEAR(q[5], 0.7415311855993945, tol);
        EXPECT_NEAR(q[6], 0.9491079123427585, tol);

        EXPECT_NEAR(w[0], 0.1294849661688697, tol);
        EXPECT_NEAR(w[1], 0.2797053914892766, tol);
        EXPECT_NEAR(w[2], 0.3818300505051189, tol);
        EXPECT_NEAR(w[3], 0.4179591836734694, tol);
        EXPECT_NEAR(w[4], 0.3818300505051189, tol);
        EXPECT_NEAR(w[5], 0.2797053914892766, tol);
        EXPECT_NEAR(w[6], 0.1294849661688697, tol);
    }
}

TYPED_TEST(GaussJacobiQuadratureTest, ChebychevGaussQuadrature) {
    // Chebychev-Gauss gaussLegendreQuadrature
    using T = typename TestFixture::value_t;

    const auto& chebychevGaussQuadrature = this->chebychevGaussQuadrature;
    const std::size_t& numNodes1D        = this->chebychevGaussQuadrature.numNodes1D;

    const T& tol = std::numeric_limits<T>::epsilon() * 20;

    // Chebychev-Gauss Quadrature
    const auto& q = chebychevGaussQuadrature.getIntegrationNodes1D(-1.0, 1.0);
    const auto& w = chebychevGaussQuadrature.getWeights1D(-1.0, 1.0);

    T x;
    T w_k = M_PI / numNodes1D;
    for (unsigned k = 0; k < numNodes1D; ++k) {
        x = -Kokkos::cos((2.0 * static_cast<T>(k) + 1.0) * M_PI / (2.0 * numNodes1D));

        EXPECT_NEAR(q[k], x, tol);
        EXPECT_NEAR(w[k], w_k, tol);
    }
}