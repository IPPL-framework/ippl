
#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

// Verifies getWeightsForRefElement() equals the Dim-fold tensor product of the 1D weights.
// Templated on the concrete quadrature type so every rule (Midpoint, Gauss-Legendre,
// Gauss-Lobatto, ...) shares one reference implementation of the expected layout.
template <typename QuadratureType>
void expectWeightsAreTensorProduct(const QuadratureType& quadrature, unsigned numNodes1D,
                                   unsigned dim) {
    const auto& w1D = quadrature.getWeights1D(0.0, 1.0);
    const auto& w   = quadrature.getWeightsForRefElement();

    if (dim == 1) {
        ASSERT_EQ(w1D.dim, w.dim);

        for (unsigned i = 0; i < numNodes1D; ++i) {
            EXPECT_DOUBLE_EQ(w1D[i], w[i]);
        }
    } else if (dim == 2) {
        ASSERT_EQ(pow(w1D.dim, 2), w.dim);

        for (unsigned i = 0; i < numNodes1D; ++i) {
            for (unsigned j = 0; j < numNodes1D; ++j) {
                EXPECT_DOUBLE_EQ(w1D[i] * w1D[j], w[i * numNodes1D + j]);
            }
        }
    } else if (dim == 3) {
        ASSERT_EQ(pow(w1D.dim, 3), w.dim);

        for (unsigned i = 0; i < numNodes1D; ++i) {
            for (unsigned j = 0; j < numNodes1D; ++j) {
                for (unsigned k = 0; k < numNodes1D; ++k) {
                    EXPECT_DOUBLE_EQ(w1D[i] * w1D[j] * w1D[k],
                                     w[k * (numNodes1D * numNodes1D) + i * numNodes1D + j]);
                }
            }
        }
    } else {
        FAIL();
    }
}

// Verifies getIntegrationNodesForRefElement() equals the Dim-fold tensor product of the 1D nodes
// (and that every 1D node lies inside the [0, 1] reference interval).
template <typename QuadratureType>
void expectNodesAreTensorProduct(const QuadratureType& quadrature, unsigned numNodes1D,
                                 unsigned dim) {
    const auto& q1D = quadrature.getIntegrationNodes1D(0.0, 1.0);

    if (dim == 1) {
        const auto& q = quadrature.getIntegrationNodesForRefElement();

        ASSERT_EQ(q1D.dim, q.dim);

        for (unsigned i = 0; i < numNodes1D; ++i) {
            EXPECT_LE(q1D[i], 1.0);
            EXPECT_GE(q1D[i], 0.0);

            EXPECT_DOUBLE_EQ(q1D[i], q[i][0]);
        }
    } else if (dim == 2) {
        const auto& q = quadrature.getIntegrationNodesForRefElement();

        ASSERT_EQ(pow(q1D.dim, 2), q.dim);

        for (unsigned y = 0; y < numNodes1D; ++y) {
            for (unsigned x = 0; x < numNodes1D; ++x) {
                EXPECT_LE(q1D[x], 1.0);
                EXPECT_GE(q1D[x], 0.0);
                EXPECT_LE(q1D[y], 1.0);
                EXPECT_GE(q1D[y], 0.0);

                EXPECT_DOUBLE_EQ(q1D[x], q[y * numNodes1D + x][0]);
                EXPECT_DOUBLE_EQ(q1D[y], q[y * numNodes1D + x][1]);
            }
        }
    } else if (dim == 3) {
        const auto& q = quadrature.getIntegrationNodesForRefElement();

        ASSERT_EQ(pow(q1D.dim, 3), q.dim);

        for (unsigned z = 0; z < numNodes1D; ++z) {
            for (unsigned y = 0; y < numNodes1D; ++y) {
                for (unsigned x = 0; x < numNodes1D; ++x) {
                    EXPECT_LE(q1D[x], 1.0);
                    EXPECT_GE(q1D[x], 0.0);
                    EXPECT_LE(q1D[y], 1.0);
                    EXPECT_GE(q1D[y], 0.0);
                    EXPECT_LE(q1D[z], 1.0);
                    EXPECT_GE(q1D[z], 0.0);

                    EXPECT_DOUBLE_EQ(q1D[x],
                                     q[z * (numNodes1D * numNodes1D) + y * numNodes1D + x][0]);
                    EXPECT_DOUBLE_EQ(q1D[y],
                                     q[z * (numNodes1D * numNodes1D) + y * numNodes1D + x][1]);
                    EXPECT_DOUBLE_EQ(q1D[z],
                                     q[z * (numNodes1D * numNodes1D) + y * numNodes1D + x][2]);
                }
            }
        }
    } else {
        FAIL();
    }
}

template <typename>
class QuadratureTest;

template <typename T, typename ExecSpace, unsigned NumNodes1D, unsigned Dim>
class QuadratureTest<Parameters<T, ExecSpace, Rank<NumNodes1D>, Rank<Dim>>>
    : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t                        = T;
    constexpr static unsigned dim        = Dim;
    constexpr static unsigned numNodes1D = NumNodes1D;

    static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");

    using ElementType = std::conditional_t<
        Dim == 1, ippl::EdgeElement<T>,
        std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    QuadratureTest()
        : ref_element()
        , midpointQuadrature(ref_element)
        , gaussLegendreQuadrature(ref_element) {}

    const ElementType ref_element;
    const ippl::MidpointQuadrature<T, NumNodes1D, ElementType> midpointQuadrature;
    const ippl::GaussLegendreQuadrature<T, NumNodes1D, ElementType> gaussLegendreQuadrature;
};

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using NumNodes   = TestParams::Ranks<1, 2, 3>;
using Dims       = TestParams::Ranks<1, 2, 3>;
using Combos     = CreateCombinations<Precisions, Spaces, NumNodes, Dims>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(QuadratureTest, Tests);

TYPED_TEST(QuadratureTest, getWeightsForRefElement) {
    expectWeightsAreTensorProduct(this->midpointQuadrature, TestFixture::numNodes1D,
                                  TestFixture::dim);
    expectWeightsAreTensorProduct(this->gaussLegendreQuadrature, TestFixture::numNodes1D,
                                  TestFixture::dim);
}

TYPED_TEST(QuadratureTest, getIntegrationNodesForRefElement) {
    expectNodesAreTensorProduct(this->midpointQuadrature, TestFixture::numNodes1D,
                                TestFixture::dim);
    expectNodesAreTensorProduct(this->gaussLegendreQuadrature, TestFixture::numNodes1D,
                                TestFixture::dim);
}

// ---------------------------------------------------------------------------------------------
// Gauss-Lobatto-Legendre tensor-product coverage.
//
// GLL fixes the two endpoints, so a 1-node rule is undefined and GaussLobattoQuadrature
// static_asserts NumNodes1D >= 2. It therefore cannot join QuadratureTest, whose node list is
// Ranks<1, 2, 3> (the N == 1 instantiation would fail to compile). This dedicated fixture uses
// Ranks<2, 3> and reuses the same tensor-product helpers to check getWeightsForRefElement() and
// getIntegrationNodesForRefElement() across Dim = 1, 2, 3.
// ---------------------------------------------------------------------------------------------
template <typename>
class GaussLobattoQuadratureRefElementTest;

template <typename T, typename ExecSpace, unsigned NumNodes1D, unsigned Dim>
class GaussLobattoQuadratureRefElementTest<Parameters<T, ExecSpace, Rank<NumNodes1D>, Rank<Dim>>>
    : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t                        = T;
    constexpr static unsigned dim        = Dim;
    constexpr static unsigned numNodes1D = NumNodes1D;

    static_assert(NumNodes1D >= 2, "Gauss-Lobatto quadrature requires NumNodes1D >= 2");
    static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");

    using ElementType = std::conditional_t<
        Dim == 1, ippl::EdgeElement<T>,
        std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    GaussLobattoQuadratureRefElementTest()
        : ref_element()
        , gaussLobattoQuadrature(ref_element) {}

    const ElementType ref_element;
    const ippl::GaussLobattoQuadrature<T, NumNodes1D, ElementType> gaussLobattoQuadrature;
};

using GaussLobattoNumNodes = TestParams::Ranks<2, 3>;
using GaussLobattoCombos = CreateCombinations<Precisions, Spaces, GaussLobattoNumNodes, Dims>::type;
using GaussLobattoTests  = TestForTypes<GaussLobattoCombos>::type;
TYPED_TEST_CASE(GaussLobattoQuadratureRefElementTest, GaussLobattoTests);

TYPED_TEST(GaussLobattoQuadratureRefElementTest, getWeightsForRefElement) {
    expectWeightsAreTensorProduct(this->gaussLobattoQuadrature, TestFixture::numNodes1D,
                                  TestFixture::dim);
}

TYPED_TEST(GaussLobattoQuadratureRefElementTest, getIntegrationNodesForRefElement) {
    expectNodesAreTensorProduct(this->gaussLobattoQuadrature, TestFixture::numNodes1D,
                                TestFixture::dim);
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
