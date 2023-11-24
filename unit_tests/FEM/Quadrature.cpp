
#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class QuadratureTest;

template <typename T, typename ExecSpace, unsigned NumNodes1D, unsigned Dim>
class QuadratureTest<Parameters<T, ExecSpace, Rank<NumNodes1D>, Rank<Dim>>>
    : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t = T;

    using ElementType = std::conditional_t<
        Dim == 1, ippl::EdgeElement<T>,
        std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    using QuadratureType = ippl::MidpointQuadrature<T, NumNodes1D, ElementType>;

    QuadratureTest()
        : ref_element()
        , quadrature(ref_element) {}

    const ElementType ref_element;
    const QuadratureType quadrature;
};

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using NumNodes   = TestParams::Ranks<1, 2, 3>;
using Dims       = TestParams::Ranks<1, 2, 3>;
using Combos     = CreateCombinations<Precisions, Spaces, NumNodes, Dims>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(QuadratureTest, Tests);

TYPED_TEST(QuadratureTest, getWeightsForRefElement) {
    const auto& quadrature        = this->quadrature;
    const std::size_t& numNodes1D = this->quadrature.numNodes1D;
    const std::size_t& dim        = this->ref_element.dim;

    // get 1D weights
    const auto& w1D = quadrature.getWeights1D(0.0, 1.0);

    if (dim == 1) {
        const auto& w = quadrature.getWeightsForRefElement();

        ASSERT_EQ(w1D.dim, w.dim);

        for (unsigned i = 0; i < numNodes1D; ++i) {
            EXPECT_DOUBLE_EQ(w1D[i], w[i]);
        }
    } else if (dim == 2) {
        const auto& w = quadrature.getWeightsForRefElement();

        ASSERT_EQ(pow(w1D.dim, 2), w.dim);

        for (unsigned i = 0; i < numNodes1D; ++i) {
            for (unsigned j = 0; j < numNodes1D; ++j) {
                EXPECT_DOUBLE_EQ(w1D[i] * w1D[j], w[i * numNodes1D + j]);
            }
        }
    } else if (dim == 3) {
        const auto& w = quadrature.getWeightsForRefElement();

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

TYPED_TEST(QuadratureTest, getIntegrationNodesForRefElement) {
    const auto& quadrature        = this->quadrature;
    const std::size_t& numNodes1D = this->quadrature.numNodes1D;
    const std::size_t& dim        = this->ref_element.dim;

    // get 1D nodes
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

                // std::cout << "x: " << x << ", y: " << y << ", q_x: " << q[y * numNodes1D + x][0]
                //           << ", q_y: " << q[y * numNodes1D + x][1] << std::endl;
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