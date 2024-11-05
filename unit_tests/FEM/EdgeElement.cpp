

#include "Ippl.h"

#include <limits>
#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class EdgeElementTest;

template <typename T, typename ExecSpace, unsigned Seed>
class EdgeElementTest<Parameters<T, ExecSpace, Rank<Seed>>> : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t         = T;
    using point_t         = ippl::EdgeElement<T>::point_t;
    using vertex_points_t = ippl::EdgeElement<T>::vertex_points_t;

    static constexpr unsigned NumEdges = 3;

    EdgeElementTest()
        : rng(Seed) {
        const T interval_size = std::numeric_limits<T>::max() / 100.0;

        std::uniform_real_distribution<T> dist(-interval_size / 2.0, interval_size / 2.0);

        for (unsigned i = 0; i < NumEdges; i++) {
            edges[i][0][0] = dist(rng);
            edges[i][1][0] = dist(rng);
        }
    }

    std::mt19937 rng;

    ippl::EdgeElement<T> edge_element;

    const vertex_points_t local_points = {{0.0}, {1.0}};
    const point_t local_mid_point                      = {0.5};

    ippl::Vector<vertex_points_t, NumEdges> edges;
};

using Tests = TestParams::tests<42>;
TYPED_TEST_CASE(EdgeElementTest, Tests);

TYPED_TEST(EdgeElementTest, LocalVertices) {
    auto& edge_element = this->edge_element;

    for (unsigned i = 0; i < this->local_points.dim; i++) {
        ASSERT_EQ(edge_element.getLocalVertices()[i].dim, 1);
        ASSERT_EQ(edge_element.getLocalVertices()[i][0], this->local_points[i][0]);
    }
}

TYPED_TEST(EdgeElementTest, LocalToGlobal) {
    using T                               = typename TestFixture::value_t;
    using point_t                         = typename TestFixture::point_t;
    using vertex_points_t = typename TestFixture::vertex_points_t;

    auto& edge_element = this->edge_element;

    for (const vertex_points_t& edge : this->edges) {
        point_t transformed_start_point = edge_element.localToGlobal(edge, this->local_points[0]);
        point_t transformed_mid_point   = edge_element.localToGlobal(edge, this->local_mid_point);
        point_t transformed_end_point   = edge_element.localToGlobal(edge, this->local_points[1]);

        if (std::is_same<T, double>::value) {
            ASSERT_DOUBLE_EQ(transformed_start_point[0], edge[0][0]);
            ASSERT_DOUBLE_EQ(transformed_mid_point[0], 0.5 * (edge[0][0] + edge[1][0]));
            ASSERT_DOUBLE_EQ(transformed_end_point[0], edge[1][0]);
        } else if (std::is_same<T, float>::value) {
            ASSERT_FLOAT_EQ(transformed_start_point[0], edge[0][0]);
            ASSERT_FLOAT_EQ(transformed_mid_point[0], 0.5 * (edge[0][0] + edge[1][0]));
            ASSERT_FLOAT_EQ(transformed_end_point[0], edge[1][0]);
        } else {
            FAIL();
        }
    }
}

TYPED_TEST(EdgeElementTest, GlobalToLocal) {
    using T                               = typename TestFixture::value_t;
    using point_t                         = typename TestFixture::point_t;
    using vertex_points_t = typename TestFixture::vertex_points_t;

    auto& edge_element = this->edge_element;

    for (const vertex_points_t& edge : this->edges) {
        point_t transformed_start_point = edge_element.globalToLocal(edge, edge[0]);
        point_t transformed_mid_point = edge_element.globalToLocal(edge, 0.5 * (edge[0] + edge[1]));
        point_t transformed_end_point = edge_element.globalToLocal(edge, edge[1]);

        if (std::is_same<T, double>::value) {
            ASSERT_DOUBLE_EQ(transformed_start_point[0], this->local_points[0][0]);
            ASSERT_DOUBLE_EQ(transformed_mid_point[0], this->local_mid_point[0]);
            ASSERT_DOUBLE_EQ(transformed_end_point[0], this->local_points[1][0]);
        } else if (std::is_same<T, float>::value) {
            ASSERT_FLOAT_EQ(transformed_start_point[0], this->local_points[0][0]);
            ASSERT_FLOAT_EQ(transformed_mid_point[0], this->local_mid_point[0]);
            ASSERT_FLOAT_EQ(transformed_end_point[0], this->local_points[1][0]);
        } else {
            FAIL();
        }
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
