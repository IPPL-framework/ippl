
#include "Ippl.h"

#include <limits>
#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class HexahedralElementTest;

template <typename T, typename ExecSpace, unsigned Seed>
class HexahedralElementTest<Parameters<T, ExecSpace, Rank<Seed>>> : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t = T;
    using point_t = typename ippl::HexahedralElement<T>::point_t;
    using vertex_points_t =
        typename ippl::HexahedralElement<T>::vertex_points_t;

    static constexpr unsigned NumHexs = 3;

    HexahedralElementTest()
        : rng(Seed) {
        const T interval_size = 1.0;  // std::numeric_limits<T>::max() / 100.0;

        std::uniform_real_distribution<T> dist(-interval_size / 2.0, interval_size / 2.0);

        for (unsigned i = 0; i < NumHexs; i++) {
            // Determine only two points randomly of the hexahedral (Since this is only a scaling
            // and translation transformation, and the hex is parallel to the axes).

            // Point 0
            hexs[i][0][0] = dist(rng);
            hexs[i][0][1] = dist(rng);
            hexs[i][0][2] = dist(rng);

            // Point 6
            hexs[i][6][0] = dist(rng);
            hexs[i][6][1] = dist(rng);
            hexs[i][6][2] = dist(rng);

            // Point 1
            hexs[i][1][0] = hexs[i][6][0];
            hexs[i][1][1] = hexs[i][0][1];
            hexs[i][1][2] = hexs[i][0][2];

            // Point 2
            hexs[i][2][0] = hexs[i][6][0];
            hexs[i][2][1] = hexs[i][6][1];
            hexs[i][2][2] = hexs[i][0][2];

            // Point 3
            hexs[i][3][0] = hexs[i][0][0];
            hexs[i][3][1] = hexs[i][6][1];
            hexs[i][3][2] = hexs[i][0][2];

            // Point 4
            hexs[i][4][0] = hexs[i][0][0];
            hexs[i][4][1] = hexs[i][0][1];
            hexs[i][4][2] = hexs[i][6][2];

            // Point 5
            hexs[i][5][0] = hexs[i][6][0];
            hexs[i][5][1] = hexs[i][0][1];
            hexs[i][5][2] = hexs[i][6][2];

            // Point 7
            hexs[i][7][0] = hexs[i][0][0];
            hexs[i][7][1] = hexs[i][6][1];
            hexs[i][7][2] = hexs[i][6][2];
        }
    }

    std::mt19937 rng;

    ippl::HexahedralElement<T> hex_element;

    const vertex_points_t local_points = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}};
    const point_t local_mid_point = {0.5, 0.5, 0.5};

    ippl::Vector<vertex_points_t, NumHexs> hexs;
};

using Tests = TestParams::tests<42>;
TYPED_TEST_CASE(HexahedralElementTest, Tests);

TYPED_TEST(HexahedralElementTest, LocalVertices) {
    auto& hex_element = this->hex_element;

    for (unsigned i = 0; i < this->local_points.dim; i++) {
        ASSERT_EQ(hex_element.getLocalVertices()[i].dim, 3);
        ASSERT_EQ(hex_element.getLocalVertices()[i][0], this->local_points[i][0]);
    }
}

TYPED_TEST(HexahedralElementTest, LocalToGlobal) {
    using T = typename TestFixture::value_t;
    // using point_t      = typename TestFixture::point_t;
    using vertex_points_t = typename TestFixture::vertex_points_t;

    auto& hex_element = this->hex_element;

    for (unsigned i = 0; i < 1; i++) {  // TODO this->hexs.dim; i++) {
        vertex_points_t transformed_points;
        for (unsigned p = 0; p < 8; p++) {
            transformed_points[p] = hex_element.localToGlobal(this->hexs[i], this->local_points[p]);

            if (std::is_same<T, double>::value) {
                ASSERT_DOUBLE_EQ(transformed_points[p][0], this->hexs[i][p][0]);
                ASSERT_DOUBLE_EQ(transformed_points[p][1], this->hexs[i][p][1]);
                ASSERT_DOUBLE_EQ(transformed_points[p][2], this->hexs[i][p][2]);
            } else if (std::is_same<T, float>::value) {
                ASSERT_FLOAT_EQ(transformed_points[p][0], this->hexs[i][p][0]);
                ASSERT_FLOAT_EQ(transformed_points[p][1], this->hexs[i][p][1]);
                ASSERT_FLOAT_EQ(transformed_points[p][2], this->hexs[i][p][2]);
            } else {
                FAIL();
            }
        }
    }
}

TYPED_TEST(HexahedralElementTest, GlobalToLocal) {
    using T = typename TestFixture::value_t;

    using vertex_points_t = typename TestFixture::vertex_points_t;

    auto& hex_element = this->hex_element;

    for (unsigned i = 0; i < this->hexs.dim; i++) {
        vertex_points_t transformed_points;
        for (unsigned p = 0; p < 8; p++) {
            transformed_points[p] = hex_element.globalToLocal(this->hexs[i], this->hexs[i][p]);

            if (std::is_same<T, double>::value) {
                ASSERT_DOUBLE_EQ(transformed_points[p][0], this->local_points[p][0]);
                ASSERT_DOUBLE_EQ(transformed_points[p][1], this->local_points[p][1]);
                ASSERT_DOUBLE_EQ(transformed_points[p][2], this->local_points[p][2]);
            } else if (std::is_same<T, float>::value) {
                ASSERT_FLOAT_EQ(transformed_points[p][0], this->local_points[p][0]);
                ASSERT_FLOAT_EQ(transformed_points[p][1], this->local_points[p][1]);
                ASSERT_FLOAT_EQ(transformed_points[p][2], this->local_points[p][2]);
            } else {
                FAIL();
            }
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
