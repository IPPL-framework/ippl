

#include "Ippl.h"

#include <limits>
#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class QuadrilateralElementTest;

template <typename T, typename ExecSpace, unsigned Seed>
class QuadrilateralElementTest<Parameters<T, ExecSpace, Rank<Seed>>> : public ::testing::Test {
protected:
    void SetUp() override { CHECK_SKIP_SERIAL; }

public:
    using value_t      = T;
    using point_t      = ippl::QuadrilateralElement<T>::point_t;
    using mesh_element_vertex_vec_t = ippl::QuadrilateralElement<T>::mesh_element_vertex_vec_t;

    static constexpr unsigned NumQuads = 3;

    QuadrilateralElementTest()
        : rng(Seed) {
        CHECK_SKIP_SERIAL_CONSTRUCTOR;

        const T interval_size = std::numeric_limits<T>::max() / 100.0;

        std::uniform_real_distribution<T> dist(-interval_size / 2.0, interval_size / 2.0);

        for (unsigned i = 0; i < NumQuads; i++) {
            // Determine only two points randomly of the quadrilateral (Since this is only a scaling
            // and translation transformation, and the quad is parallel to the axes).

            // first corner point
            quads[i][0][0] = dist(rng);
            quads[i][0][1] = dist(rng);

            // second corner point
            quads[i][3][0] = dist(rng);
            quads[i][3][1] = dist(rng);

            // Compute the third point
            quads[i][1][0] = quads[i][3][0];
            quads[i][1][1] = quads[i][0][1];

            // Compute the fourth point
            quads[i][2][0] = quads[i][0][0];
            quads[i][2][1] = quads[i][3][1];
        }
    }

    std::mt19937 rng;

    ippl::QuadrilateralElement<T> quad_element;

    const mesh_element_vertex_vec_t local_points = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const point_t local_mid_point   = {0.5, 0.5};

    ippl::Vector<mesh_element_vertex_vec_t, NumQuads> quads;
};

using Tests = TestParams::tests<42>;
TYPED_TEST_CASE(QuadrilateralElementTest, Tests);

TYPED_TEST(QuadrilateralElementTest, LocalVertices) {
    auto& quad_element = this->quad_element;

    for (unsigned i = 0; i < this->local_points.dim; i++) {
        ASSERT_EQ(quad_element.getLocalVertices()[i].dim, 2);
        ASSERT_EQ(quad_element.getLocalVertices()[i][0], this->local_points[i][0]);
    }
}

TYPED_TEST(QuadrilateralElementTest, LocalToGlobal) {
    using T = typename TestFixture::value_t;
    // using point_t      = typename TestFixture::point_t;
    using mesh_element_vertex_vec_t = typename TestFixture::mesh_element_vertex_vec_t;

    auto& quad_element = this->quad_element;

    for (unsigned i = 0; i < this->quads.dim; i++) {
        mesh_element_vertex_vec_t transformed_points;
        for (unsigned p = 0; p < 4; p++) {
            transformed_points[p] =
                quad_element.localToGlobal(this->quads[i], this->local_points[p]);

            if (std::is_same<T, double>::value) {
                ASSERT_DOUBLE_EQ(transformed_points[p][0], this->quads[i][p][0]);
                ASSERT_DOUBLE_EQ(transformed_points[p][1], this->quads[i][p][1]);
            } else if (std::is_same<T, float>::value) {
                ASSERT_FLOAT_EQ(transformed_points[p][0], this->quads[i][p][0]);
                ASSERT_FLOAT_EQ(transformed_points[p][1], this->quads[i][p][1]);
            } else {
                FAIL();
            }
        }
    }
}

TYPED_TEST(QuadrilateralElementTest, GlobalToLocal) {
    using T = typename TestFixture::value_t;
    // using point_t      = typename TestFixture::point_t;
    using mesh_element_vertex_vec_t = typename TestFixture::mesh_element_vertex_vec_t;

    auto& quad_element = this->quad_element;

    for (unsigned i = 0; i < this->quads.dim; i++) {
        mesh_element_vertex_vec_t transformed_points;
        for (unsigned p = 0; p < 4; p++) {
            transformed_points[p] = quad_element.globalToLocal(this->quads[i], this->quads[i][p]);

            if (std::is_same<T, double>::value) {
                ASSERT_DOUBLE_EQ(transformed_points[p][0], this->local_points[p][0]);
                ASSERT_DOUBLE_EQ(transformed_points[p][1], this->local_points[p][1]);
            } else if (std::is_same<T, float>::value) {
                ASSERT_FLOAT_EQ(transformed_points[p][0], this->local_points[p][0]);
                ASSERT_FLOAT_EQ(transformed_points[p][1], this->local_points[p][1]);
            } else {
                FAIL();
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int success = 1;
    TestParams::checkArgs(argc, argv);
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
