

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
    using vertex_vec_t = ippl::QuadrilateralElement<T>::vertex_vec_t;

    static constexpr unsigned NumQuads = 3;

    QuadrilateralElementTest()
        : rng(Seed) {
        CHECK_SKIP_SERIAL_CONSTRUCTOR;

        const T interval_size = std::numeric_limits<T>::max() / 100.0;

        std::uniform_real_distribution<T> dist(-interval_size / 2.0, interval_size / 2.0);

        for (unsigned i = 0; i < NumQuads; i++) {
            for (unsigned p = 0; p < 4; p++) {
                quads[i][p][0] = dist(rng);
                quads[i][p][1] = dist(rng);
            }
        }
    }

    std::mt19937 rng;

    ippl::QuadrilateralElement<T> quad_element;

    const vertex_vec_t local_points = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const point_t local_mid_point   = {0.5, 0.5};

    ippl::Vector<vertex_vec_t, NumQuads> quads;
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
