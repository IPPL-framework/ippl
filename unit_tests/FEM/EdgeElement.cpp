

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
    void SetUp() override { CHECK_SKIP_SERIAL; }

public:
    using value_t                      = T;
    static constexpr unsigned NumEdges = 3;

    EdgeElementTest()
        : rng(Seed) {
        CHECK_SKIP_SERIAL_CONSTRUCTOR;

        const T max_double = std::numeric_limits<T>::max();
        const T min_double = std::numeric_limits<T>::lowest();

        std::uniform_real_distribution<T> dist(min_double, max_double);

        for (unsigned i = 0; i < NumEdges; i++) {
            global_edge_vertices[i][0] = dist(rng);
            global_edge_vertices[i][1] = dist(rng);
        }
    }

    std::mt19937 rng;

    ippl::EdgeElement<T> edge_element;

    const T local_start_point = 0.0;
    const T local_end_point   = 1.0;
    const T local_mid_point   = (local_end_point - local_start_point) / 2.0;

    ippl::Vector<ippl::Vector<T, 2>, NumEdges> global_edge_vertices;
};

using Tests = TestParams::tests<42>;
TYPED_TEST_CASE(EdgeElementTest, Tests);

// TYPED_TEST(EdgeElementTest, Jacobian) {}

// TYPED_TEST(EdgeElementTest, InverseJacobian) {}

TYPED_TEST(EdgeElementTest, LocalToGlobal) {
    using T = typename TestFixture::value_t;

    auto& edge_element = this->edge_element;

    for (unsigned i = 0; i < this->global_edge_vertices.dim; i++) {
        const T global_start_point = this->global_edge_vertices[i][0];
        const T global_end_point   = this->global_edge_vertices[i][1];

        // we need to pass a vector with one element because this is the type
        // that is also used in higher dimensions
        const ippl::Vector<ippl::Vector<T, 1>, 2> global_edge_vertices = {{global_start_point},
                                                                          {global_end_point}};

        ippl::Vector<double, 1> transformed_start_point =
            edge_element.localToGlobal(global_edge_vertices, this->local_start_point);
        ippl::Vector<double, 1> transformed_mid_point =
            edge_element.localToGlobal(global_edge_vertices, this->local_mid_point);
        ippl::Vector<double, 1> transformed_end_point =
            edge_element.localToGlobal(global_edge_vertices, this->local_end_point);

        ASSERT_EQ(transformed_start_point[0], global_start_point);
        ASSERT_EQ(transformed_end_point[0], global_end_point);
    }
}

// TYPED_TEST(EdgeElementTest, GlobalToLocal) {}

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