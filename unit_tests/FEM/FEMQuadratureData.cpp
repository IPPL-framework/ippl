#include "Ippl.h"

#include "FEM/FEMQuadratureData.h"

#include "gtest/gtest.h"

TEST(FEMQuadratureDataTest, DataAccess) {
    using T = double;
    constexpr unsigned Dim       = 2;
    constexpr unsigned numDOFs   = 4;
    using point_t                = ippl::Vector<T, Dim>;

    ippl::Vector<T, numDOFs> val;
    val[0] = 0.1;
    val[1] = 0.2;
    val[2] = 0.3;
    val[3] = 0.4;

    ippl::Vector<point_t, numDOFs> deriv;
    deriv[0] = point_t(1.0, 0.0);
    deriv[1] = point_t(0.0, 1.0);
    deriv[2] = point_t(-1.0, 2.0);
    deriv[3] = point_t(0.5, -0.5);

    const ippl::QuadratureData<T, point_t, numDOFs> qd{val, deriv};

    EXPECT_DOUBLE_EQ(qd.val_q[0], 0.1);
    EXPECT_DOUBLE_EQ(qd.val_q[2], 0.3);
    EXPECT_DOUBLE_EQ(qd.deriv_q[1](0), 0.0);
    EXPECT_DOUBLE_EQ(qd.deriv_q[1](1), 1.0);
    EXPECT_DOUBLE_EQ(qd.deriv_q[3](0), 0.5);
    EXPECT_DOUBLE_EQ(qd.deriv_q[3](1), -0.5);

    val[2] = 0.35;
    deriv[0] = point_t(2.0, 3.0);
    EXPECT_DOUBLE_EQ(qd.val_q[2], 0.35);
    EXPECT_DOUBLE_EQ(qd.deriv_q[0](0), 2.0);
    EXPECT_DOUBLE_EQ(qd.deriv_q[0](1), 3.0);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
