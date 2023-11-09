

#include "Ippl.h"

#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class LagrangeSpaceTest;

template <typename T, typename ExecSpace, unsigned Order, unsigned Dim>
class LagrangeSpaceTest<Parameters<T, ExecSpace, Rank<Order>, Rank<Dim>>> : public ::testing::Test {
protected:
    void SetUp() override { CHECK_SKIP_SERIAL; }

public:
    using value_t = T;

    using ElementType =
        std::conditional_t<Dim == 1, ippl::EdgeElement<T>, ippl::QuadrilateralElement<T>>;
    // std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    using QuadratureType = ippl::MidpointQuadrature<T, 1, ElementType>;

    LagrangeSpaceTest()
        : rng(42)
        , ref_element()
        , mesh(ippl::LagrangeSpace<T, Dim, Order, QuadratureType>::makeNDIndex(
                   ippl::Vector<unsigned, Dim>(2u)),
               ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(0.0))
        , quadrature(ref_element)
        , lagrange_space(mesh, ref_element, quadrature) {
        CHECK_SKIP_SERIAL_CONSTRUCTOR;

        // fill the global reference DOFs
    }

    std::mt19937 rng;

    const ElementType ref_element;
    const ippl::UniformCartesian<T, Dim> mesh;
    const QuadratureType quadrature;
    ippl::LagrangeSpace<T, Dim, Order, QuadratureType> lagrange_space;
};

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using Orders     = TestParams::Ranks<1>;
using Dimensions = TestParams::Ranks<1, 2>;
using Combos     = CreateCombinations<Precisions, Spaces, Orders, Dimensions>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(LagrangeSpaceTest, Tests);

TYPED_TEST(LagrangeSpaceTest, getLocalDOFIndex) {
    FAIL();
}

TYPED_TEST(LagrangeSpaceTest, getGlobalDOFIndex) {
    FAIL();
}

TYPED_TEST(LagrangeSpaceTest, getLocalDOFIndices) {
    auto& lagrange_space = this->lagrange_space;
    // const auto& dim = lagrange_space.dim;
    // const auto& order = lagrange_space.order;
    const auto& numElementDOFs = lagrange_space.numElementDOFs;

    auto local_dof_indices = lagrange_space.getLocalDOFIndices();

    ASSERT_EQ(local_dof_indices.dim, numElementDOFs);
    for (unsigned i = 0; i < numElementDOFs; i++) {
        ASSERT_EQ(local_dof_indices[i], i);
    }
}

TYPED_TEST(LagrangeSpaceTest, getGlobalDOFIndices) {
    auto& lagrange_space     = this->lagrange_space;
    const std::size_t& dim   = lagrange_space.dim;
    const std::size_t& order = lagrange_space.order;

    if (dim == 1) {
        auto global_dof_indices = lagrange_space.getGlobalDOFIndices(1);
        if (order == 1) {
            ASSERT_EQ(global_dof_indices[0], 1);
            ASSERT_EQ(global_dof_indices[1], 2);
        } else if (order == 2) {
            ASSERT_EQ(global_dof_indices[0], 3);
            ASSERT_EQ(global_dof_indices[1], 5);
            ASSERT_EQ(global_dof_indices[2], 4);
        } else {
            FAIL();
        }
    } else if (dim == 2) {
        auto global_dof_indices = lagrange_space.getGlobalDOFIndices(3);
        if (order == 1) {
            ASSERT_EQ(global_dof_indices[0], 4);
            ASSERT_EQ(global_dof_indices[1], 5);
            ASSERT_EQ(global_dof_indices[2], 7);
            ASSERT_EQ(global_dof_indices[3], 8);
        } else if (order == 2) {
            ASSERT_EQ(global_dof_indices[0], 12);
            ASSERT_EQ(global_dof_indices[1], 14);
            ASSERT_EQ(global_dof_indices[2], 24);
            ASSERT_EQ(global_dof_indices[3], 22);
        } else {
            FAIL();
        }
    } else {
        FAIL();
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