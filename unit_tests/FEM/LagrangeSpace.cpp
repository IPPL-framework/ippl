

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
        , meshSizes(3)
        , ref_element()
        , mesh(ippl::NDIndex<Dim>(meshSizes), ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(0.0))
        , quadrature(ref_element)
        , lagrange_space(mesh, ref_element, quadrature) {
        CHECK_SKIP_SERIAL_CONSTRUCTOR;

        // fill the global reference DOFs
    }

    std::mt19937 rng;

    const ippl::Vector<unsigned, Dim> meshSizes;
    const ElementType ref_element;
    const ippl::UniformCartesian<T, Dim> mesh;
    const QuadratureType quadrature;
    const ippl::LagrangeSpace<T, Dim, Order, QuadratureType> lagrange_space;
};

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using Orders     = TestParams::Ranks<1>;
using Dimensions = TestParams::Ranks<1, 2>;
using Combos     = CreateCombinations<Precisions, Spaces, Orders, Dimensions>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(LagrangeSpaceTest, Tests);

TYPED_TEST(LagrangeSpaceTest, getLocalDOFIndex) {
    const auto& lagrange_space = this->lagrange_space;
    const std::size_t& dim     = lagrange_space.dim;
    const std::size_t& order   = lagrange_space.order;

    std::size_t localDOFIndex     = static_cast<unsigned>(-1);
    const std::size_t numElements = (1 << dim);
    const std::size_t numDOFs     = static_cast<unsigned>(pow(3, dim));

    std::vector<std::vector<unsigned>> globalElementDOFs;

    std::cout << "Dim: " << dim << std::endl;

    if (dim == 1) {
        globalElementDOFs = {// Element 0
                             {0, 1},
                             // Element 1
                             {1, 2}};
    } else if (dim == 2) {
        globalElementDOFs = {// Element 0
                             {0, 1, 4, 3},
                             // Element 1
                             {1, 2, 5, 4},
                             // Element 2
                             {3, 4, 7, 6},
                             // Element 3
                             {4, 5, 8, 7}};
    } else {
        // This dimension was not handled
        FAIL();
    }

    if (order == 1) {
        for (std::size_t el_i = 0; el_i < numElements; el_i++) {
            for (std::size_t dof_i = 0; dof_i < numDOFs; dof_i++) {
                const auto it = std::find(globalElementDOFs[el_i].begin(),
                                          globalElementDOFs[el_i].end(), dof_i);

                const std::size_t index = it - globalElementDOFs[el_i].begin();

                try {
                    localDOFIndex = lagrange_space.getLocalDOFIndex(el_i, dof_i);
                } catch (std::exception& e) {
                    std::cout << "Element " << el_i << " does not contain DOF " << dof_i
                              << std::endl;
                    ASSERT_EQ(it, globalElementDOFs[el_i].end());
                }

                if (it != globalElementDOFs[el_i].end()) {
                    std::cout << "Found DOF " << dof_i << " in element " << el_i << std::endl;
                    ASSERT_EQ(localDOFIndex, index);
                }
            }
        }
    } else {
        // This order was not handled
        FAIL();
    }
}

TYPED_TEST(LagrangeSpaceTest, getGlobalDOFIndex) {
    auto& lagrange_space     = this->lagrange_space;
    const std::size_t& dim   = lagrange_space.dim;
    const std::size_t& order = lagrange_space.order;

    if (order == 1) {
        if (dim == 1) {
            // start element
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(0, 0), 0);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(0, 1), 1);

            // end element
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(1, 0), 1);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(1, 1), 2);

        } else if (dim == 2) {
            // lower left element
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(0, 0), 0);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(0, 1), 1);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(0, 2), 4);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(0, 3), 3);

            // lower right element
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(1, 0), 1);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(1, 1), 2);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(1, 2), 5);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(1, 3), 4);

            // upper left element
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(2, 0), 3);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(2, 1), 4);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(2, 2), 7);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(2, 3), 6);

            // upper right element
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(3, 0), 4);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(3, 1), 5);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(3, 2), 8);
            ASSERT_EQ(lagrange_space.getGlobalDOFIndex(3, 3), 7);
        } else {
            FAIL();
        }
    } else {
        FAIL();
    }
}

TYPED_TEST(LagrangeSpaceTest, getLocalDOFIndices) {
    const auto& lagrange_space = this->lagrange_space;
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
            ASSERT_EQ(global_dof_indices.dim, 2);
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
            ASSERT_EQ(global_dof_indices.dim, 4);
            ASSERT_EQ(global_dof_indices[0], 4);
            ASSERT_EQ(global_dof_indices[1], 5);
            ASSERT_EQ(global_dof_indices[2], 8);
            ASSERT_EQ(global_dof_indices[3], 7);
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