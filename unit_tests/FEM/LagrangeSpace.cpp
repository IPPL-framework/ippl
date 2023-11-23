

#include "Ippl.h"

#include <functional>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class LagrangeSpaceTest;

template <typename T, typename ExecSpace, unsigned Order, unsigned Dim>
class LagrangeSpaceTest<Parameters<T, ExecSpace, Rank<Order>, Rank<Dim>>> : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t = T;

    using ElementType =
        std::conditional_t<Dim == 1, ippl::EdgeElement<T>, ippl::QuadrilateralElement<T>>;
    // std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    using QuadratureType = ippl::MidpointQuadrature<T, 1, ElementType>;

    LagrangeSpaceTest()
        : meshSizes(3)
        , ref_element()
        , mesh(ippl::NDIndex<Dim>(meshSizes), ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(0.0))
        , quadrature(ref_element)
        , lagrangeSpace(mesh, ref_element, quadrature) {
        // fill the global reference DOFs
    }

    const ippl::Vector<unsigned, Dim> meshSizes;
    const ElementType ref_element;
    const ippl::UniformCartesian<T, Dim> mesh;
    const QuadratureType quadrature;
    const ippl::LagrangeSpace<T, Dim, Order, QuadratureType> lagrangeSpace;
};

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using Orders     = TestParams::Ranks<1>;
using Dimensions = TestParams::Ranks<1, 2>;
using Combos     = CreateCombinations<Precisions, Spaces, Orders, Dimensions>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(LagrangeSpaceTest, Tests);

TYPED_TEST(LagrangeSpaceTest, numGlobalDOFs) {
    const auto& lagrangeSpace = this->lagrangeSpace;
    const std::size_t& dim    = lagrangeSpace.dim;
    const std::size_t& order  = lagrangeSpace.order;

    ASSERT_EQ(lagrangeSpace.numGlobalDOFs(), static_cast<std::size_t>(pow(3.0 * order, dim)));
}

TYPED_TEST(LagrangeSpaceTest, getLocalDOFIndex) {
    const auto& lagrangeSpace = this->lagrangeSpace;
    const std::size_t& dim    = lagrangeSpace.dim;
    const std::size_t& order  = lagrangeSpace.order;

    std::size_t localDOFIndex     = static_cast<unsigned>(-1);
    const std::size_t numElements = (1 << dim);
    const std::size_t numDOFs     = static_cast<unsigned>(pow(3, dim));

    std::vector<std::vector<unsigned>> globalElementDOFs;

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
                    localDOFIndex = lagrangeSpace.getLocalDOFIndex(el_i, dof_i);
                } catch (std::exception& e) {
                    ASSERT_EQ(it, globalElementDOFs[el_i].end());
                }

                if (it != globalElementDOFs[el_i].end()) {
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
    auto& lagrangeSpace      = this->lagrangeSpace;
    const std::size_t& dim   = lagrangeSpace.dim;
    const std::size_t& order = lagrangeSpace.order;

    if (order == 1) {
        if (dim == 1) {
            // start element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 0), 0);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 1), 1);

            // end element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 0), 1);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 1), 2);

        } else if (dim == 2) {
            // lower left element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 0), 0);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 1), 1);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 2), 4);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 3), 3);

            // lower right element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 0), 1);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 1), 2);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 2), 5);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 3), 4);

            // upper left element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 0), 3);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 1), 4);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 2), 7);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 3), 6);

            // upper right element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 0), 4);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 1), 5);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 2), 8);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 3), 7);
        } else {
            FAIL();
        }
    } else {
        FAIL();
    }
}

TYPED_TEST(LagrangeSpaceTest, getLocalDOFIndices) {
    const auto& lagrangeSpace = this->lagrangeSpace;
    // const auto& dim = lagrangeSpace.dim;
    // const auto& order = lagrangeSpace.order;
    const auto& numElementDOFs = lagrangeSpace.numElementDOFs;

    auto local_dof_indices = lagrangeSpace.getLocalDOFIndices();

    ASSERT_EQ(local_dof_indices.dim, numElementDOFs);
    for (unsigned i = 0; i < numElementDOFs; i++) {
        ASSERT_EQ(local_dof_indices[i], i);
    }
}

TYPED_TEST(LagrangeSpaceTest, getGlobalDOFIndices) {
    auto& lagrangeSpace      = this->lagrangeSpace;
    const std::size_t& dim   = lagrangeSpace.dim;
    const std::size_t& order = lagrangeSpace.order;

    if (dim == 1) {
        auto globalDOFIndices = lagrangeSpace.getGlobalDOFIndices(1);
        if (order == 1) {
            ASSERT_EQ(globalDOFIndices.dim, 2);
            ASSERT_EQ(globalDOFIndices[0], 1);
            ASSERT_EQ(globalDOFIndices[1], 2);
        } else if (order == 2) {
            ASSERT_EQ(globalDOFIndices[0], 3);
            ASSERT_EQ(globalDOFIndices[1], 5);
            ASSERT_EQ(globalDOFIndices[2], 4);
        } else {
            FAIL();
        }
    } else if (dim == 2) {
        auto globalDOFIndices = lagrangeSpace.getGlobalDOFIndices(3);

        if (order == 1) {
            ASSERT_EQ(globalDOFIndices.dim, 4);
            ASSERT_EQ(globalDOFIndices[0], 4);
            ASSERT_EQ(globalDOFIndices[1], 5);
            ASSERT_EQ(globalDOFIndices[2], 8);
            ASSERT_EQ(globalDOFIndices[3], 7);
        } else if (order == 2) {
            ASSERT_EQ(globalDOFIndices[0], 12);
            ASSERT_EQ(globalDOFIndices[1], 14);
            ASSERT_EQ(globalDOFIndices[2], 24);
            ASSERT_EQ(globalDOFIndices[3], 22);
        } else {
            FAIL();
        }
    } else {
        FAIL();
    }
}

TYPED_TEST(LagrangeSpaceTest, evaluateRefElementBasis) {
    auto& lagrangeSpace      = this->lagrangeSpace;
    const std::size_t& dim   = lagrangeSpace.dim;
    const std::size_t& order = lagrangeSpace.order;
    using T                  = typename TestFixture::value_t;

    T tolerance = 1e-7;

    if (order == 1) {
        if (dim == 1) {
            for (T x = 0.0; x < 1.0; x += 0.05) {
                ASSERT_NEAR(lagrangeSpace.evaluateRefElementBasis(0, x), 1.0 - x, tolerance);
                ASSERT_NEAR(lagrangeSpace.evaluateRefElementBasis(1, x), x, tolerance);
            }
        } else if (dim == 2) {
            ippl::Vector<T, lagrangeSpace.dim> point;
            for (T x = 0.0; x < 1.0; x += 0.05) {
                point[0] = x;
                for (T y = 0.0; y < 1.0; y += 0.05) {
                    point[1] = y;
                    ASSERT_NEAR(lagrangeSpace.evaluateRefElementBasis(0, point),
                                x * y - x - y + 1.0, tolerance);
                    ASSERT_NEAR(lagrangeSpace.evaluateRefElementBasis(1, point), x * (1.0 - y),
                                tolerance);
                    ASSERT_NEAR(lagrangeSpace.evaluateRefElementBasis(2, point), x * y, tolerance);
                    ASSERT_NEAR(lagrangeSpace.evaluateRefElementBasis(3, point), y * (1.0 - x),
                                tolerance);
                }
            }

        } else {
            FAIL();
        }
    } else {
        FAIL();
    }
}

TYPED_TEST(LagrangeSpaceTest, evaluateRefElementBasisGradient) {
    auto& lagrangeSpace      = this->lagrangeSpace;
    const std::size_t& dim   = lagrangeSpace.dim;
    const std::size_t& order = lagrangeSpace.order;
    using T                  = typename TestFixture::value_t;

    T tolerance = 1e-7;

    if (order == 1) {
        if (dim == 1) {
            for (T x = 0.0; x < 1.0; x += 0.05) {
                const auto grad_0 = lagrangeSpace.evaluateRefElementBasisGradient(0, x);
                const auto grad_1 = lagrangeSpace.evaluateRefElementBasisGradient(1, x);

                ASSERT_NEAR(grad_0[0], -1.0, tolerance);
                ASSERT_NEAR(grad_1[0], 1.0, tolerance);
            }
        } else if (dim == 2) {
            ippl::Vector<T, lagrangeSpace.dim> point;
            for (T x = 0.0; x < 1.0; x += 0.05) {
                point[0] = x;
                for (T y = 0.0; y < 1.0; y += 0.05) {
                    point[1] = y;

                    const auto grad_0 = lagrangeSpace.evaluateRefElementBasisGradient(0, point);
                    const auto grad_1 = lagrangeSpace.evaluateRefElementBasisGradient(1, point);
                    const auto grad_2 = lagrangeSpace.evaluateRefElementBasisGradient(2, point);
                    const auto grad_3 = lagrangeSpace.evaluateRefElementBasisGradient(3, point);

                    ASSERT_NEAR(grad_0[0], y - 1.0, tolerance);
                    ASSERT_NEAR(grad_0[1], x - 1.0, tolerance);

                    ASSERT_NEAR(grad_1[0], 1.0 - y, tolerance);
                    ASSERT_NEAR(grad_1[1], -x, tolerance);

                    ASSERT_NEAR(grad_2[0], y, tolerance);
                    ASSERT_NEAR(grad_2[1], x, tolerance);

                    ASSERT_NEAR(grad_3[0], -y, tolerance);
                    ASSERT_NEAR(grad_3[1], 1.0 - x, tolerance);
                }
            }
        } else {
            FAIL();
        }
    } else {
        FAIL();
    }
}

TYPED_TEST(LagrangeSpaceTest, evaluateAx) {
    using T = typename TestFixture::value_t;

    auto& lagrangeSpace      = this->lagrangeSpace;
    const std::size_t& dim   = lagrangeSpace.dim;
    const std::size_t& order = lagrangeSpace.order;

    // const std::size_t& dim           = lagrangeSpace.dim;
    const std::size_t numGlobalDOFs = lagrangeSpace.numGlobalDOFs();

    Kokkos::View<T*> x("x", numGlobalDOFs);
    Kokkos::View<T*> z("z", numGlobalDOFs);
    Kokkos::View<T**> A("A_transpose", numGlobalDOFs, numGlobalDOFs);

    // Build the discrete poisson eqation matrix to test the assembly function against
    Kokkos::View<T**> A_ref;
    if (order == 1) {
        if (dim == 1) {
            A_ref = Kokkos::View<T**>("A_ref", numGlobalDOFs, numGlobalDOFs);
            for (std::size_t i = 0; i < numGlobalDOFs; ++i) {
                for (std::size_t j = 0; j < numGlobalDOFs; ++j) {
                    if (i == j) {
                        if (i == 0 || i == numGlobalDOFs - 1) {
                            A_ref(i, j) = 1.0;
                        } else {
                            A_ref(i, j) = 2.0;
                        }
                    } else if (i + 1 == j || j + 1 == i) {
                        A_ref(i, j) = -1.0;
                    } else {
                        A_ref(i, j) = 0.0;
                    }
                }
            }
        } else {
            FAIL();
        }
    } else {
        FAIL();
    }

    for (std::size_t i = 0; i < numGlobalDOFs; ++i) {
        if (i > 0)
            x(i - 1) = 0.0;

        x(i) = 1.0;

        // reset z to zero
        for (std::size_t j = 0; j < numGlobalDOFs; ++j) {
            z(j) = 0.0;
        }

        lagrangeSpace.evaluateAx(x, z);

        // Set the the i-th row-vector of A to z
        for (std::size_t j = 0; j < numGlobalDOFs; ++j) {
            // TODO check if there is a different way in Kokkos to do this
            A(j, i) = z(j);
        }
    }

    std::cout << "A = " << std::endl;
    for (std::size_t i = 0; i < numGlobalDOFs; ++i) {
        for (std::size_t j = 0; j < numGlobalDOFs; ++j) {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "A_ref = " << std::endl;
    for (std::size_t i = 0; i < numGlobalDOFs; ++i) {
        for (std::size_t j = 0; j < numGlobalDOFs; ++j) {
            std::cout << A_ref(i, j) << " ";
        }
        std::cout << std::endl;
    }

    for (std::size_t i = 0; i < numGlobalDOFs; ++i) {
        for (std::size_t j = 0; j < numGlobalDOFs; ++j) {
            ASSERT_NEAR(A(i, j), A_ref(i, j), 1e-7);
        }
        std::cout << std::endl;
    }
}

TYPED_TEST(LagrangeSpaceTest, evaluateLoadVector) {
    FAIL();
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