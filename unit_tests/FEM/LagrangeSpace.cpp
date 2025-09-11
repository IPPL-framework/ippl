

#include "Ippl.h"

#include <functional>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class LagrangeSpaceTest;

template <typename Tlhs, unsigned Dim, unsigned numElemDOFs>
struct EvalFunctor {
    const ippl::Vector<Tlhs, Dim> DPhiInvT;
    const Tlhs absDetDPhi;

    EvalFunctor(ippl::Vector<Tlhs, Dim> DPhiInvT, Tlhs absDetDPhi)
        : DPhiInvT(DPhiInvT)
        , absDetDPhi(absDetDPhi) {}

    KOKKOS_FUNCTION auto operator()(const size_t& i, const size_t& j,
                    const ippl::Vector<ippl::Vector<Tlhs, Dim>, numElemDOFs>& grad_b_q_k) const {
        return dot((DPhiInvT * grad_b_q_k[j]), (DPhiInvT * grad_b_q_k[i])).apply() * absDetDPhi;
    }
};

template <typename T, typename ExecSpace, unsigned Order, unsigned Dim>
class LagrangeSpaceTest<Parameters<T, ExecSpace, Rank<Order>, Rank<Dim>>> : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t = T;
    static constexpr unsigned dim = Dim;

    static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");

    using MeshType    = ippl::UniformCartesian<T, Dim>;
    using ElementType = std::conditional_t<
        Dim == 1, ippl::EdgeElement<T>,
        std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    using QuadratureType       = ippl::MidpointQuadrature<T, 1, ElementType>;
    using BetterQuadratureType = ippl::GaussLegendreQuadrature<T, 5, ElementType>;
    using FieldType            = ippl::Field<T, Dim, MeshType, typename MeshType::DefaultCentering>;
    using BCType               = ippl::BConds<FieldType, Dim>;

    using LagrangeType = ippl::LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldType, FieldType>;
    using LagrangeTypeBetter = ippl::LagrangeSpace<T, Dim, Order, ElementType, BetterQuadratureType, FieldType, FieldType>;

    LagrangeSpaceTest()
        : ref_element()
        , mesh(ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)), ippl::Vector<T, Dim>(1.0),
               ippl::Vector<T, Dim>(0.0))
        , biggerMesh(ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(5)), ippl::Vector<T, Dim>(1.0),
                     ippl::Vector<T, Dim>(0.0))
        , symmetricMesh(ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(5)),
                        ippl::Vector<T, Dim>(0.5), ippl::Vector<T, Dim>(-1.0))
        , quadrature(ref_element)
        , betterQuadrature(ref_element)
        , lagrangeSpace(mesh, ref_element, quadrature,
                        ippl::FieldLayout<Dim>(MPI_COMM_WORLD,
                                               ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)),
                                               std::array<bool, Dim>{true}))
        , lagrangeSpaceBigger(
              biggerMesh, ref_element, quadrature,
              ippl::FieldLayout<Dim>(MPI_COMM_WORLD,
                                     ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(5)),
                                     std::array<bool, Dim>{true}))
        , symmetricLagrangeSpace(
              symmetricMesh, ref_element, betterQuadrature,
              ippl::FieldLayout<Dim>(MPI_COMM_WORLD,
                                     ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(5)),
                                     std::array<bool, Dim>{true})) {
        // fill the global reference DOFs
    }

    ElementType ref_element;
    MeshType mesh;
    MeshType biggerMesh;
    MeshType symmetricMesh;
    const QuadratureType quadrature;
    const BetterQuadratureType betterQuadrature;
    const LagrangeType lagrangeSpace;
    const LagrangeType lagrangeSpaceBigger;
    const LagrangeTypeBetter symmetricLagrangeSpace;
};

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using Orders     = TestParams::Ranks<1>;
using Dimensions = TestParams::Ranks<1, 2, 3>;
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
    } else if (dim == 3) {
        globalElementDOFs = {// Element 0
                             {0, 1, 4, 3, 9, 10, 13, 12},
                             // Element 1
                             {1, 2, 5, 4, 10, 11, 14, 13},
                             // Element 2
                             {3, 4, 7, 6, 12, 13, 16, 15},
                             // Element 3
                             {4, 5, 8, 7, 13, 14, 17, 16},
                             // Element 4
                             {9, 10, 13, 12, 18, 19, 22, 21},
                             // Element 5
                             {10, 11, 14, 13, 19, 20, 23, 22},
                             // Element 6
                             {12, 13, 16, 15, 21, 22, 25, 24},
                             // Element 7
                             {13, 14, 17, 16, 22, 23, 26, 25}};
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
        } else if (dim == 3) {
            // lower left front element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 0), 0);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 1), 1);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 2), 4);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 3), 3);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 4), 9);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 5), 10);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 6), 13);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(0, 7), 12);

            // lower right front element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 0), 1);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 1), 2);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 2), 5);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 3), 4);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 4), 10);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 5), 11);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 6), 14);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(1, 7), 13);

            // upper left front element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 0), 3);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 1), 4);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 2), 7);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 3), 6);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 4), 12);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 5), 13);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 6), 16);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(2, 7), 15);

            // upper right front element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 0), 4);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 1), 5);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 2), 8);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 3), 7);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 4), 13);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 5), 14);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 6), 17);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(3, 7), 16);

            // lower left back element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(4, 0), 9);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(4, 1), 10);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(4, 2), 13);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(4, 3), 12);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(4, 4), 18);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(4, 5), 19);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(4, 6), 22);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(4, 7), 21);

            // lower right back element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(5, 0), 10);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(5, 1), 11);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(5, 2), 14);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(5, 3), 13);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(5, 4), 19);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(5, 5), 20);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(5, 6), 23);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(5, 7), 22);

            // upper left back element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(6, 0), 12);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(6, 1), 13);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(6, 2), 16);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(6, 3), 15);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(6, 4), 21);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(6, 5), 22);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(6, 6), 25);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(6, 7), 24);

            // upper right back element
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(7, 0), 13);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(7, 1), 14);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(7, 2), 17);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(7, 3), 16);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(7, 4), 22);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(7, 5), 23);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(7, 6), 26);
            ASSERT_EQ(lagrangeSpace.getGlobalDOFIndex(7, 7), 25);
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
    } else if (dim == 3) {
        auto globalDOFIndices = lagrangeSpace.getGlobalDOFIndices(7);

        if (order == 1) {
            ASSERT_EQ(globalDOFIndices.dim, 8);
            ASSERT_EQ(globalDOFIndices[0], 13);
            ASSERT_EQ(globalDOFIndices[1], 14);
            ASSERT_EQ(globalDOFIndices[2], 17);
            ASSERT_EQ(globalDOFIndices[3], 16);
            ASSERT_EQ(globalDOFIndices[4], 22);
            ASSERT_EQ(globalDOFIndices[5], 23);
            ASSERT_EQ(globalDOFIndices[6], 26);
            ASSERT_EQ(globalDOFIndices[7], 25);
        } else if (order == 2) {
            ASSERT_EQ(globalDOFIndices[0], 48);
            ASSERT_EQ(globalDOFIndices[1], 50);
            ASSERT_EQ(globalDOFIndices[2], 56);
            ASSERT_EQ(globalDOFIndices[3], 54);
            ASSERT_EQ(globalDOFIndices[4], 72);
            ASSERT_EQ(globalDOFIndices[5], 74);
            ASSERT_EQ(globalDOFIndices[6], 80);
            ASSERT_EQ(globalDOFIndices[7], 78);
        } else {
            FAIL();
        }

    } else {
        FAIL();
    }
}

TYPED_TEST(LagrangeSpaceTest, evaluateRefElementShapeFunction) {
    auto& lagrangeSpace      = this->lagrangeSpace;
    static constexpr std::size_t dim = TestFixture::dim;
    const std::size_t& order = lagrangeSpace.order;
    using T                  = typename TestFixture::value_t;

    T tolerance = std::numeric_limits<T>::epsilon() * 10.0;

    if (order == 1) {
        if (dim == 1) {
            for (T x = 0.0; x < 1.0; x += 0.05) {
                ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(0, x), 1.0 - x,
                            tolerance);
                ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(1, x), x, tolerance);
            }
        } else if (dim == 2) {
            ippl::Vector<T, dim> point;
            for (T x = 0.0; x < 1.0; x += 0.05) {
                point[0] = x;
                for (T y = 0.0; y < 1.0; y += 0.05) {
                    point[1] = y;
                    ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(0, point),
                                (1.0 - x) * (1.0 - y), tolerance);
                    ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(1, point),
                                x * (1.0 - y), tolerance);
                    ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(2, point), x * y,
                                tolerance);
                    ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(3, point),
                                (1.0 - x) * y, tolerance);
                }
            }
        } else if (dim == 3) {
            ippl::Vector<T, dim> point;
            for (T x = 0.0; x < 1.0; x += 0.05) {
                point[0] = x;
                for (T y = 0.0; y < 1.0; y += 0.05) {
                    point[1] = y;
                    for (T z = 0.0; z < 1.0; z += 0.05) {
                        point[2] = z;
                        ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(0, point),
                                    (1.0 - x) * (1.0 - y) * (1.0 - z), tolerance);
                        ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(1, point),
                                    x * (1.0 - y) * (1.0 - z), tolerance);
                        ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(2, point),
                                    x * y * (1.0 - z), tolerance);
                        ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(3, point),
                                    (1.0 - x) * y * (1.0 - z), tolerance);
                        ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(4, point),
                                    (1.0 - x) * (1.0 - y) * z, tolerance);
                        ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(5, point),
                                    x * (1.0 - y) * z, tolerance);
                        ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(6, point),
                                    x * y * z, tolerance);
                        ASSERT_NEAR(lagrangeSpace.evaluateRefElementShapeFunction(7, point),
                                    (1.0 - x) * y * z, tolerance);
                    }
                }
            }
        } else {
            FAIL();
        }
    } else {
        FAIL();
    }
}

TYPED_TEST(LagrangeSpaceTest, evaluateRefElementShapeFunctionGradient) {
    auto& lagrangeSpace      = this->lagrangeSpace;
    static constexpr std::size_t dim = TestFixture::dim;
    const std::size_t& order = lagrangeSpace.order;
    using T                  = typename TestFixture::value_t;

    T tolerance = 1e-7;

    if (order == 1) {
        if (dim == 1) {
            for (T x = 0.0; x < 1.0; x += 0.05) {
                const auto grad_0 = lagrangeSpace.evaluateRefElementShapeFunctionGradient(0, x);
                const auto grad_1 = lagrangeSpace.evaluateRefElementShapeFunctionGradient(1, x);

                ASSERT_NEAR(grad_0[0], -1.0, tolerance);
                ASSERT_NEAR(grad_1[0], 1.0, tolerance);
            }
        } else if (dim == 2) {
            ippl::Vector<T, dim> point;
            for (T x = 0.0; x < 1.0; x += 0.05) {
                point[0] = x;
                for (T y = 0.0; y < 1.0; y += 0.05) {
                    point[1] = y;

                    const auto grad_0 =
                        lagrangeSpace.evaluateRefElementShapeFunctionGradient(0, point);
                    const auto grad_1 =
                        lagrangeSpace.evaluateRefElementShapeFunctionGradient(1, point);
                    const auto grad_2 =
                        lagrangeSpace.evaluateRefElementShapeFunctionGradient(2, point);
                    const auto grad_3 =
                        lagrangeSpace.evaluateRefElementShapeFunctionGradient(3, point);

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
        } else if (dim == 3) {
            ippl::Vector<T, dim> point;
            for (T x = 0.0; x < 1.0; x += 0.05) {
                point[0] = x;
                for (T y = 0.0; y < 1.0; y += 0.05) {
                    point[1] = y;
                    for (T z = 0.0; z < 1.0; z += 0.05) {
                        point[2] = z;

                        const auto grad_0 =
                            lagrangeSpace.evaluateRefElementShapeFunctionGradient(0, point);
                        const auto grad_1 =
                            lagrangeSpace.evaluateRefElementShapeFunctionGradient(1, point);
                        const auto grad_2 =
                            lagrangeSpace.evaluateRefElementShapeFunctionGradient(2, point);
                        const auto grad_3 =
                            lagrangeSpace.evaluateRefElementShapeFunctionGradient(3, point);
                        const auto grad_4 =
                            lagrangeSpace.evaluateRefElementShapeFunctionGradient(4, point);
                        const auto grad_5 =
                            lagrangeSpace.evaluateRefElementShapeFunctionGradient(5, point);
                        const auto grad_6 =
                            lagrangeSpace.evaluateRefElementShapeFunctionGradient(6, point);
                        const auto grad_7 =
                            lagrangeSpace.evaluateRefElementShapeFunctionGradient(7, point);

                        ASSERT_NEAR(grad_0[0], -1.0 * (1.0 - y) * (1.0 - z), tolerance);
                        ASSERT_NEAR(grad_0[1], (1.0 - x) * -1.0 * (1.0 - z), tolerance);
                        ASSERT_NEAR(grad_0[2], (1.0 - x) * (1.0 - y) * -1.0, tolerance);

                        ASSERT_NEAR(grad_1[0], 1.0 * (1.0 - y) * (1.0 - z), tolerance);
                        ASSERT_NEAR(grad_1[1], x * -1.0 * (1.0 - z), tolerance);
                        ASSERT_NEAR(grad_1[2], x * (1.0 - y) * -1.0, tolerance);

                        ASSERT_NEAR(grad_2[0], 1.0 * y * (1.0 - z), tolerance);
                        ASSERT_NEAR(grad_2[1], x * 1.0 * (1.0 - z), tolerance);
                        ASSERT_NEAR(grad_2[2], x * y * -1.0, tolerance);

                        ASSERT_NEAR(grad_3[0], -1.0 * y * (1.0 - z), tolerance);
                        ASSERT_NEAR(grad_3[1], (1.0 - x) * 1.0 * (1.0 - z), tolerance);
                        ASSERT_NEAR(grad_3[2], (1.0 - x) * y * -1.0, tolerance);

                        ASSERT_NEAR(grad_4[0], -1.0 * (1.0 - y) * z, tolerance);
                        ASSERT_NEAR(grad_4[1], (1.0 - x) * -1.0 * z, tolerance);
                        ASSERT_NEAR(grad_4[2], (1.0 - x) * (1.0 - y) * 1.0, tolerance);

                        ASSERT_NEAR(grad_5[0], 1.0 * (1.0 - y) * z, tolerance);
                        ASSERT_NEAR(grad_5[1], x * -1.0 * z, tolerance);
                        ASSERT_NEAR(grad_5[2], x * (1.0 - y) * 1.0, tolerance);

                        ASSERT_NEAR(grad_6[0], 1.0 * y * z, tolerance);
                        ASSERT_NEAR(grad_6[1], x * 1.0 * z, tolerance);
                        ASSERT_NEAR(grad_6[2], x * y * 1.0, tolerance);

                        ASSERT_NEAR(grad_7[0], -1.0 * y * z, tolerance);
                        ASSERT_NEAR(grad_7[1], (1.0 - x) * 1.0 * z, tolerance);
                        ASSERT_NEAR(grad_7[2], (1.0 - x) * y * 1.0, tolerance);
                    }
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
    using T         = typename TestFixture::value_t;
    using FieldType = typename TestFixture::FieldType;
    using BCType    = typename TestFixture::BCType;
    using LagrangeType = typename TestFixture::LagrangeType;

    const auto& refElement           = this->ref_element;
    const auto& lagrangeSpace        = this->lagrangeSpaceBigger;
    auto mesh                        = this->biggerMesh;
    static constexpr std::size_t dim = TestFixture::dim;
    const std::size_t& order         = lagrangeSpace.order;

    if (order == 1) {
        // create layout
        ippl::NDIndex<dim> domain(
            ippl::Vector<unsigned, dim>(mesh.getGridsize(0)));

        // specifies decomposition; here all dimensions are parallel
        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, domain, isParallel);

        FieldType x(mesh, layout, 1);
        FieldType z(mesh, layout, 1);

        // Define boundary conditions
        BCType bcField;
        for (unsigned int i = 0; i < 2 * dim; ++i) {
            bcField[i] = std::make_shared<ippl::ZeroFace<FieldType>>(i);
        }
        x.setFieldBC(bcField);
        z.setFieldBC(bcField);

        // 1. Define the eval function for the evaluateAx function

        const ippl::Vector<std::size_t, dim> zeroNdIndex =
            ippl::Vector<std::size_t, dim>(0);

        // Inverse Transpose Transformation Jacobian
        const ippl::Vector<T, dim> DPhiInvT =
            refElement.getInverseTransposeTransformationJacobian(
                lagrangeSpace.getElementMeshVertexPoints(zeroNdIndex));

        // Absolute value of det Phi_K
        const T absDetDPhi = std::abs(refElement.getDeterminantOfTransformationJacobian(
            lagrangeSpace.getElementMeshVertexPoints(zeroNdIndex)));

        // Poisson equation eval function (based on the weak form)
        EvalFunctor<T, dim, LagrangeType::numElementDOFs> eval(DPhiInvT, absDetDPhi);

        if constexpr (dim == 1) {
            x = 1.25;

            x.fillHalo();
            lagrangeSpace.evaluateLoadVector(x);
            x.fillHalo();

            z = lagrangeSpace.evaluateAx(x, eval);
            z.fillHalo();

            // set up for comparison
            FieldType ref_field(mesh, layout, 1);
            auto view_ref = ref_field.getView();
            auto mirror   = Kokkos::create_mirror_view(view_ref);

            auto ldom     = layout.getLocalNDIndex();

            nestedViewLoop(mirror, 0, [&]<typename... Idx>(const Idx... args) {
                using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                index_type coords[dim] = {args...};

                // global coordinates
                // We don't take into account nghost as this causes
                // coords to be negative, which causes an overflow due
                // to the index type.
                // All below indices for setting the ref_field are 
                // shifted by 1 to include the ghost (applies to all tests).
                for (unsigned int d = 0; d < lagrangeSpace.dim; ++d) {
                    coords[d] += ldom[d].first();
                }

                // reference field
                if ((coords[0] == 2) || (coords[0] == 4)) {
                    mirror(args...) = 1.25;
                } else {
                    mirror(args...) = 0.0;
                }
            });
            Kokkos::fence();

            Kokkos::deep_copy(view_ref, mirror);

            // compare values with reference
            z  = z - ref_field;
            double err = ippl::norm(z);

            ASSERT_NEAR(err, 0.0, 1e-6);
        } else if constexpr (dim == 2) {
            if (ippl::Comm->size() == 1) {
                x = 1.0;

                x.fillHalo();
                lagrangeSpace.evaluateLoadVector(x);
                x.fillHalo();

                z = lagrangeSpace.evaluateAx(x, eval);
                z.fillHalo();

                // set up for comparison
                FieldType ref_field(mesh, layout, 1);
                auto view_ref = ref_field.getView();
                auto mirror   = Kokkos::create_mirror_view(view_ref);

                auto ldom     = layout.getLocalNDIndex();

                nestedViewLoop(mirror, 0, [&]<typename... Idx>(const Idx... args) {
                    using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                    index_type coords[dim] = {args...};

                    // global coordinates
                    for (unsigned int d = 0; d < lagrangeSpace.dim; ++d) {
                        coords[d] += ldom[d].first();
                    }
                    
                    // reference field
                    if (((coords[0] == 2) && (coords[1] == 2)) ||
                        ((coords[0] == 2) && (coords[1] == 4)) ||
                        ((coords[0] == 4) && (coords[1] == 2)) ||
                        ((coords[0] == 4) && (coords[1] == 4))) {
                        mirror(args...) = 1.5;
                    } else if (((coords[0] == 2) && (coords[1] == 3)) ||
                        ((coords[0] == 3) && (coords[1] == 2)) ||
                        ((coords[0] == 3) && (coords[1] == 4)) ||
                        ((coords[0] == 4) && (coords[1] == 3))) {
                        mirror(args...) = 1.0;
                    } else {
                        mirror(args...) = 0.0;
                    }
                });
                Kokkos::fence();

                Kokkos::deep_copy(view_ref, mirror);

                // compare values with reference
                z  = z - ref_field;
                double err = ippl::norm(z);

                ASSERT_NEAR(err, 0.0, 1e-6);
            }
        } else if constexpr (dim == 3) {
            x = 1.5;

            x.fillHalo();
            lagrangeSpace.evaluateLoadVector(x);
            x.fillHalo();

            z = lagrangeSpace.evaluateAx(x, eval);
            z.fillHalo();

            // set up for comparison
            FieldType ref_field(mesh, layout, 1);
            auto view_ref = ref_field.getView();
            auto mirror   = Kokkos::create_mirror_view(view_ref);

            auto ldom     = layout.getLocalNDIndex();

            nestedViewLoop(mirror, 0, [&]<typename... Idx>(const Idx... args) {
                using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                index_type coords[dim] = {args...};

                // global coordinates
                for (unsigned int d = 0; d < lagrangeSpace.dim; ++d) {
                    coords[d] += ldom[d].first();
                }

                // reference field
                if (((coords[0] > 1) && (coords[0] < 5)) && 
                    ((coords[1] > 1) && (coords[1] < 5)) && 
                    ((coords[2] > 1) && (coords[2] < 5))) {
                    
                    mirror(args...) = 2.53125;
                    
                    if ((coords[0] == 3) || (coords[1] == 3) || (coords[2] == 3)) {
                        mirror(args...) = 2.25;
                    }

                    if (((coords[0] == 3) && (coords[1] == 3) && (coords[2] == 2)) ||
                        ((coords[0] == 3) && (coords[1] == 2) && (coords[2] == 3)) ||
                        ((coords[0] == 2) && (coords[1] == 3) && (coords[2] == 3)) ||
                        ((coords[0] == 4) && (coords[1] == 3) && (coords[2] == 3)) ||
                        ((coords[0] == 3) && (coords[1] == 4) && (coords[2] == 3)) ||
                        ((coords[0] == 3) && (coords[1] == 3) && (coords[2] == 4))) {
                        mirror(args...) = 1.5;
                    }
                    
                    if ((coords[0] == 3) && (coords[1] == 3) && (coords[2] == 3)) {
                        mirror(args...) = 0.0;
                    }
                } else {
                    mirror(args...) = 0.0;
                }
            });
            Kokkos::fence();

            Kokkos::deep_copy(view_ref, mirror);

            // compare values with reference
            z  = z - ref_field;
            double err = ippl::norm(z);

            ASSERT_NEAR(err, 0.0, 1e-6);
        } else {
            // only 1D, 2D, 3D supported
            FAIL();
        }
    } else {
        // TODO add higher-order tests when available
        GTEST_SKIP();
    }
}

TYPED_TEST(LagrangeSpaceTest, evaluateLoadVector) {
    using FieldType = typename TestFixture::FieldType;
    using BCType    = typename TestFixture::BCType;

    const auto& lagrangeSpace = this->symmetricLagrangeSpace;
    auto mesh                 = this->symmetricMesh;
    static constexpr std::size_t dim = TestFixture::dim;
    const std::size_t& order  = lagrangeSpace.order;

    if (order == 1) {

        // initialize the RHS field
        ippl::NDIndex<dim> domain(
            ippl::Vector<unsigned, dim>(mesh.getGridsize(0)));

        // specifies decomposition; here all dimensions are parallel
        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, domain, isParallel);

        FieldType rhs_field(mesh, layout, 1);
        FieldType ref_field(mesh, layout, 1);

        // Define boundary conditions
        BCType bcField;
        for (unsigned int i = 0; i < 2 * dim; ++i) {
            bcField[i] = std::make_shared<ippl::ZeroFace<FieldType>>(i);
        }
        rhs_field.setFieldBC(bcField);

        if constexpr (dim == 1) {
            rhs_field = 2.75;

            // call evaluateLoadVector
            rhs_field.fillHalo();
            lagrangeSpace.evaluateLoadVector(rhs_field);
            rhs_field.fillHalo();

            // set up for comparison
            auto view_ref = ref_field.getView();
            auto mirror   = Kokkos::create_mirror_view(view_ref);

            auto ldom     = layout.getLocalNDIndex();

            nestedViewLoop(mirror, 0, [&]<typename... Idx>(const Idx... args) {
                using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                index_type coords[dim] = {args...};

                // global coordinates
                for (unsigned int d = 0; d < lagrangeSpace.dim; ++d) {
                    coords[d] += ldom[d].first();
                }

                // reference field
                switch (coords[0]) {
                    case 1:
                        mirror(args...) = 0.0;
                        break;
                    case 2:
                        mirror(args...) = 1.375;
                        break;
                    case 3:
                        mirror(args...) = 1.375;
                        break;
                    case 4:
                        mirror(args...) = 1.375;
                        break;
                    case 5:
                        mirror(args...) = 0.0;
                        break;
                    default:
                        mirror(args...) = 0.0;
                }
            });
            Kokkos::fence();

            Kokkos::deep_copy(view_ref, mirror);

            // compare values with reference
            rhs_field  = rhs_field - ref_field;
            double err = ippl::norm(rhs_field);

            ASSERT_NEAR(err, 0.0, 1e-6);
        } else if constexpr (dim == 2) {
            rhs_field = 3.5;

            // call evaluateLoadVector
            rhs_field.fillHalo();
            lagrangeSpace.evaluateLoadVector(rhs_field);
            rhs_field.fillHalo();

            // set up for comparison
            auto view_ref = ref_field.getView();
            auto mirror   = Kokkos::create_mirror_view(view_ref);

            auto ldom     = layout.getLocalNDIndex();

            nestedViewLoop(mirror, 0, [&]<typename... Idx>(const Idx... args) {
                using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                index_type coords[dim] = {args...};

                // global coordinates
                for (unsigned int d = 0; d < lagrangeSpace.dim; ++d) {
                    coords[d] += ldom[d].first();
                }

                // reference field
                if ((coords[0] < 2) || (coords[1] < 2) || 
                    (coords[0] > 4) || (coords[1] > 4)) {
                    mirror(args...) = 0.0;
                } else {
                    mirror(args...) = 0.875;
                }
            });
            Kokkos::fence();

            Kokkos::deep_copy(view_ref, mirror);

            // compare values with reference
            rhs_field  = rhs_field - ref_field;
            double err = ippl::norm(rhs_field);

            ASSERT_NEAR(err, 0.0, 1e-6);

        } else if constexpr (dim == 3) {

            rhs_field = 1.25;

            // call evaluateLoadVector
            rhs_field.fillHalo();
            lagrangeSpace.evaluateLoadVector(rhs_field);
            rhs_field.fillHalo();

            // set up for comparison
            auto view_ref = ref_field.getView();
            auto mirror   = Kokkos::create_mirror_view(view_ref);

            auto ldom     = layout.getLocalNDIndex();

            nestedViewLoop(mirror, 0, [&]<typename... Idx>(const Idx... args) {
                using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                index_type coords[dim] = {args...};

                // global coordinates
                for (unsigned int d = 0; d < lagrangeSpace.dim; ++d) {
                    coords[d] += ldom[d].first();
                }

                // reference field
                if ((coords[0] == 1) || (coords[1] == 1) || (coords[2] == 1) ||
                    (coords[0] == 5) || (coords[1] == 5) || (coords[2] == 5)) {
                    mirror(args...) = 0.0;
                } else {
                    mirror(args...) = 0.15625;
                }
            });
            Kokkos::fence();

            Kokkos::deep_copy(view_ref, mirror);

            // compare values with reference
            rhs_field  = rhs_field - ref_field;
            double err = ippl::norm(rhs_field);

            ASSERT_NEAR(err, 0.0, 1e-6);
        } else {
            // only dims 1, 2, 3 supported
            FAIL();
        }
    } else {
        // TODO add higher order unit tests when available
        GTEST_SKIP();
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
