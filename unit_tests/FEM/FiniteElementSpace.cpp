
#include "Ippl.h"

#include <limits>
#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class FiniteElementSpaceTest;

template <typename T, typename ExecSpace, unsigned Dim>
class FiniteElementSpaceTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
protected:
    void SetUp() override { CHECK_SKIP_SERIAL; }

public:
    using value_t = T;

    using ElementType =
        std::conditional_t<Dim == 1, ippl::EdgeElement<T>, ippl::QuadrilateralElement<T>>;
    // std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    using QuadratureType = ippl::MidpointQuadrature<T, 1, ElementType>;

    // Initialize a 2x2 mesh with 1.0 spacing and 0.0 offset.
    const ippl::Vector<unsigned, Dim> sizes = ippl::Vector<unsigned, Dim>(3u);
    // 3 nodes in each dimension, or 2 elements in each dimension
    const ippl::NDIndex<Dim> meshSize = ippl::NDIndex<Dim>(sizes);

    FiniteElementSpaceTest()
        : rng(42)
        , ref_element()
        , mesh(meshSize, ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(0.0))
        , quadrature(ref_element)
        , fem_space(mesh, ref_element, quadrature) {
        CHECK_SKIP_SERIAL_CONSTRUCTOR;
    }

    std::mt19937 rng;

    const ElementType ref_element;
    const ippl::UniformCartesian<T, Dim> mesh;
    const QuadratureType quadrature;
    const ippl::LagrangeSpace<T, Dim, 1, QuadratureType> fem_space;
};

using Tests = TestParams::tests<1, 2>;  // TODO add dim 3
TYPED_TEST_CASE(FiniteElementSpaceTest, Tests);

TYPED_TEST(FiniteElementSpaceTest, numElements) {
    const auto& fem_space  = this->fem_space;
    const auto& mesh       = this->mesh;
    const std::size_t& dim = fem_space.dim;

    EXPECT_GE(mesh.getGridsize(0) - 1u, 2u);

    unsigned num_elements = 1;
    for (unsigned d = 0; d < dim; ++d) {
        num_elements *= mesh.getGridsize(d) - 1u;
    }

    EXPECT_EQ(fem_space.numElements(), num_elements);
}

TYPED_TEST(FiniteElementSpaceTest, numElementsInDim) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    for (std::size_t d = 0; d < dim; ++d) {
        EXPECT_EQ(fem_space.numElementsInDim(d), 2u);
    }
}

TYPED_TEST(FiniteElementSpaceTest, getMeshVertexNDIndex) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    const std::size_t vertex_index = static_cast<double>(pow(3.0, static_cast<double>(dim))) - 1;
    const auto vertex_nd_index     = fem_space.getMeshVertexNDIndex(vertex_index);

    // std::cout << "Expected NDIndex for vertex " << vertex_index << ":\n";
    // std::cout << "[ ";
    // for (std::size_t d = 0; d < dim; ++d) {
    //     std::cout << 2 << " ";
    // }
    // std::cout << "]\n";

    // std::cout << "Computed NDIndex for vertex " << vertex_index << "\n";
    // std::cout << "[ ";
    // for (std::size_t d = 0; d < dim; ++d) {
    //     std::cout << vertex_nd_index[d] << " ";
    // }
    // std::cout << "]\n";

    ASSERT_EQ(vertex_nd_index.dim, dim);
    for (std::size_t d = 0; d < dim; ++d) {
        EXPECT_EQ(vertex_nd_index[d], 2u);
    }
}

TYPED_TEST(FiniteElementSpaceTest, getMeshVertexIndex) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    const std::size_t expected_vertex_index =
        static_cast<double>(pow(3.0, static_cast<double>(dim))) - 1;
    const ippl::Vector<std::size_t, fem_space.dim> vertex_nd_index(2);
    const std::size_t computed_vertex_nd_index = fem_space.getMeshVertexIndex(vertex_nd_index);

    EXPECT_EQ(computed_vertex_nd_index, expected_vertex_index);
}

TYPED_TEST(FiniteElementSpaceTest, getElementMeshVertexNDIndices) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    if (dim == 1) {
        const auto indices = fem_space.getElementMeshVertexNDIndices(1);
        ASSERT_EQ(indices.dim, 2);
        ASSERT_EQ(indices[0][0], 1);
        ASSERT_EQ(indices[1][0], 2);
    } else if (dim == 2) {
        const unsigned element_index = 3;
        const ippl::Vector<unsigned, fem_space.dim> element_indices =
            fem_space.getElementNDIndex(element_index);  // {1, 1}
        const auto indices = fem_space.getElementMeshVertexNDIndices(element_indices);

        // std::cout << "Expected indices:\n";
        // std::cout << 7 << " - " << 8 << "\n";
        // std::cout << "| " << element_index << " |\n";
        // std::cout << 4 << " - " << 5 << "\n";

        // std::cout << "Computed indices:\n";
        // std::cout << indices[2][0] << " - " << indices[3][0] << "\n";
        // std::cout << "| " << element_index << " |\n";
        // std::cout << indices[0][0] << " - " << indices[1][0] << "\n";

        ASSERT_EQ(indices.dim, 4);
        ASSERT_EQ(indices[0][0], 1);
        ASSERT_EQ(indices[0][1], 1);

        ASSERT_EQ(indices[1][0], 2);
        ASSERT_EQ(indices[1][1], 1);

        ASSERT_EQ(indices[2][0], 1);
        ASSERT_EQ(indices[2][1], 2);

        ASSERT_EQ(indices[3][0], 2);
        ASSERT_EQ(indices[3][1], 2);
    }
}

TYPED_TEST(FiniteElementSpaceTest, getElementNDIndex) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    const std::size_t element_index = (1 << dim) - 1;  // 2^dim - 1

    const auto element_nd_index = fem_space.getElementNDIndex(element_index);

    std::cout << "Expected NDIndex for element " << element_index << ":\n";
    std::cout << "[ ";
    for (std::size_t d = 0; d < dim; ++d) {
        std::cout << 1 << " ";
    }
    std::cout << "]\n";

    std::cout << "Computed NDIndex for element " << element_index << ":\n";
    std::cout << "[ ";
    for (std::size_t d = 0; d < dim; ++d) {
        std::cout << element_nd_index[d] << " ";
    }
    std::cout << "]\n";

    ASSERT_EQ(element_nd_index.dim, dim);
    for (std::size_t d = 0; d < dim; ++d) {
        EXPECT_EQ(element_nd_index[d], 1);
    }
}

TYPED_TEST(FiniteElementSpaceTest, getElementMeshVertexIndices) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    if (dim == 1) {
        const auto indices = fem_space.getElementMeshVertexIndices(1);
        ASSERT_EQ(indices.dim, 2);
        ASSERT_EQ(indices[0], 1);
        ASSERT_EQ(indices[1], 2);
    } else if (dim == 2) {
        const unsigned element_index = 3;
        const ippl::Vector<unsigned, fem_space.dim> element_indices =
            fem_space.getElementNDIndex(element_index);  // {1, 1}
        const auto indices = fem_space.getElementMeshVertexIndices(element_indices);

        // std::cout << "Expected indices:\n";
        // std::cout << 7 << " - " << 8 << "\n";
        // std::cout << "| " << element_index << " |\n";
        // std::cout << 4 << " - " << 5 << "\n";

        // std::cout << "Computed indices:\n";
        // std::cout << indices[2] << " - " << indices[3] << "\n";
        // std::cout << "| " << element_index << " |\n";
        // std::cout << indices[0] << " - " << indices[1] << "\n";

        ASSERT_EQ(indices.dim, 4);
        ASSERT_EQ(indices[0], 4);
        ASSERT_EQ(indices[1], 5);
        ASSERT_EQ(indices[2], 7);
        ASSERT_EQ(indices[3], 8);
    } else {
        FAIL();
    }
}

TYPED_TEST(FiniteElementSpaceTest, getElementMeshVertexPoints) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    const auto element_ndindex = ippl::Vector<unsigned, fem_space.dim>(1);

    if (dim == 1) {
        const auto indices = fem_space.getElementMeshVertexPoints(element_ndindex);
        ASSERT_EQ(indices.dim, 2);
        ASSERT_EQ(indices[0][0], 1.0);
        ASSERT_EQ(indices[1][0], 2.0);
    } else if (dim == 2) {
        const auto indices = fem_space.getElementMeshVertexPoints(element_ndindex);

        std::cout << "Expected points:\n";
        std::cout << "(" << 1.0 << "," << 2.0 << ") - (" << 2.0 << "," << 2.0 << ")\n";
        std::cout << "(" << 1.0 << "," << 1.0 << ") - (" << 2.0 << "," << 1.0 << ")\n";

        std::cout << "Computed points:\n";
        std::cout << "(" << indices[2][0] << "," << indices[2][1] << ") - (" << indices[3][0] << ","
                  << indices[3][1] << ")\n";
        std::cout << "(" << indices[0][0] << "," << indices[0][1] << ") - (" << indices[1][0] << ","
                  << indices[1][1] << ")\n";

        ASSERT_EQ(indices.dim, 4);

        ASSERT_EQ(indices[0][0], 1.0);
        ASSERT_EQ(indices[0][1], 1.0);

        ASSERT_EQ(indices[1][0], 2.0);
        ASSERT_EQ(indices[1][1], 1.0);

        ASSERT_EQ(indices[2][0], 1.0);
        ASSERT_EQ(indices[2][1], 2.0);

        ASSERT_EQ(indices[3][0], 2.0);
        ASSERT_EQ(indices[3][1], 2.0);
    } else {
        FAIL();
    }
}