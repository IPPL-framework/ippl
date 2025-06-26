
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
    void SetUp() override {}

public:
    using value_t = T;

    static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");

    using MeshType    = ippl::UniformCartesian<T, Dim>;
    using ElementType = std::conditional_t<
        Dim == 1, ippl::EdgeElement<T>,
        std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    using QuadratureType = ippl::MidpointQuadrature<T, 1, ElementType>;
    using FieldType      = ippl::Field<T, Dim, MeshType, typename MeshType::DefaultCentering>;

    // Initialize a 4x4 mesh with 1.0 spacing and 0.0 offset.
    // 4 nodes in each dimension, or 3 elements in each dimension

    FiniteElementSpaceTest()
        : rng(42)
        , meshSizes(4)
        , ref_element()
        , mesh(ippl::NDIndex<Dim>(meshSizes), ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(-1.0))
        , quadrature(ref_element)
        , fem_space(mesh, ref_element, quadrature,
                    ippl::FieldLayout<Dim>(MPI_COMM_WORLD, ippl::NDIndex<Dim>(meshSizes),
                                           std::array<bool, Dim>{true})) {}

    std::mt19937 rng;

    const ippl::Vector<unsigned, Dim> meshSizes;
    ElementType ref_element;
    MeshType mesh;
    const QuadratureType quadrature;
    const ippl::LagrangeSpace<T, Dim, 1, ElementType, QuadratureType, FieldType, FieldType> fem_space;
};

using Tests = TestParams::tests<2, 3>;
TYPED_TEST_CASE(FiniteElementSpaceTest, Tests);

TYPED_TEST(FiniteElementSpaceTest, numElements) {
    const auto& fem_space  = this->fem_space;
    const auto& meshSizes  = this->meshSizes;
    const std::size_t& dim = fem_space.dim;

    unsigned num_elements = 1;
    for (unsigned d = 0; d < dim; ++d) {
        num_elements *= meshSizes[d] - 1;
    }

    EXPECT_EQ(fem_space.numElements(), num_elements);
}

TYPED_TEST(FiniteElementSpaceTest, numElementsInDim) {
    const auto& fem_space  = this->fem_space;
    const auto& meshSizes  = this->meshSizes;
    const std::size_t& dim = fem_space.dim;

    for (std::size_t d = 0; d < dim; ++d) {
        EXPECT_EQ(fem_space.numElementsInDim(d), meshSizes[d] - 1);
    }
}

TYPED_TEST(FiniteElementSpaceTest, getMeshVertexNDIndex) {
    const auto& fem_space  = this->fem_space;
    const auto& meshSizes  = this->meshSizes;
    const std::size_t& dim = fem_space.dim;

    // compute the number of vertices
    std::size_t num_vertices = 1;
    for (std::size_t d = 0; d < dim; ++d) {
        num_vertices *= meshSizes[d];
    }

    ippl::Vector<std::size_t, fem_space.dim> ndindexCounter(0);

    for (std::size_t vertex_index = 0; vertex_index < num_vertices; ++vertex_index) {
        const auto computed_vertex_ndindex = fem_space.getMeshVertexNDIndex(vertex_index);

        ASSERT_EQ(computed_vertex_ndindex.dim, dim);

        for (std::size_t d = 0; d < dim; ++d) {
            EXPECT_EQ(computed_vertex_ndindex[d], ndindexCounter[d]);

            if (ndindexCounter[d] < meshSizes[d] - 1) {
                ndindexCounter[d] += 1;
                break;
            } else {
                ndindexCounter[d] = 0;
            }
        }
    }
}

TYPED_TEST(FiniteElementSpaceTest, getMeshVertexIndex) {
    const auto& fem_space  = this->fem_space;
    const auto& meshSizes  = this->meshSizes;
    const std::size_t& dim = fem_space.dim;

    // compute the number of vertices
    std::size_t num_vertices = 1;
    for (std::size_t d = 0; d < dim; ++d) {
        num_vertices *= meshSizes[d];
    }

    ippl::Vector<std::size_t, fem_space.dim> ndindexCounter(0);

    for (std::size_t vertex_index = 0; vertex_index < num_vertices; ++vertex_index) {
        const std::size_t computed_vertex_index = fem_space.getMeshVertexIndex(ndindexCounter);

        ASSERT_EQ(vertex_index, computed_vertex_index);

        for (std::size_t d = 0; d < dim; ++d) {
            if (ndindexCounter[d] < meshSizes[d] - 1) {
                ndindexCounter[d] += 1;
                break;
            } else {
                ndindexCounter[d] = 0;
            }
        }
    }
}

TYPED_TEST(FiniteElementSpaceTest, getElementMeshVertexNDIndices) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    if (dim == 1) {
        const auto indices = fem_space.getElementMeshVertexNDIndices(2);
        ASSERT_EQ(indices.dim, 2);
        ASSERT_EQ(indices[0][0], 2);
        ASSERT_EQ(indices[1][0], 3);
    } else if (dim == 2) {
        const unsigned element_index = 8;
        const ippl::Vector<unsigned, fem_space.dim> elementNDIndex =
            fem_space.getElementNDIndex(element_index);  // {2, 2}
        const auto indices = fem_space.getElementMeshVertexNDIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 4);
        ASSERT_EQ(indices[0][0], 2);
        ASSERT_EQ(indices[0][1], 2);

        ASSERT_EQ(indices[1][0], 3);
        ASSERT_EQ(indices[1][1], 2);

        ASSERT_EQ(indices[2][0], 2);
        ASSERT_EQ(indices[2][1], 3);

        ASSERT_EQ(indices[3][0], 3);
        ASSERT_EQ(indices[3][1], 3);
    }
}

TYPED_TEST(FiniteElementSpaceTest, getElementNDIndex) {
    const auto& fem_space          = this->fem_space;
    const std::size_t& dim         = fem_space.dim;
    const std::size_t& numElements = fem_space.numElements();

    std::vector<std::vector<std::size_t>> element_nd_indices(dim);

    if (dim == 1) {
        element_nd_indices = {{0}, {1}, {2}};
    } else if (dim == 2) {
        element_nd_indices = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1},
                              {2, 1}, {0, 2}, {1, 2}, {2, 2}};
    } else if (dim == 3) {
        element_nd_indices = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0},
                              {0, 2, 0}, {1, 2, 0}, {2, 2, 0}, {0, 0, 1}, {1, 0, 1}, {2, 0, 1},
                              {0, 1, 1}, {1, 1, 1}, {2, 1, 1}, {0, 2, 1}, {1, 2, 1}, {2, 2, 1},
                              {0, 0, 2}, {1, 0, 2}, {2, 0, 2}, {0, 1, 2}, {1, 1, 2}, {2, 1, 2},
                              {0, 2, 2}, {1, 2, 2}, {2, 2, 2}};
    } else {
        FAIL();
    }

    ippl::Vector<std::size_t, fem_space.dim> element_nd_index;

    for (std::size_t i = 0; i < numElements; ++i) {
        element_nd_index = fem_space.getElementNDIndex(i);

        ASSERT_EQ(element_nd_index.dim, dim);
        ASSERT_EQ(element_nd_index.dim, element_nd_indices.at(i).size());

        for (std::size_t d = 0; d < dim; ++d) {
            EXPECT_EQ(element_nd_index[d], element_nd_indices.at(i).at(d));
        }
    }
}

TYPED_TEST(FiniteElementSpaceTest, getElementIndex) {
    const auto& fem_space          = this->fem_space;
    constexpr size_t dim           = fem_space.dim;
    const std::size_t& numElements = fem_space.numElements();

    std::vector<ippl::Vector<size_t, dim>> element_nd_indices(numElements);

    if (dim == 1) {
        element_nd_indices = {{0}, {1}, {2}};
    } else if (dim == 2) {
        element_nd_indices = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1},
                              {2, 1}, {0, 2}, {1, 2}, {2, 2}};
    } else if (dim == 3) {
        element_nd_indices = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0},
                              {0, 2, 0}, {1, 2, 0}, {2, 2, 0}, {0, 0, 1}, {1, 0, 1}, {2, 0, 1},
                              {0, 1, 1}, {1, 1, 1}, {2, 1, 1}, {0, 2, 1}, {1, 2, 1}, {2, 2, 1},
                              {0, 0, 2}, {1, 0, 2}, {2, 0, 2}, {0, 1, 2}, {1, 1, 2}, {2, 1, 2},
                              {0, 2, 2}, {1, 2, 2}, {2, 2, 2}};
    } else {
        FAIL();
    }

    for (std::size_t i = 0; i < numElements; ++i) {
        int index = fem_space.getElementIndex(element_nd_indices[i]);
        EXPECT_EQ(i, index);
    }
}

TYPED_TEST(FiniteElementSpaceTest, getElementMeshVertexIndices) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    ippl::Vector<std::size_t, fem_space.dim> elementNDIndex;

    if (dim == 1) {
        // start element
        elementNDIndex[0] = 0;
        auto indices      = fem_space.getElementMeshVertexIndices(elementNDIndex);
        ASSERT_EQ(indices.dim, 2);
        ASSERT_EQ(indices[0], 0);
        ASSERT_EQ(indices[1], 1);

        // end element
        elementNDIndex[0] = 2;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);
        ASSERT_EQ(indices.dim, 2);
        ASSERT_EQ(indices[0], 2);
        ASSERT_EQ(indices[1], 3);
    } else if (dim == 2) {
        // bottom left element
        elementNDIndex[0] = 0;
        elementNDIndex[1] = 0;
        auto indices      = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 4);
        ASSERT_EQ(indices[0], 0);
        ASSERT_EQ(indices[1], 1);
        ASSERT_EQ(indices[2], 4);
        ASSERT_EQ(indices[3], 5);

        // bottom right element
        elementNDIndex[0] = 2;
        elementNDIndex[1] = 0;

        indices = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 4);
        ASSERT_EQ(indices[0], 2);
        ASSERT_EQ(indices[1], 3);
        ASSERT_EQ(indices[2], 6);
        ASSERT_EQ(indices[3], 7);

        // top left element
        elementNDIndex[0] = 0;
        elementNDIndex[1] = 2;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 4);
        ASSERT_EQ(indices[0], 8);
        ASSERT_EQ(indices[1], 9);
        ASSERT_EQ(indices[2], 12);
        ASSERT_EQ(indices[3], 13);

        // top right element
        elementNDIndex[0] = 2;
        elementNDIndex[1] = 2;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 4);
        ASSERT_EQ(indices[0], 10);
        ASSERT_EQ(indices[1], 11);
        ASSERT_EQ(indices[2], 14);
        ASSERT_EQ(indices[3], 15);
    } else if (dim == 3) {
        // bottom left front element
        elementNDIndex[0] = 0;
        elementNDIndex[1] = 0;
        elementNDIndex[2] = 0;
        auto indices      = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 8);
        ASSERT_EQ(indices[0], 0);
        ASSERT_EQ(indices[1], 1);
        ASSERT_EQ(indices[2], 4);
        ASSERT_EQ(indices[3], 5);
        ASSERT_EQ(indices[4], 16);
        ASSERT_EQ(indices[5], 17);
        ASSERT_EQ(indices[6], 20);
        ASSERT_EQ(indices[7], 21);

        // bottom right front element
        elementNDIndex[0] = 2;
        elementNDIndex[1] = 0;
        elementNDIndex[2] = 0;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 8);
        ASSERT_EQ(indices[0], 2);
        ASSERT_EQ(indices[1], 3);
        ASSERT_EQ(indices[2], 6);
        ASSERT_EQ(indices[3], 7);
        ASSERT_EQ(indices[4], 18);
        ASSERT_EQ(indices[5], 19);
        ASSERT_EQ(indices[6], 22);
        ASSERT_EQ(indices[7], 23);

        // bottom left back element
        elementNDIndex[0] = 0;
        elementNDIndex[1] = 2;
        elementNDIndex[2] = 0;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 8);
        ASSERT_EQ(indices[0], 8);
        ASSERT_EQ(indices[1], 9);
        ASSERT_EQ(indices[2], 12);
        ASSERT_EQ(indices[3], 13);
        ASSERT_EQ(indices[4], 24);
        ASSERT_EQ(indices[5], 25);
        ASSERT_EQ(indices[6], 28);
        ASSERT_EQ(indices[7], 29);

        // bottom right back element
        elementNDIndex[0] = 2;
        elementNDIndex[1] = 2;
        elementNDIndex[2] = 0;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 8);
        ASSERT_EQ(indices[0], 10);
        ASSERT_EQ(indices[1], 11);
        ASSERT_EQ(indices[2], 14);
        ASSERT_EQ(indices[3], 15);
        ASSERT_EQ(indices[4], 26);
        ASSERT_EQ(indices[5], 27);
        ASSERT_EQ(indices[6], 30);
        ASSERT_EQ(indices[7], 31);

        // top left front element
        elementNDIndex[0] = 0;
        elementNDIndex[1] = 0;
        elementNDIndex[2] = 2;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 8);
        ASSERT_EQ(indices[0], 32);
        ASSERT_EQ(indices[1], 33);
        ASSERT_EQ(indices[2], 36);
        ASSERT_EQ(indices[3], 37);
        ASSERT_EQ(indices[4], 48);
        ASSERT_EQ(indices[5], 49);
        ASSERT_EQ(indices[6], 52);
        ASSERT_EQ(indices[7], 53);

        // top right front element
        elementNDIndex[0] = 2;
        elementNDIndex[1] = 0;
        elementNDIndex[2] = 2;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 8);
        ASSERT_EQ(indices[0], 34);
        ASSERT_EQ(indices[1], 35);
        ASSERT_EQ(indices[2], 38);
        ASSERT_EQ(indices[3], 39);
        ASSERT_EQ(indices[4], 50);
        ASSERT_EQ(indices[5], 51);
        ASSERT_EQ(indices[6], 54);
        ASSERT_EQ(indices[7], 55);

        // top left back element
        elementNDIndex[0] = 0;
        elementNDIndex[1] = 2;
        elementNDIndex[2] = 2;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 8);
        ASSERT_EQ(indices[0], 40);
        ASSERT_EQ(indices[1], 41);
        ASSERT_EQ(indices[2], 44);
        ASSERT_EQ(indices[3], 45);
        ASSERT_EQ(indices[4], 56);
        ASSERT_EQ(indices[5], 57);
        ASSERT_EQ(indices[6], 60);
        ASSERT_EQ(indices[7], 61);

        // top right back element
        elementNDIndex[0] = 2;
        elementNDIndex[1] = 2;
        elementNDIndex[2] = 2;
        indices           = fem_space.getElementMeshVertexIndices(elementNDIndex);

        ASSERT_EQ(indices.dim, 8);
        ASSERT_EQ(indices[0], 42);
        ASSERT_EQ(indices[1], 43);
        ASSERT_EQ(indices[2], 46);
        ASSERT_EQ(indices[3], 47);
        ASSERT_EQ(indices[4], 58);
        ASSERT_EQ(indices[5], 59);
        ASSERT_EQ(indices[6], 62);
        ASSERT_EQ(indices[7], 63);
    } else {
        FAIL();
    }
}

TYPED_TEST(FiniteElementSpaceTest, getElementMeshVertexPoints) {
    const auto& fem_space  = this->fem_space;
    const std::size_t& dim = fem_space.dim;

    const auto element_ndindex = ippl::Vector<unsigned, fem_space.dim>(2);

    if (dim == 1) {
        const auto indices = fem_space.getElementMeshVertexPoints(element_ndindex);
        ASSERT_EQ(indices.dim, 2);
        ASSERT_EQ(indices[0][0], 1.0);
        ASSERT_EQ(indices[1][0], 2.0);
    } else if (dim == 2) {
        const auto indices = fem_space.getElementMeshVertexPoints(element_ndindex);

        ASSERT_EQ(indices.dim, 4);

        ASSERT_EQ(indices[0][0], 1.0);
        ASSERT_EQ(indices[0][1], 1.0);

        ASSERT_EQ(indices[1][0], 2.0);
        ASSERT_EQ(indices[1][1], 1.0);

        ASSERT_EQ(indices[2][0], 1.0);
        ASSERT_EQ(indices[2][1], 2.0);

        ASSERT_EQ(indices[3][0], 2.0);
        ASSERT_EQ(indices[3][1], 2.0);
    } else if (dim == 3) {
        const auto indices = fem_space.getElementMeshVertexPoints(element_ndindex);

        ASSERT_EQ(indices.dim, 8);

        ASSERT_EQ(indices[0][0], 1.0);
        ASSERT_EQ(indices[0][1], 1.0);
        ASSERT_EQ(indices[0][2], 1.0);

        ASSERT_EQ(indices[1][0], 2.0);
        ASSERT_EQ(indices[1][1], 1.0);
        ASSERT_EQ(indices[1][2], 1.0);

        ASSERT_EQ(indices[2][0], 1.0);
        ASSERT_EQ(indices[2][1], 2.0);
        ASSERT_EQ(indices[2][2], 1.0);

        ASSERT_EQ(indices[3][0], 2.0);
        ASSERT_EQ(indices[3][1], 2.0);
        ASSERT_EQ(indices[3][2], 1.0);

        ASSERT_EQ(indices[4][0], 1.0);
        ASSERT_EQ(indices[4][1], 1.0);
        ASSERT_EQ(indices[4][2], 2.0);

        ASSERT_EQ(indices[5][0], 2.0);
        ASSERT_EQ(indices[5][1], 1.0);
        ASSERT_EQ(indices[5][2], 2.0);

        ASSERT_EQ(indices[6][0], 1.0);
        ASSERT_EQ(indices[6][1], 2.0);
        ASSERT_EQ(indices[6][2], 2.0);

        ASSERT_EQ(indices[7][0], 2.0);
        ASSERT_EQ(indices[7][1], 2.0);
        ASSERT_EQ(indices[7][2], 2.0);
    } else {
        FAIL();
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
