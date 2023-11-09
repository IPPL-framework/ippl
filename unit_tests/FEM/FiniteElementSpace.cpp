
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

    FiniteElementSpaceTest()
        : rng(42) {
        CHECK_SKIP_SERIAL_CONSTRUCTOR;
    }

    std::mt19937 rng;

    ippl::LagrangeSpace<T, Dim, 1, QuadratureType> finite_element_space;
};

using Tests = TestParams::tests<1, 2>;  // TODO add dim 3
TYPED_TEST_CASE(FiniteElementSpaceTest, Tests);

TYPED_TEST(FiniteElementSpaceTest, numElements) {
    FAIL();
}

TYPED_TEST(FiniteElementSpaceTest, numElementsInDim) {
    FAIL();
}

TYPED_TEST(FiniteElementSpaceTest, getMeshVertexNDIndex) {
    FAIL();
}

TYPED_TEST(FiniteElementSpaceTest, getElementNDIndex) {
    FAIL();
}

TYPED_TEST(FiniteElementSpaceTest, getElementMeshVertexIndices) {
    FAIL();
}

TYPED_TEST(FiniteElementSpaceTest, getElementMeshVertexPoints) {
    FAIL();
}