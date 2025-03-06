#include "Ippl.h"

#include <functional>

#include "TestUtils.h"
#include "gtest/gtest.h"


template <typename>
class NedelecSpaceTest;

template <typename T, unsigned Order, unsigned Dim>
class NedelecSpaceTest<Parameters<T, Rank<Order>, Rank<Dim>>> : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t = T;

    static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");
    static_assert(Order == 1, "Currently only order 2 is supported");

    using MeshType    = ippl::UniformCartesian<T, Dim>;
    using ElementType = std::conditional_t<Dim == 1, ippl::EdgeElement<T>,std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, ippl::HexahedralElement<T>>>;

    using QuadratureType = ippl::GaussJacobiQuadrature<T, 5, ElementType>;
    using FieldType            = ippl::Field<T, Dim, MeshType, typename MeshType::DefaultCentering>;
    using BCType               = ippl::BConds<FieldType, Dim>;

    NedelecSpaceTest()
        : ref_element()
        , mesh(ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)), ippl::Vector<T, Dim>(1.0),
               ippl::Vector<T, Dim>(0.0))
        , quadrature(ref_element, 0.0, 0.0)
        , nedelecSpace(mesh, ref_element, quadrature,
                        ippl::FieldLayout<Dim>(MPI_COMM_WORLD,
                                               ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)),
                                               std::array<bool, Dim>{true})) {
        // fill the global reference DOFs
    }

    ElementType ref_element;
    MeshType mesh;
    const QuadratureType quadrature;
    const ippl::NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType, FieldType> nedelecSpace;
};



using Precisions = TestParams::Precisions;
using Orders     = TestParams::Ranks<1>;
using Dimensions = TestParams::Ranks<2, 3>;
using Combos     = CreateCombinations<Precisions, Orders, Dimensions>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(NedelecSpaceTest, Tests);


TYPED_TEST(NedelecSpaceTest, numLocalDOFS) {
  if (this->nedelecSpace.dim == 2) {
    ASSERT_EQ(this->nedelecSpace.numElementDOFs, 4);
  } else{
    ASSERT_EQ(this->nedelecSpace.numElementDOFs, 12);
  }
}

TYPED_TEST(NedelecSpaceTest, numGlobalDOFs) {
  if (this->nedelecSpace.dim == 2) {
    ASSERT_EQ(this->nedelecSpace.numGlobalDOFs(), 12);
  } else if (this->nedelecSpace.dim == 3) {
    ASSERT_EQ(this->nedelecSpace.numGlobalDOFs(), 36+18);
  }
}


TYPED_TEST(NedelecSpaceTest, getGlobalDOFIndices) {
  // Here we just check some random element and see if it is equivalent to a handcrafted one,
  // probably the simplest way of doing this.

  if (this->nedelecSpace.dim == 2) {
    auto idxSet = this->nedelecSpace.getGlobalDOFIndices(1);
    ASSERT_EQ(idxSet[0],1);
    ASSERT_EQ(idxSet[1],3);
    ASSERT_EQ(idxSet[2],8);
    ASSERT_EQ(idxSet[3],10);

    idxSet = this->nedelecSpace.getGlobalDOFIndices(3);
    ASSERT_EQ(idxSet[0],3);
    ASSERT_EQ(idxSet[1],5);
    ASSERT_EQ(idxSet[2],9);
    ASSERT_EQ(idxSet[3],11);
  } else if (this->nedelecSpace.dim == 3) {
    auto idxSet = this->nedelecSpace.getGlobalDOFIndices(7);
    ASSERT_EQ(idxSet[0], 15);
    ASSERT_EQ(idxSet[1], 17);
    ASSERT_EQ(idxSet[2], 27);
    ASSERT_EQ(idxSet[3], 29);
    ASSERT_EQ(idxSet[4], 21);
    ASSERT_EQ(idxSet[5], 23);
    ASSERT_EQ(idxSet[6], 33);
    ASSERT_EQ(idxSet[7], 35);
    ASSERT_EQ(idxSet[8], 45);
    ASSERT_EQ(idxSet[9], 47);
    ASSERT_EQ(idxSet[10],51);
    ASSERT_EQ(idxSet[11],53);
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
