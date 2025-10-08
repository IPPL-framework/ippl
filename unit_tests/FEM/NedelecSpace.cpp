#include "Ippl.h"

#include <functional>

#include "TestUtils.h"
#include "gtest/gtest.h"


template <typename>
class NedelecSpaceTest;

template <typename T, unsigned Dim, unsigned numElementDOFs>
struct DummyFunctor {
    KOKKOS_FUNCTION const T operator()(size_t i, size_t j,
        [[maybe_unused]] const ippl::Vector<ippl::Vector<T, Dim>, numElementDOFs>& curl_b_q_k,
        [[maybe_unused]] const ippl::Vector<ippl::Vector<T, Dim>, numElementDOFs>& val_b_q_k) const {
        return i==j; //val_b_q_k<:i:>.dot(val_b_q_k<:j:>);
    }
};

template <typename T, unsigned Order, unsigned Dim>
class NedelecSpaceTest<Parameters<T, Rank<Order>, Rank<Dim>>> : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t = T;
    static constexpr unsigned dim = Dim;

    static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");
    static_assert(Order == 1, "Currently only order 1 is supported");

    using MeshType    = ippl::UniformCartesian<T, Dim>;
    using ElementType = std::conditional_t<Dim == 1, ippl::EdgeElement<T>,
                            std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>,
                                ippl::HexahedralElement<T>>>;

    using QuadratureType = ippl::GaussJacobiQuadrature<T, 5, ElementType>;
    using FieldType            = ippl::Field<T, Dim, MeshType, typename MeshType::DefaultCentering>;
    using BCType               = ippl::BConds<FieldType, Dim>;
    using Layout               = ippl::FieldLayout<Dim>;

    using NedelecType = ippl::NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>;

    NedelecSpaceTest()
        : ref_element()
        , mesh(ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(5)),
            ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(0.0))
        , meshSmall(ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)),
            ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(0.0))
        , quadrature(ref_element, 0.0, 0.0)
        , layout(MPI_COMM_WORLD, ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(5)),
                                               std::array<bool, Dim>{true})
        , layoutSmall(MPI_COMM_WORLD, ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)),
                                               std::array<bool, Dim>{true})
        , nedelecSpace(mesh, ref_element, quadrature,layout)
        , nedelecSpaceSmall(meshSmall, ref_element, quadrature, layoutSmall) {
        // fill the global reference DOFs
    }

    ElementType ref_element;
    MeshType mesh;
    MeshType meshSmall;
    const QuadratureType quadrature;
    Layout layout;
    Layout layoutSmall;
    const NedelecType nedelecSpace;
    const NedelecType nedelecSpaceSmall;
};



using Precisions = TestParams::Precisions;
using Orders     = TestParams::Ranks<1>;
using Dimensions = TestParams::Ranks<2, 3>;
using Combos     = CreateCombinations<Precisions, Orders, Dimensions>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(NedelecSpaceTest, Tests);


TYPED_TEST(NedelecSpaceTest, numLocalDOFS) {
  const auto& nedelecSpace = this->nedelecSpace;
  int numElementDOFs       = nedelecSpace.numElementDOFs;
  static constexpr std::size_t dim = TestFixture::dim;

  if (dim == 2) {
    EXPECT_EQ(numElementDOFs, 4);
  } else{
    EXPECT_EQ(numElementDOFs, 12);
  }
}

TYPED_TEST(NedelecSpaceTest, numGlobalDOFs) {
    static constexpr std::size_t dim = TestFixture::dim;
    if (dim == 2) {
      EXPECT_EQ(this->nedelecSpace.numGlobalDOFs(), 40);
    } else{
      EXPECT_EQ(this->nedelecSpace.numGlobalDOFs(), 300);
    }
  }


TYPED_TEST(NedelecSpaceTest, getGlobalDOFIndices) {
  // Here we just check some random element and see if it is equivalent to a handcrafted one,
  // probably the simplest way of doing this.
  static constexpr std::size_t dim = TestFixture::dim;

  if (dim == 2) {
    auto idxSet = this->nedelecSpace.getGlobalDOFIndices(3);
    EXPECT_EQ(idxSet[0],3);
    EXPECT_EQ(idxSet[1],7);
    EXPECT_EQ(idxSet[2],12);
    EXPECT_EQ(idxSet[3],8);

    idxSet = this->nedelecSpace.getGlobalDOFIndices(15);
    EXPECT_EQ(idxSet[0],30);
    EXPECT_EQ(idxSet[1],34);
    EXPECT_EQ(idxSet[2],39);
    EXPECT_EQ(idxSet[3],35);
  } else if (dim == 3) {
    auto idxSet = this->nedelecSpaceSmall.getGlobalDOFIndices(7);
    EXPECT_EQ(idxSet[0], 27);
    EXPECT_EQ(idxSet[1], 29);
    EXPECT_EQ(idxSet[2], 32);
    EXPECT_EQ(idxSet[3], 30);
    EXPECT_EQ(idxSet[4], 37);
    EXPECT_EQ(idxSet[5], 38);
    EXPECT_EQ(idxSet[6], 41);
    EXPECT_EQ(idxSet[7], 40);
    EXPECT_EQ(idxSet[8], 48);
    EXPECT_EQ(idxSet[9], 50);
    EXPECT_EQ(idxSet[10],53);
    EXPECT_EQ(idxSet[11],51);
  }

}

TYPED_TEST(NedelecSpaceTest, isDOFOnBoundary) {
    static constexpr std::size_t dim = TestFixture::dim;
    if (dim == 2) {
        // check all the ones which are true
        // south boundary
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(0));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(1));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(2));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(3));
        // north boundary
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(36));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(37));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(38));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(39));
        // west boundary
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(4));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(13));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(22));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(31));
        // east boundary
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(8));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(17));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(26));
        EXPECT_TRUE(this->nedelecSpace.isDOFOnBoundary(35));


        // check the ones which are false
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(5));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(6));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(7));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(9));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(10));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(11));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(12));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(14));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(15));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(16));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(18));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(19));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(20));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(21));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(23));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(24));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(25));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(27));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(28));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(29));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(30));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(32));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(33));
        EXPECT_FALSE(this->nedelecSpace.isDOFOnBoundary(34));
    } else {
        // check all the ones which are true
        // south boundary
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(0));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(12));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(13));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(34));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(35));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(43));
        // north boundary
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(10));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(18));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(19));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(40));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(41));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(53));
        // west boundary
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(2));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(15));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(18));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(23));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(36));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(42));
        // east boundary
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(4));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(9));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(25));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(30));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(35));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(41));
        // ground boundary
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(5));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(6));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(7));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(8));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(9));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(11));
        // space boundary
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(44));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(45));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(48));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(50));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(51));
        EXPECT_TRUE(this->nedelecSpaceSmall.isDOFOnBoundary(52));


        // check some of the ones which are false
        EXPECT_FALSE(this->nedelecSpaceSmall.isDOFOnBoundary(16));
        EXPECT_FALSE(this->nedelecSpaceSmall.isDOFOnBoundary(24));
        EXPECT_FALSE(this->nedelecSpaceSmall.isDOFOnBoundary(26));
        EXPECT_FALSE(this->nedelecSpaceSmall.isDOFOnBoundary(27));
        EXPECT_FALSE(this->nedelecSpaceSmall.isDOFOnBoundary(29));
        EXPECT_FALSE(this->nedelecSpaceSmall.isDOFOnBoundary(37));
    }
}


TYPED_TEST(NedelecSpaceTest, getBoundarySide) {
  static constexpr std::size_t dim = TestFixture::dim;
  if (dim == 2) {
    // check all the ones which are true
    // south boundary
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(0),0);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(1),0);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(2),0);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(3),0);
    // north boundary
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(36),2);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(37),2);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(38),2);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(39),2);
    // west boundary
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(4),1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(13),1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(22),1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(31),1);
    // east boundary
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(8),3);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(17),3);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(26),3);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(35),3);


    // check the ones which are false
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(5),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(6),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(7),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(9),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(10),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(11),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(12),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(14),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(15),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(16),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(18),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(19),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(20),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(21),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(23),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(24),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(25),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(27),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(28),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(29),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(30),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(32),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(33),-1);
    EXPECT_EQ(this->nedelecSpace.getBoundarySide(34),-1);

  } else {
    // check all the ones which are true
    // south boundary
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(13),0);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(21),0);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(22),0);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(34),0);
    // north boundary
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(19),2);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(31),2);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(32),2);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(40),2);
    // west boundary
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(15),1);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(23),1);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(28),1);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(36),1);
    // east boundary
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(17),3);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(25),3);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(30),3);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(38),3);
    // ground boundary
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(3),4);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(5),4);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(6),4);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(8),4);
    // space boundary
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(45),5);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(47),5);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(48),5);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(50),5);

    // check some of the ones which are false
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(16),-1);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(24),-1);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(26),-1);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(27),-1);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(29),-1);
    EXPECT_EQ(this->nedelecSpaceSmall.getBoundarySide(37),-1);
  }
}

TYPED_TEST(NedelecSpaceTest, evaluateRefElementShapeFunction) {
    // This might seem useless to test the reference element shape function
    // values using the same formulae as we are using in the NedelecSpace class
    // to create them, but we do this, as maybe in the future we are changing
    // the implementation in the NedelecSpace class and for this case we still
    // have this test here. So while it right now might not be the most useful
    // thing, it could become useful in the future.
    
    static constexpr std::size_t dim = TestFixture::dim;
    using T = typename TestFixture::value_t;
    using point_t = ippl::Vector<T, dim>;

    point_t point;
    point_t dif;
    T tolerance = std::numeric_limits<T>::epsilon() * 10.0;

    if constexpr (dim == 2) {
        for (T x = 0; x <= 1; x += 0.05) {
            point[0] = x;
            for (T y = 0; y <= 1; y += 0.05) {
                point[1] = y;
                dif = this->nedelecSpace.evaluateRefElementShapeFunction(0, point) - point_t(1-y,0.);
                ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                dif = this->nedelecSpace.evaluateRefElementShapeFunction(1, point) - point_t(0.,1-x);
                ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                dif = this->nedelecSpace.evaluateRefElementShapeFunction(2, point) - point_t(y,0.);
                ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                dif = this->nedelecSpace.evaluateRefElementShapeFunction(3, point) - point_t(0.,x);
                ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

            }
        }
    } else if constexpr (dim == 3) {
        for (T x = 0; x <= 1; x += 0.05) {
            point[0] = x;
            for (T y = 0; y <= 1; y += 0.05) {
                point[1] = y;
                for (T z = 0; z <= 1; z += 0.05) {
                    point[2] = z;
                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(0, point) 
                        - point_t(y*z-y-z+1,0.,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(1, point) 
                        - point_t(0.,x*z-x-z+1,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(2, point) 
                        - point_t(y*(1-z),0.,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(3, point) 
                        - point_t(0.,x*(1-z),0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(4, point) 
                        - point_t(0.,0.,x*y-x-y+1);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(5, point) 
                        - point_t(0.,0.,x*(1-y));
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(6, point) 
                        - point_t(0.,0.,x*y);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(7, point) 
                        - point_t(0.,0.,y*(1-x));
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(8, point) 
                        - point_t(z*(1-y),0.,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(9, point) 
                        - point_t(0.,z*(1-x),0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(10, point) 
                        - point_t(y*z,0.,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunction(11, point) 
                        - point_t(0.,x*z,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                }
            }
        }

    }

}


TYPED_TEST(NedelecSpaceTest, evaluateRefElementShapeFunctionCurl) {
    // This might seem useless to test the reference element shape function curl
    // values using the same formulae as we are using in the NedelecSpace class
    // to create them, but we do this, as maybe in the future we are changing
    // the implementation in the NedelecSpace class and for this case we still
    // have this test here. So while it right now might not be the most useful
    // thing, it could become useful in the future.
    
    using T = typename TestFixture::value_t;
    static constexpr std::size_t dim = TestFixture::dim;
    using point_t = ippl::Vector<T, dim>;

    point_t point;
    point_t dif;
    T tolerance = std::numeric_limits<T>::epsilon() * 10.0;

    if constexpr (dim == 2) {
        for (T x = 0; x <= 1; x += 0.05) {
            point[0] = x;
            for (T y = 0; y <= 1; y += 0.05) {
                point[1] = y;
                dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(0, point)
                    - point_t(1.,0.);
                ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(1, point)
                    - point_t(-1.,0.);
                ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(2, point)
                    - point_t(-1.,0.);
                ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(3, point)
                    - point_t(1.,0.);
                ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

            }
        }
    } else if constexpr (dim == 3) {
        for (T x = 0; x <= 1; x += 0.05) {
            point[0] = x;
            for (T y = 0; y <= 1; y += 0.05) {
                point[1] = y;
                for (T z = 0; z <= 1; z += 0.05) {
                    point[2] = z;
                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(0, point)
                        - point_t(0.,-1+y,1-z);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(1, point)
                        - point_t(1-x,0.,-1+z);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(2, point)
                        - point_t(0.,-y,-1+z);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(3, point)
                        - point_t(x,0.,1-z);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(4, point)
                        - point_t(-1+x,1-y,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(5, point)
                        - point_t(-x,-1+y,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(6, point)
                        - point_t(x,-y,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(7, point)
                        - point_t(1-x,y,0.);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(8, point)
                        - point_t(0.,1-y,z);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(9, point)
                        - point_t(-1+x,0.,-z);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(10, point)
                        - point_t(0.,y,-z);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                    dif = this->nedelecSpace.evaluateRefElementShapeFunctionCurl(11, point)
                        - point_t(-x,0.,z);
                    ASSERT_NEAR(dif.dot(dif), 0.,tolerance);

                }
            }
        }
    }

}





TYPED_TEST(NedelecSpaceTest, createFEMVector) {
    // Note that this test will start to fail in case the implementation how the
    // domain decomposition is done or how the FEMVector entires are ordered is 
    // changed. Also be aware of the domain boundaries, as there the
    // halo cells are not necessarily uniquely defined. All in all this test is
    // custom tailored to the current way we have implemented it and if in the
    // future changes to those two parts are made the test need to be adjusted.
    // Note that currently this test is implemented for using 1, 2, or 3 MPI
    // ranks, if more ranks are used the test is skipped.

    static constexpr std::size_t dim = TestFixture::dim;
    
    if constexpr (dim == 2) {
        auto vec = this->nedelecSpace.createFEMVector();
        vec = ippl::Comm->rank();
        vec.fillHalo();
        vec.accumulateHalo();
        auto view = vec.getView();
        auto hView = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(hView, view);


        if (ippl::Comm->size() == 1) {
            ASSERT_EQ(vec.size(), 84);
            for (size_t i = 0; i < hView.extent(0); ++i) {
                ASSERT_EQ(hView(i),0);
            }
        }
        if (ippl::Comm->size() == 2) {
            if (ippl::Comm->rank() == 0) {
                ASSERT_EQ(vec.size(), 45);

                ASSERT_EQ(hView(0),0);
                ASSERT_EQ(hView(1),0);
                ASSERT_EQ(hView(2),0);
                ASSERT_EQ(hView(3),0);
                ASSERT_EQ(hView(4),0);
                ASSERT_EQ(hView(5),0);
                ASSERT_EQ(hView(6),0);
                ASSERT_EQ(hView(7),0);
                ASSERT_EQ(hView(8),0);
                ASSERT_EQ(hView(9),0);
                ASSERT_EQ(hView(10),0);
                ASSERT_EQ(hView(11),0);
                ASSERT_EQ(hView(12),0);
                ASSERT_EQ(hView(13),1);
                ASSERT_EQ(hView(14),0);
                ASSERT_EQ(hView(15),0);
                ASSERT_EQ(hView(16),0);
                ASSERT_EQ(hView(17),0);
                ASSERT_EQ(hView(18),0);
                ASSERT_EQ(hView(19),0);
                ASSERT_EQ(hView(20),1);
                ASSERT_EQ(hView(21),0);
                ASSERT_EQ(hView(22),0);
                ASSERT_EQ(hView(23),0);
                ASSERT_EQ(hView(24),0);
                ASSERT_EQ(hView(25),0);
                ASSERT_EQ(hView(26),0);
                ASSERT_EQ(hView(27),1);
                ASSERT_EQ(hView(28),0);
                ASSERT_EQ(hView(29),0);
                ASSERT_EQ(hView(30),0);
                ASSERT_EQ(hView(31),0);
                ASSERT_EQ(hView(32),0);
                ASSERT_EQ(hView(33),0);
                ASSERT_EQ(hView(34),1);
                ASSERT_EQ(hView(35),0);
                ASSERT_EQ(hView(36),0);
                ASSERT_EQ(hView(37),0);
                ASSERT_EQ(hView(38),0);
                ASSERT_EQ(hView(39),0);
                ASSERT_EQ(hView(40),0);
                ASSERT_EQ(hView(41),1);
                ASSERT_EQ(hView(42),0);
                ASSERT_EQ(hView(43),0);
                ASSERT_EQ(hView(44),0);

            }
            if (ippl::Comm->rank() == 1) {
                ASSERT_EQ(vec.size(), 58);

                ASSERT_EQ(hView(0),1);
                ASSERT_EQ(hView(1),1);
                ASSERT_EQ(hView(2),1);
                ASSERT_EQ(hView(3),1);
                ASSERT_EQ(hView(4),1);
                ASSERT_EQ(hView(5),1);
                ASSERT_EQ(hView(6),1);
                ASSERT_EQ(hView(7),1);
                ASSERT_EQ(hView(8),1);
                ASSERT_EQ(hView(9),0);
                ASSERT_EQ(hView(10),1);
                ASSERT_EQ(hView(11),1);
                ASSERT_EQ(hView(12),1);
                ASSERT_EQ(hView(13),0);
                ASSERT_EQ(hView(14),2);
                ASSERT_EQ(hView(15),1);
                ASSERT_EQ(hView(16),1);
                ASSERT_EQ(hView(17),1);
                ASSERT_EQ(hView(18),0);
                ASSERT_EQ(hView(19),1);
                ASSERT_EQ(hView(20),1);
                ASSERT_EQ(hView(21),1);
                ASSERT_EQ(hView(22),0);
                ASSERT_EQ(hView(23),2);
                ASSERT_EQ(hView(24),1);
                ASSERT_EQ(hView(25),1);
                ASSERT_EQ(hView(26),1);
                ASSERT_EQ(hView(27),0);
                ASSERT_EQ(hView(28),1);
                ASSERT_EQ(hView(29),1);
                ASSERT_EQ(hView(30),1);
                ASSERT_EQ(hView(31),0);
                ASSERT_EQ(hView(32),2);
                ASSERT_EQ(hView(33),1);
                ASSERT_EQ(hView(34),1);
                ASSERT_EQ(hView(35),1);
                ASSERT_EQ(hView(36),0);
                ASSERT_EQ(hView(37),1);
                ASSERT_EQ(hView(38),1);
                ASSERT_EQ(hView(39),1);
                ASSERT_EQ(hView(40),0);
                ASSERT_EQ(hView(41),2);
                ASSERT_EQ(hView(42),1);
                ASSERT_EQ(hView(43),1);
                ASSERT_EQ(hView(44),1);
                ASSERT_EQ(hView(45),0);
                ASSERT_EQ(hView(46),1);
                ASSERT_EQ(hView(47),1);
                ASSERT_EQ(hView(48),1);
                ASSERT_EQ(hView(49),0);
                ASSERT_EQ(hView(50),2);
                ASSERT_EQ(hView(51),1);
                ASSERT_EQ(hView(52),1);
                ASSERT_EQ(hView(53),1);
                ASSERT_EQ(hView(54),0);
                ASSERT_EQ(hView(55),1);
                ASSERT_EQ(hView(56),1);
                ASSERT_EQ(hView(57),1);

            }
        }
        if (ippl::Comm->size() == 3) {
            if (ippl::Comm->rank() == 0) {
                ASSERT_EQ(vec.size(), 45);

                ASSERT_EQ(hView(0),0);
                ASSERT_EQ(hView(1),0);
                ASSERT_EQ(hView(2),0);
                ASSERT_EQ(hView(3),0);
                ASSERT_EQ(hView(4),0);
                ASSERT_EQ(hView(5),0);
                ASSERT_EQ(hView(6),0);
                ASSERT_EQ(hView(7),0);
                ASSERT_EQ(hView(8),0);
                ASSERT_EQ(hView(9),0);
                ASSERT_EQ(hView(10),0);
                ASSERT_EQ(hView(11),0);
                ASSERT_EQ(hView(12),0);
                ASSERT_EQ(hView(13),1);
                ASSERT_EQ(hView(14),0);
                ASSERT_EQ(hView(15),0);
                ASSERT_EQ(hView(16),0);
                ASSERT_EQ(hView(17),0);
                ASSERT_EQ(hView(18),0);
                ASSERT_EQ(hView(19),0);
                ASSERT_EQ(hView(20),1);
                ASSERT_EQ(hView(21),0);
                ASSERT_EQ(hView(22),0);
                ASSERT_EQ(hView(23),0);
                ASSERT_EQ(hView(24),0);
                ASSERT_EQ(hView(25),0);
                ASSERT_EQ(hView(26),0);
                ASSERT_EQ(hView(27),1);
                ASSERT_EQ(hView(28),0);
                ASSERT_EQ(hView(29),0);
                ASSERT_EQ(hView(30),0);
                ASSERT_EQ(hView(31),0);
                ASSERT_EQ(hView(32),0);
                ASSERT_EQ(hView(33),0);
                ASSERT_EQ(hView(34),1);
                ASSERT_EQ(hView(35),0);
                ASSERT_EQ(hView(36),0);
                ASSERT_EQ(hView(37),0);
                ASSERT_EQ(hView(38),0);
                ASSERT_EQ(hView(39),0);
                ASSERT_EQ(hView(40),0);
                ASSERT_EQ(hView(41),1);
                ASSERT_EQ(hView(42),0);
                ASSERT_EQ(hView(43),0);
                ASSERT_EQ(hView(44),0);
            }
            if (ippl::Comm->rank() == 1) {
                ASSERT_EQ(vec.size(), 45);

                ASSERT_EQ(hView(0),1);
                ASSERT_EQ(hView(1),1);
                ASSERT_EQ(hView(2),1);
                ASSERT_EQ(hView(3),1);
                ASSERT_EQ(hView(4),1);
                ASSERT_EQ(hView(5),1);
                ASSERT_EQ(hView(6),1);
                ASSERT_EQ(hView(7),0);
                ASSERT_EQ(hView(8),1);
                ASSERT_EQ(hView(9),2);
                ASSERT_EQ(hView(10),0);
                ASSERT_EQ(hView(11),2);
                ASSERT_EQ(hView(12),2);
                ASSERT_EQ(hView(13),2);
                ASSERT_EQ(hView(14),0);
                ASSERT_EQ(hView(15),1);
                ASSERT_EQ(hView(16),2);
                ASSERT_EQ(hView(17),0);
                ASSERT_EQ(hView(18),2);
                ASSERT_EQ(hView(19),2);
                ASSERT_EQ(hView(20),2);
                ASSERT_EQ(hView(21),0);
                ASSERT_EQ(hView(22),1);
                ASSERT_EQ(hView(23),2);
                ASSERT_EQ(hView(24),0);
                ASSERT_EQ(hView(25),2);
                ASSERT_EQ(hView(26),2);
                ASSERT_EQ(hView(27),2);
                ASSERT_EQ(hView(28),0);
                ASSERT_EQ(hView(29),1);
                ASSERT_EQ(hView(30),2);
                ASSERT_EQ(hView(31),0);
                ASSERT_EQ(hView(32),2);
                ASSERT_EQ(hView(33),2);
                ASSERT_EQ(hView(34),2);
                ASSERT_EQ(hView(35),0);
                ASSERT_EQ(hView(36),1);
                ASSERT_EQ(hView(37),2);
                ASSERT_EQ(hView(38),0);
                ASSERT_EQ(hView(39),2);
                ASSERT_EQ(hView(40),2);
                ASSERT_EQ(hView(41),2);
                ASSERT_EQ(hView(42),0);
                ASSERT_EQ(hView(43),1);
                ASSERT_EQ(hView(44),2);


            }
            if (ippl::Comm->rank() == 2) {
                ASSERT_EQ(vec.size(), 32);

                ASSERT_EQ(hView(0),2);
                ASSERT_EQ(hView(1),2);
                ASSERT_EQ(hView(2),2);
                ASSERT_EQ(hView(3),2);
                ASSERT_EQ(hView(4),2);
                ASSERT_EQ(hView(5),1);
                ASSERT_EQ(hView(6),2);
                ASSERT_EQ(hView(7),1);
                ASSERT_EQ(hView(8),4);
                ASSERT_EQ(hView(9),2);
                ASSERT_EQ(hView(10),1);
                ASSERT_EQ(hView(11),2);
                ASSERT_EQ(hView(12),1);
                ASSERT_EQ(hView(13),4);
                ASSERT_EQ(hView(14),2);
                ASSERT_EQ(hView(15),1);
                ASSERT_EQ(hView(16),2);
                ASSERT_EQ(hView(17),1);
                ASSERT_EQ(hView(18),4);
                ASSERT_EQ(hView(19),2);
                ASSERT_EQ(hView(20),1);
                ASSERT_EQ(hView(21),2);
                ASSERT_EQ(hView(22),1);
                ASSERT_EQ(hView(23),4);
                ASSERT_EQ(hView(24),2);
                ASSERT_EQ(hView(25),1);
                ASSERT_EQ(hView(26),2);
                ASSERT_EQ(hView(27),1);
                ASSERT_EQ(hView(28),4);
                ASSERT_EQ(hView(29),2);
                ASSERT_EQ(hView(30),1);
                ASSERT_EQ(hView(31),2);
            }
        }
    }

    if constexpr (dim == 3) {
        // Due to the fact, that for the 3D case we have a lot of values we now
        // do not check all the values in the FEMVector, but only the ones which
        // are involved in the halo exchange operations.
        auto vec = this->nedelecSpaceSmall.createFEMVector();
        vec = ippl::Comm->rank();
        vec.fillHalo();
        vec.accumulateHalo();
        auto view = vec.getView();
        auto hView = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(hView, view);

        if (ippl::Comm->size() == 1) {
            ASSERT_EQ(vec.size(), 300);
            for (size_t i = 0; i < hView.extent(0); ++i) {
                ASSERT_EQ(hView(i),0);
            }
        }
        if (ippl::Comm->size() == 2) {
            if (ippl::Comm->rank() == 0) {
                ASSERT_EQ(vec.size(), 170);

                ASSERT_EQ(hView(43),0);
                ASSERT_EQ(hView(45),0);
                ASSERT_EQ(hView(46),1);
                ASSERT_EQ(hView(48),0);
                ASSERT_EQ(hView(50),0);
                ASSERT_EQ(hView(51),1);
                ASSERT_EQ(hView(53),0);
                ASSERT_EQ(hView(55),0);
                ASSERT_EQ(hView(56),1);
                ASSERT_EQ(hView(58),0);
                ASSERT_EQ(hView(63),0);
                ASSERT_EQ(hView(64),1);
                ASSERT_EQ(hView(66),0);
                ASSERT_EQ(hView(67),1);
                ASSERT_EQ(hView(69),0);
                ASSERT_EQ(hView(70),1);
                ASSERT_EQ(hView(72),0);
                ASSERT_EQ(hView(73),1);
                ASSERT_EQ(hView(80),0);
                ASSERT_EQ(hView(82),0);
                ASSERT_EQ(hView(83),1);
                ASSERT_EQ(hView(85),0);
                ASSERT_EQ(hView(87),0);
                ASSERT_EQ(hView(88),1);
                ASSERT_EQ(hView(90),0);
                ASSERT_EQ(hView(92),0);
                ASSERT_EQ(hView(93),1);
                ASSERT_EQ(hView(95),0);
                ASSERT_EQ(hView(100),0);
                ASSERT_EQ(hView(101),1);
                ASSERT_EQ(hView(103),0);
                ASSERT_EQ(hView(104),1);
                ASSERT_EQ(hView(106),0);
                ASSERT_EQ(hView(107),1);
                ASSERT_EQ(hView(109),0);
                ASSERT_EQ(hView(110),1);
                ASSERT_EQ(hView(117),0);
                ASSERT_EQ(hView(119),0);
                ASSERT_EQ(hView(120),1);
                ASSERT_EQ(hView(122),0);
                ASSERT_EQ(hView(124),0);
                ASSERT_EQ(hView(125),1);
                ASSERT_EQ(hView(127),0);
                ASSERT_EQ(hView(129),0);
                ASSERT_EQ(hView(130),1);
                ASSERT_EQ(hView(132),0);
                ASSERT_EQ(hView(137),0);
                ASSERT_EQ(hView(138),1);
                ASSERT_EQ(hView(140),0);
                ASSERT_EQ(hView(141),1);
                ASSERT_EQ(hView(143),0);
                ASSERT_EQ(hView(144),1);
                ASSERT_EQ(hView(146),0);
                ASSERT_EQ(hView(147),1);
                ASSERT_EQ(hView(154),0);
                ASSERT_EQ(hView(156),0);
                ASSERT_EQ(hView(157),1);
                ASSERT_EQ(hView(159),0);
                ASSERT_EQ(hView(161),0);
                ASSERT_EQ(hView(162),1);
                ASSERT_EQ(hView(164),0);
                ASSERT_EQ(hView(166),0);
                ASSERT_EQ(hView(167),1);
                ASSERT_EQ(hView(169),0);

            }
            if (ippl::Comm->rank() == 1) {
                ASSERT_EQ(vec.size(), 235);

                ASSERT_EQ(hView(58),0);
                ASSERT_EQ(hView(61),0);
                ASSERT_EQ(hView(62),2);
                ASSERT_EQ(hView(65),0);
                ASSERT_EQ(hView(68),0);
                ASSERT_EQ(hView(69),2);
                ASSERT_EQ(hView(72),0);
                ASSERT_EQ(hView(75),0);
                ASSERT_EQ(hView(76),2);
                ASSERT_EQ(hView(79),0);
                ASSERT_EQ(hView(86),0);
                ASSERT_EQ(hView(87),2);
                ASSERT_EQ(hView(90),0);
                ASSERT_EQ(hView(91),2);
                ASSERT_EQ(hView(94),0);
                ASSERT_EQ(hView(95),2);
                ASSERT_EQ(hView(98),0);
                ASSERT_EQ(hView(99),2);
                ASSERT_EQ(hView(109),0);
                ASSERT_EQ(hView(112),0);
                ASSERT_EQ(hView(113),2);
                ASSERT_EQ(hView(116),0);
                ASSERT_EQ(hView(119),0);
                ASSERT_EQ(hView(120),2);
                ASSERT_EQ(hView(123),0);
                ASSERT_EQ(hView(126),0);
                ASSERT_EQ(hView(127),2);
                ASSERT_EQ(hView(130),0);
                ASSERT_EQ(hView(137),0);
                ASSERT_EQ(hView(138),2);
                ASSERT_EQ(hView(141),0);
                ASSERT_EQ(hView(142),2);
                ASSERT_EQ(hView(145),0);
                ASSERT_EQ(hView(146),2);
                ASSERT_EQ(hView(149),0);
                ASSERT_EQ(hView(150),2);
                ASSERT_EQ(hView(160),0);
                ASSERT_EQ(hView(163),0);
                ASSERT_EQ(hView(164),2);
                ASSERT_EQ(hView(167),0);
                ASSERT_EQ(hView(170),0);
                ASSERT_EQ(hView(171),2);
                ASSERT_EQ(hView(174),0);
                ASSERT_EQ(hView(177),0);
                ASSERT_EQ(hView(178),2);
                ASSERT_EQ(hView(181),0);
                ASSERT_EQ(hView(188),0);
                ASSERT_EQ(hView(189),2);
                ASSERT_EQ(hView(192),0);
                ASSERT_EQ(hView(193),2);
                ASSERT_EQ(hView(196),0);
                ASSERT_EQ(hView(197),2);
                ASSERT_EQ(hView(200),0);
                ASSERT_EQ(hView(201),2);
                ASSERT_EQ(hView(211),0);
                ASSERT_EQ(hView(214),0);
                ASSERT_EQ(hView(215),2);
                ASSERT_EQ(hView(218),0);
                ASSERT_EQ(hView(221),0);
                ASSERT_EQ(hView(222),2);
                ASSERT_EQ(hView(225),0);
                ASSERT_EQ(hView(228),0);
                ASSERT_EQ(hView(229),2);
                ASSERT_EQ(hView(232),0);            
            }
        }
        if (ippl::Comm->size() == 3) {
            if (ippl::Comm->rank() == 0) {
                ASSERT_EQ(vec.size(), 170);

                ASSERT_EQ(hView(43),0);
                ASSERT_EQ(hView(45),0);
                ASSERT_EQ(hView(46),1);
                ASSERT_EQ(hView(48),0);
                ASSERT_EQ(hView(50),0);
                ASSERT_EQ(hView(51),1);
                ASSERT_EQ(hView(53),0);
                ASSERT_EQ(hView(55),0);
                ASSERT_EQ(hView(56),1);
                ASSERT_EQ(hView(58),0);
                ASSERT_EQ(hView(63),0);
                ASSERT_EQ(hView(64),1);
                ASSERT_EQ(hView(66),0);
                ASSERT_EQ(hView(67),1);
                ASSERT_EQ(hView(69),0);
                ASSERT_EQ(hView(70),1);
                ASSERT_EQ(hView(72),0);
                ASSERT_EQ(hView(73),1);
                ASSERT_EQ(hView(80),0);
                ASSERT_EQ(hView(82),0);
                ASSERT_EQ(hView(83),1);
                ASSERT_EQ(hView(85),0);
                ASSERT_EQ(hView(87),0);
                ASSERT_EQ(hView(88),1);
                ASSERT_EQ(hView(90),0);
                ASSERT_EQ(hView(92),0);
                ASSERT_EQ(hView(93),1);
                ASSERT_EQ(hView(95),0);
                ASSERT_EQ(hView(100),0);
                ASSERT_EQ(hView(101),1);
                ASSERT_EQ(hView(103),0);
                ASSERT_EQ(hView(104),1);
                ASSERT_EQ(hView(106),0);
                ASSERT_EQ(hView(107),1);
                ASSERT_EQ(hView(109),0);
                ASSERT_EQ(hView(110),1);
                ASSERT_EQ(hView(117),0);
                ASSERT_EQ(hView(119),0);
                ASSERT_EQ(hView(120),1);
                ASSERT_EQ(hView(122),0);
                ASSERT_EQ(hView(124),0);
                ASSERT_EQ(hView(125),1);
                ASSERT_EQ(hView(127),0);
                ASSERT_EQ(hView(129),0);
                ASSERT_EQ(hView(130),1);
                ASSERT_EQ(hView(132),0);
                ASSERT_EQ(hView(137),0);
                ASSERT_EQ(hView(138),1);
                ASSERT_EQ(hView(140),0);
                ASSERT_EQ(hView(141),1);
                ASSERT_EQ(hView(143),0);
                ASSERT_EQ(hView(144),1);
                ASSERT_EQ(hView(146),0);
                ASSERT_EQ(hView(147),1);
                ASSERT_EQ(hView(154),0);
                ASSERT_EQ(hView(156),0);
                ASSERT_EQ(hView(157),1);
                ASSERT_EQ(hView(159),0);
                ASSERT_EQ(hView(161),0);
                ASSERT_EQ(hView(162),1);
                ASSERT_EQ(hView(164),0);
                ASSERT_EQ(hView(166),0);
                ASSERT_EQ(hView(167),1);
                ASSERT_EQ(hView(169),0);
            }
            if (ippl::Comm->rank() == 1) {
                ASSERT_EQ(vec.size(), 170);

                ASSERT_EQ(hView(42),0);
                ASSERT_EQ(hView(43),2);
                ASSERT_EQ(hView(44),0);
                ASSERT_EQ(hView(45),3);
                ASSERT_EQ(hView(46),2);
                ASSERT_EQ(hView(47),0);
                ASSERT_EQ(hView(48),2);
                ASSERT_EQ(hView(49),0);
                ASSERT_EQ(hView(50),3);
                ASSERT_EQ(hView(51),2);
                ASSERT_EQ(hView(52),0);
                ASSERT_EQ(hView(53),2);
                ASSERT_EQ(hView(54),0);
                ASSERT_EQ(hView(55),3);
                ASSERT_EQ(hView(56),2);
                ASSERT_EQ(hView(57),0);
                ASSERT_EQ(hView(58),2);
                ASSERT_EQ(hView(62),0);
                ASSERT_EQ(hView(63),3);
                ASSERT_EQ(hView(64),2);
                ASSERT_EQ(hView(65),0);
                ASSERT_EQ(hView(66),3);
                ASSERT_EQ(hView(67),2);
                ASSERT_EQ(hView(68),0);
                ASSERT_EQ(hView(69),3);
                ASSERT_EQ(hView(70),2);
                ASSERT_EQ(hView(71),0);
                ASSERT_EQ(hView(72),3);
                ASSERT_EQ(hView(73),2);
                ASSERT_EQ(hView(79),0);
                ASSERT_EQ(hView(80),2);
                ASSERT_EQ(hView(81),0);
                ASSERT_EQ(hView(82),3);
                ASSERT_EQ(hView(83),2);
                ASSERT_EQ(hView(84),0);
                ASSERT_EQ(hView(85),2);
                ASSERT_EQ(hView(86),0);
                ASSERT_EQ(hView(87),3);
                ASSERT_EQ(hView(88),2);
                ASSERT_EQ(hView(89),0);
                ASSERT_EQ(hView(90),2);
                ASSERT_EQ(hView(91),0);
                ASSERT_EQ(hView(92),3);
                ASSERT_EQ(hView(93),2);
                ASSERT_EQ(hView(94),0);
                ASSERT_EQ(hView(95),2);
                ASSERT_EQ(hView(99),0);
                ASSERT_EQ(hView(100),3);
                ASSERT_EQ(hView(101),2);
                ASSERT_EQ(hView(102),0);
                ASSERT_EQ(hView(103),3);
                ASSERT_EQ(hView(104),2);
                ASSERT_EQ(hView(105),0);
                ASSERT_EQ(hView(106),3);
                ASSERT_EQ(hView(107),2);
                ASSERT_EQ(hView(108),0);
                ASSERT_EQ(hView(109),3);
                ASSERT_EQ(hView(110),2);
                ASSERT_EQ(hView(116),0);
                ASSERT_EQ(hView(117),2);
                ASSERT_EQ(hView(118),0);
                ASSERT_EQ(hView(119),3);
                ASSERT_EQ(hView(120),2);
                ASSERT_EQ(hView(121),0);
                ASSERT_EQ(hView(122),2);
                ASSERT_EQ(hView(123),0);
                ASSERT_EQ(hView(124),3);
                ASSERT_EQ(hView(125),2);
                ASSERT_EQ(hView(126),0);
                ASSERT_EQ(hView(127),2);
                ASSERT_EQ(hView(128),0);
                ASSERT_EQ(hView(129),3);
                ASSERT_EQ(hView(130),2);
                ASSERT_EQ(hView(131),0);
                ASSERT_EQ(hView(132),2);
                ASSERT_EQ(hView(136),0);
                ASSERT_EQ(hView(137),3);
                ASSERT_EQ(hView(138),2);
                ASSERT_EQ(hView(139),0);
                ASSERT_EQ(hView(140),3);
                ASSERT_EQ(hView(141),2);
                ASSERT_EQ(hView(142),0);
                ASSERT_EQ(hView(143),3);
                ASSERT_EQ(hView(144),2);
                ASSERT_EQ(hView(145),0);
                ASSERT_EQ(hView(146),3);
                ASSERT_EQ(hView(147),2);
                ASSERT_EQ(hView(153),0);
                ASSERT_EQ(hView(154),2);
                ASSERT_EQ(hView(155),0);
                ASSERT_EQ(hView(156),3);
                ASSERT_EQ(hView(157),2);
                ASSERT_EQ(hView(158),0);
                ASSERT_EQ(hView(159),2);
                ASSERT_EQ(hView(160),0);
                ASSERT_EQ(hView(161),3);
                ASSERT_EQ(hView(162),2);
                ASSERT_EQ(hView(163),0);
                ASSERT_EQ(hView(164),2);
                ASSERT_EQ(hView(165),0);
                ASSERT_EQ(hView(166),3);
                ASSERT_EQ(hView(167),2);
                ASSERT_EQ(hView(168),0);
                ASSERT_EQ(hView(169),2);
            }
            if (ippl::Comm->rank() == 2) {
                ASSERT_EQ(vec.size(), 170);

                ASSERT_EQ(hView(42),1);
                ASSERT_EQ(hView(44),1);
                ASSERT_EQ(hView(45),4);
                ASSERT_EQ(hView(47),1);
                ASSERT_EQ(hView(49),1);
                ASSERT_EQ(hView(50),4);
                ASSERT_EQ(hView(52),1);
                ASSERT_EQ(hView(54),1);
                ASSERT_EQ(hView(55),4);
                ASSERT_EQ(hView(57),1);
                ASSERT_EQ(hView(62),1);
                ASSERT_EQ(hView(63),4);
                ASSERT_EQ(hView(65),1);
                ASSERT_EQ(hView(66),4);
                ASSERT_EQ(hView(68),1);
                ASSERT_EQ(hView(69),4);
                ASSERT_EQ(hView(71),1);
                ASSERT_EQ(hView(72),4);
                ASSERT_EQ(hView(79),1);
                ASSERT_EQ(hView(81),1);
                ASSERT_EQ(hView(82),4);
                ASSERT_EQ(hView(84),1);
                ASSERT_EQ(hView(86),1);
                ASSERT_EQ(hView(87),4);
                ASSERT_EQ(hView(89),1);
                ASSERT_EQ(hView(91),1);
                ASSERT_EQ(hView(92),4);
                ASSERT_EQ(hView(94),1);
                ASSERT_EQ(hView(99),1);
                ASSERT_EQ(hView(100),4);
                ASSERT_EQ(hView(102),1);
                ASSERT_EQ(hView(103),4);
                ASSERT_EQ(hView(105),1);
                ASSERT_EQ(hView(106),4);
                ASSERT_EQ(hView(108),1);
                ASSERT_EQ(hView(109),4);
                ASSERT_EQ(hView(116),1);
                ASSERT_EQ(hView(118),1);
                ASSERT_EQ(hView(119),4);
                ASSERT_EQ(hView(121),1);
                ASSERT_EQ(hView(123),1);
                ASSERT_EQ(hView(124),4);
                ASSERT_EQ(hView(126),1);
                ASSERT_EQ(hView(128),1);
                ASSERT_EQ(hView(129),4);
                ASSERT_EQ(hView(131),1);
                ASSERT_EQ(hView(136),1);
                ASSERT_EQ(hView(137),4);
                ASSERT_EQ(hView(139),1);
                ASSERT_EQ(hView(140),4);
                ASSERT_EQ(hView(142),1);
                ASSERT_EQ(hView(143),4);
                ASSERT_EQ(hView(145),1);
                ASSERT_EQ(hView(146),4);
                ASSERT_EQ(hView(153),1);
                ASSERT_EQ(hView(155),1);
                ASSERT_EQ(hView(156),4);
                ASSERT_EQ(hView(158),1);
                ASSERT_EQ(hView(160),1);
                ASSERT_EQ(hView(161),4);
                ASSERT_EQ(hView(163),1);
                ASSERT_EQ(hView(165),1);
                ASSERT_EQ(hView(166),4);
                ASSERT_EQ(hView(168),1);

            }
        }

    }
}


TYPED_TEST(NedelecSpaceTest, evaluateLoadVector) {
    using T = typename TestFixture::value_t;
    T tolerance = std::numeric_limits<T>::epsilon() * 10.0;
    using NedelecType = typename TestFixture::NedelecType;
    static constexpr std::size_t dim = TestFixture::dim;
    
    if (ippl::Comm->size() ==1) {
        if (dim == 2){
            auto fModel = this->nedelecSpace.createFEMVector();
            
            auto f = fModel.template skeletonCopy<ippl::Vector<T,dim>>();
            f = ippl::Vector<T,dim>(1.);
            
            ippl::FEMVector<T> out = this->nedelecSpace.evaluateLoadVector(f);

            auto view = out.getView();
            auto hView = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hView, view);

            auto ldom = this->layout.getLocalNDIndex();
            for (size_t elementIndex = 0; elementIndex < 20; ++ elementIndex) {
                const ippl::Vector<size_t, NedelecType::numElementDOFs> global_dofs =
                    this->nedelecSpace.getGlobalDOFIndices(elementIndex);

                const ippl::Vector<size_t, NedelecType::numElementDOFs> vectorIndices =
                    this->nedelecSpace.getFEMVectorDOFIndices(elementIndex, ldom);


                for (size_t i = 0; i < NedelecType::numElementDOFs; ++i) {
                    size_t I = global_dofs[i];
                    if (this->nedelecSpace.isDOFOnBoundary(I)) {
                        continue;
                    } else {
                        ASSERT_NEAR(hView(vectorIndices<:i:>), 1., tolerance);
                    }

                }
            }
        }

        if (dim == 3){
            auto fModel = this->nedelecSpaceSmall.createFEMVector();
            
            auto f = fModel.template skeletonCopy<ippl::Vector<T,dim>>();
            f = ippl::Vector<T,dim>(1.);
            
            ippl::FEMVector<T> out = this->nedelecSpaceSmall.evaluateLoadVector(f);

            auto view = out.getView();
            auto hView = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hView, view);

            auto ldom = this->layoutSmall.getLocalNDIndex();
            for (size_t elementIndex = 0; elementIndex < 8; ++ elementIndex) {
                const ippl::Vector<size_t, NedelecType::numElementDOFs> global_dofs =
                    this->nedelecSpaceSmall.getGlobalDOFIndices(elementIndex);

                const ippl::Vector<size_t, NedelecType::numElementDOFs> vectorIndices =
                    this->nedelecSpaceSmall.getFEMVectorDOFIndices(elementIndex, ldom);


                for (size_t i = 0; i < NedelecType::numElementDOFs; ++i) {
                    size_t I = global_dofs[i];
                    if (this->nedelecSpaceSmall.isDOFOnBoundary(I)) {
                        continue;
                    } else {
                        ASSERT_NEAR(hView(vectorIndices<:i:>), 1., tolerance);
                    }

                }
            }
        }
    } else {
        GTEST_SKIP();
    }
}


TYPED_TEST(NedelecSpaceTest, evaluateAx) {
    using T         = typename TestFixture::value_t;
    T tolerance = std::numeric_limits<T>::epsilon() * 100.0;
    using NedelecType = typename TestFixture::NedelecType;
    static constexpr std::size_t dim = TestFixture::dim;

    if (ippl::Comm->size() ==1) {
        if (dim == 2) {
            auto f = DummyFunctor<T, dim, NedelecType::numElementDOFs>();

            auto x = this->nedelecSpace.createFEMVector();
            x = 1;
            ippl::FEMVector<T> out = this->nedelecSpace.evaluateAx(x,f);

            auto view = out.getView();
            auto hView = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hView, view);

            auto ldom = this->layout.getLocalNDIndex();
            for (size_t elementIndex = 0; elementIndex < 20; ++ elementIndex) {
                const ippl::Vector<size_t, NedelecType::numElementDOFs> global_dofs =
                    this->nedelecSpace.getGlobalDOFIndices(elementIndex);

                const ippl::Vector<size_t, NedelecType::numElementDOFs> vectorIndices =
                    this->nedelecSpace.getFEMVectorDOFIndices(elementIndex, ldom);


                for (size_t i = 0; i < NedelecType::numElementDOFs; ++i) {
                    size_t I = global_dofs[i];
                    if (this->nedelecSpace.isDOFOnBoundary(I)) {
                        continue;
                    } else {
                        ASSERT_NEAR(hView(vectorIndices<:i:>), 2., tolerance);
                    }
                }
            }
        }
        if (dim == 3) {
            auto f = DummyFunctor<T, dim, NedelecType::numElementDOFs>();

            auto x = this->nedelecSpaceSmall.createFEMVector();
            x = 1;
            ippl::FEMVector<T> out = this->nedelecSpaceSmall.evaluateAx(x,f);

            auto view = out.getView();
            auto hView = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hView, view);

            auto ldom = this->layoutSmall.getLocalNDIndex();
            for (size_t elementIndex = 0; elementIndex < 8; ++ elementIndex) {
                const ippl::Vector<size_t, NedelecType::numElementDOFs> global_dofs =
                    this->nedelecSpaceSmall.getGlobalDOFIndices(elementIndex);

                const ippl::Vector<size_t, NedelecType::numElementDOFs> vectorIndices =
                    this->nedelecSpaceSmall.getFEMVectorDOFIndices(elementIndex, ldom);


                for (size_t i = 0; i < NedelecType::numElementDOFs; ++i) {
                    size_t I = global_dofs[i];
                    if (this->nedelecSpaceSmall.isDOFOnBoundary(I)) {
                        continue;
                    } else {
                        ASSERT_NEAR(hView(vectorIndices<:i:>), 4., tolerance);
                    }
                }
            }

        }
    } else {
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
