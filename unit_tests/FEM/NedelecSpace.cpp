#include "Ippl.h"

#include <functional>

#include "TestUtils.h"
#include "gtest/gtest.h"


template <typename>
class NedelecSpaceTest;

template <typename T, unsigned Dim, unsigned numElementDOFs>
struct DummyFunctor {
    KOKKOS_FUNCTION const T operator()(size_t i, size_t j,
        const ippl::Vector<ippl::Vector<T, Dim>, numElementDOFs>& curl_b_q_k,
        const ippl::Vector<ippl::Vector<T, Dim>, numElementDOFs>& val_b_q_k) const {
        return 1.0;
    }
};

template <typename T, unsigned Order, unsigned Dim>
class NedelecSpaceTest<Parameters<T, Rank<Order>, Rank<Dim>>> : public ::testing::Test {
protected:
    void SetUp() override {}

public:
    using value_t = T;

    static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");
    static_assert(Order == 1, "Currently only order 1 is supported");

    using MeshType    = ippl::UniformCartesian<T, Dim>;
    using ElementType = std::conditional_t<Dim == 1, ippl::EdgeElement<T>,
                            std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>,
                                ippl::HexahedralElement<T>>>;

    using QuadratureType = ippl::GaussJacobiQuadrature<T, 5, ElementType>;
    using FieldType            = ippl::Field<T, Dim, MeshType, typename MeshType::DefaultCentering>;
    using BCType               = ippl::BConds<FieldType, Dim>;

    NedelecSpaceTest()
        : ref_element()
        , mesh(ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(5)),
            ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(0.0))
        , meshSmall(ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)),
            ippl::Vector<T, Dim>(1.0), ippl::Vector<T, Dim>(0.0))
        , quadrature(ref_element, 0.0, 0.0)
        , nedelecSpace(mesh, ref_element, quadrature,
                        ippl::FieldLayout<Dim>(MPI_COMM_WORLD,
                                               ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)),
                                               std::array<bool, Dim>{true}))
        , nedelecSpaceSmall(meshSmall, ref_element, quadrature,
                        ippl::FieldLayout<Dim>(MPI_COMM_WORLD,
                                               ippl::NDIndex<Dim>(ippl::Vector<unsigned, Dim>(3)),
                                               std::array<bool, Dim>{true})) {
        // fill the global reference DOFs
    }

    ElementType ref_element;
    MeshType mesh;
    MeshType meshSmall;
    const QuadratureType quadrature;
    const ippl::NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType> nedelecSpace;
    const ippl::NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
        nedelecSpaceSmall;
};



using Precisions = TestParams::Precisions;
using Orders     = TestParams::Ranks<1>;
using Dimensions = TestParams::Ranks<2, 3>;
using Combos     = CreateCombinations<Precisions, Orders, Dimensions>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(NedelecSpaceTest, Tests);


TYPED_TEST(NedelecSpaceTest, numLocalDOFS) {
  if (this->nedelecSpace.dim == 2) {
    EXPECT_EQ(this->nedelecSpace.numElementDOFs, 4);
  } else{
    EXPECT_EQ(this->nedelecSpace.numElementDOFs, 12);
  }
}

TYPED_TEST(NedelecSpaceTest, numGlobalDOFs) {
    if (this->nedelecSpace.dim == 2) {
      EXPECT_EQ(this->nedelecSpace.numGlobalDOFs(), 40);
    } else{
      EXPECT_EQ(this->nedelecSpace.numGlobalDOFs(), 300);
    }
  }


TYPED_TEST(NedelecSpaceTest, getGlobalDOFIndices) {
  // Here we just check some random element and see if it is equivalent to a handcrafted one,
  // probably the simplest way of doing this.

  if (this->nedelecSpace.dim == 2) {
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
  } else if (this->nedelecSpaceSmall.dim == 3) {
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
    if (this->nedelecSpace.dim == 2) {
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
    } else <%
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
    %>
}


TYPED_TEST(NedelecSpaceTest, getBoundarySide) {
  if (this->nedelecSpace.dim == 2) {
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

  } else <%
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
  %>
}

TYPED_TEST(NedelecSpaceTest, evaluateRefElementShapeFunction) {
    // This might seem useless to test the reference element shape function
    // values using the same formulae as we are using in the NedelecSpace class
    // to create them, but we do this, as maybe in the future we are changing
    // the implementation in the NedelecSpace class and for this case we still
    // have this test here. So while it right now might not be the most useful
    // thing, it could become useful in the future.
    
    using T = typename TestFixture::value_t;
    using point_t = ippl::Vector<T, this->nedelecSpace.dim>;

    point_t point;
    point_t dif;
    T tolerance = std::numeric_limits<T>::epsilon() * 10.0;

    if constexpr (this->nedelecSpace.dim == 2) {
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
    } else if constexpr (this->nedelecSpace.dim == 3) {
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
    using point_t = ippl::Vector<T, this->nedelecSpace.dim>;

    point_t point;
    point_t dif;
    T tolerance = std::numeric_limits<T>::epsilon() * 10.0;

    if constexpr (this->nedelecSpace.dim == 2) {
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
    } else if constexpr (this->nedelecSpace.dim == 3) {
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


TYPED_TEST(NedelecSpaceTest, evaluateAx) {
    using T         = typename TestFixture::value_t;

    if (this->nedelecSpace.dim == 2) {
        auto f = DummyFunctor<T, this->nedelecSpace.dim, this->nedelecSpace.numElementDOFs>();

        ippl::FEMVector<T> x(this->nedelecSpace.numGlobalDOFs());
        x = 1;
        std::cout << "Start ax\n";
        ippl::FEMVector<T> out = this->nedelecSpace.evaluateAx(x,f);
        std::cout << "End ax\n";

        auto view = out.getView();
        auto hView = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(hView, view);

        for (size_t i = 0; i < hView.extent(0); ++i) {
            std::cout << i << ": " << hView(i) << "\n";
        }
    }
} 


TYPED_TEST(NedelecSpaceTest, evaluateLoadVector) {
    using T = typename TestFixture::value_t;

    if constexpr (this->nedelecSpace.dim == 2) {
        ippl::FEMVector<ippl::Vector<T,2>> f(this->nedelecSpace.numGlobalDOFs());
        f = ippl::Vector<T,2>(1.);
        
        std::cout << "Start eval\n";
        ippl::FEMVector<T> out = this->nedelecSpace.evaluateLoadVector(f);
        std::cout << "End eval\n";

        auto view = out.getView();
        auto hView = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(hView, view);

        for (size_t i = 0; i < hView.extent(0); ++i) {
            std::cout << i << ": " << hView(i) << "\n";
        }
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
