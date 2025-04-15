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
    static_assert(Order == 1, "Currently only order 2 is supported");

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
    const ippl::NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType> nedelecSpace;
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
  } else if (this->nedelecSpace.dim == 3) {
    /*
    auto idxSet = this->nedelecSpace.getGlobalDOFIndices(7);
    EXPECT_EQ(idxSet[0], 15);
    EXPECT_EQ(idxSet[1], 17);
    EXPECT_EQ(idxSet[2], 27);
    EXPECT_EQ(idxSet[3], 29);
    EXPECT_EQ(idxSet[4], 21);
    EXPECT_EQ(idxSet[5], 23);
    EXPECT_EQ(idxSet[6], 33);
    EXPECT_EQ(idxSet[7], 35);
    EXPECT_EQ(idxSet[8], 45);
    EXPECT_EQ(idxSet[9], 47);
    EXPECT_EQ(idxSet[10],51);
    EXPECT_EQ(idxSet[11],53);
    */
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
    }
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
