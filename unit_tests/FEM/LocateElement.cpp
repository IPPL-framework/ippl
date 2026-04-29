#include "Ippl.h"
#include "TestUtils.h"

#include "gtest/gtest.h"


template <unsigned Dim, typename T>
struct MeshStub {
  ippl::Vector<int,Dim> nr;   // vertices per axis
  ippl::Vector<T,Dim>   h;    // spacing
  ippl::Vector<T,Dim>   org;  // origin

  KOKKOS_INLINE_FUNCTION ippl::Vector<int,Dim> getGridsize()    const { return nr; }
  KOKKOS_INLINE_FUNCTION ippl::Vector<T,Dim>   getMeshSpacing() const { return h; }
  KOKKOS_INLINE_FUNCTION ippl::Vector<T,Dim>   getOrigin()      const { return org; }
};

template <typename> 
class LocateElementTest;

template <typename T, unsigned Dim>
class LocateElementTest<Parameters<T, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type  = T;
    static constexpr unsigned dim = Dim;

    static constexpr T tol() {
        return std::numeric_limits<T>::epsilon() * 10.0;
    }
};

using Precisions = TestParams::Precisions;
using Ranks      = TestParams::Ranks<1, 2, 3>;
using Tests      = TestForTypes<CreateCombinations<Precisions, Ranks>::type>::type;

TYPED_TEST_SUITE(LocateElementTest, Tests);

TYPED_TEST(LocateElementTest, DeterministicPoints) {
    using T = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;


    MeshStub<Dim,T> M{};

    // Fixed, arbitrary choice of origin, spacing and number of vertices
    for (unsigned d=0; d<Dim; ++d) {
        M.nr[d]  = (d+4);           // e.g., 4,5,6 vertices → 3,4,5 cells
        M.h[d]   = T(0.3 + 0.2*d);  // non-uniform spacings
        M.org[d] = T(-1.2 + 0.4*d); // non-zero origin
    }
    
    // Build two interior points:
    // s1 = i + 0.25 (expect e=1, xi=0.25), s2 = i + 0.75 (expect e=2, xi=0.75)
    ippl::Vector<T,Dim> s1, s2, x1, x2;
    for (unsigned d=0; d<Dim; ++d) { 
        s1[d] = T(1) + T(0.25);
        s2[d] = T(2) + T(0.75); 
    }

    for (unsigned d=0; d<Dim; ++d) { 
        x1[d] = M.org[d] + M.h[d]*s1[d];
        x2[d] = M.org[d] + M.h[d]*s2[d]; 
    }

    
    ippl::Vector<size_t,Dim> e_nd{};
    ippl::Vector<T,Dim>      xi{};

    ippl::locate_element_nd_and_xi<T,Dim>(M.h, M.org, x1, e_nd, xi);

    for (unsigned d=0; d<Dim; ++d) {
        EXPECT_EQ(e_nd[d], size_t(1));
        EXPECT_NEAR(xi[d], T(0.25), TestFixture::tol());
    }
    
    ippl::locate_element_nd_and_xi<T,Dim>(M.h, M.org, x2, e_nd, xi);
    for (unsigned d=0; d<Dim; ++d) {
      EXPECT_EQ(e_nd[d], size_t(2));
      EXPECT_NEAR(xi[d], T(0.75), TestFixture::tol());
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
