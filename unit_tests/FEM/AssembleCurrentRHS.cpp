#include "Ippl.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }

    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};



template <unsigned Dim, typename T> struct ElemSelector;

template <typename T>
struct ElemSelector<1, T> {
  using Elem = ippl::EdgeElement<T>;
  using Quad = ippl::MidpointQuadrature<T, 1, Elem>;

  static Elem make_elem() { return Elem{}; }
  static Quad make_quad(const Elem& e) { return Quad(e); }
};

template <typename T>
struct ElemSelector<2, T> {
  using Elem = ippl::QuadrilateralElement<T>;
  using Quad = ippl::MidpointQuadrature<T, 1, Elem>;

  static Elem make_elem() { return Elem{}; }
  static Quad make_quad(const Elem& e) { return Quad(e); }
};

template <typename T>
struct ElemSelector<3, T> {
  using Elem = ippl::HexahedralElement<T>;
  using Quad = ippl::MidpointQuadrature<T, 1, Elem>;

  static Elem make_elem() { return Elem{}; }
  static Quad make_quad(const Elem& e) { return Quad(e); }
};


template <typename>
class AssembleCurrentTest;

template <typename T, unsigned Dim>
class AssembleCurrentTest<Parameters<T, Rank<Dim>>> : public ::testing::Test {
public:
  using value_type = T;
  static constexpr unsigned dim = Dim;

  using Mesh_t      = ippl::UniformCartesian<T, dim>;
  using Centering_t = typename Mesh_t::DefaultCentering;

  using ElemSel     = ElemSelector<dim, T>;
  using Elem        = typename ElemSel::Elem;
  using Quad        = typename ElemSel::Quad;
  // Somehow the field type is completely irrelevant for the fem space??
  using FieldType   = ippl::Field<T, Dim, Mesh_t, typename Mesh_t::DefaultCentering>;
  using Layout      = ippl::FieldLayout<Dim>;

  using NedelecType = ippl::NedelecSpace<T, Dim, 1, Elem, Quad, FieldType>;

  using playout_t   = ippl::ParticleSpatialLayout<T, dim>;
  using bunch_t     = Bunch<playout_t>;

  static ippl::NDIndex<dim> make_owned_nd(int nx) {
    ippl::Index I0(nx);
    if constexpr (dim == 1)
      return ippl::NDIndex<1>(I0);
    else if constexpr (dim == 2)
      return ippl::NDIndex<2>(I0, I0);
    else
      return ippl::NDIndex<3>(I0, I0, I0);
  }

  static Layout make_layout(const ippl::NDIndex<dim>& owned) {
    std::array<bool, dim> par{}; par.fill(true);
    return ippl::FieldLayout<dim>(MPI_COMM_WORLD, owned, par);
  }

  static Mesh_t make_mesh(const ippl::NDIndex<dim>& owned,
                          const ippl::Vector<T, dim>& h,
                          const ippl::Vector<T, dim>& origin) {
    return Mesh_t(owned, h, origin);
  }

  static NedelecType make_space(Mesh_t& mesh, Layout l) {
    Elem e = ElemSel::make_elem();
    Quad q = ElemSel::make_quad(e);
    return NedelecType(mesh, e, q, l);
  }
};

using Precisions = TestParams::Precisions;
using Ranks      = TestParams::Ranks<2, 3>;
using Tests      = TestForTypes<CreateCombinations<Precisions, Ranks>::type>::type;

TYPED_TEST_SUITE(AssembleCurrentTest, Tests);

// ------------------ Actual minimal test ------------------

TYPED_TEST(AssembleCurrentTest, SingleParticle_Smoke) {
  using T   = typename TestFixture::value_type;
  constexpr unsigned Dim = TestFixture::dim;

  using bunch_t    = typename TestFixture::bunch_t;
  using playout_t  = typename TestFixture::playout_t;

  int nx = 4;

  ippl::Vector<T, Dim> origin(0.0);
  ippl::Vector<T, Dim> h(1.0);

  auto owned  = TestFixture::make_owned_nd(nx);
  auto layout = TestFixture::make_layout(owned);
  auto mesh   = TestFixture::make_mesh(owned, h, origin);
  auto space  = TestFixture::make_space(mesh, layout);

  playout_t playout(layout, mesh);
  bunch_t   bunch(playout);

  // --- create 1 particle ---
  bunch.create(1);
  {
    auto R_host = bunch.R.getHostMirror();
    auto Q_host = bunch.Q.getHostMirror();
    for (unsigned d=0; d<Dim; ++d)
      R_host(0)[d] = T(0.5) * h[d];  // inside domain
    Q_host(0) = 1.0;
    Kokkos::deep_copy(bunch.R.getView(), R_host);
    Kokkos::deep_copy(bunch.Q.getView(), Q_host);
    bunch.update();
  }

  // --- mock "next" positions (slightly moved) ---
  bunch_t bunch_next(playout);
  bunch_next.create(1);
  {
    auto Rn_host = bunch_next.R.getHostMirror();
    for (unsigned d=0; d<Dim; ++d)
      Rn_host(0)[d] = T(0.6) * h[d];
    Kokkos::deep_copy(bunch_next.R.getView(), Rn_host);
    bunch_next.update();
  }

  auto policy = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
  auto fem_vector = space.createFEMVector();
  fem_vector = T(0);

  T dt = T(1.0);
  ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch_next.R, fem_vector, space, policy, dt);
  fem_vector.accumulateHalo();


  SUCCEED() << "assemble_current_whitney1 ran without error for 1 particle in "
            << Dim << "D.";
}

TYPED_TEST(AssembleCurrentTest, SingleAxis_X_SameCell_ExactValues) {
  using T = typename TestFixture::value_type;
  constexpr unsigned Dim = TestFixture::dim;

  using bunch_t   = typename TestFixture::bunch_t;
  using playout_t = typename TestFixture::playout_t;

  int nx = 4;
  ippl::Vector<T, Dim> origin(0.0);
  ippl::Vector<T, Dim> h(1.0);

  auto owned  = TestFixture::make_owned_nd(nx);
  auto layout = TestFixture::make_layout(owned);
  auto mesh   = TestFixture::make_mesh(owned, h, origin);
  auto space  = TestFixture::make_space(mesh, layout);


  //create the two positions used for the test, representing pure x-motion within one cell)
  playout_t playout(layout, mesh);
  bunch_t bunch(playout);
  bunch.create(1);
  {
    auto R_host = bunch.R.getHostMirror();
    auto Q_host = bunch.Q.getHostMirror();
    R_host(0)[0] = T(0.25);
    for (unsigned d = 1; d < Dim; ++d) R_host(0)[d] = T(0.50);
    Q_host(0) = T(1.0);
    Kokkos::deep_copy(bunch.R.getView(), R_host);
    Kokkos::deep_copy(bunch.Q.getView(), Q_host);
    bunch.update();
  }

  bunch_t bunch_next(playout);
  bunch_next.create(1);
  {
    auto Rn_host = bunch_next.R.getHostMirror();
    Rn_host(0)[0] = T(0.75);
    for (unsigned d = 1; d < Dim; ++d) Rn_host(0)[d] = T(0.50);
    Kokkos::deep_copy(bunch_next.R.getView(), Rn_host);
    bunch_next.update();
  }

  //Compute current contribution 
  auto policy     = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
  auto fem_vector = space.createFEMVector();
  fem_vector = T(0);

  T dt = T(1.0);
  ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch_next.R,
                                  fem_vector, space, policy, dt);
  fem_vector.accumulateHalo();

  typename TestFixture::NedelecType::indices_t cellIdx(0);
  auto ldom   = space.getLocalNDIndex();
  auto dofIdx = space.getFEMVectorDOFIndices(cellIdx, ldom);

  auto view      = fem_vector.getView();
  auto view_host = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(view_host, view);
  Kokkos::fence();

  const T tol = std::numeric_limits<T>::epsilon() * T(100);

  //Check that current contributions are correct, which is easy here, as the current should distribute equally over all x-aligned axis
  if constexpr (Dim == 2) {
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[0])), 0.25, static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[1])), 0.0,  static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[2])), 0.25, static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[3])), 0.0,  static_cast<double>(tol));
  } 
  else if constexpr (Dim == 3) {
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[0])),  0.125, static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[1])),  0.0,   static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[2])),  0.125, static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[3])),  0.0,   static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[4])),  0.0,   static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[5])),  0.0,   static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[6])),  0.0,   static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[7])),  0.0,   static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[8])),  0.125, static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[9])),  0.0,   static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[10])), 0.125, static_cast<double>(tol));
    EXPECT_NEAR(static_cast<double>(view_host(dofIdx[11])), 0.0,   static_cast<double>(tol));
  }
}

TYPED_TEST(AssembleCurrentTest, DiagonalPath_ThreeCells_ExactValues) {
  using T = typename TestFixture::value_type;
  constexpr unsigned Dim = TestFixture::dim;

  if constexpr (Dim != 2) {
    GTEST_SKIP() << "Exact value check only implemented for 2D";
  } else {
    using bunch_t   = typename TestFixture::bunch_t;
    using playout_t = typename TestFixture::playout_t;

    int nx = 4;
    ippl::Vector<T, Dim> origin(0.0);
    ippl::Vector<T, Dim> h(1.0);

    auto owned  = TestFixture::make_owned_nd(nx);
    auto layout = TestFixture::make_layout(owned);
    auto mesh   = TestFixture::make_mesh(owned, h, origin);
    auto space  = TestFixture::make_space(mesh, layout);

    playout_t playout(layout, mesh);
    bunch_t bunch(playout);
    //path of particle: (0.75,0.50) -> (1.50,1.25), crossing three cells: (0,0), (1,0), (1,1)
    bunch.create(1);
    {
      auto R_host = bunch.R.getHostMirror();
      auto Q_host = bunch.Q.getHostMirror();
      R_host(0)[0] = T(0.75);
      R_host(0)[1] = T(0.50);
      Q_host(0) = T(1.0);
      Kokkos::deep_copy(bunch.R.getView(), R_host);
      Kokkos::deep_copy(bunch.Q.getView(), Q_host);
      bunch.update();
    }

    bunch_t bunch_next(playout);
    bunch_next.create(1);
    {
      auto Rn_host = bunch_next.R.getHostMirror();
      Rn_host(0)[0] = T(1.50);
      Rn_host(0)[1] = T(1.25);
      Kokkos::deep_copy(bunch_next.R.getView(), Rn_host);
      bunch_next.update();
    }

    auto policy     = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
    auto fem_vector = space.createFEMVector();
    fem_vector = T(0);

    T dt = T(1.0);

    ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch_next.R,
                                    fem_vector, space, policy, dt);
    fem_vector.accumulateHalo();

    auto ldom = space.getLocalNDIndex();

    auto view      = fem_vector.getView();
    auto view_host = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(view_host, view);
    Kokkos::fence();

    const T tol = std::numeric_limits<T>::epsilon() * T(100);

    // Each segment contributes to the 4 edge DOFs of its cell. Edges shared
    // between adjacent cells map to the same FEM vector entry, so those entries
    // add contributions from both neighbouring segments.

    //Results calculated on paper:
    {
      typename TestFixture::NedelecType::indices_t cell00(0);
      auto dof = space.getFEMVectorDOFIndices(cell00, ldom);
      EXPECT_NEAR(static_cast<double>(view_host(dof[0])), 0.09375, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[1])), 0.03125, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[2])), 0.15625, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[3])), 0.4375,  static_cast<double>(tol));
    }
    {
      typename TestFixture::NedelecType::indices_t cell10(0);
      cell10[0] = 1;
      auto dof = space.getFEMVectorDOFIndices(cell10, ldom);
      EXPECT_NEAR(static_cast<double>(view_host(dof[0])), 0.03125, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[1])), 0.4375,  static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[2])), 0.4375,  static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[3])), 0.03125, static_cast<double>(tol));
    }
    {
      typename TestFixture::NedelecType::indices_t cell11(0);
      cell11[0] = 1; cell11[1] = 1;
      auto dof = space.getFEMVectorDOFIndices(cell11, ldom);
      EXPECT_NEAR(static_cast<double>(view_host(dof[0])), 0.4375,  static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[1])), 0.15625, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[2])), 0.03125, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[3])), 0.09375, static_cast<double>(tol));
    }
  }
}

TYPED_TEST(AssembleCurrentTest, DiagonalPath_VertexHit_3D) {
  using T = typename TestFixture::value_type;
  constexpr unsigned Dim = TestFixture::dim;

  if constexpr (Dim != 3) {
    GTEST_SKIP() << "Vertex-hit crossing test only implemented for 3D";
  } else {
    using bunch_t   = typename TestFixture::bunch_t;
    using playout_t = typename TestFixture::playout_t;

    int nx = 4;
    ippl::Vector<T, Dim> origin(0.0);
    ippl::Vector<T, Dim> h(1.0);

    auto owned  = TestFixture::make_owned_nd(nx);
    auto layout = TestFixture::make_layout(owned);
    auto mesh   = TestFixture::make_mesh(owned, h, origin);
    auto space  = TestFixture::make_space(mesh, layout);

    playout_t playout(layout, mesh);
    bunch_t bunch(playout);
    // path: (0.9, 0.9, 0.8) -> (1.1, 1.1, 1.2)
    // all three axis crossings at t=0.5 (vertex hit), 2 real segments:
    // seg0 in cell (0,0,0), seg1 in cell (1,1,1)
    bunch.create(1);
    {
      auto R_host = bunch.R.getHostMirror();
      auto Q_host = bunch.Q.getHostMirror();
      R_host(0)[0] = T(0.9);
      R_host(0)[1] = T(0.9);
      R_host(0)[2] = T(0.8);
      Q_host(0) = T(1.0);
      Kokkos::deep_copy(bunch.R.getView(), R_host);
      Kokkos::deep_copy(bunch.Q.getView(), Q_host);
      bunch.update();
    }

    bunch_t bunch_next(playout);
    bunch_next.create(1);
    {
      auto Rn_host = bunch_next.R.getHostMirror();
      Rn_host(0)[0] = T(1.1);
      Rn_host(0)[1] = T(1.1);
      Rn_host(0)[2] = T(1.2);
      Kokkos::deep_copy(bunch_next.R.getView(), Rn_host);
      bunch_next.update();
    }

    auto policy     = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
    auto fem_vector = space.createFEMVector();
    fem_vector = T(0);

    T dt = T(1.0);
    ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch_next.R,
                                    fem_vector, space, policy, dt);
    fem_vector.accumulateHalo();

    auto ldom = space.getLocalNDIndex();

    auto view      = fem_vector.getView();
    auto view_host = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(view_host, view);
    Kokkos::fence();

    const T tol = std::numeric_limits<T>::epsilon() * T(100);

    // Cell (0,0,0): midpoint xi=(0.95, 0.95, 0.9), dp=(0.1, 0.1, 0.2)
    // No shared edges with cell (1,1,1), so simple computation
    {
      typename TestFixture::NedelecType::indices_t cell000(0);
      auto dof = space.getFEMVectorDOFIndices(cell000, ldom);
      EXPECT_NEAR(static_cast<double>(view_host(dof[0])),  0.0005, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[1])),  0.0005, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[2])),  0.0095, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[3])),  0.0095, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[4])),  0.0005, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[5])),  0.0095, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[6])),  0.1805, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[7])),  0.0095, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[8])),  0.0045, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[9])),  0.0045, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[10])), 0.0855, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[11])), 0.0855, static_cast<double>(tol));
    }
    // Cell (1,1,1): indentical to first cell, except different order as cell is mirrored across vertex
    {
      typename TestFixture::NedelecType::indices_t cell111(0);
      cell111[0] = 1; cell111[1] = 1; cell111[2] = 1;
      auto dof = space.getFEMVectorDOFIndices(cell111, ldom);
      EXPECT_NEAR(static_cast<double>(view_host(dof[0])),  0.0855, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[1])),  0.0855, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[2])),  0.0045, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[3])),  0.0045, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[4])),  0.1805, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[5])),  0.0095, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[6])),  0.0005, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[7])),  0.0095, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[8])),  0.0095, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[9])),  0.0095, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[10])), 0.0005, static_cast<double>(tol));
      EXPECT_NEAR(static_cast<double>(view_host(dof[11])), 0.0005, static_cast<double>(tol));
    }
  }
}

int main(int argc, char* argv[]) {
  ippl::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  ippl::finalize();
  return result;
}
