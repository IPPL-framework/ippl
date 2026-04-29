#include "Ippl.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
        this->addAttribute(R_next);
    }

    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
    typename PLayout::particle_position_type R_next;
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

  bunch.create(ippl::Comm->rank() == 0 ? 1 : 0);
  if (ippl::Comm->rank() == 0) {
    auto R_host  = bunch.R.getHostMirror();
    auto Rn_host = bunch.R_next.getHostMirror();
    auto Q_host  = bunch.Q.getHostMirror();
    for (unsigned d=0; d<Dim; ++d) {
      R_host(0)[d]  = T(0.5) * h[d];
      Rn_host(0)[d] = T(0.6) * h[d];
    }
    Q_host(0) = 1.0;
    Kokkos::deep_copy(bunch.R.getView(), R_host);
    Kokkos::deep_copy(bunch.R_next.getView(), Rn_host);
    Kokkos::deep_copy(bunch.Q.getView(), Q_host);
  }
  bunch.update();

  auto policy = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
  auto fem_vector = space.createFEMVector();
  fem_vector = T(0);

  T dt = T(1.0);
  ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch.R_next, fem_vector, space, policy, dt);
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
  bunch.create(ippl::Comm->rank() == 0 ? 1 : 0);
  if (ippl::Comm->rank() == 0) {
    auto R_host  = bunch.R.getHostMirror();
    auto Rn_host = bunch.R_next.getHostMirror();
    auto Q_host  = bunch.Q.getHostMirror();
    R_host(0)[0]  = T(0.25);
    Rn_host(0)[0] = T(0.75);
    for (unsigned d = 1; d < Dim; ++d) { R_host(0)[d] = T(0.50); Rn_host(0)[d] = T(0.50); }
    Q_host(0) = T(1.0);
    Kokkos::deep_copy(bunch.R.getView(), R_host);
    Kokkos::deep_copy(bunch.R_next.getView(), Rn_host);
    Kokkos::deep_copy(bunch.Q.getView(), Q_host);
  }
  bunch.update();

  auto policy     = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
  auto fem_vector = space.createFEMVector();
  fem_vector = T(0);

  T dt = T(1.0);
  ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch.R_next,
                                  fem_vector, space, policy, dt);
  fem_vector.accumulateHalo();

  typename TestFixture::NedelecType::indices_t cellIdx(0);
  auto ldom = space.getLocalNDIndex();

  auto view      = fem_vector.getView();
  auto view_host = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(view_host, view);
  Kokkos::fence();

  const T tol = std::numeric_limits<T>::epsilon() * T(100);

  auto in_ldom = [&](const auto& cell) {
    for (unsigned d = 0; d < Dim; ++d)
      if (static_cast<int>(cell[d]) < ldom.first()[d] || static_cast<int>(cell[d]) > ldom.last()[d]) return false;
    return true;
  };

  // Extract value on owning rank, broadcast to all via SUM so every rank asserts.
  auto check_dof = [&](const auto& cell, int i, double expected) {
    double local_val = 0.0;
    if (in_ldom(cell)) {
      auto dof = space.getFEMVectorDOFIndices(cell, ldom);
      local_val = static_cast<double>(view_host(dof[i]));
    }
    double global_val = 0.0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_NEAR(global_val, expected, static_cast<double>(tol));
  };

  if constexpr (Dim == 2) {
    check_dof(cellIdx, 0, 0.25);
    check_dof(cellIdx, 1, 0.0);
    check_dof(cellIdx, 2, 0.25);
    check_dof(cellIdx, 3, 0.0);
  } else if constexpr (Dim == 3) {
    check_dof(cellIdx, 0,  0.125);
    check_dof(cellIdx, 1,  0.0);
    check_dof(cellIdx, 2,  0.125);
    check_dof(cellIdx, 3,  0.0);
    check_dof(cellIdx, 4,  0.0);
    check_dof(cellIdx, 5,  0.0);
    check_dof(cellIdx, 6,  0.0);
    check_dof(cellIdx, 7,  0.0);
    check_dof(cellIdx, 8,  0.125);
    check_dof(cellIdx, 9,  0.0);
    check_dof(cellIdx, 10, 0.125);
    check_dof(cellIdx, 11, 0.0);
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
    bunch.create(ippl::Comm->rank() == 0 ? 1 : 0);
    if (ippl::Comm->rank() == 0) {
      auto R_host  = bunch.R.getHostMirror();
      auto Rn_host = bunch.R_next.getHostMirror();
      auto Q_host  = bunch.Q.getHostMirror();
      R_host(0)[0]  = T(0.75); R_host(0)[1]  = T(0.50);
      Rn_host(0)[0] = T(1.50); Rn_host(0)[1] = T(1.25);
      Q_host(0) = T(1.0);
      Kokkos::deep_copy(bunch.R.getView(), R_host);
      Kokkos::deep_copy(bunch.R_next.getView(), Rn_host);
      Kokkos::deep_copy(bunch.Q.getView(), Q_host);
    }
    bunch.update();

    auto policy     = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
    auto fem_vector = space.createFEMVector();
    fem_vector = T(0);

    T dt = T(1.0);

    ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch.R_next,
                                    fem_vector, space, policy, dt);
    fem_vector.accumulateHalo();

    auto ldom = space.getLocalNDIndex();

    auto view      = fem_vector.getView();
    auto view_host = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(view_host, view);
    Kokkos::fence();

    const T tol = std::numeric_limits<T>::epsilon() * T(100);

    auto in_ldom = [&](const auto& cell) {
      for (unsigned d = 0; d < Dim; ++d)
        if (static_cast<int>(cell[d]) < ldom.first()[d] || static_cast<int>(cell[d]) > ldom.last()[d]) return false;
      return true;
    };

    auto check_dof = [&](const auto& cell, int i, double expected) {
      double local_val = 0.0;
      if (in_ldom(cell)) {
        auto dof = space.getFEMVectorDOFIndices(cell, ldom);
        local_val = static_cast<double>(view_host(dof[i]));
      }
      double global_val = 0.0;
      MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      EXPECT_NEAR(global_val, expected, static_cast<double>(tol));
    };

    // Each segment contributes to the 4 edge DOFs of its cell. Edges shared
    // between adjacent cells map to the same FEM vector entry, so those entries
    // add contributions from both neighbouring segments.

    //Results calculated on paper:
    {
      typename TestFixture::NedelecType::indices_t cell00(0);
      check_dof(cell00, 0, 0.09375);
      check_dof(cell00, 1, 0.03125);
      check_dof(cell00, 2, 0.15625);
      check_dof(cell00, 3, 0.4375);
    }
    {
      typename TestFixture::NedelecType::indices_t cell10(0);
      cell10[0] = 1;
      check_dof(cell10, 0, 0.03125);
      check_dof(cell10, 1, 0.4375);
      check_dof(cell10, 2, 0.4375);
      check_dof(cell10, 3, 0.03125);
    }
    {
      typename TestFixture::NedelecType::indices_t cell11(0);
      cell11[0] = 1; cell11[1] = 1;
      check_dof(cell11, 0, 0.4375);
      check_dof(cell11, 1, 0.15625);
      check_dof(cell11, 2, 0.03125);
      check_dof(cell11, 3, 0.09375);
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
    bunch.create(ippl::Comm->rank() == 0 ? 1 : 0);
    if (ippl::Comm->rank() == 0) {
      auto R_host  = bunch.R.getHostMirror();
      auto Rn_host = bunch.R_next.getHostMirror();
      auto Q_host  = bunch.Q.getHostMirror();
      R_host(0)[0]  = T(0.9); R_host(0)[1]  = T(0.9); R_host(0)[2]  = T(0.8);
      Rn_host(0)[0] = T(1.1); Rn_host(0)[1] = T(1.1); Rn_host(0)[2] = T(1.2);
      Q_host(0) = T(1.0);
      Kokkos::deep_copy(bunch.R.getView(), R_host);
      Kokkos::deep_copy(bunch.R_next.getView(), Rn_host);
      Kokkos::deep_copy(bunch.Q.getView(), Q_host);
    }
    bunch.update();

    auto policy     = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
    auto fem_vector = space.createFEMVector();
    fem_vector = T(0);

    T dt = T(1.0);
    ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch.R_next,
                                    fem_vector, space, policy, dt);
    fem_vector.accumulateHalo();

    auto ldom = space.getLocalNDIndex();

    auto view      = fem_vector.getView();
    auto view_host = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(view_host, view);
    Kokkos::fence();

    const T tol = std::numeric_limits<T>::epsilon() * T(100);

    auto in_ldom = [&](const auto& cell) {
      for (unsigned d = 0; d < Dim; ++d)
        if (static_cast<int>(cell[d]) < ldom.first()[d] || static_cast<int>(cell[d]) > ldom.last()[d]) return false;
      return true;
    };

    auto check_dof = [&](const auto& cell, int i, double expected) {
      double local_val = 0.0;
      if (in_ldom(cell)) {
        auto dof = space.getFEMVectorDOFIndices(cell, ldom);
        local_val = static_cast<double>(view_host(dof[i]));
      }
      double global_val = 0.0;
      MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      EXPECT_NEAR(global_val, expected, static_cast<double>(tol));
    };

    // Cell (0,0,0): midpoint xi=(0.95, 0.95, 0.9), dp=(0.1, 0.1, 0.2)
    // No shared edges with cell (1,1,1), so simple computation
    {
      typename TestFixture::NedelecType::indices_t cell000(0);
      check_dof(cell000, 0,  0.0005);
      check_dof(cell000, 1,  0.0005);
      check_dof(cell000, 2,  0.0095);
      check_dof(cell000, 3,  0.0095);
      check_dof(cell000, 4,  0.0005);
      check_dof(cell000, 5,  0.0095);
      check_dof(cell000, 6,  0.1805);
      check_dof(cell000, 7,  0.0095);
      check_dof(cell000, 8,  0.0045);
      check_dof(cell000, 9,  0.0045);
      check_dof(cell000, 10, 0.0855);
      check_dof(cell000, 11, 0.0855);
    }
    // Cell (1,1,1): identical to first cell, except different order as cell is mirrored across vertex
    {
      typename TestFixture::NedelecType::indices_t cell111(0);
      cell111[0] = 1; cell111[1] = 1; cell111[2] = 1;
      check_dof(cell111, 0,  0.0855);
      check_dof(cell111, 1,  0.0855);
      check_dof(cell111, 2,  0.0045);
      check_dof(cell111, 3,  0.0045);
      check_dof(cell111, 4,  0.1805);
      check_dof(cell111, 5,  0.0095);
      check_dof(cell111, 6,  0.0005);
      check_dof(cell111, 7,  0.0095);
      check_dof(cell111, 8,  0.0095);
      check_dof(cell111, 9,  0.0095);
      check_dof(cell111, 10, 0.0005);
      check_dof(cell111, 11, 0.0005);
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
