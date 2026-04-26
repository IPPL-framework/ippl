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

  // Call function (no assertions yet)
  ippl::assemble_current_whitney1(mesh, bunch.Q, bunch.R, bunch_next.R, fem_vector, space, policy);


  SUCCEED() << "assemble_current_whitney1 ran without error for 1 particle in "
            << Dim << "D.";
}

int main(int argc, char* argv[]) {
  ippl::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  ippl::finalize();
  return result;
}
