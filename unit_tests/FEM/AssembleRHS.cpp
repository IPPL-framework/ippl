#include "Ippl.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

#include <random>

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
  template<class Field>
  using Space = ippl::LagrangeSpace<T, 1, 1, Elem, Quad, Field, Field>;

  static Elem make_elem() { return Elem{}; }
  static Quad make_quad(const Elem& e) { return Quad(e); }
};

template <typename T>
struct ElemSelector<2, T> {
  using Elem = ippl::QuadrilateralElement<T>;
  using Quad = ippl::MidpointQuadrature<T, 1, Elem>;
  template<class Field>
  using Space = ippl::LagrangeSpace<T, 2, 1, Elem, Quad, Field, Field>;

  static Elem make_elem() { return Elem{}; }
  static Quad make_quad(const Elem& e) { return Quad(e); }
};

template <typename T>
struct ElemSelector<3, T> {
  using Elem = ippl::HexahedralElement<T>;
  using Quad = ippl::MidpointQuadrature<T, 1, Elem>;
  template<class Field>
  using Space = ippl::LagrangeSpace<T, 3, 1, Elem, Quad, Field, Field>;

  static Elem make_elem() { return Elem{}; }
  static Quad make_quad(const Elem& e) { return Quad(e); }
};


template <typename> 
class AssembleRHSTest;

template <typename T, unsigned Dim>
class AssembleRHSTest<Parameters<T, Rank<Dim>>> : public ::testing::Test {
public:
  using value_type = T;
  static constexpr unsigned dim = Dim;

  using Mesh_t      = ippl::UniformCartesian<T, dim>;
  using Centering_t = typename Mesh_t::DefaultCentering;
  using field_t     = ippl::Field<T, dim, Mesh_t, Centering_t>;

  using ElemSel     = ElemSelector<dim, T>;
  using Elem        = typename ElemSel::Elem;
  using Quad        = typename ElemSel::Quad;
  using Space       = typename ElemSel::template Space<field_t>;

  using playout_t   = ippl::ParticleSpatialLayout<T, dim>;
  using bunch_t     = Bunch<playout_t>;

  static ippl::NDIndex<dim> make_owned_nd(int nx) {
    ippl::Index I0(nx);
    if constexpr (dim == 1) {
      return ippl::NDIndex<1>(I0);
    } else if constexpr (dim == 2) {
      ippl::Index I1(nx);
      return ippl::NDIndex<2>(I0, I1);
    } else {
      ippl::Index I1(nx), I2(nx);
      return ippl::NDIndex<3>(I0, I1, I2);
    }
  }


  static ippl::FieldLayout<dim> make_layout(const ippl::NDIndex<dim>& owned) {
    std::array<bool, dim> par{}; par.fill(true);
    return ippl::FieldLayout<dim>(MPI_COMM_WORLD, owned, par);
  }

  static Mesh_t make_mesh(const ippl::NDIndex<dim>& owned,
                          const ippl::Vector<T, dim>& h,
                          const ippl::Vector<T, dim>& origin) {
    return Mesh_t(owned, h, origin);
  }
  
  static field_t make_zero_field(Mesh_t& mesh, 
                                 ippl::FieldLayout<dim>& layout) {
    field_t f; 
    f.initialize(mesh, layout);
    Kokkos::deep_copy(f.getView(), T(0));
    return f;
  }

  static Space make_space(Mesh_t& mesh) {
    Elem e = ElemSel::make_elem();
    Quad q = ElemSel::make_quad(e);
    return Space(mesh, e, q);
  }

  // Axis-aligned domain bounds from mesh & nx
  static void domain_bounds(const Mesh_t& mesh, int nx,
                            ippl::Vector<T, dim>& xmin,
                            ippl::Vector<T, dim>& xmax) {
    const auto org = mesh.getOrigin();
    const auto h   = mesh.getMeshSpacing();
    for (unsigned d=0; d<dim; ++d) {
      xmin[d] = org[d];
      xmax[d] = org[d] + h[d] * T(nx - 1);
    }
  }

  static double fill_uniform_particles(bunch_t& bunch, int n_local,
                                       const ippl::Vector<T, dim>& xmin,
                                       const ippl::Vector<T, dim>& xmax,
                                       uint64_t seed = 1337) {
    bunch.create(n_local);

    std::mt19937_64 eng(seed + ippl::Comm->rank());

    std::uniform_real_distribution<T> wdist(0, 1);

    auto R_host = bunch.R.getHostMirror();
    auto Q_host = bunch.Q.getHostMirror();

    double local_sum = 0.0;
    for (int i=0; i < n_local; ++i) {
      ippl::Vector<T, dim> r{};
      for (unsigned d=0; d<dim; ++d) {
        std::uniform_real_distribution<T> pd(xmin[d], xmax[d]);
        r[d] = pd(eng);
      }
      const T q = wdist(eng);
      R_host(i) = r; Q_host(i) = q;
      local_sum += static_cast<double>(q);
    }

    Kokkos::deep_copy(bunch.R.getView(), R_host);
    Kokkos::deep_copy(bunch.Q.getView(), Q_host);
    bunch.update();
    return local_sum;
  }
  
  static double mpi_sum(double x) {
    double g = 0.0;
    MPI_Allreduce(&x, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return g;
  }

  static constexpr T tol() {
      return std::numeric_limits<T>::epsilon() * 10.0;
  }
};

using Precisions = TestParams::Precisions;
using Ranks      = TestParams::Ranks<1, 2, 3>;  
using Tests      = TestForTypes<CreateCombinations<Precisions, Ranks>::type>::type;

TYPED_TEST_SUITE(AssembleRHSTest, Tests);


TYPED_TEST(AssembleRHSTest, ConservationOfTotalWeight) {
  using T   = typename TestFixture::value_type;
  constexpr unsigned Dim = TestFixture::dim;

  using bunch_t = typename TestFixture::bunch_t;
  using playout_t = typename TestFixture::playout_t;
  using field_t = typename TestFixture::field_t;
  using exec_space = typename field_t::view_type::execution_space;


  int nx = 32;

  // Fixed, arbitrary choice of origin and spacing
  ippl::Vector<T, Dim> origin;
  for (unsigned d=0; d<Dim; ++d) {
    origin[d] = T(-0.7 + 0.11*d);
  }

  ippl::Vector<T, Dim> h;
  for (unsigned d=0; d<Dim; ++d) {
    h[d] = T(0.31 + 0.07*d);
  }

  auto owned  = TestFixture::make_owned_nd(nx);
  auto layout = TestFixture::make_layout(owned);
  auto mesh   = TestFixture::make_mesh(owned, h, origin);

  auto rhs    = TestFixture::make_zero_field(mesh, layout);
  auto space  = TestFixture::make_space(mesh);

  playout_t playout(layout, mesh);
  bunch_t   bunch(playout);

  ippl::Vector<T, Dim> xmin, xmax;
  TestFixture::domain_bounds(mesh, nx, xmin, xmax);

  const int    n_local = 1000;
  const double local_w = TestFixture::fill_uniform_particles(bunch, n_local, xmin, xmax);

  auto policy = Kokkos::RangePolicy<exec_space>(0, bunch.getLocalNum());
  ippl::assemble_rhs_from_particles(bunch.Q, rhs, bunch.R, device_space, policy);

  const double global_w = TestFixture::mpi_sum(local_w);
  const double rhs_sum  = static_cast<double>(rhs.sum());

  const double abs_tol = static_cast<double>(TestFixture::tol());
  const double rel_tol = static_cast<double>(TestFixture::tol());

  const double diff   = std::abs(global_w - rhs_sum);
  const double scale  = std::max(std::abs(global_w), std::abs(rhs_sum));
  const double allow  = abs_tol + rel_tol * scale;

  EXPECT_LE(diff, allow)
    << "Conservation check failed:\n"
    << "  rhs_sum               = " << rhs_sum << "\n"
    << "  sum(weights)          = " << global_w << "\n"
    << "  |diff|                = " << diff << "\n"
    << "  abs_tol               = " << abs_tol << "\n"
    << "  rel_tol * max(|.|)    = " << (rel_tol * scale) << "\n"
    << "  allowed (abs+rel * scale)     = " << allow << "\n";
}


int main(int argc, char* argv[]) {
  ippl::initialize(argc, argv);
  int result = 0;
  {
    ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();
  }
  ippl::finalize();
  return result;
}
