#include "Ippl.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

#include <limits>
#include <random>
#include <algorithm>

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
  using Quad = ippl::MidpointQuadrature<T, 2, Elem>;
  template<class Field>
  using Space = ippl::LagrangeSpace<T, 2, 1, Elem, Quad, Field, Field>;
  static Elem make_elem() { return Elem{}; }
  static Quad make_quad(const Elem& e) { return Quad(e); }
};

template <typename T>
struct ElemSelector<3, T> {
  using Elem = ippl::HexahedralElement<T>;
  using Quad = ippl::MidpointQuadrature<T, 3, Elem>;
  template<class Field>
  using Space = ippl::LagrangeSpace<T, 3, 1, Elem, Quad, Field, Field>;
  static Elem make_elem() { return Elem{}; }
  static Quad make_quad(const Elem& e) { return Quad(e); }
};

template <typename> class InterpTest;

template <typename T, unsigned Dim>
class InterpTest<Parameters<T, Rank<Dim>>> : public ::testing::Test {
public:
  using value_type = T;
  static constexpr unsigned dim = Dim;

  using Mesh_t      = ippl::UniformCartesian<T, dim>;
  using Centering_t = typename Mesh_t::DefaultCentering;
  using field_t     = ippl::Field<T, dim, Mesh_t, Centering_t>;
  using ElemSel     = ElemSelector<dim, T>;
  using Space       = typename ElemSel::template Space<field_t>;

  using playout_t   = ippl::ParticleSpatialLayout<T, dim>;
  using bunch_t     = Bunch<playout_t>;

  static constexpr T tol() {
    return std::numeric_limits<T>::epsilon() * 10.0;
  }

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

  static Space make_space(Mesh_t& mesh, const ippl::FieldLayout<dim>& layout) {
    auto elem = ElemSel::make_elem();
    auto quad = ElemSel::make_quad(elem);
    return Space(mesh, elem, quad, layout);
  }


  template <class Func>
  static void set_field_from_function(field_t& coeffs, const Func& u) {
    constexpr unsigned D = dim;
    using exec_space = typename field_t::view_type::execution_space;

    auto view  = coeffs.getView();
    auto M     = coeffs.get_mesh();
    auto lDom  = coeffs.getLayout().getLocalNDIndex();
    const int nghost = coeffs.getNghost();

    ippl::Vector<int,D> first = lDom.first();
    ippl::Vector<int,D> last  = lDom.last();
    ippl::Vector<int,D> ext;
    for (unsigned d=0; d<D; ++d) ext[d] = last[d] - first[d] + 1;

    ippl::Vector<int,D> size;                               // ext + 2*nghost per axis
    for (unsigned d=0; d<D; ++d) size[d] = ext[d] + 2*nghost;

    if constexpr (D == 1) {
      Kokkos::parallel_for("fill_field_fun_1d",
        Kokkos::RangePolicy<exec_space>(0, size[0]),
        KOKKOS_LAMBDA(const int i0) {
          const int v0 = first[0] + (i0 - nghost);
          ippl::Vector<T,D> x;
          x[0] = M.getOrigin()[0] + M.getMeshSpacing()[0] * T(v0);
          view(size_t(i0)) = u(x);
        });
    } else if constexpr (D == 2) {
      Kokkos::parallel_for("fill_field_fun_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>({0,0}, {size[0], size[1]}),
        KOKKOS_LAMBDA(const int i0, const int i1) {
          const int v0 = first[0] + (i0 - nghost);
          const int v1 = first[1] + (i1 - nghost);
          ippl::Vector<T,D> x;
          x[0] = M.getOrigin()[0] + M.getMeshSpacing()[0] * T(v0);
          x[1] = M.getOrigin()[1] + M.getMeshSpacing()[1] * T(v1);
          view(size_t(i0), size_t(i1)) = u(x);
        });
    } else { // D == 3
      Kokkos::parallel_for("fill_field_fun_3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec_space>({0,0,0}, {size[0], size[1], size[2]}),
        KOKKOS_LAMBDA(const int i0, const int i1, const int i2) {
          const int v0 = first[0] + (i0 - nghost);
          const int v1 = first[1] + (i1 - nghost);
          const int v2 = first[2] + (i2 - nghost);
          ippl::Vector<T,D> x;
          x[0] = M.getOrigin()[0] + M.getMeshSpacing()[0] * T(v0);
          x[1] = M.getOrigin()[1] + M.getMeshSpacing()[1] * T(v1);
          x[2] = M.getOrigin()[2] + M.getMeshSpacing()[2] * T(v2);
          view(size_t(i0), size_t(i1), size_t(i2)) = u(x);
        });
    }

    Kokkos::fence();
  }

  static void fill_field_constant(field_t& coeffs, T c) {
    Kokkos::deep_copy(coeffs.getView(), c);
  }


  // Create particles uniformly in the GLOBAL domain 
  static void fill_uniform_positions(bunch_t& bunch, int n_local,
                                            const Mesh_t& M,
                                            uint64_t seed = 4242) {
    bunch.create(n_local);
    std::mt19937_64 eng(seed + ippl::Comm->rank());
    auto R_host = bunch.R.getHostMirror();

    ippl::Vector<T,dim> xmin, xmax;
    domain_bounds(M, xmin, xmax);

    // small interior margin vs locator’s open upper bound
    ippl::Vector<T,dim> h = M.getMeshSpacing();

    for (int i=0; i<n_local; ++i) {
      ippl::Vector<T,dim> r{};
      for (unsigned d=0; d<dim; ++d) {
        std::uniform_real_distribution<T> pd(xmin[d] + 3 * h[d], xmax[d] - 3 * h[d]);
        r[d] = pd(eng);
      }
      R_host(i) = r;
    }
    Kokkos::deep_copy(bunch.R.getView(), R_host);
    bunch.update(); // migrate to owning ranks
  }

  static void domain_bounds(const Mesh_t& M,
                                   ippl::Vector<T,dim>& xmin,
                                   ippl::Vector<T,dim>& xmax) {
    const auto org = M.getOrigin();
    const auto h   = M.getMeshSpacing();
    const auto nr  = M.getGridsize();
    for (unsigned d=0; d<dim; ++d) {
      xmin[d] = org[d];
      xmax[d] = org[d] + h[d]*T(nr[d] - 1);
    }
  }

  static double mpi_max(double x) {
    double g = 0.0;
    MPI_Allreduce(&x, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return g;
  }

};

using Precisions = TestParams::Precisions;
using Ranks      = TestParams::Ranks<1,2,3>;
using Tests      = TestForTypes<CreateCombinations<Precisions, Ranks>::type>::type;

TYPED_TEST_SUITE(InterpTest, Tests);

template <typename TestFixture>
void InterpolatesAffineExactly() {
  using T          = typename TestFixture::value_type;
  constexpr unsigned Dim = TestFixture::dim;
  using field_t    = typename TestFixture::field_t;
  using exec_space = typename field_t::view_type::execution_space;
  using playout_t  = typename TestFixture::playout_t;
  using bunch_t    = typename TestFixture::bunch_t;

  // Fixed, arbitrary choice of origin and spacing
  const int nx = 10; 
  ippl::Vector<T, Dim> origin, h;
  for (unsigned d=0; d<Dim; ++d) { 
    origin[d] = T(-0.7 + 0.11*d); 
    h[d] = T(0.31 + 0.07*d); 
  }

  auto owned  = TestFixture::make_owned_nd(nx);
  auto layout = TestFixture::make_layout(owned);
  auto mesh   = TestFixture::make_mesh(owned, h, origin);


  // analytic affine function u(x) = c + sum a_d * x[d]
  const T c = T(10.0);
  ippl::Vector<T,Dim> a; 
  for (unsigned d=0; d<Dim; ++d) {
    a[d] = T(0.21 + 0.1*d); 
  }

  auto u = KOKKOS_LAMBDA(const ippl::Vector<T,Dim>& x) -> T {
    T s = c;
    for (unsigned d=0; d<Dim; ++d) s += a[d]*x[d];
    return s;
  };

  auto coeffs = TestFixture::make_zero_field(mesh, layout);
  TestFixture::set_field_from_function(coeffs, u);

  auto view  = coeffs.getView();

  auto space = TestFixture::make_space(mesh, layout);

  playout_t playout(layout, mesh);
  bunch_t   bunch(playout);

  const int n_local = 1000;
  TestFixture::fill_uniform_positions(bunch, n_local, mesh, /*seed=*/42);

  auto policy = Kokkos::RangePolicy<exec_space>(0, bunch.getLocalNum());
  Kokkos::deep_copy(bunch.Q.getView(), T(0));
  ippl::interpolate_to_diracs(bunch.Q, coeffs, bunch.R, space, policy);

  auto d_pos = bunch.R.getView();
  auto d_out = bunch.Q.getView();

  double max_err = 0.0, max_ref = 0.0;
  Kokkos::parallel_reduce("interp_affine_err", policy,
    KOKKOS_LAMBDA(const int p, double& lmax) {
      const T exact = u(d_pos(p));
      const T err   = Kokkos::fabs(d_out(p) - exact);
      if (double(err) > lmax) lmax = double(err);
    }, Kokkos::Max<double>(max_err));

  Kokkos::parallel_reduce("interp_affine_ref", policy,
    KOKKOS_LAMBDA(const int p, double& lmax) {
      const T exact = u(d_pos(p));
      const double ref = std::abs(double(exact));
      if (ref > lmax) lmax = ref;
    }, Kokkos::Max<double>(max_ref));

    // global (MPI) max across ranks
  const double g_err = TestFixture::mpi_max(max_err);
  const double g_ref = TestFixture::mpi_max(max_ref);

  // combined abs+rel tolerance based on fixture tol()
  const double abs_tol = static_cast<double>(TestFixture::tol());
  const double rel_tol = static_cast<double>(TestFixture::tol());
  const double allow   = abs_tol + rel_tol * g_ref;

  EXPECT_LE(g_err, allow)
    << "Interpolation error too large for affine function over global domain:\n"
    << "  max |u_h(x_p) - u(x_p)| = " << g_err << "\n"
    << "  ref scale max|u(x_p)|   = " << g_ref << "\n"
    << "  allowed (abs+rel)       = " << allow
    << " (abs_tol=" << abs_tol << ", rel_tol*ref=" << rel_tol * g_ref << ")";

  std::cout << max_err << std::endl;
}

TYPED_TEST(InterpTest, InterpolatesAffineExactly) {
  InterpolatesAffineExactly<TestFixture>();
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
