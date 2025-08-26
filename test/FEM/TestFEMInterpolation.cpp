#include "Ippl.h"

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



// Translate global x -> (element ND index, local reference xi) for a UniformCartesian mesh
template <class T, unsigned Dim, class Mesh>
inline void global_to_element_local(
    const Mesh& mesh,
    const ippl::Vector<T,Dim>& x,
    ippl::Vector<size_t,Dim>& e_nd,          // OUT: element ND index
    ippl::Vector<T,Dim>& xi                  // OUT: local [0,1]^Dim coord
) {
  const auto nr   = mesh.getGridsize();      // vertices per axis
  const auto h    = mesh.getMeshSpacing();   // spacing per axis
  const auto org  = mesh.getOrigin();        // origin

  for (unsigned d=0; d<Dim; ++d) {
    const T s     = (x[d] - org[d]) / h[d];           // scaled global → cell units
    const T cells = static_cast<T>(nr[d] - 1);        // cells per axis

    if (s <= T(0)) {
      e_nd[d] = 0; xi[d] = T(0);
    } else if (s >= cells) {
      e_nd[d] = nr[d] - 2; xi[d] = T(1);
    } else {
      const auto e = static_cast<size_t>(std::floor(s));
      e_nd[d] = (e < (nr[d]-1)) ? e : (nr[d]-2);
      T loc = s - static_cast<T>(e_nd[d]);           // in [0,1)
      if (loc < T(0)) loc = T(0);
      if (loc > T(1)) loc = T(1);
      xi[d] = loc;
    }
  }
}

template <class Space, class T, unsigned Dim, class Mesh>
inline std::pair<size_t, ippl::Vector<T,Dim>>
global_to_element_linear_and_xi(
    const Mesh& mesh, const Space& space, const ippl::Vector<T,Dim>& x)
{
  ippl::Vector<size_t,Dim> e_nd;
  ippl::Vector<T,Dim> xi;
  global_to_element_local<T,Dim>(mesh, x, e_nd, xi);
  const size_t e_lin = space.getElementIndex(e_nd);
  return {e_lin, xi};
}



template <typename Attrib1, typename Field, typename Attrib2, typename Space,
          typename policy_type = Kokkos::RangePolicy<typename Field::execution_space>>
inline void scatter(const Attrib1& attrib, Field& f, const Attrib2& pp,
                    const Space& space, policy_type iteration_policy)
{

  constexpr unsigned Dim = Field::dim;

  using T               = typename Field::value_type;
  using view_type       = typename Field::view_type;
  using mesh_type       = typename Field::Mesh_t;
  using vector_type     = typename mesh_type::vector_type;

  static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter(P1-FEM)");
  IpplTimings::startTimer(scatterTimer);

  // Field view (execution-space)
  view_type view = f.getView();

  // Mesh / layout (used for locating and indexing the field view)
  const mesh_type& mesh  = f.get_mesh();
  const vector_type origin = mesh.getOrigin();
  const vector_type dx     = mesh.getMeshSpacing();

  const ippl::FieldLayout<Dim>& layout = f.getLayout();
  const ippl::NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
  const int nghost               = f.getNghost();

  // Particle attribute/device views
  auto d_attr = attrib.getView();   // scalar weight per particle (e.g. charge)
  auto d_pos  = pp.getView();       // positions (Vector<T,Dim>) per particle

    auto atomic_add_vertex = [&](const ippl::Vector<size_t,Dim>& args, T v) KOKKOS_FUNCTION {
      if constexpr (Dim == 1) {
        Kokkos::atomic_add(&view(args[0]), v);
      } else if constexpr (Dim == 2) {
        Kokkos::atomic_add(&view(args[0], args[1]), v);
      } else if constexpr (Dim == 3) {
        Kokkos::atomic_add(&view(args[0], args[1], args[2]), v);
      } else {
      }
    };

    Kokkos::parallel_for("ParticleAttrib::scatter_FEM_P1", iteration_policy,
    KOKKOS_LAMBDA(const size_t p) {


        const auto x   = d_pos(p);     // ippl::Vector<T,Dim>
        const T    val = d_attr(p);    // scalar weight

        const auto located = global_to_element_linear_and_xi<Space, T, Dim, mesh_type>(mesh, space, x);
        const size_t e_lin = located.first;
        const ippl::Vector<T,Dim> xi = located.second;

        const auto e_nd  = space.getElementNDIndex(e_lin);
        const auto verts = space.getElementMeshVertexIndices(e_nd);
        const auto dofs  = space.getGlobalDOFIndices(e_lin);  


        for (size_t a = 0; a < dofs.dim; ++a) {
          const size_t local = space.getLocalDOFIndex(e_lin, dofs[a]);
          const T w = space.evaluateRefElementShapeFunction(local, xi);

          const auto v_nd = space.getMeshVertexNDIndex(verts[a]); // ND integer coords
          ippl::Vector<size_t,Dim> args;                          // indices into view
          for (unsigned d = 0; d < Dim; ++d)
            args[d] = static_cast<size_t>(v_nd[d] - lDom.first()[d] + nghost);

          atomic_add_vertex(args, val * w);
        }

    });
}



template <typename AttribOut, typename Field, typename PosAttrib, typename Space,
          typename policy_type = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>>
inline void gather(AttribOut& attrib_out, const Field& coeffs, const PosAttrib& pp,
                   const Space& space, policy_type iteration_policy)
{
  constexpr unsigned Dim = Field::dim;
  using T   = typename AttribOut::value_type;   // scalar written to particles
  using field_value  = typename Field::value_type;       // scalar nodal coefficient
  using view_type    = typename Field::view_type;  // read-only view of coeffs
  using mesh_type    = typename Field::Mesh_t;
  using vector_type  = typename mesh_type::vector_type;


  static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather(P1-FEM)");
  IpplTimings::startTimer(gatherTimer);

  view_type view     = coeffs.getConstView();
  const mesh_type& mesh = coeffs.get_mesh();

  const ippl::FieldLayout<Dim>& layout = coeffs.getLayout();
  const ippl::NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
  const int nghost               = coeffs.getNghost();

  ippl::Vector<int,Dim> start;
  for (unsigned d=0; d<Dim; ++d) start[d] = lDom.first()[d];

  auto d_pos = pp.getView();
  auto d_out = attrib_out.getView();


  Kokkos::parallel_for("gather_P1_FEM_policy", iteration_policy,
    KOKKOS_LAMBDA(const size_t p) {
      const auto x = d_pos(p);

      const auto located = global_to_element_linear_and_xi<Space, T, Dim, mesh_type>(mesh, space, x);
      const size_t e_lin = located.first;
      const ippl::Vector<T,Dim> xi = located.second;
      const auto e_nd  = space.getElementNDIndex(e_lin);
      
      const auto verts = space.getElementMeshVertexIndices(e_nd);
      const auto dofs  = space.getGlobalDOFIndices(e_nd);
      field_value up = field_value(0);
      for (size_t a = 0; a < dofs.size(); ++a) {
        const size_t local = space.getLocalDOFIndex(e_lin, dofs[a]);
        const field_value w = space.evaluateRefElementShapeFunction(local, xi);

        const auto v_nd = space.getMeshVertexNDIndex(verts[a]);
        ippl::Vector<size_t,Dim> args;
        for (unsigned d=0; d<Dim; ++d)
          args[d] = static_cast<size_t>(v_nd[d] - start[d] + nghost);

        const field_value ua = read_vertex<Dim>(view, args);
        up += ua * w;
      }

      d_out(p) = static_cast<T>(up);

    });
  IpplTimings::stopTimer(gatherTimer);

}


int main(int argc, char* argv[]) {
  ippl::initialize(argc, argv);
  {
    using T = double; constexpr unsigned Dim = 1;
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef Bunch<playout_type> bunch_type;
    using Mesh_t = ippl::UniformCartesian<T, Dim>;
    using Centering_t = typename Mesh_t::DefaultCentering;

    int nx = 16;
    ippl::Index I(nx);
    ippl::NDIndex<Dim> owned(I);

    std::array<bool, Dim> par{}; par.fill(true);
    ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, par);

    ippl::Vector<T, Dim> h{1.0 / T(nx - 1)};
    ippl::Vector<T, Dim> origin{0.0};
    Mesh_t mesh(owned, h, origin);

    using field_t = ippl::Field<T, Dim, Mesh_t, Centering_t>;
    field_t coeffs; coeffs.initialize(mesh, layout);
    field_t rhs;    rhs.initialize(mesh, layout);

    using Elem = ippl::EdgeElement<T>;
    Elem ref_elem;
    using Quad = ippl::MidpointQuadrature<T, 1, Elem>;
    Quad quad(ref_elem);
    using Space = ippl::LagrangeSpace<T, Dim, 1, Elem, Quad, field_t, field_t>;
    Space space(mesh, ref_elem, quad);

    playout_type playout(layout, mesh);
    bunch_type bunch(playout);
    int n = 8;
    bunch.create(n);

    std::mt19937_64 eng;
    std::uniform_real_distribution<double> unif(0, 1);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    typename bunch_type::charge_container_type::HostMirror Q_host  = bunch.Q.getHostMirror();
    for (int i = 0; i < n; ++i) {
        ippl::Vector<double, 1> r = {unif(eng)};
        R_host(i)                 = r;
        Q_host(i)                 = 1.0;
    }
    Kokkos::deep_copy(bunch.R.getView(), R_host);
    Kokkos::deep_copy(bunch.Q.getView(), Q_host);

    bunch.update();

    auto policy = Kokkos::RangePolicy<field_t::execution_space>(0, bunch.getLocalNum());

    scatter(bunch.Q, rhs, bunch.R, space, policy);
    std::cout << rhs.sum() << std::endl;
  }

  ippl::finalize();
  return 0;
}
