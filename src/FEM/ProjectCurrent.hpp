#ifndef IPPL_PROJECT_CURRENT_H
#define IPPL_PROJECT_CURRENT_H

namespace ippl {


template <typename Mesh,
          typename ChargeAttrib,
          typename PosAttrib,
          typename FEMVector,
          typename NedelecSpace,
          typename policy_type = Kokkos::RangePolicy<>>
inline void assemble_current_whitney1(const Mesh& mesh,
                                      const ChargeAttrib& q_attrib,
                                      const PosAttrib& X0,
                                      const PosAttrib& X1,
                                      FEMVector& fem_vector,
                                      const NedelecSpace& space,
                                      policy_type iteration_policy)
{

  using T = Mesh::value_type;

  const auto org_v = mesh.getOrigin();       // ippl::Vector<T,Dim>
  const auto h_v   = mesh.getMeshSpacing();  // ippl::Vector<T,Dim>
  constexpr unsigned Dim = Mesh::Dimension;

  std::array<T,Dim> origin_a{};
  std::array<T,Dim> h_a{};
  for (unsigned d = 0; d < Dim; ++d) {
    origin_a[d] = org_v[d];
    h_a[d]      = h_v[d];
  }

  Kokkos::parallel_for("assemble_current_whitney1_make_segments", iteration_policy,
    KOKKOS_LAMBDA(const std::size_t p) {

    std::cout << X0(p) << std::endl;
    std::cout << X1(p) << std::endl;
    
  });


}

}
#endif
