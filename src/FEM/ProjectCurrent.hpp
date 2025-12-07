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

  const auto origin = mesh.getOrigin();
  const auto h   = mesh.getMeshSpacing();
  constexpr unsigned Dim = Mesh::Dimension;

  Kokkos::parallel_for("assemble_current_whitney1_make_segments", iteration_policy,
    KOKKOS_LAMBDA(const std::size_t p) {

    auto segs = ippl::GridPathSegmenter<Dim, T, ippl::DefaultCellCrossingRule>
                    ::split(X0(p), X1(p), origin, h);
    
     

    
  });


}

}
#endif
