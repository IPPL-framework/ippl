#ifndef IPPL_PROJECT_CURRENT_H
#define IPPL_PROJECT_CURRENT_H

namespace ippl {

/**
 * @brief Assemble the current density RHS vector for a Nedelec FEM space.
 *
 * For each particle p moving from X0(p) to X1(p) during one time step dt,
 * the particle trajectory is split into sub-segments that each lie within a
 * single mesh cell (via GridPathSegmenter).
 * Each sub-segment's contribution to the current density is computed and then 
 * scattered onto the edge DOFs of the cell that contains the sub-segment's midpoint,
 * using the Whitney-1 basis functions evaluated at the midpoint (equivalent to linear interpolation).  
 * 
 */
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
                                      policy_type iteration_policy,
                                      typename Mesh::value_type dt)
{
    using T = typename Mesh::value_type;
    constexpr unsigned Dim = Mesh::Dimension;

    const auto origin = mesh.getOrigin();
    const auto h      = mesh.getMeshSpacing();
    auto ldom         = space.getLocalNDIndex();

    // Atomic view for safe concurrent scatter from multiple particles.
    using AtomicViewType = Kokkos::View<T*, Kokkos::MemoryTraits<Kokkos::Atomic>>;
    AtomicViewType atomic_view = fem_vector.getView();

    constexpr unsigned numDOFs = NedelecSpace::numElementDOFs;

    Kokkos::parallel_for("assemble_current_whitney1", iteration_policy,
        KOKKOS_LAMBDA(const std::size_t p) {

        // Split trajectory
        auto segs = GridPathSegmenter<Dim, T, DefaultCellCrossingRule>
                        ::split(X0(p), X1(p), origin, h);

        const T q_over_dt = q_attrib(p) / dt;

        for (unsigned i = 0; i < Dim + 1; ++i) {
            const auto& seg = segs[i];

            // Calcualte displacement of this sub-segment
            Vector<T, Dim> dp{};
            T len_sq = T(0);
            for (unsigned d = 0; d < Dim; ++d) {
                dp[d] = seg.p1[d] - seg.p0[d];
                len_sq += dp[d] * dp[d];
            }
            //Skip unphisical zero-segments
            if (len_sq == T(0)) continue;

            // Calculate midpoint of the segment
            Vector<T, Dim> mid{};
            for (unsigned d = 0; d < Dim; ++d)
                mid[d] = T(0.5) * (seg.p0[d] + seg.p1[d]);

            // Index of the cell that contains the midpoint
            typename NedelecSpace::indices_t cellIdx{};
            for (unsigned d = 0; d < Dim; ++d)
                cellIdx[d] = static_cast<size_t>((mid[d] - origin[d]) / h[d]);

            // Calculate local reference coordinate
            typename NedelecSpace::point_t xi{};
            for (unsigned d = 0; d < Dim; ++d)
                xi[d] = (mid[d] - origin[d]) / h[d] - T(cellIdx[d]);

            // FEM-vector indices for the edge DOFs of this cell.
            auto dofIdx = space.getFEMVectorDOFIndices(cellIdx, ldom);

            // Scatter onto each edge DOF k.
            for (unsigned k = 0; k < numDOFs; ++k) {
                auto phi_k = space.evaluateRefElementShapeFunction(k, xi);
                T contrib = T(0);
                for (unsigned d = 0; d < Dim; ++d)
                    contrib += q_over_dt * dp[d] * phi_k[d];
                atomic_view(dofIdx[k]) += contrib;
            }
        }
    });
}

} // namespace ippl
#endif
