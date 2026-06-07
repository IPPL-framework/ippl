#ifndef IPPL_CURRENT_DEPOSITION_H
#define IPPL_CURRENT_DEPOSITION_H

#include "FEM/GridPathSegmenter.h"

namespace ippl {

/**
 * @brief Deposit current density onto a co-located (non-staggered) grid.
 *
 * For each particle p moving from X0(p) to X1(p) during one time step dt,
 * the trajectory is split into sub-segments each lying within a single mesh
 * cell (via GridPathSegmenter). For each segment, the contribution is
 * scattered onto all 2^Dim surrounding grid nodes using CIC weights 
 * evaluated at the midpoint (all J components share the same grid
 * location on a co-located grid).
 *
 * Writes q*dp[c]/(dt*cell_volume) into component [c+1] of the field vector.
 * Component [0] is reserved for the scalar potential / charge density.
 * This matches the co-located 4-potential FDTD formulation where the source
 * field stores (phi, Jx, Jy, Jz) at each node.
 */
template <typename Mesh,
          typename ChargeAttrib,
          typename PosAttrib,
          typename JField,
          typename policy_type = Kokkos::RangePolicy<>>
inline void assemble_current_collocated(const Mesh& mesh,
                                        const ChargeAttrib& q_attrib,
                                        const PosAttrib& X0,
                                        const PosAttrib& X1,
                                        JField& J_field,
                                        policy_type iteration_policy,
                                        typename Mesh::value_type dt)
{
    using T = typename Mesh::value_type;
    constexpr unsigned Dim = Mesh::Dimension;
    static_assert(Dim == 2 || Dim == 3,
                  "assemble_current_collocated only supports 2D and 3D");

    const auto origin = mesh.getOrigin();
    const auto h      = mesh.getMeshSpacing();
    auto ldom         = J_field.getLayout().getLocalNDIndex();
    const int nghost  = J_field.getNghost();
    auto view         = J_field.getView();

    T volume = T(1);
    for (unsigned d = 0; d < Dim; ++d)
        volume *= h[d];

    Kokkos::parallel_for("assemble_current_collocated", iteration_policy,
        KOKKOS_LAMBDA(const std::size_t p) {

        auto segs = GridPathSegmenter<Dim, T, DefaultCellCrossingRule>
                        ::split(X0(p), X1(p), origin, h);

        const T q_over_dt_vol = q_attrib(p) / (dt * volume);

        for (unsigned i = 0; i < Dim + 1; ++i) {
            const auto& seg = segs[i];

            Vector<T, Dim> dp{};
            T len_sq = T(0);
            for (unsigned d = 0; d < Dim; ++d) {
                dp[d] = seg.p1[d] - seg.p0[d];
                len_sq += dp[d] * dp[d];
            }
            if (len_sq == T(0)) continue;

            Vector<T, Dim> mid{};
            for (unsigned d = 0; d < Dim; ++d)
                mid[d] = T(0.5) * (seg.p0[d] + seg.p1[d]);

            Kokkos::Array<size_t, Dim> cellIdx;
            for (unsigned d = 0; d < Dim; ++d)
                cellIdx[d] = static_cast<size_t>((mid[d] - origin[d]) / h[d]);

            Kokkos::Array<T, Dim> xi;
            for (unsigned d = 0; d < Dim; ++d)
                xi[d] = (mid[d] - origin[d]) / h[d] - T(cellIdx[d]);

            // Scatter to all 2^Dim corners with CIC weights.
            // All J components share the same weight and index set because
            // the grid is co-located (no per-component staggering).
            for (unsigned corner = 0; corner < (1u << Dim); ++corner) {
                size_t idx[Dim];
                T weight = T(1);
                for (unsigned d = 0; d < Dim; ++d) {
                    const unsigned offset = (corner >> d) & 1u;
                    weight *= offset ? xi[d] : (T(1) - xi[d]);
                    idx[d] = cellIdx[d] - ldom.first()[d] + nghost + offset;
                }

                for (unsigned c = 0; c < Dim; ++c) {
                    Kokkos::atomic_add(&(apply(view, idx)[c + 1]),
                                       q_over_dt_vol * dp[c] * weight);
                }
            }
        }
    });
}

} // namespace ippl
#endif
