#ifndef IPPL_PROJECT_CURRENT_FDTD_H
#define IPPL_PROJECT_CURRENT_FDTD_H

#include "FEM/GridPathSegmenter.h"

namespace ippl {

/**
 * @brief Deposit current density onto a Yee-staggered grid.
 *
 * For each particle p moving from X0(p) to X1(p) during one time step dt,
 * the trajectory is split into sub-segments each lying within a single mesh
 * cell (via GridPathSegmenter). Each segment's contribution to J is
 * scattered onto the 2^(Dim-1) Yee-grid nodes using linear
 * interpolation weights evaluated at the midpoint.
 *
 */
template <typename Mesh,
          typename ChargeAttrib,
          typename PosAttrib,
          typename JField,
          typename policy_type = Kokkos::RangePolicy<>>
inline void assemble_current_yee(const Mesh& mesh,
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
                  "assemble_current_yee only supports 2D and 3D");

    const auto origin = mesh.getOrigin();
    const auto h      = mesh.getMeshSpacing();
    auto ldom         = J_field.getLayout().getLocalNDIndex();
    const int nghost  = J_field.getNghost();
    auto view         = J_field.getView();

    Kokkos::parallel_for("assemble_current_yee", iteration_policy,
        KOKKOS_LAMBDA(const std::size_t p) {

        auto segs = ippl::GridPathSegmenter<Dim, T, ippl::DefaultCellCrossingRule>
                        ::split(X0(p), X1(p), origin, h);

        const T q_over_dt = q_attrib(p) / dt;

        for (unsigned i = 0; i < Dim + 1; ++i) {
            const auto& seg = segs[i];

            ippl::Vector<T, Dim> dp{};
            T len_sq = T(0);
            for (unsigned d = 0; d < Dim; ++d) {
                dp[d] = seg.p1[d] - seg.p0[d];
                len_sq += dp[d] * dp[d];
            }
            if (len_sq == T(0)) continue;

            ippl::Vector<T, Dim> mid{};
            for (unsigned d = 0; d < Dim; ++d)
                mid[d] = T(0.5) * (seg.p0[d] + seg.p1[d]);

            size_t cellIdx[Dim];
            for (unsigned d = 0; d < Dim; ++d)
                cellIdx[d] = static_cast<size_t>((mid[d] - origin[d]) / h[d]);

            T xi[Dim];
            for (unsigned d = 0; d < Dim; ++d)
                xi[d] = (mid[d] - origin[d]) / h[d] - T(cellIdx[d]);

            // For component c, scatter to 2^(Dim-1) transverse neighbours.
            // In direction c (staggered): fixed cell face, no weight.
            // In each transverse direction d: linear CIC weight between
            // nodes cellIdx[d] and cellIdx[d]+1.
            for (unsigned c = 0; c < Dim; ++c) {
                const T val_c = q_over_dt * dp[c];

                for (unsigned corner = 0; corner < (1u << (Dim - 1)); ++corner) {
                    size_t idx[Dim];
                    T weight = T(1);
                    unsigned bit = 0;
                    for (unsigned d = 0; d < Dim; ++d) {
                        if (d == c) {
                            idx[d] = cellIdx[d] - ldom.first()[d] + nghost;
                        } else {
                            const unsigned offset = (corner >> bit) & 1u;
                            weight *= offset ? xi[d] : (T(1) - xi[d]);
                            idx[d] = cellIdx[d] - ldom.first()[d] + nghost + offset;
                            ++bit;
                        }
                    }

                    if constexpr (Dim == 2) {
                        Kokkos::atomic_add(&(view(idx[0], idx[1])[c]),
                                           val_c * weight);
                    } else {
                        Kokkos::atomic_add(&(view(idx[0], idx[1], idx[2])[c]),
                                           val_c * weight);
                    }
                }
            }
        }
    });
}

} // namespace ippl
#endif
