/**
 * @file Nodes1DDetail.hpp
 * @brief Internal helpers for Nodes1D (not part of the public API).
 * @internal Included by computeGauss* View overloads; do not include directly.
 *
 * Host-only: fillHostThenDeepCopy runs quadrature generation on the host and copies
 * results into views via Kokkos::deep_copy (no parallel_for for setup).
 */
#ifndef IPPL_NODES1D_DETAIL_HPP
#define IPPL_NODES1D_DETAIL_HPP

#include <cassert>
#include <cstddef>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Types/Vector.h"

namespace ippl {
namespace nodes1d {
namespace detail {

    /**
     * @internal Fill fixed-size Vector outputs via a runtime-n fill callback (host only).
     *
     * fill(n, x, w) must write n nodes/weights into x and w; n is always N.
     */
    template <typename T, unsigned N, typename Scalar, typename FillFn>
    void fillFixedVectors(Vector<T, N>& nodes, Vector<T, N>& weights, FillFn&& fill) {
        Scalar x[N];
        Scalar w[N];
        fill(static_cast<std::size_t>(N), x, w);
        for (unsigned i = 0; i < N; ++i) {
            nodes[i]   = static_cast<T>(x[i]);
            weights[i] = static_cast<T>(w[i]);
        }
    }

    /**
     * @internal Compute nodes/weights on the host via fill(n, nodes, weights), then deep-copy
     * into the destination views. Does not run quadrature generation on device.
     */
    template <typename ExecSpace, typename RealType, typename FillFn>
    void fillHostThenDeepCopy(Kokkos::View<RealType*, typename ExecSpace::memory_space>& nodes,
                              Kokkos::View<RealType*, typename ExecSpace::memory_space>& weights,
                              FillFn&& fill) {
        const int n = static_cast<int>(nodes.extent(0));
        assert(weights.extent(0) == nodes.extent(0));
        assert(n >= 1);

        auto h_nodes   = Kokkos::create_mirror_view(Kokkos::HostSpace(), nodes);
        auto h_weights = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);

        std::vector<RealType> nbuf(static_cast<std::size_t>(n));
        std::vector<RealType> wbuf(static_cast<std::size_t>(n));
        fill(static_cast<std::size_t>(n), nbuf.data(), wbuf.data());
        for (int i = 0; i < n; ++i) {
            h_nodes(i)   = nbuf[static_cast<std::size_t>(i)];
            h_weights(i) = wbuf[static_cast<std::size_t>(i)];
        }
        Kokkos::deep_copy(nodes, h_nodes);
        Kokkos::deep_copy(weights, h_weights);
    }

}  // namespace detail
}  // namespace nodes1d
}  // namespace ippl

#endif
