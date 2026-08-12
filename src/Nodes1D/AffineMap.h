/**
 * @file AffineMap.h
 * @brief Affine maps for 1D quadrature nodes and weights between intervals.
 *
 * Classical Nodes1D rules are defined on [-1, 1]. Use these helpers to map nodes and weights
 * onto another interval [dstA, dstB] while preserving quadrature exactness.
 */
 
#ifndef IPPL_NODES1D_AFFINE_MAP_H
#define IPPL_NODES1D_AFFINE_MAP_H

#include <Kokkos_Macros.hpp>

#include "Types/Vector.h"

namespace ippl {
namespace nodes1d {

    /**
     * @brief Affine map of a point from [srcA, srcB] onto [dstA, dstB].
     *
     * @tparam T Scalar type (typically double).
     * @param x Point in the source interval.
     * @param srcA Source interval left endpoint.
     * @param srcB Source interval right endpoint.
     * @param dstA Destination interval left endpoint.
     * @param dstB Destination interval right endpoint.
     * @return Mapped point in the destination interval.
     * @pre srcB != srcA (non-degenerate source interval).
     */
    template <typename T>
    KOKKOS_INLINE_FUNCTION T affineMapPoint(T x, T srcA, T srcB, T dstA, T dstB) {
        return dstA + (x - srcA) * (dstB - dstA) / (srcB - srcA);
    }

    /**
     * @brief Scale a quadrature weight when the integration interval is affinely mapped.
     *
     * If nodes are mapped with affineMapPoint from [srcA, srcB] to [dstA, dstB], multiply each
     * weight by (dstB - dstA) / (srcB - srcA).
     *
     * @tparam T Scalar type (typically double).
     * @param w Weight on the source interval.
     * @param srcA Source interval left endpoint.
     * @param srcB Source interval right endpoint.
     * @param dstA Destination interval left endpoint.
     * @param dstB Destination interval right endpoint.
     * @return Weight on the destination interval.
     * @pre srcB != srcA.
     */
    template <typename T>
    KOKKOS_INLINE_FUNCTION T affineMapWeight(T w, T srcA, T srcB, T dstA, T dstB) {
        return w * (dstB - dstA) / (srcB - srcA);
    }

    /**
     * @brief In-place affine map of nodes and weights from [srcA, srcB] to [dstA, dstB].
     *
     * @tparam T Element type stored in the vectors.
     * @tparam N Fixed size of both nodes and weights.
     * @param nodes Node array on [srcA, srcB]; overwritten with mapped nodes.
     * @param weights Weight array paired with nodes; overwritten with scaled weights.
     * @param srcA Source interval left endpoint.
     * @param srcB Source interval right endpoint.
     * @param dstA Destination interval left endpoint.
     * @param dstB Destination interval right endpoint.
     * @pre srcB != srcA.
     */
    template <typename T, unsigned N>
    void affineMapNodesWeights(Vector<T, N>& nodes, Vector<T, N>& weights, T srcA, T srcB, T dstA,
                               T dstB) {
        for (unsigned i = 0; i < N; ++i) {
            nodes[i]   = affineMapPoint(nodes[i], srcA, srcB, dstA, dstB);
            weights[i] = affineMapWeight(weights[i], srcA, srcB, dstA, dstB);
        }
    }

}  // namespace nodes1d
}  // namespace ippl

#endif
