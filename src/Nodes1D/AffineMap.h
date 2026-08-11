//
// AffineMap — map 1D nodes/weights between intervals.
// Native Nodes1D interval for classical GL/GLL/Jacobi is [-1, 1].
//
#ifndef IPPL_NODES1D_AFFINE_MAP_H
#define IPPL_NODES1D_AFFINE_MAP_H

#include <Kokkos_Macros.hpp>

#include "Types/Vector.h"

namespace ippl {
namespace nodes1d {

    /**
     * @brief Affine map of a scalar from [srcA, srcB] onto [dstA, dstB].
     */
    template <typename T>
    KOKKOS_INLINE_FUNCTION T affineMapPoint(T x, T srcA, T srcB, T dstA, T dstB) {
        return dstA + (x - srcA) * (dstB - dstA) / (srcB - srcA);
    }

    /**
     * @brief Scale a quadrature weight when mapping from [srcA, srcB] to [dstA, dstB].
     */
    template <typename T>
    KOKKOS_INLINE_FUNCTION T affineMapWeight(T w, T srcA, T srcB, T dstA, T dstB) {
        return w * (dstB - dstA) / (srcB - srcA);
    }

    /**
     * @brief In-place map of nodes and weights from [srcA, srcB] to [dstA, dstB].
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
