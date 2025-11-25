//
// High-Order Interpolation
//   Generic templated scatter/gather operations for particle-field interactions
//   using any kernel that provides width() and operator()(x).
//
#ifndef IPPL_HIGH_ORDER_INTERPOLATION_H
#define IPPL_HIGH_ORDER_INTERPOLATION_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <cmath>

#include "Types/Vector.h"

namespace ippl {
namespace detail {

    /**
     * @brief Compute Morton code for spatial sorting
     *
     * Interleaves bits from each dimension to create a single sortable key
     * that preserves spatial locality (Z-order curve).
     *
     * @tparam Dim Number of dimensions
     * @tparam IndexType Integer type for indices
     */
    template <unsigned Dim, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION
    uint64_t computeMortonCode(const Vector<IndexType, Dim>& gridIndices) {
        uint64_t morton = 0;
        constexpr int bits_per_dim = 21;  // Safe for 3D (21*3 = 63 bits)

        for (int bit = 0; bit < bits_per_dim; ++bit) {
            for (unsigned d = 0; d < Dim; ++d) {
                if (gridIndices[d] & (IndexType(1) << bit)) {
                    morton |= (uint64_t(1) << (bit * Dim + d));
                }
            }
        }
        return morton;
    }

    /**
     * @brief Scatter a particle value to 1D field using any higher-order kernel
     *
     * @tparam W Kernel width (compile-time)
     * @tparam View Field view type
     * @tparam Kernel Kernel type (must provide operator()(T))
     * @tparam T Value type
     * @tparam IndexType Index type
     */
    template <int W, typename View, typename Kernel, typename T, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION
    void scatterHighOrderKernel1D(const View& view,
                                  const Kernel& kernel,
                                  IndexType idx0,
                                  T sx,
                                  const typename View::value_type& val,
                                  IndexType ngrid) {
        constexpr T inv_hw = T(2.0) / W;

        for (int i = 0; i < W; ++i) {
            IndexType idx = idx0 + i;
            if (idx >= ngrid) idx -= ngrid;

            T dx = (sx - static_cast<T>(idx0 + i)) * inv_hw;
            T weight = kernel(dx);

            Kokkos::atomic_add(&view(idx), val * weight);
        }
    }

    /**
     * @brief Scatter a particle value to 2D field using any higher-order kernel
     */
    template <int W, typename View, typename Kernel, typename T, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION
    void scatterHighOrderKernel2D(const View& view,
                                  const Kernel& kernel,
                                  const Vector<IndexType, 2>& idx0,
                                  const Vector<T, 2>& sx,
                                  const typename View::value_type& val,
                                  const Vector<IndexType, 2>& ngrid) {
        constexpr T inv_hw = T(2.0) / W;

        // Precompute kernel values
        T kernel_x[W], kernel_y[W];
        for (int i = 0; i < W; ++i) {
            kernel_x[i] = kernel((sx[0] - static_cast<T>(idx0[0] + i)) * inv_hw);
            kernel_y[i] = kernel((sx[1] - static_cast<T>(idx0[1] + i)) * inv_hw);
        }

        for (int j = 0; j < W; ++j) {
            IndexType idy = idx0[1] + j;
            if (idy >= ngrid[1]) idy -= ngrid[1];

            for (int i = 0; i < W; ++i) {
                IndexType idx = idx0[0] + i;
                if (idx >= ngrid[0]) idx -= ngrid[0];

                T weight = kernel_x[i] * kernel_y[j];
                Kokkos::atomic_add(&view(idx, idy), val * weight);
            }
        }
    }

    /**
     * @brief Scatter a particle value to 3D field using any higher-order kernel
     */
    template <int W, typename View, typename Kernel, typename T, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION
    void scatterHighOrderKernel3D(const View& view,
                                  const Kernel& kernel,
                                  const Vector<IndexType, 3>& idx0,
                                  const Vector<T, 3>& sx,
                                  const typename View::value_type& val,
                                  const Vector<IndexType, 3>& ngrid) {
        constexpr T inv_hw = T(2.0) / W;

        // Precompute kernel values
        T kernel_x[W], kernel_y[W], kernel_z[W];
        for (int i = 0; i < W; ++i) {
            kernel_x[i] = kernel((sx[0] - static_cast<T>(idx0[0] + i)) * inv_hw);
            kernel_y[i] = kernel((sx[1] - static_cast<T>(idx0[1] + i)) * inv_hw);
            kernel_z[i] = kernel((sx[2] - static_cast<T>(idx0[2] + i)) * inv_hw);
        }

        for (int k = 0; k < W; ++k) {
            IndexType idz = idx0[2] + k;
            if (idz >= ngrid[2]) idz -= ngrid[2];

            for (int j = 0; j < W; ++j) {
                IndexType idy = idx0[1] + j;
                if (idy >= ngrid[1]) idy -= ngrid[1];

                T weight_yz = kernel_y[j] * kernel_z[k];

                for (int i = 0; i < W; ++i) {
                    IndexType idx = idx0[0] + i;
                    if (idx >= ngrid[0]) idx -= ngrid[0];

                    T weight = kernel_x[i] * weight_yz;
                    Kokkos::atomic_add(&view(idx, idy, idz), val * weight);
                }
            }
        }
    }

    /**
     * @brief Gather field value at particle position using any higher-order kernel (1D)
     */
    template <int W, typename View, typename Kernel, typename T, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION
    typename View::value_type gatherHighOrderKernel1D(const View& view,
                                                      const Kernel& kernel,
                                                      IndexType idx0,
                                                      T sx,
                                                      IndexType ngrid) {
        using value_type = typename View::value_type;
        constexpr T inv_hw = T(2.0) / W;

        value_type result{};

        for (int i = 0; i < W; ++i) {
            IndexType idx = idx0 + i;
            if (idx >= ngrid) idx -= ngrid;

            T dx = (sx - static_cast<T>(idx0 + i)) * inv_hw;
            T weight = kernel(dx);

            result += view(idx) * weight;
        }

        return result;
    }

    /**
     * @brief Gather field value at particle position using any higher-order kernel (2D)
     */
    template <int W, typename View, typename Kernel, typename T, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION
    typename View::value_type gatherHighOrderKernel2D(const View& view,
                                                      const Kernel& kernel,
                                                      const Vector<IndexType, 2>& idx0,
                                                      const Vector<T, 2>& sx,
                                                      const Vector<IndexType, 2>& ngrid) {
        using value_type = typename View::value_type;
        constexpr T inv_hw = T(2.0) / W;

        // Precompute kernel values
        T kernel_x[W], kernel_y[W];
        for (int i = 0; i < W; ++i) {
            kernel_x[i] = kernel((sx[0] - static_cast<T>(idx0[0] + i)) * inv_hw);
            kernel_y[i] = kernel((sx[1] - static_cast<T>(idx0[1] + i)) * inv_hw);
        }

        value_type result{};

        for (int j = 0; j < W; ++j) {
            IndexType idy = idx0[1] + j;
            if (idy >= ngrid[1]) idy -= ngrid[1];

            for (int i = 0; i < W; ++i) {
                IndexType idx = idx0[0] + i;
                if (idx >= ngrid[0]) idx -= ngrid[0];

                T weight = kernel_x[i] * kernel_y[j];
                result += view(idx, idy) * weight;
            }
        }

        return result;
    }

    /**
     * @brief Gather field value at particle position using any higher-order kernel (3D)
     */
    template <int W, typename View, typename Kernel, typename T, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION
    typename View::value_type gatherHighOrderKernel3D(const View& view,
                                                      const Kernel& kernel,
                                                      const Vector<IndexType, 3>& idx0,
                                                      const Vector<T, 3>& sx,
                                                      const Vector<IndexType, 3>& ngrid) {
        using value_type = typename View::value_type;
        constexpr T inv_hw = T(2.0) / W;

        // Precompute kernel values
        T kernel_x[W], kernel_y[W], kernel_z[W];
        for (int i = 0; i < W; ++i) {
            kernel_x[i] = kernel((sx[0] - static_cast<T>(idx0[0] + i)) * inv_hw);
            kernel_y[i] = kernel((sx[1] - static_cast<T>(idx0[1] + i)) * inv_hw);
            kernel_z[i] = kernel((sx[2] - static_cast<T>(idx0[2] + i)) * inv_hw);
        }

        value_type result{};

        for (int k = 0; k < W; ++k) {
            IndexType idz = idx0[2] + k;
            if (idz >= ngrid[2]) idz -= ngrid[2];

            for (int j = 0; j < W; ++j) {
                IndexType idy = idx0[1] + j;
                if (idy >= ngrid[1]) idy -= ngrid[1];

                T weight_yz = kernel_y[j] * kernel_z[k];

                for (int i = 0; i < W; ++i) {
                    IndexType idx = idx0[0] + i;
                    if (idx >= ngrid[0]) idx -= ngrid[0];

                    T weight = kernel_x[i] * weight_yz;
                    result += view(idx, idy, idz) * weight;
                }
            }
        }

        return result;
    }

    /**
     * @brief Runtime dispatch for scatter based on kernel width
     */
    template <int W, int MaxW, typename View, typename Kernel, typename T, typename IndexType, unsigned Dim>
    struct ScatterHighOrderDispatcher {
        KOKKOS_INLINE_FUNCTION
        static void dispatch(int w, const View& view, const Kernel& kernel,
                            const Vector<IndexType, Dim>& idx0,
                            const Vector<T, Dim>& sx,
                            const typename View::value_type& val,
                            const Vector<IndexType, Dim>& ngrid) {
            if (w == W) {
                if constexpr (Dim == 1) {
                    scatterHighOrderKernel1D<W>(view, kernel, idx0[0], sx[0], val, ngrid[0]);
                } else if constexpr (Dim == 2) {
                    scatterHighOrderKernel2D<W>(view, kernel, idx0, sx, val, ngrid);
                } else if constexpr (Dim == 3) {
                    scatterHighOrderKernel3D<W>(view, kernel, idx0, sx, val, ngrid);
                }
            } else if constexpr (W + 1 < MaxW) {
                ScatterHighOrderDispatcher<W + 1, MaxW, View, Kernel, T, IndexType, Dim>::dispatch(
                    w, view, kernel, idx0, sx, val, ngrid);
            }
        }
    };

    /**
     * @brief Runtime dispatch for gather based on kernel width
     */
    template <int W, int MaxW, typename View, typename Kernel, typename T, typename IndexType, unsigned Dim>
    struct GatherHighOrderDispatcher {
        KOKKOS_INLINE_FUNCTION
        static typename View::value_type dispatch(int w, const View& view, const Kernel& kernel,
                                                   const Vector<IndexType, Dim>& idx0,
                                                   const Vector<T, Dim>& sx,
                                                   const Vector<IndexType, Dim>& ngrid) {
            if (w == W) {
                if constexpr (Dim == 1) {
                    return gatherHighOrderKernel1D<W>(view, kernel, idx0[0], sx[0], ngrid[0]);
                } else if constexpr (Dim == 2) {
                    return gatherHighOrderKernel2D<W>(view, kernel, idx0, sx, ngrid);
                } else if constexpr (Dim == 3) {
                    return gatherHighOrderKernel3D<W>(view, kernel, idx0, sx, ngrid);
                }
            } else if constexpr (W + 1 < MaxW) {
                return GatherHighOrderDispatcher<W + 1, MaxW, View, Kernel, T, IndexType, Dim>::dispatch(
                    w, view, kernel, idx0, sx, ngrid);
            }
            return typename View::value_type{};
        }
    };

    /**
     * @brief Compute starting grid index for higher-order kernel interpolation
     *
     * @tparam T Floating point type
     * @param sx Scaled position [0, ngrid)
     * @param w Kernel width
     * @param ngrid Grid size
     * @return Starting index for the kernel stencil
     */
    template <typename T, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION
    IndexType computeHighOrderStartIndex(T sx, int w, IndexType ngrid) {
        const int hw = w / 2;
        const bool odd = (w & 1);

        IndexType idx0 = odd
            ? static_cast<IndexType>(Kokkos::round(sx)) - hw
            : static_cast<IndexType>(sx) + 1 - hw;

        // Handle periodic boundary
        if (idx0 < 0) idx0 += ngrid;
        else if (idx0 >= static_cast<int64_t>(ngrid)) idx0 -= ngrid;

        return idx0;
    }

}  // namespace detail
}  // namespace ippl

#include "Interpolation/HighOrder.hpp"

#endif  // IPPL_HIGH_ORDER_INTERPOLATION_H
