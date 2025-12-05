#ifndef IPPL_INTERPOLATION_UTIL_H
#define IPPL_INTERPOLATION_UTIL_H

#include "Types/IpplTypes.h"

#include "Index/Index.h"

namespace ippl::Interpolation::detail {
    // (paul) I saw quite non-uniform usage of 32/64-bit types in IPPL. This just obtains what an
    // Index gives.
    using local_index_type = decltype(Index{}.first());
    using size_type        = ippl::detail::size_type;

    /** @brief Returns a wrapped and scaled point in [0, n_grid)
     *
     * This scales an input point x so that the result is in [0, n_grid). The function is
     * periodic in [0, 2*pi). This is numerically inaccuracte for very large inputs.
     *
     * @tparam T The floating-point type
     * @param x Non-uniform point, periodic in [0, 2*pi]
     * @param n_grid (Total) grid size
     * @return Point wrapped periodically and scaled to grid indices
     */
    template <typename T>
    KOKKOS_INLINE_FUNCTION T scale_to_grid_indices(T x, size_type n_grid) {
        constexpr T inv_two_pi = T(0.5) / std::numbers::pi_v<T>;
        // [0, 2*pi] -> [0, 1]
        T k = x * inv_two_pi;
        // Wrap all values to [0, 1]
        k -= Kokkos::floor(k);
        // -> [0, n_grid]
        T sx = k * n_grid;

        return sx;
    }

    /** @brief Takes a grid point x_i in a NUFFT and assigns to it a grid index in a grid [0,n_grid).
     *
     * Let i be the return value. The index i is chosen such that, if the width of the kernel is W,
     * the non-zero contributions to the grid from this point are contained within the indices
     * [i - (w-1)/2, i - (w-1)/2 + w). The w input is necessary because for even kernel values,
     * we always have w/2 non-zero grid points with contributions to the left and w/2 to the right
     * of x, while for uneven w, we have w/2 + 1 contributions on the side that x lies closer to.
     *
     * This function takes points in [0, n_grid), already scaled to the grid
     *
     * @tparam T The floating-point type
     * @param x Non-uniform grid point in [0, n_grid)
     * @param n_grid (Total) grid size
     * @param w Width of the kernel
     * @return Index as described above
     */
    template <typename T>
    KOKKOS_INLINE_FUNCTION local_index_type grid_point_to_grid_idx(T sx, size_type n_grid,
                                                                      int w) {
        // See comment above
        const bool odd_w = w & 1;
        int idx_0        = odd_w ? static_cast<local_index_type>(Kokkos::round(sx))
                                 : static_cast<local_index_type>(sx);
        // idx_0 might now be exactly n_grid. We return a value [0, n_grid)
        // if (idx_0 == static_cast<local_index_type>(n_grid)) {
        //     idx_0 -= static_cast<local_index_type>(n_grid);
        // }

        return idx_0;
    }

    /** @brief Takes a point x_i in a NUFFT and assigns to it a grid index in a grid [0,n_grid).
     *
     * Let i be the return value. The index i is chosen such that, if the width of the kernel is W,
     * the non-zero contributions to the grid from this point are contained within the indices
     * [i - (w-1)/2, i - (w-1)/2 + w]. The w input is necessary because for even kernel values,
     * we always have w/2 non-zero grid points with contributions to the left and w/2 to the right
     * of x, while for uneven w, we have w/2 + 1 contributions on the side that x lies closer to.
     *
     * This function wraps inputs to [0, 2*pi], so arbitrary inputs are allowed. Numerical accuracy
     * will be bad for very large inputs.
     *
     * @tparam T The floating-point type
     * @param x Non-uniform point, periodic in [0, 2*pi]
     * @param n_grid (Total) grid size
     * @param w Width of the kernel
     * @return Index as described above
     */
    template <typename T>
    KOKKOS_INLINE_FUNCTION local_index_type fourier_point_to_grid_idx(T x, size_type n_grid,
                                                                      int w) {
        T sx = scale_to_grid_indices(x, n_grid);
        return grid_point_to_grid_idx(sx, n_grid, w);
    }

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_INTERPOLATION_UTIL_H
