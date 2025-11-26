#ifndef IPPL_ATOMIC_GATHER_H
#define IPPL_ATOMIC_GATHER_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

namespace ippl {
namespace Interpolation {
namespace detail {

    /**
     * @brief Generic atomic gather functor for grid-to-particle operations
     *
     * This is a simple implementation that works with any kernel function
     * and any dimension. It reads from the grid and accumulates values at particle locations.
     *
     * Template parameters:
     * @tparam Dim The dimension (1, 2, or 3)
     * @tparam RealType The floating point type
     * @tparam ExecSpace The Kokkos execution space
     * @tparam KernelType The kernel function type
     * @tparam ValueType The type of values being gathered
     * @tparam GridViewType The grid view type
     */
    template<unsigned Dim, typename RealType, typename ExecSpace, typename KernelType,
             typename ValueType, typename GridViewType>
    struct AtomicGatherFunctor {
        using real_type = RealType;
        using value_type = ValueType;
        using memory_space = typename ExecSpace::memory_space;
        using size_type = typename memory_space::size_type;

        // Input data
        Kokkos::View<real_type*[Dim], memory_space> x;  // Particle positions in GRID coordinates [0, n_grid)
        GridViewType grid;  // Input grid
        Kokkos::View<value_type*, memory_space> values;  // Output values

        // Parameters
        int n_grid[Dim];  // Grid dimensions
        int w;  // kernel width
        int nghost;  // ghost cell offset
        real_type inv_hw;  // 1 / half_width for kernel scaling
        bool add_to_attribute;  // If true, add to existing values; otherwise overwrite
        KernelType kernel;

        KOKKOS_INLINE_FUNCTION
        void operator()(const size_type j) const {
            // Particle position in grid coordinates
            real_type pos[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                pos[d] = x(j, d);
            }

            // Starting index for kernel stencil
            const bool odd = (w & 1);
            const int hw = w / 2;

            int idx0[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                idx0[d] = odd ? static_cast<int>(Kokkos::round(pos[d])) - hw
                              : static_cast<int>(pos[d]) + 1 - hw;
            }

            // Gather from all w^Dim grid points in the stencil
            value_type result(0);

            if constexpr (Dim == 3) {
                for (int k = 0; k < w; ++k) {
                    for (int jj = 0; jj < w; ++jj) {
                        for (int i = 0; i < w; ++i) {
                            int gi = idx0[0] + i;
                            int gj = idx0[1] + jj;
                            int gk = idx0[2] + k;

                            // Periodic wrapping
                            while (gi < 0) gi += n_grid[0];
                            while (gi >= n_grid[0]) gi -= n_grid[0];
                            while (gj < 0) gj += n_grid[1];
                            while (gj >= n_grid[1]) gj -= n_grid[1];
                            while (gk < 0) gk += n_grid[2];
                            while (gk >= n_grid[2]) gk -= n_grid[2];

                            real_type ker_i = kernel((pos[0] - static_cast<real_type>(idx0[0] + i)) * inv_hw);
                            real_type ker_j = kernel((pos[1] - static_cast<real_type>(idx0[1] + jj)) * inv_hw);
                            real_type ker_k = kernel((pos[2] - static_cast<real_type>(idx0[2] + k)) * inv_hw);
                            real_type kernel_val = ker_i * ker_j * ker_k;

                            result += grid(gi + nghost, gj + nghost, gk + nghost) * kernel_val;
                        }
                    }
                }
            } else if constexpr (Dim == 2) {
                for (int jj = 0; jj < w; ++jj) {
                    for (int i = 0; i < w; ++i) {
                        int gi = idx0[0] + i;
                        int gj = idx0[1] + jj;

                        while (gi < 0) gi += n_grid[0];
                        while (gi >= n_grid[0]) gi -= n_grid[0];
                        while (gj < 0) gj += n_grid[1];
                        while (gj >= n_grid[1]) gj -= n_grid[1];

                        real_type ker_i = kernel((pos[0] - static_cast<real_type>(idx0[0] + i)) * inv_hw);
                        real_type ker_j = kernel((pos[1] - static_cast<real_type>(idx0[1] + jj)) * inv_hw);
                        real_type kernel_val = ker_i * ker_j;

                        result += grid(gi + nghost, gj + nghost) * kernel_val;
                    }
                }
            } else if constexpr (Dim == 1) {
                for (int i = 0; i < w; ++i) {
                    int gi = idx0[0] + i;

                    while (gi < 0) gi += n_grid[0];
                    while (gi >= n_grid[0]) gi -= n_grid[0];

                    real_type kernel_val = kernel((pos[0] - static_cast<real_type>(idx0[0] + i)) * inv_hw);

                    result += grid(gi + nghost) * kernel_val;
                }
            }

            if (add_to_attribute) {
                values(j) += result;
            } else {
                values(j) = result;
            }
        }
    };

}  // namespace detail
}  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_ATOMIC_GATHER_H
