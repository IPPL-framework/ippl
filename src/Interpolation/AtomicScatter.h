#ifndef IPPL_ATOMIC_SCATTER_H
#define IPPL_ATOMIC_SCATTER_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>

namespace ippl {
    namespace Interpolation {
        namespace detail {

            /**
             * @brief Generic atomic scatter functor for particle-to-grid operations
             *
             * This is a simple atomic implementation that works with any kernel function
             * and any dimension.
             *
             * Template parameters:
             * @tparam Dim The dimension (1, 2, or 3)
             * @tparam RealType The floating point type
             * @tparam ExecSpace The Kokkos execution space
             * @tparam KernelType The kernel function type
             * @tparam ValueType The type of values being scattered
             * @tparam GridViewType The grid view type
             * @tparam PositionViewType The type of the position view (default:
             * View<Vector<RealType, Dim>*>)
             */
            template <unsigned Dim, typename RealType, typename ExecSpace, typename KernelType,
                      typename ValueType, typename GridViewType,
                      typename PositionViewType = Kokkos::View<ippl::Vector<RealType, Dim>*,
                                                               typename ExecSpace::memory_space>>
            struct AtomicScatterFunctor {
                using real_type    = RealType;
                using value_type   = ValueType;
                using memory_space = typename ExecSpace::memory_space;
                using size_type    = typename memory_space::size_type;

                // Input data
                PositionViewType x;  // Particle positions in PHYSICAL coordinates (e.g., [-pi, pi])
                Kokkos::View<value_type*, memory_space> values;  // Values to scatter
                GridViewType grid;                               // Output grid

                // Parameters
                int n_grid[Dim];        // GLOBAL grid dimensions (for coordinate transformation)
                int n_grid_local[Dim];  // LOCAL grid dimensions (owned by this rank)
                int local_offset[Dim];  // First global index of local domain
                int w;                  // kernel width
                int nghost;             // ghost cell offset
                real_type inv_hw;       // 1 / half_width for kernel scaling
                KernelType kernel;

                KOKKOS_INLINE_FUNCTION void operator()(const size_type j) const {
                    const value_type& val = values(j);

                    // Transform from physical coordinates [-pi, pi] to grid coordinates [0, n_grid)
                    constexpr RealType inv_two_pi = RealType(0.5) / std::numbers::pi_v<RealType>;

                    real_type pos[Dim];  // Grid coordinates
                    for (unsigned d = 0; d < Dim; ++d) {
                        pos[d] = scale_to_grid_indices(x(j)[d], n_grid[d]);
                    }

                    int idx0[Dim];
                    for (unsigned d = 0; d < Dim; ++d) {
                        idx0[d] = grid_point_to_grid_idx(pos[d], n_grid[d], w) - (w - 1) / 2;
                    }

                    // Spread to all w^Dim grid points in the stencil
                    int64_t total = 1;
                    for (unsigned d = 0; d < Dim; ++d)
                        total *= w;

                    for (int64_t flat = 0; flat < total; ++flat) {
                        int idx_global[Dim];  // Global grid indices
                        int idx_local[Dim];   // Local grid indices
                        real_type kernel_val = 1.0;
                        int64_t tmp          = flat;
                        bool in_bounds       = true;

                        // Decode flat index to Dim-dimensional index and convert to local
                        for (int d = Dim - 1; d >= 0; --d) {
                            int k = tmp % w;
                            tmp /= w;
                            idx_global[d] = idx0[d] + k;

                            // Convert to local index
                            idx_local[d] = idx_global[d] - local_offset[d];

                            // Multiply by kernel value
                            kernel_val *=
                                kernel((pos[d] - static_cast<real_type>(idx0[d] + k)) * inv_hw);
                        }

                        // Atomic add to grid (with ghost offset)
                        // Handle different combinations of value and grid types
                        constexpr bool val_is_complex =
                            std::is_same_v<value_type, Kokkos::complex<real_type>>;
                        using grid_element_type = std::remove_reference_t<decltype(grid(0))>;
                        constexpr bool grid_is_complex =
                            std::is_same_v<grid_element_type, Kokkos::complex<real_type>>;

                        if constexpr (Dim == 3) {
                            if constexpr (val_is_complex && grid_is_complex) {
                                // Complex values to complex grid
                                Kokkos::atomic_add(
                                    &grid(idx_local[0] + nghost, idx_local[1] + nghost,
                                          idx_local[2] + nghost)
                                         .real(),
                                    val.real() * kernel_val);
                                Kokkos::atomic_add(
                                    &grid(idx_local[0] + nghost, idx_local[1] + nghost,
                                          idx_local[2] + nghost)
                                         .imag(),
                                    val.imag() * kernel_val);
                            } else if constexpr (!val_is_complex && grid_is_complex) {
                                // Real values to complex grid (scatter to real part only)
                                Kokkos::atomic_add(
                                    &grid(idx_local[0] + nghost, idx_local[1] + nghost,
                                          idx_local[2] + nghost)
                                         .real(),
                                    val * kernel_val);
                            } else {
                                // Real values to real grid
                                Kokkos::atomic_add(
                                    &grid(idx_local[0] + nghost, idx_local[1] + nghost,
                                          idx_local[2] + nghost),
                                    val * kernel_val);
                            }
                        } else if constexpr (Dim == 2) {
                            if constexpr (val_is_complex && grid_is_complex) {
                                Kokkos::atomic_add(
                                    &grid(idx_local[0] + nghost, idx_local[1] + nghost).real(),
                                    val.real() * kernel_val);
                                Kokkos::atomic_add(
                                    &grid(idx_local[0] + nghost, idx_local[1] + nghost).imag(),
                                    val.imag() * kernel_val);
                            } else if constexpr (!val_is_complex && grid_is_complex) {
                                Kokkos::atomic_add(
                                    &grid(idx_local[0] + nghost, idx_local[1] + nghost).real(),
                                    val * kernel_val);
                            } else {
                                Kokkos::atomic_add(
                                    &grid(idx_local[0] + nghost, idx_local[1] + nghost),
                                    val * kernel_val);
                            }
                        } else if constexpr (Dim == 1) {
                            if constexpr (val_is_complex && grid_is_complex) {
                                Kokkos::atomic_add(&grid(idx_local[0] + nghost).real(),
                                                   val.real() * kernel_val);
                                Kokkos::atomic_add(&grid(idx_local[0] + nghost).imag(),
                                                   val.imag() * kernel_val);
                            } else if constexpr (!val_is_complex && grid_is_complex) {
                                Kokkos::atomic_add(&grid(idx_local[0] + nghost).real(),
                                                   val * kernel_val);
                            } else {
                                Kokkos::atomic_add(&grid(idx_local[0] + nghost), val * kernel_val);
                            }
                        }
                    }
                }
            };

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_ATOMIC_SCATTER_H
