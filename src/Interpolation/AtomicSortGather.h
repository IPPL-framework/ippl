#ifndef IPPL_ATOMIC_SORT_GATHER_H
#define IPPL_ATOMIC_SORT_GATHER_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include "InterpolationUtil.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

            /**
             * @brief Generic atomic gather functor for grid-to-particle operations
             *
             * Works with any kernel function and dimension. Reads from the grid and
             * accumulates values at particle locations.
             *
             * This version supports a permutation array so that the functor can
             * iterate over particles in sorted/bin order while still writing the
             * gathered value to the original particle index.
             *
             * Template parameters:
             * @tparam Dim The dimension (1, 2, or 3)
             * @tparam RealType The floating point type
             * @tparam ExecSpace The Kokkos execution space
             * @tparam KernelType The kernel function type
             * @tparam ValueType The type of values being gathered
             * @tparam GridViewType The grid view type
             * @tparam PositionViewType The type of the position view (default:
             *         View<Vector<RealType, Dim>*>)
             * @tparam PermuteViewType The type of the permutation view (indices into x/values)
             */
            template <unsigned Dim,
                      typename RealType,
                      typename ExecSpace,
                      typename KernelType,
                      typename ValueType,
                      typename GridViewType,
                      typename PositionViewType =
                          Kokkos::View<ippl::Vector<RealType, Dim>*,
                                       typename ExecSpace::memory_space>,
                      typename PermuteViewType =
                          Kokkos::View<std::size_t*, typename ExecSpace::memory_space>>
            struct AtomicSortGatherFunctor {
                using real_type    = RealType;
                using value_type   = ValueType;
                using memory_space = typename ExecSpace::memory_space;
                using size_type    = typename memory_space::size_type;

                // Input data
                PositionViewType x;      // particle positions in PHYSICAL coordinates
                GridViewType     grid;   // input grid
                PermuteViewType  permute; // permutation: sorted_index -> particle_index

                // Output data
                Kokkos::View<value_type*, memory_space> values;  // per-particle gathered values

                // Parameters
                int n_grid[Dim];        // GLOBAL grid dimensions
                int n_grid_local[Dim];  // LOCAL grid dimensions
                int local_offset[Dim];  // first global index of local domain
                int w;                  // kernel width
                int nghost;             // ghost cell offset
                real_type inv_hw;       // 1 / half_width for kernel scaling
                bool add_to_attribute;  // if true, add to existing values; otherwise overwrite
                KernelType kernel;

                KOKKOS_INLINE_FUNCTION void operator()(const size_type j_sorted) const {
                    // Map sorted index -> actual particle index
                    const size_type p = permute(j_sorted);

                    // Transform from physical coordinates to grid coordinates
                    Kokkos::Array<real_type, Dim> pos;
                    for (int d = 0; d < Dim; ++d) {
                        pos[d] = scale_to_grid_indices(x(p)[d], n_grid[d]);
                    }

                    Kokkos::Array<int, Dim> idx0;
                    for (int d = 0; d < Dim; ++d) {
                        idx0[d] = grid_point_to_grid_idx(pos[d], n_grid[d], w) - (w - 1) / 2;
                    }

                    // Result type matches what we read from the grid
                    using grid_element_type = std::remove_reference_t<decltype(grid(0))>;
                    grid_element_type result(0);

                    if constexpr (Dim == 3) {
                        for (int k = 0; k < w; ++k) {
                            for (int jj = 0; jj < w; ++jj) {
                                for (int i = 0; i < w; ++i) {
                                    int gi_global = idx0[0] + i;
                                    int gj_global = idx0[1] + jj;
                                    int gk_global = idx0[2] + k;

                                    // Convert to LOCAL indices
                                    int gi_local = gi_global - local_offset[0];
                                    int gj_local = gj_global - local_offset[1];
                                    int gk_local = gk_global - local_offset[2];

#ifndef NDEBUG
                                    // Check if within local domain (including ghosts)
                                    assert(gi_local >= -nghost
                                           && gi_local < n_grid_local[0] + nghost);
                                    assert(gj_local >= -nghost
                                           && gj_local < n_grid_local[1] + nghost);
                                    assert(gk_local >= -nghost
                                           && gk_local < n_grid_local[2] + nghost);
#endif

                                    real_type ker_i = kernel(
                                        (pos[0] - static_cast<real_type>(idx0[0] + i)) * inv_hw);
                                    real_type ker_j = kernel(
                                        (pos[1] - static_cast<real_type>(idx0[1] + jj)) * inv_hw);
                                    real_type ker_k = kernel(
                                        (pos[2] - static_cast<real_type>(idx0[2] + k)) * inv_hw);
                                    real_type kernel_val = ker_i * ker_j * ker_k;

                                    result += grid(gi_local + nghost,
                                                   gj_local + nghost,
                                                   gk_local + nghost)
                                              * kernel_val;
                                }
                            }
                        }
                    } else if constexpr (Dim == 2) {
                        for (int jj = 0; jj < w; ++jj) {
                            for (int i = 0; i < w; ++i) {
                                int gi = idx0[0] + i;
                                int gj = idx0[1] + jj;

                                real_type ker_i =
                                    kernel((pos[0] - static_cast<real_type>(idx0[0] + i)) * inv_hw);
                                real_type ker_j =
                                    kernel((pos[1] - static_cast<real_type>(idx0[1] + jj)) * inv_hw);
                                real_type kernel_val = ker_i * ker_j;

                                result += grid(gi + nghost, gj + nghost) * kernel_val;
                            }
                        }
                    } else if constexpr (Dim == 1) {
                        for (int i = 0; i < w; ++i) {
                            int gi = idx0[0] + i;

                            real_type kernel_val =
                                kernel((pos[0] - static_cast<real_type>(idx0[0] + i)) * inv_hw);

                            result += grid(gi + nghost) * kernel_val;
                        }
                    }

                    // Write result to output, extracting real part if needed
                    constexpr bool val_is_complex =
                        std::is_same_v<value_type, Kokkos::complex<real_type>>;
                    constexpr bool grid_is_complex =
                        std::is_same_v<grid_element_type, Kokkos::complex<real_type>>;

                    if constexpr (grid_is_complex && !val_is_complex) {
                        // Grid is complex but output is real - extract real part
                        if (add_to_attribute) {
                            values(p) += result.real();
                        } else {
                            values(p) = result.real();
                        }
                    } else {
                        // Types match (both real or both complex)
                        if (add_to_attribute) {
                            values(p) += result;
                        } else {
                            values(p) = result;
                        }
                    }
                }
            };

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_ATOMIC_SORT_GATHER_H
