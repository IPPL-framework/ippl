#ifndef IPPL_ATOMIC_SORT_GATHER_3D_H
#define IPPL_ATOMIC_SORT_GATHER_3D_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include "InterpolationUtil.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

            /**
             * @brief  gather functor with precomputed kernel values
             *
             * Template parameters:
             * @tparam W Compile-time kernel width
             * @tparam RealType The floating point type
             * @tparam ExecSpace The Kokkos execution space
             * @tparam KernelType The kernel function type
             * @tparam ValueType The type of values being gathered
             * @tparam GridViewType The grid view type
             * @tparam PositionViewType The type of the position view
             * @tparam PermuteViewType The type of the permutation view
             */
            template <int W,
                      typename RealType,
                      typename ExecSpace,
                      typename KernelType,
                      typename ValueType,
                      typename GridViewType,
                      typename PositionViewType =
                          Kokkos::View<ippl::Vector<RealType, 3>*,
                                       typename ExecSpace::memory_space>,
                      typename PermuteViewType =
                          Kokkos::View<std::size_t*, typename ExecSpace::memory_space>>
            struct AtomicSortGatherFunctor3D {
                using real_type    = RealType;
                using value_type   = ValueType;
                using memory_space = typename ExecSpace::memory_space;
                using size_type    = typename memory_space::size_type;

                static constexpr int Dim = 3;

                // Input data
                const std::decay_t<PositionViewType> x;       // particle positions in PHYSICAL coordinates
                const std::decay_t<GridViewType>     grid;    // input grid
                const std::decay_t<PermuteViewType>  permute; // permutation: sorted_index -> particle_index

                // Output data
                const Kokkos::View<value_type*, memory_space> values;  // per-particle gathered values

                // Parameters
                const Vector<int, Dim> n_grid;        // GLOBAL grid dimensions
                const Vector<int, Dim> n_grid_local;  // LOCAL grid dimensions
                const Vector<int, Dim> local_offset;  // first global index of local domain
                const int nghost;                      // ghost cell offset
                const real_type inv_hw;                // 1 / half_width for kernel scaling
                const bool add_to_attribute;           // if true, add to existing values; otherwise overwrite
                const std::decay_t<KernelType> kernel;

                KOKKOS_INLINE_FUNCTION void operator()(const size_type j_sorted) const {
                    // Map sorted index -> actual particle index
                    const size_type p = permute(j_sorted);

                    // Transform from physical coordinates to grid coordinates
                    real_type pos[Dim];
                    for (int d = 0; d < Dim; ++d) {
                        pos[d] = scale_to_grid_indices(x(p)[d], n_grid[d]);
                    }

                    // Compute base grid indices
                    int idx0[Dim];
                    for (int d = 0; d < Dim; ++d) {
                        idx0[d] = grid_point_to_grid_idx(pos[d], n_grid[d], W) - (W - 1) / 2;
                    }

                    // Precompute kernel values in each cardinal direction
                    real_type kernel_x[W];
                    real_type kernel_y[W];
                    real_type kernel_z[W];

                    for (int i = 0; i < W; ++i) {
                        kernel_x[i] = kernel((pos[0] - static_cast<real_type>(idx0[0] + i)) * inv_hw);
                        kernel_y[i] = kernel((pos[1] - static_cast<real_type>(idx0[1] + i)) * inv_hw);
                        kernel_z[i] = kernel((pos[2] - static_cast<real_type>(idx0[2] + i)) * inv_hw);
                    }

                    // Convert base indices to local coordinates
                    const int base_i = idx0[0] - local_offset[0] + nghost;
                    const int base_j = idx0[1] - local_offset[1] + nghost;
                    const int base_k = idx0[2] - local_offset[2] + nghost;

#ifndef NDEBUG
                    // Check bounds in debug mode
                    assert(base_i >= 0 && base_i + W <= n_grid_local[0] + 2 * nghost);
                    assert(base_j >= 0 && base_j + W <= n_grid_local[1] + 2 * nghost);
                    assert(base_k >= 0 && base_k + W <= n_grid_local[2] + 2 * nghost);
#endif

                    // Result type matches what we read from the grid
                    using grid_element_type = std::remove_reference_t<decltype(grid(0, 0, 0))>;
                    grid_element_type result(0);

                    // Gather with precomputed kernel values
                    for (int k = 0; k < W; ++k) {
                        const real_type kz = kernel_z[k];
                        const int gk = base_k + k;

                        for (int j = 0; j < W; ++j) {
                            const real_type kyz = kernel_y[j] * kz;
                            const int gj = base_j + j;

                            for (int i = 0; i < W; ++i) {
                                const real_type kernel_val = kernel_x[i] * kyz;
                                result += grid(base_i + i, gj, gk) * kernel_val;
                            }
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

            /**
             * @brief Dispatcher for 3D gather with runtime kernel width
             */
            template <typename RealType,
                      typename ExecSpace,
                      typename KernelType,
                      typename ValueType,
                      typename GridViewType,
                      typename PositionViewType,
                      typename PermuteViewType,
                      typename OutputViewType>
            void dispatch_gather_3d(
                const PositionViewType& x,
                const GridViewType& grid,
                const PermuteViewType& permute,
                const OutputViewType& values,
                const Vector<int, 3>& n_grid,
                const Vector<int, 3>& n_grid_local,
                const Vector<int, 3>& local_offset,
                int w,
                int nghost,
                RealType inv_hw,
                bool add_to_attribute,
                const KernelType& kernel,
                size_t n_particles) {

                auto create_and_run = [&]<int W>() {
                    AtomicSortGatherFunctor3D<W, RealType, ExecSpace, KernelType, ValueType,
                                              GridViewType, PositionViewType, PermuteViewType>
                        functor{x, grid, permute, values, n_grid, n_grid_local, local_offset,
                                nghost, inv_hw, add_to_attribute, kernel};

                    // (paul) Remove the fences here with caution. HIP gives invalid memory
                    //         access errors with the current rocm (old) 6.0.2
                    Kokkos::fence();
                    Kokkos::parallel_for(
                        "AtomicSortGather3D",
                        Kokkos::RangePolicy<ExecSpace>(0, n_particles),
                        functor);
                    Kokkos::fence();
                };

                switch (w) {
                    case 1: create_and_run.template operator()<1>(); break;
                    case 2: create_and_run.template operator()<2>(); break;
                    case 3: create_and_run.template operator()<3>(); break;
                    case 4: create_and_run.template operator()<4>(); break;
                    case 5: create_and_run.template operator()<5>(); break;
                    case 6: create_and_run.template operator()<6>(); break;
                    case 7: create_and_run.template operator()<7>(); break;
                    case 8: create_and_run.template operator()<8>(); break;
                    case 9: create_and_run.template operator()<9>(); break;
                    case 10: create_and_run.template operator()<10>(); break;
                    case 11: create_and_run.template operator()<11>(); break;
                    case 12: create_and_run.template operator()<12>(); break;
                    case 13: create_and_run.template operator()<13>(); break;
                    case 14: create_and_run.template operator()<14>(); break;

                    default:
                        Kokkos::abort("AtomicSortGatherFunctor3D: unsupported kernel width");
                }
            }

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_ATOMIC_SORT_GATHER_3D_H