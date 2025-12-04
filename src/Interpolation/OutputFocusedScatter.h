#ifndef IPPL_OUTPUTFOCUSED_SCATTER_H
#define IPPL_OUTPUTFOCUSED_SCATTER_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>

#include "InterpolationUtil.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {
            /**
             * @brief Generic scatter functor with the julia-like algorithm
             *
             * Template parameters:
             * @tparam W The kernel width (compile-time constant for optimization)
             * @tparam RealType The floating point type (float or double)
             * @tparam ExecSpace The Kokkos execution space
             * @tparam KernelType The kernel function type (must have operator()(RealType) and
             * width())
             * @tparam ValueType The type of values being scattered (can be scalar or complex)
             * @tparam GridViewType The type of the grid view (can differ from ValueType, e.g. real
             * values to complex grid)
             * @tparam PositionViewType The type of the position view (View<T*[3]> or
             * View<Vector<T,3>*>)
             */
            template <int W, typename RealType, typename ExecSpace, typename KernelType,
                      typename ValueType, typename GridViewType,
                      typename PositionViewType =
                          Kokkos::View<RealType* [3], typename ExecSpace::memory_space>>
            struct OutputFocusedScatterFunctor3D {
                using real_type        = RealType;
                using value_type       = ValueType;
                using memory_space     = typename ExecSpace::memory_space;
                using size_type        = typename memory_space::size_type;
                using team_policy      = Kokkos::TeamPolicy<ExecSpace>;
                using team_member      = typename team_policy::member_type;
                using scratch_space    = typename ExecSpace::scratch_memory_space;
                using shared_real_view = Kokkos::View<real_type*, scratch_space,
                                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

                // Input data
                Kokkos::View<size_type*, memory_space> bin_offsets;
                Kokkos::View<size_type*, memory_space> permute;
                PositionViewType x;  // Particle positions in coordinates [-pi, pi]
                Kokkos::View<value_type*, memory_space> values;  // Values to scatter
                GridViewType grid;                               // Output grid

                // Parameters
                Kokkos::Array<size_type, 3> n_grid;        // GLOBAL grid dimensions
                Kokkos::Array<size_type, 3> n_grid_local;  // LOCAL grid dimensions
                Kokkos::Array<int, 3> local_offset;        // First global index of local domain
                Kokkos::Array<size_type, 3> num_tiles;
                int tile_size_x, tile_size_y, tile_size_z;
                int z_tiles;       // z-dimension splitting for parallelism
                int nghost;        // ghost cell offset for field
                real_type inv_hw;  // 1 / half_width for kernel scaling
                KernelType kernel;

                // Compile-time constants
                static constexpr int w         = W;
                static constexpr int hw        = W / 2;
                static constexpr bool odd      = (W & 1);
                static constexpr int half_left = (W - 1) / 2;

                // Helper to access position component - works with both View<T*[3]> and
                // View<Vector<T,3>*>
                template <typename PosView>
                KOKKOS_INLINE_FUNCTION static auto get_component(const PosView& pos, size_type i,
                                                                 int d) -> decltype(pos(i, d)) {
                    return pos(i, d);  // For View<T*[3]>
                }

                template <typename PosView>
                KOKKOS_INLINE_FUNCTION static auto get_component(const PosView& pos, size_type i,
                                                                 int d) -> decltype(pos(i)[d]) {
                    return pos(i)[d];  // For View<Vector<T,3>*>
                }

                // Compile-time histogram size calculations
                KOKKOS_INLINE_FUNCTION constexpr int hist_size_x() const { return tile_size_x + W; }

                KOKKOS_INLINE_FUNCTION constexpr int hist_size_y() const { return tile_size_y + W; }

                KOKKOS_INLINE_FUNCTION constexpr int hist_size_z() const {
                    return tile_size_z + z_tiles;
                }

                KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
                    const int team_id = team.league_rank();
                    // const int num_threads = team.team_size();
                    // const int thread_id   = team.team_rank();

                    // Decode team_id to tile and z-slice
                    // We split the w loop into threads_per_tile loops in the z direction
                    // z_tiles is the number of work iterms from the wz loop we work in in this
                    // block
                    const int threads_per_tile  = (z_tiles + w - 1) / z_tiles;
                    const size_type tile_linear = team_id / threads_per_tile;
                    const int tile_thread_idx   = team_id % threads_per_tile;
                    const int z_offset          = z_tiles * tile_thread_idx;

                    // const int tiles_per_xy = num_tiles[0] * num_tiles[1];

                    const int tile_x = tile_linear % num_tiles[0];
                    const int tile_y = (tile_linear / num_tiles[0]) % num_tiles[1];
                    const int tile_z = tile_linear / (num_tiles[0] * num_tiles[1]);

                    // Tile bounds in grid coordinates
                    const int tile_x0 = tile_x * tile_size_x;
                    const int tile_y0 = tile_y * tile_size_y;
                    const int tile_z0 = tile_z * tile_size_z;

                    const int hx = hist_size_x();
                    const int hy = hist_size_y();
                    const int hz = hist_size_z();

                    // Allocate shared memory histogram (separate real/complex parts)
                    // Determine if grid is complex (values might be real even if grid is complex)
                    using grid_element_type = std::remove_reference_t<decltype(grid(0, 0, 0))>;
                    constexpr bool value_is_complex =
                        std::is_same_v<value_type, Kokkos::complex<real_type>>;
                    constexpr bool grid_is_complex =
                        std::is_same_v<grid_element_type, Kokkos::complex<real_type>>;

                    const size_t hist_total = hx * hy * hz;

                    // Allocate shared memory - for complex grid, we need two arrays
                    shared_real_view hist_r(team.team_scratch(0), hist_total);
                    shared_real_view hist_c;
                    if constexpr (grid_is_complex) {
                        hist_c = shared_real_view(team.team_scratch(0), hist_total);
                    }

                    // Initialize histogram to zero
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hist_total), [&](int i) {
                        hist_r(i) = 0;
                        if constexpr (grid_is_complex) {
                            hist_c(i) = 0;
                        }
                    });
                    team.team_barrier();

                    // Allocate shared memory for kernel values
                    const int z_count = (tile_thread_idx == threads_per_tile - 1)
                                            ? (W - (threads_per_tile - 1) * z_tiles)
                                            : z_tiles;
                    shared_real_view kernel_vals(team.team_scratch(0), 2 * W + z_count);

                    // Get particles in this tile
                    const size_type pstart = bin_offsets(tile_linear);
                    const size_type pend   = bin_offsets(tile_linear + 1);

                    for (int i = pstart; i < pend; ++i) {
                        const size_type j    = permute(i);
                        const value_type val = values(j);

                        real_type s[3];
                        int idx[3];

                        for (int d = 0; d < 3; ++d) {
                            s[d]   = scale_to_grid_indices(get_component(x, j, d), n_grid[d]);
                            idx[d] = grid_point_to_grid_idx(s[d], n_grid[d], w) - half_left;
                        }

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, 2 * W + z_count), [&](int flat_w) {
                                int d = flat_w / w;
                                int k = flat_w % w;

                                kernel_vals[W * d + k] = kernel(
                                    (s[d]
                                     - static_cast<real_type>(idx[d] + k + (d == 2) * z_offset))
                                    * inv_hw);
                            });

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadMDRange(team, W, W, z_count),
                            [&](int wx, int wy, int wz) {
                                const real_type kernel_val =
                                    kernel_vals[wx] * kernel_vals[W + wy] * kernel_vals[2 * W + wz];

                                const int point_tile_x = idx[0] + half_left - tile_x0;
                                const int point_tile_y = idx[1] + half_left - tile_y0;
                                const int point_tile_z = idx[2] + half_left - tile_z0;

                                const int hist_idx =
                                    (((point_tile_z + wz) * hy + (point_tile_y + wy)) * hx
                                     + (point_tile_x + wx));

                                if constexpr (value_is_complex && grid_is_complex) {
                                    // Complex values to complex grid
                                    hist_r(hist_idx) += val.real() * kernel_val;
                                    hist_c(hist_idx) += val.imag() * kernel_val;
                                } else if constexpr (!value_is_complex && grid_is_complex) {
                                    hist_r(hist_idx) += val * kernel_val;
                                } else {
                                    // Real values to real grid
                                    hist_r(hist_idx) += val * kernel_val;
                                }
                            });
                    }
                    team.team_barrier();

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, hist_total), [&](int hist_idx) {
                            int hist_x = hist_idx % hx;
                            int hist_y = (hist_idx / hx) % hy;
                            int hist_z = hist_idx / (hx * hy);

                            // Compute GLOBAL grid indices
                            int global_x = tile_x0 + hist_x - half_left;
                            int global_y = tile_y0 + hist_y - half_left;
                            int global_z = tile_z0 + hist_z - half_left + z_offset;

                            // Convert to LOCAL indices
                            int local_x = global_x - local_offset[0];
                            int local_y = global_y - local_offset[1];
                            int local_z = global_z - local_offset[2];

                            // Check if within LOCAL domain (including ghosts)
                            if (local_x < -nghost
                                || local_x >= static_cast<int>(n_grid_local[0]) + nghost
                                || local_y < -nghost
                                || local_y >= static_cast<int>(n_grid_local[1]) + nghost
                                || local_z < -nghost
                                || local_z >= static_cast<int>(n_grid_local[2]) + nghost) {
                                return;
                            }

                            // Use local indices for grid access
                            if constexpr (grid_is_complex) {
#ifdef KOKKOS_ENABLE_CUDA
                                if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
                                    double* addr_as_double = reinterpret_cast<double*>(&grid(
                                        local_x + nghost, local_y + nghost, local_z + nghost));
                                    Kokkos::atomic_add(&addr_as_double[0], hist_r(hist_idx));
                                    Kokkos::atomic_add(&addr_as_double[1], hist_c(hist_idx));
                                } else
#endif
                                {
                                    Kokkos::atomic_add(
                                        &grid(local_x + nghost, local_y + nghost, local_z + nghost),
                                        Kokkos::complex<real_type>(hist_r(hist_idx),
                                                                   hist_c(hist_idx)));
                                }
                            } else {
                                Kokkos::atomic_add(
                                    &grid(local_x + nghost, local_y + nghost, local_z + nghost),
                                    hist_r(hist_idx));
                            }
                        });
                }
            };

            /**
             * @brief Dispatcher for scatter with different kernel widths
             * Uses template recursion to dispatch to the correct kernel width at runtime
             */
            template <int W, int MaxW>
            struct OutputFocusedScatterDispatcher {
                template <typename RealType, typename ExecSpace, typename KernelType,
                          typename ValueType, typename GridViewType, typename PositionViewType,
                          typename PermuteViewType, typename BinOffsetsViewType>
                static void dispatch_3d(
                    int w, BinOffsetsViewType bin_offsets, PermuteViewType permute,
                    PositionViewType x,
                    Kokkos::View<ValueType*, typename ExecSpace::memory_space> values,
                    GridViewType grid,
                    Kokkos::Array<typename ExecSpace::memory_space::size_type, 3> n_grid,
                    Kokkos::Array<typename ExecSpace::memory_space::size_type, 3> n_grid_local,
                    Kokkos::Array<int, 3> local_offset,
                    Kokkos::Array<typename ExecSpace::memory_space::size_type, 3> num_tiles,
                    int tile_size_x, int tile_size_y, int tile_size_z, int z_tiles, int nghost,
                    RealType inv_hw, const KernelType& kernel, int team_size) {
                    if constexpr (W <= MaxW) {
                        if (w == W) {
                            // Use generic Kokkos functor for other execution spaces
                            using size_type = typename ExecSpace::memory_space::size_type;

                            // Create functor with templated W
                            OutputFocusedScatterFunctor3D<W, RealType, ExecSpace, KernelType,
                                                          ValueType, GridViewType, PositionViewType>
                                functor{bin_offsets,  permute,      x,
                                        values,       grid,         n_grid,
                                        n_grid_local, local_offset, num_tiles,
                                        tile_size_x,  tile_size_y,  tile_size_z,
                                        z_tiles,      nghost,       inv_hw,
                                        kernel};

                            // Calculate scratch memory size
                            const size_t hist_size = functor.hist_size_x() * functor.hist_size_y()
                                                     * functor.hist_size_z();
                            using grid_element_type =
                                std::remove_reference_t<decltype(grid(0, 0, 0))>;
                            constexpr bool is_complex =
                                std::is_same_v<grid_element_type, Kokkos::complex<RealType>>;
                            const size_t scratch_size  = is_complex ? (2 * hist_size + (2 * W + z_tiles)) : (hist_size + (2 * W + z_tiles));
                            const size_t scratch_bytes = scratch_size * sizeof(RealType);

                            // Launch team policy
                            const size_type n_tiles_total =
                                num_tiles[0] * num_tiles[1] * num_tiles[2];
                            const int threads_per_tile = (z_tiles + W - 1) / z_tiles;
                            const size_type n_teams    = n_tiles_total * threads_per_tile;

                            using team_policy = Kokkos::TeamPolicy<ExecSpace>;
                            team_policy policy(n_teams, team_size);
                            policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

                            Kokkos::parallel_for("output_focused_spread", policy, functor);
                        } else {
                            OutputFocusedScatterDispatcher<W + 1, MaxW>::template dispatch_3d<
                                RealType, ExecSpace, KernelType, ValueType, GridViewType,
                                PositionViewType, PermuteViewType, BinOffsetsViewType>(
                                w, bin_offsets, permute, x, values, grid, n_grid, n_grid_local,
                                local_offset, num_tiles, tile_size_x, tile_size_y, tile_size_z,
                                z_tiles, nghost, inv_hw, kernel, team_size);
                        }
                    } else {
                        throw std::runtime_error("Kernel width exceeds maximum supported width");
                    }
                }
            };

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_TILED_SCATTER_H
