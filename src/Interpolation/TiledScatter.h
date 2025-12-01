#ifndef IPPL_TILED_SCATTER_H
#define IPPL_TILED_SCATTER_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>

#include "InterpolationUtil.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

#ifdef KOKKOS_ENABLE_CUDA
            /**
             * @brief Optimized CUDA kernel for tiled scatter operation
             *
             * This kernel uses block-level parallelism where each block processes one tile.
             * All threads in a block collaborate to:
             * 1. Build a shared memory histogram for the tile
             * 2. Process all particles that belong to the tile
             * 3. Flush the histogram to global memory with atomic operations
             *
             * @tparam w Kernel width (compile-time constant)
             * @tparam RealType Floating point type (float or double)
             * @tparam PositionViewType Type of position view
             * @tparam PermuteViewType Type of permutation view
             * @tparam BinOffsetsViewType Type of bin offsets view
             * @tparam ValueViewType Type of values view
             * @tparam GridViewType Type of grid view
             * @tparam KernelType Type of kernel functor
             */
            template <int w, typename RealType, typename PositionViewType, typename PermuteViewType,
                      typename BinOffsetsViewType, typename ValueViewType, typename GridViewType,
                      typename KernelType>
            __global__ void tiled_scatter_3d_cuda_kernel(
                PositionViewType x,       // Particle positions in PHYSICAL coordinates [-pi, pi]
                PermuteViewType permute,  // Permutation array
                BinOffsetsViewType bin_offsets,  // Offsets for each tile in permutation array
                ValueViewType values,            // Input values to scatter
                GridViewType grid,               // Output grid with ghosts
                int tile_size_x, int tile_size_y, int tile_size_z,
                int z_tiles,             // z-dimension splitting
                int n0, int n1, int n2,  // Grid dimensions
                int nghost,
                RealType inv_hw,  // 1 / half_width for kernel scaling
                KernelType kernel) {
                constexpr bool odd            = w % 2 == 1;
                constexpr int hw              = w / 2;
                constexpr int half_left       = (w - 1) / 2;
                constexpr RealType inv_two_pi = RealType(0.5) / RealType(3.14159265358979323846);
                using size_type               = typename PermuteViewType::value_type;
                using value_type              = typename ValueViewType::value_type;
                using grid_element_type       = std::remove_reference_t<decltype(grid(0, 0, 0))>;

                // Each block processes one tile
                const int threads_per_tile = (z_tiles + w - 1) / z_tiles;
                const int tile_linear      = blockIdx.x / threads_per_tile;
                const int tile_thread_idx  = blockIdx.x % threads_per_tile;
                const int num_tiles_x      = (n0 + tile_size_x - 1) / tile_size_x;
                const int num_tiles_y      = (n1 + tile_size_y - 1) / tile_size_y;
                const int num_tiles_z      = (n2 + tile_size_z - 1) / tile_size_z;

                const int tile_x = tile_linear % num_tiles_x;
                const int tile_y = (tile_linear / num_tiles_x) % num_tiles_y;
                const int tile_z = tile_linear / (num_tiles_x * num_tiles_y);

                // Shared memory histogram for this tile
                const int hist_size_x = tile_size_x + w;
                const int hist_size_y = tile_size_y + w;
                const int hist_size_z = tile_size_z + z_tiles;
                const int hist_size   = hist_size_x * hist_size_y * hist_size_z;

                const int z_offset = z_tiles * tile_thread_idx;

                extern __shared__ char shared_mem[];
                grid_element_type* hist = reinterpret_cast<grid_element_type*>(shared_mem);

                // Initialize histogram to zero collaboratively
                for (int idx = threadIdx.x; idx < hist_size; idx += blockDim.x) {
                    hist[idx] = grid_element_type(0);
                }
                __syncthreads();

                // Helper to access position component
                auto get_pos = [&](size_type particle_idx, int d) -> RealType {
                    if constexpr (std::is_same_v<PositionViewType,
                                                 Kokkos::View<RealType* [3], Kokkos::CudaSpace>>) {
                        return x(particle_idx, d);
                    } else {
                        return x(particle_idx)[d];
                    }
                };

                // Transform from physical coordinates [-pi, pi] to grid coordinates [0, n_grid)
                auto transform_coord = [&](RealType pos_phys, int grid_size) -> RealType {
                    RealType k = pos_phys * inv_two_pi;
                    k          = k - Kokkos::floor(k);
                    return k * grid_size;
                };

                // Get range of particles for this tile
                const size_type tile_idx_1d =
                    tile_x + tile_y * num_tiles_x + tile_z * num_tiles_x * num_tiles_y;
                const size_type particle_start        = bin_offsets(tile_idx_1d);
                const size_type particle_end          = bin_offsets(tile_idx_1d + 1);
                const size_type num_particles_in_tile = particle_end - particle_start;

                // Tile origin in global grid
                const int tile_origin_x = tile_x * tile_size_x;
                const int tile_origin_y = tile_y * tile_size_y;
                const int tile_origin_z = tile_z * tile_size_z;

                // Each thread processes particles in strided fashion
                for (size_type p_idx = threadIdx.x; p_idx < num_particles_in_tile;
                     p_idx += blockDim.x) {
                    const size_type sorted_idx   = particle_start + p_idx;
                    const size_type particle_idx = permute(sorted_idx);

                    // Transform from physical coordinates [-pi, pi] to grid coordinates [0, n_grid)
                    RealType kx = get_pos(particle_idx, 0) * inv_two_pi;
                    kx          = kx - ::floor(kx);
                    RealType sx = kx * n0;

                    RealType ky = get_pos(particle_idx, 1) * inv_two_pi;
                    ky          = ky - ::floor(ky);
                    RealType sy = ky * n1;

                    RealType kz = get_pos(particle_idx, 2) * inv_two_pi;
                    kz          = kz - ::floor(kz);
                    RealType sz = kz * n2;

                    // Compute grid indices (matching Kokkos functor)
                    int idx_x = odd ? static_cast<int>(sx + RealType{0.5}) : static_cast<int>(sx);
                    int idx_y = odd ? static_cast<int>(sy + RealType{0.5}) : static_cast<int>(sy);
                    int idx_z = odd ? static_cast<int>(sz + RealType{0.5}) : static_cast<int>(sz);

                    // Compute kernel differences
                    RealType diff_x = (sx - idx_x + half_left) * inv_hw;
                    RealType diff_y = (sy - idx_y + half_left) * inv_hw;
                    RealType diff_z = (sz - idx_z + half_left) * inv_hw;

                    // Convert to tile-local coordinates
                    idx_x = idx_x - tile_origin_x;
                    idx_y = idx_y - tile_origin_y;
                    idx_z = idx_z - tile_origin_z;

                    // Compute kernel values (matching Kokkos functor formula)
                    RealType kernel_x[w];
                    RealType kernel_y[w];
                    RealType kernel_z[w];

                    // #pragma unroll
                    for (int wx = 0; wx < w; ++wx) {
                        kernel_x[wx] = kernel(inv_hw * wx - diff_x);
                    }
                    // #pragma unroll
                    for (int wy = 0; wy < w; ++wy) {
                        kernel_y[wy] = kernel(inv_hw * wy - diff_y);
                    }
                    const int z_count = (tile_thread_idx == threads_per_tile - 1)
                                            ? (w - (threads_per_tile - 1) * z_tiles)
                                            : z_tiles;
                    // #pragma unroll
                    for (int wz = 0; wz < z_count; ++wz) {
                        kernel_z[wz] = kernel(inv_hw * (wz + z_offset) - diff_z);
                    }

                    // Get particle value
                    value_type val = values(particle_idx);

                    // Scatter to local histogram with atomic operations
                    // #pragma unroll
                    for (int wz = 0; wz < z_count; ++wz) {
                        // #pragma unroll
                        for (int wy = 0; wy < w; ++wy) {
                            // #pragma unroll
                            for (int wx = 0; wx < w; ++wx) {
                                const int hist_idx =
                                    ((idx_z + wz) * hist_size_y + (idx_y + wy)) * hist_size_x
                                    + (idx_x + wx);
                                const RealType kernel_val =
                                    kernel_x[wx] * kernel_y[wy] * kernel_z[wz];

                                if constexpr (std::is_same_v<grid_element_type,
                                                             Kokkos::complex<RealType>>) {
                                    if constexpr (std::is_same_v<value_type,
                                                                 Kokkos::complex<RealType>>) {
                                        // Complex value to complex grid
                                        RealType contrib_real = val.real() * kernel_val;
                                        RealType contrib_imag = val.imag() * kernel_val;
                                        atomicAdd(&hist[hist_idx].real(), contrib_real);
                                        atomicAdd(&hist[hist_idx].imag(), contrib_imag);
                                    } else {
                                        // Real value to complex grid
                                        RealType contrib = val * kernel_val;
                                        atomicAdd(&hist[hist_idx].real(), contrib);
                                    }
                                } else {
                                    // Real value to real grid
                                    RealType contrib = val * kernel_val;
                                    atomicAdd(&hist[hist_idx], contrib);
                                }
                            }
                        }
                    }
                }

                __syncthreads();

                // Flush histogram to global grid with atomic operations
                // Match Kokkos functor logic: tile_x0 + hist_x - half_left
                for (int idx = threadIdx.x; idx < hist_size; idx += blockDim.x) {
                    const int hist_x = idx % hist_size_x;
                    const int hist_y = (idx / hist_size_x) % hist_size_y;
                    const int hist_z = idx / (hist_size_x * hist_size_y);

                    const int global_x = tile_origin_x + hist_x - half_left;
                    const int global_y = tile_origin_y + hist_y - half_left;
                    const int global_z = tile_origin_z + hist_z - half_left + z_offset;

                    grid_element_type hist_val = hist[idx];

                    if constexpr (std::is_same_v<grid_element_type, Kokkos::complex<RealType>>) {
                        atomicAdd(
                            &grid(global_x + nghost, global_y + nghost, global_z + nghost).real(),
                            hist_val.real());
                        atomicAdd(
                            &grid(global_x + nghost, global_y + nghost, global_z + nghost).imag(),
                            hist_val.imag());
                    } else {
                        atomicAdd(&grid(global_x + nghost, global_y + nghost, global_z + nghost),
                                  hist_val);
                    }
                }
            }

            /**
             * @brief CUDA-specific dispatcher for scatter kernel
             * Uses template recursion to dispatch to the correct kernel width at runtime
             */
            template <int W, int MaxW>
            struct CudaScatterDispatcher {
                template <typename RealType, typename PositionViewType, typename PermuteViewType,
                          typename BinOffsetsViewType, typename ValueViewType,
                          typename GridViewType, typename KernelType>
                static void dispatch_3d(
                    int w, BinOffsetsViewType bin_offsets, PermuteViewType permute,
                    PositionViewType x, ValueViewType values, GridViewType grid,
                    Kokkos::Array<typename Kokkos::CudaSpace::size_type, 3> n_grid,
                    Kokkos::Array<typename Kokkos::CudaSpace::size_type, 3> num_tiles,
                    int tile_size_x, int tile_size_y, int tile_size_z, int z_tiles, int nghost,
                    RealType inv_hw, const KernelType& kernel) {
                    if constexpr (W <= MaxW) {
                        if (w == W) {
                            const int threads_per_tile = (z_tiles + w - 1) / z_tiles;

                            // Launch one block per tile
                            const int num_tiles_total = num_tiles[0] * num_tiles[1] * num_tiles[2];
                            const int threads_per_block = 256;

                            // Calculate shared memory size for histogram
                            const int hist_size_x = tile_size_x + W;
                            const int hist_size_y = tile_size_y + W;
                            const int hist_size_z = tile_size_z + z_tiles;
                            const int hist_size   = hist_size_x * hist_size_y * hist_size_z;

                            using grid_element_type =
                                std::remove_reference_t<decltype(grid(0, 0, 0))>;
                            size_t shared_mem_bytes = hist_size * sizeof(grid_element_type);

                            tiled_scatter_3d_cuda_kernel<W, RealType, PositionViewType,
                                                         PermuteViewType, BinOffsetsViewType,
                                                         ValueViewType, GridViewType, KernelType>
                                <<<threads_per_tile * num_tiles_total, threads_per_block,
                                   shared_mem_bytes>>>(x, permute, bin_offsets, values, grid,
                                                       tile_size_x, tile_size_y, tile_size_z,
                                                       z_tiles, n_grid[0], n_grid[1], n_grid[2],
                                                       nghost, inv_hw, kernel);

                            cudaError_t err = cudaGetLastError();
                            if (err != cudaSuccess) {
                                throw std::runtime_error(std::string("CUDA kernel launch failed: ")
                                                         + cudaGetErrorString(err));
                            }
                        } else {
                            CudaScatterDispatcher<W + 1, MaxW>::template dispatch_3d<
                                RealType, PositionViewType, PermuteViewType, BinOffsetsViewType,
                                ValueViewType, GridViewType, KernelType>(
                                w, bin_offsets, permute, x, values, grid, n_grid, num_tiles,
                                tile_size_x, tile_size_y, tile_size_z, z_tiles, nghost, inv_hw,
                                kernel);
                        }
                    } else {
                        throw std::runtime_error("Kernel width exceeds maximum supported width");
                    }
                }
            };
#endif  // KOKKOS_ENABLE_CUDA

            /**
             * @brief Generic tiled scatter functor for 3D particle-to-grid operations
             *
             * This is a generic implementation that works with any kernel function.
             * Uses shared memory histogram per tile to reduce global memory atomics.
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
            struct TiledScatterFunctor3D {
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
                    const int team_id     = team.league_rank();
                    const int num_threads = team.team_size();
                    const int thread_id   = team.team_rank();

                    // Decode team_id to tile and z-slice
                    // We split the w loop into threads_per_tile loops in the z direction
                    // z_tiles is the number of work iterms from the wz loop we work in in this
                    // block
                    const int threads_per_tile  = (z_tiles + w - 1) / z_tiles;
                    const size_type tile_linear = team_id / threads_per_tile;
                    const int tile_thread_idx   = team_id % threads_per_tile;
                    const int z_offset          = z_tiles * tile_thread_idx;

                    const int tiles_per_xy = num_tiles[0] * num_tiles[1];

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

                    // Get particles in this tile
                    const size_type pstart = bin_offsets(tile_linear);
                    const size_type pend   = bin_offsets(tile_linear + 1);

                    // Process particles
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, pstart, pend), [&](size_type ip) {
                            const size_type j    = permute(ip);
                            const value_type val = values(j);

                            // X dimension
                            real_type sx = scale_to_grid_indices(get_component(x, j, 0), n_grid[0]);
                            // The indices are chosen so that there are non-zero contributions from
                            // sx in the range [idx_x, idx_x + w)
                            int idx_x = grid_point_to_grid_idx(sx, n_grid[0], w) - half_left;
                            assert((idx_x + half_left) / tile_size_x == tile_x);

                            // Y dimension
                            real_type sy = scale_to_grid_indices(get_component(x, j, 1), n_grid[1]);
                            int idx_y    = grid_point_to_grid_idx(sy, n_grid[1], w) - half_left;
                            assert((idx_y + half_left) / tile_size_y == tile_y);

                            // Z dimension
                            real_type sz = scale_to_grid_indices(get_component(x, j, 2), n_grid[2]);
                            int idx_z    = grid_point_to_grid_idx(sz, n_grid[2], w) - half_left;
                            assert((idx_z + half_left) / tile_size_z == tile_z);

                            // Precompute kernel values
                            real_type kernel_x[W];
                            real_type kernel_y[W];
                            real_type kernel_z[W];

                            for (int wx = 0; wx < W; ++wx) {
                                assert((sx - static_cast<real_type>(idx_x + wx)) <= w / 2.0);
                                kernel_x[wx] =
                                    kernel((sx - static_cast<real_type>(idx_x + wx)) * inv_hw);
                            }
                            for (int wy = 0; wy < W; ++wy) {
                                assert((sy - static_cast<real_type>(idx_y + wy)) <= w / 2.0);
                                kernel_y[wy] =
                                    kernel((sy - static_cast<real_type>(idx_y + wy)) * inv_hw);
                            }

                            // Determine z range for this thread (match old code)
                            const int z_count = (tile_thread_idx == threads_per_tile - 1)
                                                    ? (W - (threads_per_tile - 1) * z_tiles)
                                                    : z_tiles;

                            for (int wz = 0; wz < z_count; ++wz) {
                                assert((sz - static_cast<real_type>(idx_z + wz + z_offset))
                                       <= w / 2.0);
                                kernel_z[wz] = kernel(
                                    (sz - static_cast<real_type>(idx_z + wz + z_offset)) * inv_hw);
                            }

                            // Spread to histogram
                            for (int wz = 0; wz < z_count; ++wz) {
                                for (int wy = 0; wy < W; ++wy) {
                                    for (int wx = 0; wx < W; ++wx) {
                                        const real_type kernel_val =
                                            kernel_x[wx] * kernel_y[wy] * kernel_z[wz];

                                        const int point_tile_x = idx_x + half_left - tile_x0;
                                        const int point_tile_y = idx_y + half_left - tile_y0;
                                        const int point_tile_z = idx_z + half_left - tile_z0;

                                        const int hist_idx =
                                            (((point_tile_z + wz) * hy + (point_tile_y + wy)) * hx
                                             + (point_tile_x + wx));

                                        assert(point_tile_x >= 0);
                                        assert(point_tile_y >= 0);
                                        assert(point_tile_z >= 0);
                                        assert(point_tile_x < tile_size_x);
                                        assert(point_tile_y < tile_size_y);
                                        assert(point_tile_z < tile_size_z);

                                        if constexpr (value_is_complex && grid_is_complex) {
                                            // Complex values to complex grid
                                            Kokkos::atomic_add(&hist_r(hist_idx),
                                                               val.real() * kernel_val);
                                            Kokkos::atomic_add(&hist_c(hist_idx),
                                                               val.imag() * kernel_val);
                                        } else if constexpr (!value_is_complex && grid_is_complex) {
                                            // Real values to complex grid (scatter to real part
                                            // only)
                                            Kokkos::atomic_add(&hist_r(hist_idx), val * kernel_val);
                                        } else {
                                            // Real values to real grid
                                            Kokkos::atomic_add(&hist_r(hist_idx), val * kernel_val);
                                        }
                                    }
                                }
                            }
                        });

                    team.team_barrier();

                    // Flush histogram to global grid
                    // No periodic wrapping - write to ghosts, will be accumulated with
                    // accumulateHalo()
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
                            if (local_x < -nghost || local_x >= static_cast<int>(n_grid_local[0]) + nghost
                                || local_y < -nghost || local_y >= static_cast<int>(n_grid_local[1]) + nghost
                                || local_z < -nghost || local_z >= static_cast<int>(n_grid_local[2]) + nghost) {
                                return;
                            }

                            // Use LOCAL indices for grid access
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
                                    Kokkos::atomic_add(&grid(local_x + nghost, local_y + nghost,
                                                             local_z + nghost),
                                                       Kokkos::complex<real_type>(
                                                           hist_r(hist_idx), hist_c(hist_idx)));
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
            struct ScatterDispatcher {
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
#ifdef KOKKOS_ENABLE_CUDA
                            // Use native CUDA kernel for CUDA execution space
                            if constexpr (false && std::is_same_v<ExecSpace, Kokkos::Cuda>) {
                                CudaScatterDispatcher<W, W>::template dispatch_3d<
                                    RealType, PositionViewType, PermuteViewType, BinOffsetsViewType,
                                    Kokkos::View<ValueType*, typename ExecSpace::memory_space>,
                                    GridViewType, KernelType>(w, bin_offsets, permute, x, values,
                                                              grid, n_grid, num_tiles, tile_size_x,
                                                              tile_size_y, tile_size_z, z_tiles,
                                                              nghost, inv_hw, kernel);
                            } else
#endif
                            {
                                // Use generic Kokkos functor for other execution spaces
                                using size_type = typename ExecSpace::memory_space::size_type;

                                // Create functor with templated W
                                TiledScatterFunctor3D<W, RealType, ExecSpace, KernelType, ValueType,
                                                      GridViewType, PositionViewType>
                                    functor{bin_offsets, permute,      x,            values,
                                            grid,        n_grid,       n_grid_local, local_offset,
                                            num_tiles,   tile_size_x,  tile_size_y,  tile_size_z,
                                            z_tiles,     nghost,       inv_hw,       kernel};

                                // Calculate scratch memory size
                                const size_t hist_size = functor.hist_size_x()
                                                         * functor.hist_size_y()
                                                         * functor.hist_size_z();
                                using grid_element_type =
                                    std::remove_reference_t<decltype(grid(0, 0, 0))>;
                                constexpr bool is_complex =
                                    std::is_same_v<grid_element_type, Kokkos::complex<RealType>>;
                                const size_t scratch_size  = is_complex ? 2 * hist_size : hist_size;
                                const size_t scratch_bytes = scratch_size * sizeof(RealType);

                                // Launch team policy
                                const size_type n_tiles_total =
                                    num_tiles[0] * num_tiles[1] * num_tiles[2];
                                const int threads_per_tile = (z_tiles + W - 1) / z_tiles;
                                const size_type n_teams    = n_tiles_total * threads_per_tile;

                                using team_policy = Kokkos::TeamPolicy<ExecSpace>;
                                team_policy policy(n_teams, team_size);
                                policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

                                Kokkos::parallel_for("tiled_spread", policy, functor);
                            }
                        } else {
                            ScatterDispatcher<W + 1, MaxW>::template dispatch_3d<
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
