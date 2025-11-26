#ifndef IPPL_TILED_SCATTER_H
#define IPPL_TILED_SCATTER_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

namespace ippl {
namespace Interpolation {
namespace detail {

    /**
     * @brief Generic tiled scatter functor for 3D particle-to-grid operations
     *
     * This is a generic implementation that works with any kernel function.
     * Uses shared memory histogram per tile to reduce global memory atomics.
     *
     * Template parameters:
     * @tparam RealType The floating point type (float or double)
     * @tparam ExecSpace The Kokkos execution space
     * @tparam KernelType The kernel function type (must have operator()(RealType) and width())
     * @tparam ValueType The type of values being scattered (can be scalar or complex)
     */
    template<typename RealType, typename ExecSpace, typename KernelType, typename ValueType>
    struct TiledScatterFunctor3D {
        using real_type = RealType;
        using value_type = ValueType;
        using memory_space = typename ExecSpace::memory_space;
        using size_type = typename memory_space::size_type;
        using team_policy = Kokkos::TeamPolicy<ExecSpace>;
        using team_member = typename team_policy::member_type;
        using scratch_space = typename ExecSpace::scratch_memory_space;
        using shared_real_view = Kokkos::View<real_type*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // Input data
        Kokkos::View<size_type*, memory_space> bin_offsets;
        Kokkos::View<size_type*, memory_space> permute;
        Kokkos::View<real_type*[3], memory_space> x;  // Particle positions in GRID coordinates [0, n_grid)
        Kokkos::View<value_type*, memory_space> values;  // Values to scatter
        Kokkos::View<value_type***, memory_space> grid;  // Output grid

        // Parameters
        Kokkos::Array<size_type, 3> n_grid;
        Kokkos::Array<size_type, 3> num_tiles;
        int w;  // kernel width
        int tile_size_x, tile_size_y, tile_size_z;
        int z_tiles;  // z-dimension splitting for parallelism
        int nghost;  // ghost cell offset for field
        real_type inv_hw;  // 1 / half_width for kernel scaling
        KernelType kernel;

        // Compile-time derived constants
        KOKKOS_INLINE_FUNCTION
        static constexpr int half_left(int W) { return (W - 1) / 2; }

        KOKKOS_INLINE_FUNCTION
        static constexpr int half_right(int W) { return W / 2; }

        KOKKOS_INLINE_FUNCTION
        int hist_size_x() const { return tile_size_x + w; }

        KOKKOS_INLINE_FUNCTION
        int hist_size_y() const { return tile_size_y + w; }

        KOKKOS_INLINE_FUNCTION
        int hist_size_z() const { return tile_size_z + z_tiles; }

        KOKKOS_INLINE_FUNCTION
        void operator()(const team_member& team) const {
            const int team_id = team.league_rank();
            const int num_threads = team.team_size();
            const int thread_id = team.team_rank();

            // Decode team_id to tile and z-slice (match old code)
            const int threads_per_tile = (z_tiles + w - 1) / z_tiles;
            const size_type tile_linear = team_id / threads_per_tile;
            const int tile_thread_idx = team_id % threads_per_tile;
            const int z_offset = z_tiles * tile_thread_idx;

            const int tiles_per_xy = num_tiles[0] * num_tiles[1];

            const size_type tile_x = tile_linear % num_tiles[0];
            const size_type tile_y = (tile_linear / num_tiles[0]) % num_tiles[1];
            const size_type tile_z = tile_linear / (num_tiles[0] * num_tiles[1]);

            // Tile bounds in grid coordinates
            const size_type tile_x0 = tile_x * tile_size_x;
            const size_type tile_y0 = tile_y * tile_size_y;
            const size_type tile_z0 = tile_z * tile_size_z;

            const int hx = hist_size_x();
            const int hy = hist_size_y();
            const int hz = hist_size_z();

            // Allocate shared memory histogram (separate real/complex parts)
            constexpr bool is_complex = std::is_same_v<value_type, Kokkos::complex<real_type>>;
            const size_t hist_total = hx * hy * hz;

            // Allocate shared memory - for complex, we need two arrays
            shared_real_view hist_r(team.team_scratch(0), hist_total);
            shared_real_view hist_c;
            if constexpr (is_complex) {
                hist_c = shared_real_view(team.team_scratch(0), hist_total);
            }

            // Initialize histogram to zero
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hist_total),
                [&](int i) {
                    hist_r(i) = 0;
                    if constexpr (is_complex) {
                        hist_c(i) = 0;
                    }
                });
            team.team_barrier();

            // Get particles in this tile
            const size_type pstart = bin_offsets(tile_linear);
            const size_type pend = bin_offsets(tile_linear + 1);

            // Process particles
            const bool odd = (w & 1);
            const int half_left = (w - 1) / 2;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, pstart, pend),
                [&](size_type ip) {
                    const size_type j = permute(ip);
                    const value_type val = values(j);

                    constexpr real_type inv_two_pi = real_type(0.5) / real_type(3.14159265358979323846);  // 0.5 / pi

                    // X dimension - transform from [-pi, pi] to grid coordinates
                    real_type kx = x(j, 0) * inv_two_pi;
                    kx = kx - Kokkos::floor(kx);
                    real_type sx = kx * n_grid[0];
                    int idx_x = (w & 1) ? static_cast<int>(sx + real_type{0.5}) : static_cast<int>(sx);
                    auto diff_x = (sx - idx_x + half_left) * inv_hw;
                    if (idx_x >= static_cast<int>(n_grid[0])) idx_x -= n_grid[0];
                    idx_x = idx_x - tile_x0;

                    // Y dimension
                    real_type ky = x(j, 1) * inv_two_pi;
                    ky = ky - Kokkos::floor(ky);
                    real_type sy = ky * n_grid[1];
                    int idx_y = (w & 1) ? static_cast<int>(sy + real_type{0.5}) : static_cast<int>(sy);
                    auto diff_y = (sy - idx_y + half_left) * inv_hw;
                    if (idx_y >= static_cast<int>(n_grid[1])) idx_y -= n_grid[1];
                    idx_y = idx_y - tile_y0;

                    // Z dimension
                    real_type kz = x(j, 2) * inv_two_pi;
                    kz = kz - Kokkos::floor(kz);
                    real_type sz = kz * n_grid[2];
                    int idx_z = (w & 1) ? static_cast<int>(sz + real_type{0.5}) : static_cast<int>(sz);
                    auto diff_z = (sz - idx_z + half_left) * inv_hw;
                    if (idx_z >= static_cast<int>(n_grid[2])) idx_z -= n_grid[2];
                    idx_z = idx_z - tile_z0;

                    // Precompute kernel values
                    real_type kernel_x[16];
                    real_type kernel_y[16];
                    real_type kernel_z[16];

                    for (int wx = 0; wx < w; ++wx) {
                        kernel_x[wx] = kernel(inv_hw * wx - diff_x);
                    }
                    for (int wy = 0; wy < w; ++wy) {
                        kernel_y[wy] = kernel(inv_hw * wy - diff_y);
                    }

                    // Determine z range for this thread (match old code)
                    const int z_count = (tile_thread_idx == threads_per_tile - 1)
                                      ? (w - (threads_per_tile - 1) * z_tiles)
                                      : z_tiles;

                    for (int wz = 0; wz < z_count; ++wz) {
                        kernel_z[wz] = kernel(inv_hw * (wz + z_offset) - diff_z);
                    }

                    // Spread to histogram
                    for (int wz = 0; wz < z_count; ++wz) {
                        for (int wy = 0; wy < w; ++wy) {
                            for (int wx = 0; wx < w; ++wx) {
                                const real_type kernel_val = kernel_x[wx] * kernel_y[wy] * kernel_z[wz];
                                const int hist_idx = ((idx_z + wz) * hy + (idx_y + wy)) * hx + (idx_x + wx);

                                if constexpr (is_complex) {
                                    Kokkos::atomic_add(&hist_r(hist_idx), val.real() * kernel_val);
                                    Kokkos::atomic_add(&hist_c(hist_idx), val.imag() * kernel_val);
                                } else {
                                    Kokkos::atomic_add(&hist_r(hist_idx), val * kernel_val);
                                }
                            }
                        }
                    }
                });

            team.team_barrier();

            // Flush histogram to global grid (match old code exactly)
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hist_total),
                [&](int hist_idx) {
                    int hist_x = hist_idx % hx;
                    int hist_y = (hist_idx / hx) % hy;
                    int hist_z = hist_idx / (hx * hy);

                    int global_x = tile_x0 + hist_x - half_left;
                    int global_y = tile_y0 + hist_y - half_left;
                    int global_z = tile_z0 + hist_z - half_left + z_offset;

                    if (global_x < 0) global_x += n_grid[0];
                    else if (global_x >= static_cast<int>(n_grid[0])) global_x -= n_grid[0];
                    if (global_y < 0) global_y += n_grid[1];
                    else if (global_y >= static_cast<int>(n_grid[1])) global_y -= n_grid[1];
                    if (global_z < 0) global_z += n_grid[2];
                    else if (global_z >= static_cast<int>(n_grid[2])) global_z -= n_grid[2];

                    if constexpr (is_complex) {
#ifdef KOKKOS_ENABLE_CUDA
                        if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
                            double* addr_as_double = reinterpret_cast<double*>(
                                &grid(global_x + nghost, global_y + nghost, global_z + nghost));
                            Kokkos::atomic_add(&addr_as_double[0], hist_r(hist_idx));
                            Kokkos::atomic_add(&addr_as_double[1], hist_c(hist_idx));
                        } else
#endif
                        {
                            Kokkos::atomic_add(&grid(global_x + nghost, global_y + nghost, global_z + nghost),
                                              Kokkos::complex<real_type>(hist_r(hist_idx), hist_c(hist_idx)));
                        }
                    } else {
                        Kokkos::atomic_add(&grid(global_x + nghost, global_y + nghost, global_z + nghost),
                                          hist_r(hist_idx));
                    }
                });
        }
    };

}  // namespace detail
}  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_TILED_SCATTER_H
