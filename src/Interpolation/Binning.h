#ifndef IPPL_INTERPOLATION_BINNING_H
#define IPPL_INTERPOLATION_BINNING_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>

#include "InterpolationUtil.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

            /**
             * @brief Generic BinOp for 3D tiled scatter/gather
             *
             * Bins particles into tiles based on their grid coordinates.
             * This is a generic implementation that works with any coordinate system.
             *
             * @tparam RealType Floating point type
             * @tparam ExecSpace Kokkos execution space
             */
            template <typename RealType, typename ExecSpace>
            struct BinOp3D {
                using size_type = typename ExecSpace::memory_space::size_type;

                Kokkos::Array<int, 3>
                    n_grid_global;  // GLOBAL grid dimensions (for coord transform)
                Kokkos::Array<int, 3> n_grid_local;  // LOCAL grid dimensions
                Kokkos::Array<int, 3> local_offset;        // First global index of local domain
                Kokkos::Array<int, 3> tile_size;           // Tile size per dimension
                Kokkos::Array<int, 3>
                    num_tiles;  // Number of tiles per dimension (in local domain)
                int w;          // kernel width

                KOKKOS_INLINE_FUNCTION int max_bins() const {
                    return (num_tiles[0] + 1) * (num_tiles[1] + 1) * (num_tiles[2] + 1) + 1;
                }

                // Helper to access position component - works with both View<T*[3]> and
                // View<Vector<T,3>*>
                template <class ViewType>
                KOKKOS_INLINE_FUNCTION static auto get_component(const ViewType& keys, int i, int d)
                    -> decltype(keys(i, d)) {
                    return keys(i, d);  // For View<T*[3]>
                }

                template <class ViewType>
                KOKKOS_INLINE_FUNCTION static auto get_component(const ViewType& keys, int i, int d)
                    -> decltype(keys(i)[d]) {
                    return keys(i)[d];  // For View<Vector<T,3>*>
                }

                template <class ViewType>
                KOKKOS_INLINE_FUNCTION int bin(ViewType& keys, int i) const {
                    // Compute tile index for each dimension
                    // keys contains particle positions in physical coordinates [0, 2*pi]
                    // Returns -1 if particle is outside local domain
                    int tile_x = compute_bin(get_component(keys, i, 0), 0);
                    int tile_y = compute_bin(get_component(keys, i, 1), 1);
                    int tile_z = compute_bin(get_component(keys, i, 2), 2);

                    // Check if particle is outside local domain
                    if (tile_x < 0 || tile_x >= static_cast<int>(num_tiles[0]) || tile_y < 0
                        || tile_y >= static_cast<int>(num_tiles[1]) || tile_z < 0
                        || tile_z >= static_cast<int>(num_tiles[2])) {
                        // This should not happen - particles should be on correct rank
                        assert(false && "Particle outside local domain during binning");
                        return max_bins() - 1;  // Put out-of-domain particles in last bin
                    }

                    // Flatten to single bin index
                    return tile_z * (num_tiles[0] * num_tiles[1]) + tile_y * num_tiles[0] + tile_x;
                }

                template <class ViewType, typename IndexType>
                KOKKOS_INLINE_FUNCTION bool operator()(ViewType& keys, IndexType i1,
                                                       IndexType i2) const {
                    // Lexicographic comparison by tile indices
                    int bin1_x = compute_bin(get_component(keys, i1, 0), 0);
                    int bin2_x = compute_bin(get_component(keys, i2, 0), 0);
                    if (bin1_x < bin2_x)
                        return true;
                    if (bin1_x > bin2_x)
                        return false;

                    int bin1_y = compute_bin(get_component(keys, i1, 1), 1);
                    int bin2_y = compute_bin(get_component(keys, i2, 1), 1);
                    if (bin1_y < bin2_y)
                        return true;
                    if (bin1_y > bin2_y)
                        return false;

                    int bin1_z = compute_bin(get_component(keys, i1, 2), 2);
                    int bin2_z = compute_bin(get_component(keys, i2, 2), 2);
                    return bin1_z < bin2_z;
                }

                KOKKOS_INLINE_FUNCTION int compute_bin(RealType val, int dim) const {
                    // Convert physical position to global grid index
                    int global_idx = detail::fourier_point_to_grid_idx(val, n_grid_global[dim], w);
                    // Convert to local grid index
                    int local_idx = global_idx - local_offset[dim];
                    // Compute local tile index
                    return local_idx / tile_size[dim];
                }
            };

            /**
             * @brief Sort particles by tile and compute bin offsets using Kokkos::BinSort
             *
             * This is a generic implementation that works with View<T*[3]> or View<Vector<T,3>*>.
             * The input coordinates x are in physical coordinates [-pi, pi] and will be transformed
             * internally. Binning is performed in LOCAL grid coordinates.
             *
             * @tparam RealType The floating point type
             * @tparam PositionViewType The position view type (View<T*[3]> or View<Vector<T,3>*>)
             * @tparam ExecSpace Kokkos execution space
             */
            template <typename RealType, typename PositionViewType, typename ExecSpace,
                      typename PermuteViewType, typename OffsetViewType>
            void bin_sort_3d(
                PositionViewType x,  // Positions in PHYSICAL coordinates [-pi, pi]
                Kokkos::Array<int, 3>
                    n_grid_global,  // GLOBAL grid dimensions
                Kokkos::Array<int, 3>
                    n_grid_local,                    // LOCAL grid dimensions
                Kokkos::Array<int, 3> local_offset,  // First global index of local domain
                Kokkos::Array<int, 3> tile_size, int w, PermuteViewType& permute,
                OffsetViewType& bin_offsets, const size_t n_particles) {
                using size_type    = typename ExecSpace::memory_space::size_type;
                using memory_space = typename ExecSpace::memory_space;

                // const size_type n_particles = x.extent(0);

                // Calculate number of tiles based on LOCAL grid
                Kokkos::Array<int, 3> num_tiles;
                num_tiles[0]      = (n_grid_local[0] + tile_size[0] - 1) / tile_size[0] + 1;
                num_tiles[1]      = (n_grid_local[1] + tile_size[1] - 1) / tile_size[1] + 1;
                num_tiles[2]      = (n_grid_local[2] + tile_size[2] - 1) / tile_size[2] + 1;
                size_type n_tiles = num_tiles[0] * num_tiles[1] * num_tiles[2];

                // Allocate outputs
                // Kokkos::realloc(permute, n_particles);
                // Kokkos::realloc(bin_offsets, n_tiles + 1);
		Kokkos::deep_copy(permute, 0);
		Kokkos::deep_copy(bin_offsets, 0);

                // Create BinOp
                BinOp3D<RealType, ExecSpace> bin_op{n_grid_global, n_grid_local, local_offset,
                                                    tile_size,     num_tiles,    w};

                // Use Kokkos::BinSort
                Kokkos::BinSort<decltype(x), BinOp3D<RealType, ExecSpace>> sorter(x, 0, n_particles,
                                                                                  bin_op, false);

                sorter.create_permute_vector();
                auto perm_view = sorter.get_permute_vector();
                Kokkos::deep_copy(Kokkos::subview(permute, Kokkos::make_pair<size_type, size_type>(0, n_particles)), Kokkos::subview(perm_view, Kokkos::make_pair<size_type, size_type>(0, n_particles)));

                // Get bin offsets from sorter
                auto offsets_view = sorter.get_bin_offsets();
                Kokkos::parallel_for(
                    "copy_bin_offsets", Kokkos::RangePolicy<ExecSpace>(0, n_tiles + 1),
                    KOKKOS_LAMBDA(size_type i) { bin_offsets(i) = offsets_view(i); });

                Kokkos::fence();
            }

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_INTERPOLATION_BINNING_H
