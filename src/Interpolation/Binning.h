#ifndef IPPL_INTERPOLATION_BINNING_H
#define IPPL_INTERPOLATION_BINNING_H

#include <Kokkos_Core.hpp>

#include "InterpolationUtil.h"
#include "Particle/SortBuffer.h"

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

                Kokkos::Array<int, 3> n_grid_global;  // GLOBAL grid dimensions (for coord transform)
                Kokkos::Array<int, 3> n_grid_local;   // LOCAL grid dimensions
                Kokkos::Array<int, 3> local_offset;   // First global index of local domain
                Kokkos::Array<int, 3> tile_size;      // Tile size per dimension
                Kokkos::Array<int, 3> num_tiles;      // Number of tiles per dimension (in local domain)
                int w;                                 // kernel width

                KOKKOS_INLINE_FUNCTION int max_bins() const {
                    return (num_tiles[0] + 1) * (num_tiles[1] + 1) * (num_tiles[2] + 1) + 1;
                }

                // Helper to access position component - works with both View<T*[3]> and View<Vector<T,3>*>
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
                KOKKOS_INLINE_FUNCTION int bin(const ViewType& keys, int i) const {
                    // Compute tile index for each dimension
                    int tile_x = compute_bin(get_component(keys, i, 0), 0);
                    int tile_y = compute_bin(get_component(keys, i, 1), 1);
                    int tile_z = compute_bin(get_component(keys, i, 2), 2);

                    // Check if particle is outside local domain
                    if (tile_x < 0 || tile_x >= static_cast<int>(num_tiles[0]) ||
                        tile_y < 0 || tile_y >= static_cast<int>(num_tiles[1]) ||
                        tile_z < 0 || tile_z >= static_cast<int>(num_tiles[2])) {
                        return max_bins() - 1;  // Put out-of-domain particles in last bin
                    }

                    // Flatten to single bin index
                    return tile_z * (num_tiles[0] * num_tiles[1]) + tile_y * num_tiles[0] + tile_x;
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
             * @brief Sort particles by tile and compute bin offsets - allocation-free implementation
             *
             * This implementation performs binning directly without using Kokkos::BinSort,
             * avoiding all internal allocations. All required storage must be pre-allocated
             * and passed in.
             *
             * @tparam RealType The floating point type
             * @tparam PositionViewType The position view type (View<T*[3]> or View<Vector<T,3>*>)
             * @tparam ExecSpace Kokkos execution space
             * @tparam PermuteViewType View type for permutation indices
             * @tparam OffsetViewType View type for bin offsets
             * @tparam CountViewType View type for bin counts (atomic)
             *
             * @param x Positions in PHYSICAL coordinates [-pi, pi]
             * @param n_grid_global GLOBAL grid dimensions
             * @param n_grid_local LOCAL grid dimensions
             * @param local_offset First global index of local domain
             * @param tile_size Tile size per dimension
             * @param w Kernel width
             * @param permute [out] Pre-allocated view for permutation vector (size >= n_particles)
             * @param bin_offsets [out] Pre-allocated view for bin offsets (size >= n_tiles + 1)
             * @param bin_count [temp] Pre-allocated view for bin counts (size >= n_tiles), will be zeroed
             * @param n_particles Number of particles to process
             */
            template <typename RealType, typename PositionViewType, typename ExecSpace,
                      typename PermuteViewType, typename OffsetViewType, typename CountViewType>
            void bin_sort_3d(
                PositionViewType x,
                Kokkos::Array<int, 3> n_grid_global,
                Kokkos::Array<int, 3> n_grid_local,
                Kokkos::Array<int, 3> local_offset,
                Kokkos::Array<int, 3> tile_size,
                int w,
                PermuteViewType& permute,
                OffsetViewType& bin_offsets,
                CountViewType& bin_count,
                const size_t n_particles) {

                using size_type = typename ExecSpace::memory_space::size_type;

                static IpplTimings::TimerRef binSortTimer = IpplTimings::getTimer("binSort");
                IpplTimings::startTimer(binSortTimer);

                // Calculate number of tiles based on LOCAL grid
                Kokkos::Array<int, 3> num_tiles;
                num_tiles[0] = (n_grid_local[0] + tile_size[0] - 1) / tile_size[0] + 1;
                num_tiles[1] = (n_grid_local[1] + tile_size[1] - 1) / tile_size[1] + 1;
                num_tiles[2] = (n_grid_local[2] + tile_size[2] - 1) / tile_size[2] + 1;
                const size_type n_bins = num_tiles[0] * num_tiles[1] * num_tiles[2];

                // Create BinOp
                BinOp3D<RealType, ExecSpace> bin_op{
                    n_grid_global, n_grid_local, local_offset,
                    tile_size, num_tiles, w
                };

                // Zero out the bin count array
                static IpplTimings::TimerRef zeroTimer = IpplTimings::getTimer("binZero");
                IpplTimings::startTimer(zeroTimer);
                Kokkos::deep_copy(bin_count, 0);
                IpplTimings::stopTimer(zeroTimer);

                // Step 1: Count particles per bin
                static IpplTimings::TimerRef countTimer = IpplTimings::getTimer("binCount");
                IpplTimings::startTimer(countTimer);

                // Create atomic view of bin_count for thread-safe incrementing
                using AtomicCountViewType = Kokkos::View<size_t*, typename CountViewType::memory_space,
                                                         Kokkos::MemoryTraits<Kokkos::Atomic>>;
                AtomicCountViewType bin_count_atomic = bin_count;

                Kokkos::parallel_for(
                    "BinSort::CountBins",
                    Kokkos::RangePolicy<ExecSpace>(0, n_particles),
                    KOKKOS_LAMBDA(const size_type i) {
                        const int bin_idx = bin_op.bin(x, i);
                        bin_count_atomic(bin_idx)++;
                    });
                Kokkos::fence();
                IpplTimings::stopTimer(countTimer);

                // Step 2: Compute bin offsets via exclusive prefix scan
                static IpplTimings::TimerRef scanTimer = IpplTimings::getTimer("binScan");
                IpplTimings::startTimer(scanTimer);

                Kokkos::parallel_scan(
                    "BinSort::ComputeOffsets",
                    Kokkos::RangePolicy<ExecSpace>(0, n_bins),
                    KOKKOS_LAMBDA(const size_type i, size_type& running_sum, const bool final) {
                        if (final) {
                            bin_offsets(i) = running_sum;
                        }
                        running_sum += bin_count(i);
                    });

                // Set the last offset (total count)
                Kokkos::parallel_for(
                    "BinSort::SetLastOffset",
                    Kokkos::RangePolicy<ExecSpace>(0, 1),
                    KOKKOS_LAMBDA(const size_type) {
                        size_type total = 0;
                        for (size_type i = 0; i < n_bins; ++i) {
                            total += bin_count(i);
                        }
                        bin_offsets(n_bins) = total;
                    });
                Kokkos::fence();
                IpplTimings::stopTimer(scanTimer);

                // Step 3: Reset bin counts and place particles into permutation array
                static IpplTimings::TimerRef binningTimer = IpplTimings::getTimer("binPlacement");
                IpplTimings::startTimer(binningTimer);

                Kokkos::deep_copy(bin_count, 0);

                Kokkos::parallel_for(
                    "BinSort::CreatePermutation",
                    Kokkos::RangePolicy<ExecSpace>(0, n_particles),
                    KOKKOS_LAMBDA(const size_type i) {
                        const int bin_idx = bin_op.bin(x, i);
                        const int local_offset_in_bin = bin_count_atomic(bin_idx)++;
                        permute(bin_offsets(bin_idx) + local_offset_in_bin) = i;
                    });
                Kokkos::fence();
                IpplTimings::stopTimer(binningTimer);

                IpplTimings::stopTimer(binSortTimer);
            }

            /**
             * @brief Overload maintaining backward compatibility - allocates bin_count internally
             *
             * Note: This version still allocates a temporary bin_count array.
             * For fully allocation-free operation, use the version that takes bin_count as parameter.
             */
            template <typename RealType, typename PositionViewType, typename ExecSpace,
                      typename PermuteViewType, typename OffsetViewType>
            void bin_sort_3d(
                PositionViewType x,
                Kokkos::Array<int, 3> n_grid_global,
                Kokkos::Array<int, 3> n_grid_local,
                Kokkos::Array<int, 3> local_offset,
                Kokkos::Array<int, 3> tile_size,
                int w,
                PermuteViewType& permute,
                OffsetViewType& bin_offsets,
                const size_t n_particles) {

                using size_type = typename ExecSpace::memory_space::size_type;
                using memory_space = typename ExecSpace::memory_space;

                // Calculate number of bins
                Kokkos::Array<int, 3> num_tiles;
                num_tiles[0] = (n_grid_local[0] + tile_size[0] - 1) / tile_size[0] + 1;
                num_tiles[1] = (n_grid_local[1] + tile_size[1] - 1) / tile_size[1] + 1;
                num_tiles[2] = (n_grid_local[2] + tile_size[2] - 1) / tile_size[2] + 1;
                const size_type n_bins = num_tiles[0] * num_tiles[1] * num_tiles[2];

                auto& buf_handler = ippl::detail::getDefaultSortBufferManager<memory_space>();

                // Kokkos::View<int*, memory_space> bin_count("bin_count", n_bins);
                auto bin_count = buf_handler.mortonKeys();

                bin_sort_3d<RealType, PositionViewType, ExecSpace>(
                    x, n_grid_global, n_grid_local, local_offset,
                    tile_size, w, permute, bin_offsets, bin_count, n_particles);
            }

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_INTERPOLATION_BINNING_H