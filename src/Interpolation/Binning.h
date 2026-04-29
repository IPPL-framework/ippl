#ifndef IPPL_INTERPOLATION_BINNING_H
#define IPPL_INTERPOLATION_BINNING_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>

#include "CoordinateTransform.h"
#include "Particle/ParticleLayout.h"
#include "Particle/SortBuffer.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

            template <int Dim, typename MemorySpace>
            struct BinningResult {
                Kokkos::View<size_type*, MemorySpace> permute;
                Kokkos::View<size_type*, MemorySpace> bin_offsets;
                Vector<int, Dim> num_tiles;
            };

            /**
             * @brief Functor to compute bin index for a particle position
             *
             * Maps physical particle positions to tile indices for tiled scatter/gather.
             * Particles are assigned to tiles based on their stencil center location.
             *
             * @tparam Dim Spatial dimension
             * @tparam RealType Floating point type
             */
            template <unsigned Dim, typename RealType>
            struct BinComputer {
                Vector<int, Dim> n_grid_global;
                Vector<int, Dim> local_offset;
                Vector<int, Dim> tile_size;
                Vector<int, Dim> num_tiles;
                int kernel_width;
                CoordinateTransform<RealType, Dim> transform;

                /**
                 * @brief Compute 1D tile index for a coordinate value
                 */
                KOKKOS_INLINE_FUNCTION int compute_tile_1d(RealType val, int dim) const {
                    const RealType grid_pos = transform.toGridCoordinate(val, dim);
                    const int center =
                        transform.getStencilCenter(grid_pos - RealType(0.5), kernel_width);
                    const int local_c = center - local_offset[dim];
                    return Kokkos::clamp(local_c / tile_size[dim], 0, num_tiles[dim] - 1);
                }

                /**
                 * @brief Compute flat bin index for a particle position
                 *
                 * Uses row-major ordering (dimension Dim-1 varies fastest) to match
                 * the decoding in TiledScatter.
                 */
                template <typename PositionType>
                KOKKOS_INLINE_FUNCTION int operator()(const PositionType& pos) const {
                    int bin_idx = 0;
                    int stride  = 1;

                    for (int d = Dim - 1; d >= 0; --d) {
                        const int tile_d = compute_tile_1d(pos[d], d);
                        bin_idx += tile_d * stride;
                        stride *= num_tiles[d];
                    }

                    return bin_idx;
                }
            };

            /**
             * @brief Sort particles by bin using Kokkos::sort_by_key and compute bin offsets
             *
             * Algorithm:
             * 1. Compute bin key for each particle
             * 2. Sort (keys, permutation) pairs by key using Kokkos::sort_by_key (maps to vendor)
             * 3. Compute bin offsets by finding transitions in sorted keys
             *
             * @tparam Dim Spatial dimension
             * @tparam RealType Floating point type
             * @tparam PositionViewType Position view type
             * @tparam ExecSpace Kokkos execution space
             * @tparam PermuteViewType Permutation view type
             * @tparam OffsetViewType Bin offset view type
             * @tparam KeyViewType Bin key view type
             *
             * @param positions Particle positions in physical coordinates
             * @param n_grid_global Global grid dimensions
             * @param n_grid_local Local grid dimensions (kept for API compatibility; unused)
             * @param local_offset First global index of local domain
             * @param tile_size Tile size per dimension
             * @param kernel_width Interpolation kernel width
             * @param origin Mesh origin
             * @param invdx Inverse mesh spacing
             * @param[out] permute Permutation array (sorted particle indices)
             * @param[out] bin_offsets Start index of each bin in permute array
             * @param[inout] bin_keys Temporary storage for bin keys
             * @param n_particles Number of particles
             * @param num_tiles Number of tiles per dimension
             */
            template <unsigned Dim, typename RealType, typename PositionViewType,
                      typename ExecSpace, typename PermuteViewType, typename OffsetViewType,
                      typename KeyViewType>
            void bin_sort(PositionViewType positions, Vector<int, Dim> n_grid_global,
                          [[maybe_unused]] Vector<int, Dim> n_grid_local,
                          Vector<int, Dim> local_offset, Vector<int, Dim> tile_size,
                          int kernel_width, Vector<RealType, Dim> origin,
                          Vector<RealType, Dim> invdx, PermuteViewType& permute,
                          OffsetViewType& bin_offsets, KeyViewType& bin_keys, size_t n_particles,
                          Vector<int, Dim> num_tiles) {
                using key_type = typename KeyViewType::non_const_value_type;

                static IpplTimings::TimerRef binSortTimer = IpplTimings::getTimer("binSort");
                IpplTimings::startTimer(binSortTimer);

                // Total number of bins
                size_t n_bins = 1;
                for (unsigned d = 0; d < Dim; ++d) {
                    n_bins *= num_tiles[d];
                }

                // Create bin computer functor
                CoordinateTransform<RealType, Dim> transform(origin, invdx, n_grid_global);
                BinComputer<Dim, RealType> bin_computer{n_grid_global, local_offset, tile_size,
                                                        num_tiles,     kernel_width, transform};

                // Step 1: Compute bin keys and initialize permutation
                static IpplTimings::TimerRef keyTimer = IpplTimings::getTimer("binComputeKeys");
                IpplTimings::startTimer(keyTimer);

                Kokkos::parallel_for(
                    "BinSort::ComputeKeys", Kokkos::RangePolicy<ExecSpace>(0, n_particles),
                    KOKKOS_LAMBDA(const size_t i) {
                        bin_keys(i) = static_cast<key_type>(bin_computer(positions(i)));
                        permute(i)  = i;
                    });
                Kokkos::fence();
                IpplTimings::stopTimer(keyTimer);

                // Step 2: Sort by key
                static IpplTimings::TimerRef sortTimer = IpplTimings::getTimer("binSortByKey");
                IpplTimings::startTimer(sortTimer);

                if (n_particles > 0) {
                    auto keys_sub =
                        Kokkos::subview(bin_keys, std::make_pair(size_t(0), n_particles));
                    auto permute_sub =
                        Kokkos::subview(permute, std::make_pair(size_t(0), n_particles));

                    // The CUB radix-sort fast path is only available when the
                    // *execution space* is Kokkos::Cuda; on a CUDA-enabled
                    // build the test fixture also instantiates Kokkos::Serial
                    // (and possibly OpenMP), where ExecSpace().cuda_stream()
                    // doesn't exist. Gate accordingly.
#if defined(KOKKOS_ENABLE_CUDA)
                    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
                        auto& bufs = ippl::detail::getDefaultBinSortBuffers<
                            typename KeyViewType::memory_space>();
                        bufs.ensureCapacity(n_particles, n_bins + 1);
                        auto keys_out_sub = Kokkos::subview(
                            bufs.keysOut(), std::make_pair(size_t(0), n_particles));
                        auto perm_out_sub = Kokkos::subview(
                            bufs.permOut(), std::make_pair(size_t(0), n_particles));

                        cudaStream_t cuda_stream = ExecSpace().cuda_stream();

                        void* d_temp      = nullptr;
                        size_t temp_bytes = 0;
                        cub::DeviceRadixSort::SortPairs(
                            d_temp, temp_bytes, keys_sub.data(), keys_out_sub.data(),
                            permute_sub.data(), perm_out_sub.data(),
                            static_cast<int>(n_particles), 0, sizeof(key_type) * 8, cuda_stream);

                        bufs.ensureTempStorage(temp_bytes);
                        d_temp = bufs.tempStorage().data();

                        auto err = cub::DeviceRadixSort::SortPairs(
                            d_temp, temp_bytes, keys_sub.data(), keys_out_sub.data(),
                            permute_sub.data(), perm_out_sub.data(),
                            static_cast<int>(n_particles), 0, sizeof(key_type) * 8, cuda_stream);

                        if (err != cudaSuccess) {
                            printf("CUB SortPairs failed: %s\n", cudaGetErrorString(err));
                            Kokkos::abort("CUB Radix Sort failed.");
                        }

                        Kokkos::deep_copy(ExecSpace(), keys_sub, keys_out_sub);
                        Kokkos::deep_copy(ExecSpace(), permute_sub, perm_out_sub);
                    } else {
                        Kokkos::Experimental::sort_by_key(ExecSpace(), keys_sub, permute_sub);
                    }
#else
                    Kokkos::Experimental::sort_by_key(ExecSpace(), keys_sub, permute_sub);
#endif
                }
                Kokkos::fence();
                IpplTimings::stopTimer(sortTimer);

                // Step 3: Compute bin offsets from sorted keys
                static IpplTimings::TimerRef offsetTimer =
                    IpplTimings::getTimer("binComputeOffsets");
                IpplTimings::startTimer(offsetTimer);

                // Initialize all offsets to n_particles (empty bins point to end)
                Kokkos::parallel_for(
                    "BinSort::InitOffsets", Kokkos::RangePolicy<ExecSpace>(0, n_bins + 1),
                    KOKKOS_LAMBDA(const size_t i) { bin_offsets(i) = n_particles; });
                Kokkos::fence();

                if (n_particles > 0) {
                    // Step A: each particle writes its index into the slot of
                    // its own bin iff it is the first particle in that bin.
                    // Each thread does O(1) work — no inner loop over the gap
                    // to the previous transition.
                    Kokkos::parallel_for(
                        "BinSort::MarkStarts", Kokkos::RangePolicy<ExecSpace>(0, n_particles),
                        KOKKOS_LAMBDA(const size_t i) {
                            const auto curr_bin = bin_keys(i);
                            if (i == 0 || bin_keys(i - 1) != curr_bin) {
                                bin_offsets(static_cast<size_t>(curr_bin)) = i;
                            }
                        });
                    Kokkos::fence();

                    // Step B: right-to-left inclusive min-scan to fill empty
                    // bins (their start = next non-empty bin's start). Empty
                    // bins still hold the n_particles sentinel after Step A.
                    // Sequential single-thread launch is intentional: the
                    // dependency is purely linear and n_bins is small relative
                    // to n_particles in any realistic configuration.
                    Kokkos::parallel_for(
                        "BinSort::PropagateEmpties",
                        Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(const int) {
                            for (size_t b = n_bins; b-- > 0;) {
                                if (bin_offsets(b) > bin_offsets(b + 1)) {
                                    bin_offsets(b) = bin_offsets(b + 1);
                                }
                            }
                        });
                }
                Kokkos::fence();
                IpplTimings::stopTimer(offsetTimer);

                IpplTimings::stopTimer(binSortTimer);
            }

            /**
             * @brief High-level interface for particle binning
             *
             * Bins particles into tiles for tiled scatter/gather operations.
             * Uses sort-based binning for efficient GPU execution.
             *
             * @tparam ParticleT Particle coordinate type
             * @tparam FieldT Field value type
             * @tparam ParticleProperties Additional particle attribute properties
             * @tparam Dim Spatial dimension
             *
             * @param particles Particle position attribute
             * @param fieldLayout Field layout
             * @param mesh Uniform Cartesian mesh
             * @param tile_size Tile size per dimension
             * @param kernel_width Interpolation kernel width
             *
             * @return Tuple of (permutation, bin_offsets, num_tiles)
             */
            template <typename ParticleT, typename FieldT, class... ParticleProperties,
                      unsigned Dim>
            auto bin_particles(
                const ParticleAttrib<Vector<ParticleT, Dim>, ParticleProperties...>& particles,
                FieldLayout<Dim> fieldLayout, UniformCartesian<FieldT, Dim> mesh,
                Vector<int, Dim> tile_size, int kernel_width) {
                using AttribType   = std::decay_t<decltype(particles)>;
                using ExecSpace    = typename AttribType::execution_space;
                using memory_space = typename AttribType::memory_space;

                // Extract grid information
                const NDIndex<Dim>& lDom = fieldLayout.getLocalNDIndex();
                const NDIndex<Dim>& gDom = fieldLayout.getDomain();

                Vector<int, Dim> ngrid_global;
                Vector<int, Dim> ngrid_local;
                Vector<int, Dim> local_offset;
                for (unsigned d = 0; d < Dim; ++d) {
                    ngrid_global[d] = gDom[d].length();
                    ngrid_local[d]  = lDom[d].length();
                    local_offset[d] = lDom[d].first();
                }

                // Compute number of tiles (+1 for boundary particles).
                Vector<int, Dim> num_tiles;
                size_t total_tiles = 1;
                for (unsigned d = 0; d < Dim; ++d) {
                    num_tiles[d] = (ngrid_local[d] + tile_size[d] - 1) / tile_size[d] + 1;
                    total_tiles *= num_tiles[d];
                }

                auto particle_view       = particles.getView();
                const auto invdx         = 1.0 / mesh.getMeshSpacing();
                const size_t n_particles = particles.getParticleCount();

                auto& bufs = ippl::detail::getDefaultBinSortBuffers<memory_space>();
                // n_bins + 1 slots needed for bin_offsets
                bufs.ensureCapacity(n_particles, total_tiles + 1);

                auto& permute     = bufs.permute();
                auto& bin_offsets = bufs.binOffsets();
                auto& bin_keys    = bufs.binKeys();

                bin_sort<Dim, ParticleT, std::decay_t<decltype(particle_view)>, ExecSpace>(
                    particle_view, ngrid_global, ngrid_local, local_offset, tile_size,
                    kernel_width, mesh.getOrigin(), invdx, permute, bin_offsets, bin_keys,
                    n_particles, num_tiles);

                return std::make_tuple(permute, bin_offsets, num_tiles);
            }

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_INTERPOLATION_BINNING_H