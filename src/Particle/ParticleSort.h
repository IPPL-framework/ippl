//
// Class ParticleSort
//   Utilities for sorting particles based on spatial locality using Morton codes.
//   Uses CUB for CUDA, rocPRIM for HIP, and Kokkos::BinSort as fallback.
//   Now with buffer management for memory reuse across multiple sort calls.
//
#ifndef IPPL_PARTICLE_SORT_H
#define IPPL_PARTICLE_SORT_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "Particle/SortBuffer.h"

#ifdef KOKKOS_ENABLE_CUDA
#include <cub/device/device_radix_sort.cuh>
#endif

#ifdef KOKKOS_ENABLE_HIP
#include <rocprim/rocprim.hpp>
#endif

namespace ippl {
    namespace detail {

        /**
         * @brief Compute Morton code (Z-order curve) for spatial sorting
         *
         * Interleaves bits from each dimension to create a single sortable key
         * that preserves spatial locality.
         *
         * @tparam Dim Number of dimensions
         * @tparam T Floating point type
         * @tparam IndexType Integer type for grid indices
         */
        template <unsigned Dim, typename T, typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION uint64_t computeMortonCode(const Vector<T, Dim>& position,
                                                          const Vector<T, Dim>& origin,
                                                          const Vector<T, Dim>& invdx,
                                                          const Vector<IndexType, Dim>& ngrid) {
            uint64_t morton            = 0;
            constexpr int bits_per_dim = 21;  // Safe for 3D (21*3 = 63 bits)

            uint32_t gridIndices[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                T sx = (position[d] - origin[d]) * invdx[d];
                // Wrap to [0, ngrid)
                sx -= ngrid[d] * Kokkos::floor(sx / ngrid[d]);
                gridIndices[d] = static_cast<uint32_t>(sx);
            }

            for (int bit = 0; bit < bits_per_dim; ++bit) {
                for (unsigned d = 0; d < Dim; ++d) {
                    if (gridIndices[d] & (1u << bit)) {
                        morton |= (uint64_t(1) << (bit * Dim + d));
                    }
                }
            }
            return morton;
        }

        /**
         * @brief Functor to compute Morton codes for all particles
         */
        template <unsigned Dim, typename PositionView, typename T, typename KeyView>
        struct ComputeMortonCodesFunctor {
            using memory_space = typename PositionView::memory_space;
            using size_type    = size_t;

            PositionView positions;
            KeyView keys;
            Vector<T, Dim> origin;
            Vector<T, Dim> invdx;
            Vector<size_type, Dim> ngrid;

            KOKKOS_INLINE_FUNCTION void operator()(size_type i) const {
                keys(i) = computeMortonCode<Dim, T, size_type>(positions(i), origin, invdx, ngrid);
            }
        };

        /**
         * @brief Sort particles on host using std::sort
         */
        template <unsigned Dim, typename T>
        void sortParticlesHost(Kokkos::View<Vector<T, Dim>*, Kokkos::HostSpace> positions,
                               Kokkos::View<size_t*, Kokkos::HostSpace> permute,
                               const Vector<T, Dim>& origin, const Vector<T, Dim>& invdx,
                               const Vector<size_t, Dim>& ngrid, size_t n) {
            // Compute Morton codes
            std::vector<std::pair<uint64_t, size_t>> key_index_pairs(n);

            for (size_t i = 0; i < n; ++i) {
                key_index_pairs[i] = {
                    computeMortonCode<Dim, T, size_t>(positions(i), origin, invdx, ngrid), i};
            }

            // Sort by Morton code
            std::sort(key_index_pairs.begin(), key_index_pairs.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });

            // Extract permutation
            for (size_t i = 0; i < n; ++i) {
                permute(i) = key_index_pairs[i].second;
            }
        }

#ifdef KOKKOS_ENABLE_CUDA
        /**
         * @brief Sort particles on CUDA using CUB RadixSort with buffer reuse
         *
         * Writes sorted indices directly to permute array to avoid extra copy.
         *
         * @param buffer_manager Optional buffer manager for memory reuse.
         *                       If nullptr, uses the default static buffer manager.
         */
        template <unsigned Dim, typename T, typename PermuteViewType>
        void sortParticlesCuda(Kokkos::View<Vector<T, Dim>*, Kokkos::CudaSpace> positions,
                               PermuteViewType permute, const Vector<T, Dim>& origin,
                               const Vector<T, Dim>& invdx, const Vector<size_t, Dim>& ngrid,
                               size_t n,
                               SortBufferManager<Kokkos::CudaSpace>* buffer_manager = nullptr) {
            using memory_space = Kokkos::CudaSpace;
            using size_type    = size_t;

            // Use provided buffer manager or get the default one
            SortBufferManager<memory_space>& buffers =
                buffer_manager ? *buffer_manager : getDefaultSortBufferManager<memory_space>();

            // Ensure buffers are large enough
            buffers.ensureCapacity(n, 0);

            // Get buffer views
            auto& keys        = buffers.mortonKeys();
            auto& keys_sorted = buffers.mortonKeysSorted();
            auto& indices     = buffers.indices();

            // Compute Morton codes
            Kokkos::parallel_for(
                "compute_morton_codes", Kokkos::RangePolicy<Kokkos::Cuda>(0, n),
                ComputeMortonCodesFunctor<Dim, decltype(positions), T, decltype(keys)>{
                    positions, keys, origin, invdx, ngrid});

            // Initialize indices
            Kokkos::parallel_for(
                "init_indices", Kokkos::RangePolicy<Kokkos::Cuda>(0, n),
                KOKKOS_LAMBDA(size_type i) { indices(i) = i; });

            Kokkos::fence();

            // Determine temporary storage requirements
            size_t temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, keys.data(),
                                            keys_sorted.data(), indices.data(),
                                            permute.data(), n);

            // Ensure temp storage is large enough
            buffers.ensureTempStorageCapacity(temp_storage_bytes);
            auto& d_temp_storage = buffers.tempStorage();

            // Run sorting operation - write directly to permute
            cub::DeviceRadixSort::SortPairs(d_temp_storage.data(), temp_storage_bytes, keys.data(),
                                            keys_sorted.data(), indices.data(),
                                            permute.data(), n);

            Kokkos::fence();
        }
#endif

#ifdef KOKKOS_ENABLE_HIP
        /**
         * @brief Sort particles on HIP using rocPRIM RadixSort with buffer reuse
         *
         * Writes sorted indices directly to permute array to avoid extra copy.
         */
        template <unsigned Dim, typename T, typename PermuteViewType>
        void sortParticlesHip(Kokkos::View<Vector<T, Dim>*, Kokkos::HIPSpace> positions,
                              PermuteViewType permute, const Vector<T, Dim>& origin,
                              const Vector<T, Dim>& invdx, const Vector<size_t, Dim>& ngrid,
                              size_t n,
                              SortBufferManager<Kokkos::HIPSpace>* buffer_manager = nullptr) {
            using memory_space = Kokkos::HIPSpace;
            using size_type    = size_t;

            // Use provided buffer manager or get the default one
            SortBufferManager<memory_space>& buffers =
                buffer_manager ? *buffer_manager : getDefaultSortBufferManager<memory_space>();

            // Ensure buffers are large enough
            buffers.ensureCapacity(n);

            // Get buffer views
            auto& keys        = buffers.mortonKeys();
            auto& keys_sorted = buffers.mortonKeysSorted();
            auto& indices     = buffers.indices();

            // Compute Morton codes
            Kokkos::parallel_for(
                "compute_morton_codes", Kokkos::RangePolicy<Kokkos::HIP>(0, n),
                ComputeMortonCodesFunctor<Dim, decltype(positions), T, decltype(keys)>{
                    positions, keys, origin, invdx, ngrid});

            // Initialize indices
            Kokkos::parallel_for(
                "init_indices", Kokkos::RangePolicy<Kokkos::HIP>(0, n),
                KOKKOS_LAMBDA(size_type i) { indices(i) = i; });

            Kokkos::fence();

            // Determine temporary storage requirements
            size_t temp_storage_bytes = 0;
            rocprim::radix_sort_pairs(nullptr, temp_storage_bytes, keys.data(), keys_sorted.data(),
                                      indices.data(), permute.data(), n, 0,
                                      sizeof(uint64_t) * 8, Kokkos::HIP().hip_stream());

            // Ensure temp storage is large enough
            buffers.ensureTempStorageCapacity(temp_storage_bytes);
            auto& d_temp_storage = buffers.tempStorage();

            // Run sorting operation - write directly to permute
            rocprim::radix_sort_pairs(d_temp_storage.data(), temp_storage_bytes, keys.data(),
                                      keys_sorted.data(), indices.data(), permute.data(), n,
                                      0, sizeof(uint64_t) * 8, Kokkos::HIP().hip_stream());

            Kokkos::fence();
        }
#endif

        /**
         * @brief Generic sort dispatcher based on execution space
         *
         * @tparam Dim Number of dimensions
         * @tparam ExecSpace Kokkos execution space
         * @tparam T Floating point type
         * @param buffer_manager Optional buffer manager for memory reuse
         */
        template <unsigned Dim, typename ExecSpace, typename T, typename PermuteViewType>
        void sortParticles(
            Kokkos::View<Vector<T, Dim>*, typename ExecSpace::memory_space> positions,
            PermuteViewType permute, const Vector<T, Dim>& origin, const Vector<T, Dim>& invdx,
            const Vector<size_t, Dim>& ngrid, size_t n,
            SortBufferManager<typename ExecSpace::memory_space>* buffer_manager) {
            using memory_space = typename ExecSpace::memory_space;

#ifdef KOKKOS_ENABLE_CUDA
            if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
                sortParticlesCuda<Dim, T>(positions, permute, origin, invdx, ngrid, n,
                                          buffer_manager);
                return;
            }
#endif

#ifdef KOKKOS_ENABLE_HIP
            if constexpr (std::is_same_v<ExecSpace, Kokkos::HIP>) {
                sortParticlesHip<Dim, T>(positions, permute, origin, invdx, ngrid, n,
                                         buffer_manager);
                return;
            }
#endif

            // Host fallback (buffer manager not used for host sort)
            if constexpr (std::is_same_v<memory_space, Kokkos::HostSpace>) {
                sortParticlesHost<Dim, T>(positions, permute, origin, invdx, ngrid, n);
            } else {
                // For other device spaces without specialized sort, copy to host
                auto positions_host =
                    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, positions);
                Kokkos::View<size_t*, Kokkos::HostSpace> permute_host("permute_host", n);

                Vector<size_t, Dim> ngrid_host;
                for (unsigned d = 0; d < Dim; ++d) {
                    ngrid_host[d] = ngrid[d];
                }

                sortParticlesHost<Dim, T>(positions_host, permute_host, origin, invdx, ngrid_host,
                                          n);

                Kokkos::deep_copy(permute, permute_host);
            }
        }

        // Backward-compatible overload without buffer manager
        template <unsigned Dim, typename ExecSpace, typename T, typename PermuteViewType>
        void sortParticles(
            Kokkos::View<Vector<T, Dim>*, typename ExecSpace::memory_space> positions,
            PermuteViewType permute, const Vector<T, Dim>& origin, const Vector<T, Dim>& invdx,
            const Vector<size_t, Dim>& ngrid, size_t n) {
            sortParticles<Dim, ExecSpace, T, PermuteViewType>(positions, permute, origin, invdx,
                                                              ngrid, n, nullptr);
        }

        /**
         * @brief Functor to apply permutation to particle data
         */
        template <typename SrcView, typename DstView, typename IndexView>
        struct ApplyPermutationFunctor {
            SrcView src;
            DstView dst;
            IndexView permute;

            KOKKOS_INLINE_FUNCTION void operator()(size_t i) const { dst(i) = src(permute(i)); }
        };

        /**
         * @brief Functor to apply inverse permutation to results
         */
        template <typename SrcView, typename DstView, typename IndexView>
        struct ApplyInversePermutationFunctor {
            SrcView src;
            DstView dst;
            IndexView permute;

            KOKKOS_INLINE_FUNCTION void operator()(size_t i) const { dst(permute(i)) = src(i); }
        };

        /**
         * @brief Apply permutation to reorder data
         *
         * @tparam ExecSpace Kokkos execution space
         * @tparam DataView Data view type
         * @tparam IndexView Index view type
         */
        template <typename ExecSpace, typename DataView, typename IndexView>
        void applyPermutation(DataView& data, const IndexView& permute, size_t n) {
            using value_type   = typename DataView::value_type;
            using memory_space = typename DataView::memory_space;

            DataView temp("temp", n);

            using policy_type = Kokkos::RangePolicy<ExecSpace>;
            Kokkos::parallel_for(
                "apply_permutation", policy_type(0, n),
                ApplyPermutationFunctor<DataView, DataView, IndexView>{data, temp, permute});

            Kokkos::deep_copy(Kokkos::subview(data, Kokkos::make_pair(size_t(0), n)), temp);
        }

        /**
         * @brief Apply inverse permutation to restore original ordering
         */
        template <typename ExecSpace, typename DataView, typename IndexView>
        void applyInversePermutation(const DataView& src, DataView& dst, const IndexView& permute,
                                     size_t n) {
            using policy_type = Kokkos::RangePolicy<ExecSpace>;
            Kokkos::parallel_for(
                "apply_inverse_permutation", policy_type(0, n),
                ApplyInversePermutationFunctor<DataView, DataView, IndexView>{src, dst, permute});
            Kokkos::fence();
        }

    }  // namespace detail
}  // namespace ippl

#endif  // IPPL_PARTICLE_SORT_H