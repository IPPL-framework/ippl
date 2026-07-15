#ifndef IPPL_PARTICLE_SORT_H
#define IPPL_PARTICLE_SORT_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
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
         * @brief Validate that all permutation indices are in bounds
         */
        template <typename ExecSpace, typename IndexView>
        void validatePermutation(const IndexView& permute, size_t n,
                                 const char* label = "sortParticles") {
            using size_type = size_t;

            size_type num_invalid = 0;
            Kokkos::parallel_reduce(
                label, Kokkos::RangePolicy<ExecSpace>(0, n),
                KOKKOS_LAMBDA(size_type i, size_type & count) {
                    if (permute(i) >= n)
                        count++;
                },
                num_invalid);
            Kokkos::fence();

            if (num_invalid > 0) {
                size_type first_invalid_idx = n;
                Kokkos::parallel_reduce(
                    "find_first_invalid", Kokkos::RangePolicy<ExecSpace>(0, n),
                    KOKKOS_LAMBDA(size_type i, size_type & first_idx) {
                        if (permute(i) >= n && i < first_idx)
                            first_idx = i;
                    },
                    Kokkos::Min<size_type>(first_invalid_idx));
                Kokkos::fence();

                auto permute_host =
                    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, permute);

                throw std::runtime_error(std::string(label) + ": Found "
                                         + std::to_string(num_invalid)
                                         + " invalid permutation indices. First invalid: permute["
                                         + std::to_string(first_invalid_idx)
                                         + "] = " + std::to_string(permute_host(first_invalid_idx))
                                         + " (n = " + std::to_string(n) + ")");
            }
        }

        /**
         * @brief Compute Morton code (Z-order curve) for spatial sorting
         */
        template <unsigned Dim, typename T, typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION uint64_t computeMortonCode(const Vector<T, Dim>& position,
                                                          const Vector<T, Dim>& origin,
                                                          const Vector<T, Dim>& invdx,
                                                          const Vector<IndexType, Dim>& ngrid) {
            uint64_t morton            = 0;
            constexpr int bits_per_dim = 21;  // 21*3 = 63 bits, safe for 3D

            uint32_t gridIndices[Dim];
            for (unsigned d = 0; d < Dim; ++d) {
                T sx = (position[d] - origin[d]) * invdx[d];
                sx -= ngrid[d] * Kokkos::floor(sx / ngrid[d]);  // wrap to [0, ngrid)
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
            using size_type = size_t;

            PositionView positions;
            KeyView keys;
            Vector<T, Dim> origin;
            Vector<T, Dim> invdx;
            Vector<size_type, Dim> ngrid;

            KOKKOS_INLINE_FUNCTION void operator()(size_type i) const {
                keys(i) = computeMortonCode<Dim, T, size_type>(positions(i), origin, invdx, ngrid);
            }
        };

        // -------------------------------------------------------------------
        // Platform-specific sort implementations
        // -------------------------------------------------------------------

        /**
         * @brief Sort particles on host using std::sort on (morton, index) pairs
         *
         * @return Subview of the buffered permute array, valid until next ensureCapacity
         */
        template <unsigned Dim, typename T>
        Kokkos::View<size_t*, Kokkos::HostSpace> sortParticlesHost(
            Kokkos::View<Vector<T, Dim>*, Kokkos::HostSpace> positions,
            const Vector<T, Dim>& origin, const Vector<T, Dim>& invdx,
            const Vector<size_t, Dim>& ngrid, size_t n) {
            auto& bufs = ippl::detail::getDefaultBinSortBuffers<Kokkos::HostSpace>();
            bufs.ensureCapacity(n, /*n_bins_p1=*/1);

            auto& permute = bufs.permute();

            // Build (key, index) pairs, sort, then extract permutation
            std::vector<std::pair<uint64_t, size_t>> pairs(n);
            for (size_t i = 0; i < n; ++i) {
                pairs[i] = {computeMortonCode<Dim, T, size_t>(positions(i), origin, invdx, ngrid),
                            i};
            }
            std::sort(pairs.begin(), pairs.end());  // pair has lexicographic < by .first

            for (size_t i = 0; i < n; ++i) {
                permute(i) = pairs[i].second;
            }

            return permute;
        }

#ifdef KOKKOS_ENABLE_CUDA
        /**
         * @brief Sort particles on CUDA using CUB DeviceRadixSort with buffered temporaries
         *
         * @return Subview of the buffered permOut array containing sorted particle indices
         */
        template <unsigned Dim, typename T>
        Kokkos::View<size_t*, Kokkos::CudaSpace> sortParticlesCuda(
            Kokkos::View<Vector<T, Dim>*, Kokkos::CudaSpace> positions,
            const Vector<T, Dim>& origin, const Vector<T, Dim>& invdx,
            const Vector<size_t, Dim>& ngrid, size_t n) {
            using size_type = size_t;
            auto& bufs      = ippl::detail::getDefaultBinSortBuffers<Kokkos::CudaSpace>();
            bufs.ensureCapacity(n, /*n_bins_p1=*/1);

            // Step 1: compute Morton codes into binKeys, init permute to identity
            auto keys    = bufs.binKeys();
            auto permute = bufs.permute();

            Kokkos::parallel_for(
                "MortonSort::ComputeKeys", Kokkos::RangePolicy<Kokkos::Cuda>(0, n),
                ComputeMortonCodesFunctor<Dim, decltype(positions), T, decltype(keys)>{
                    positions, keys, origin, invdx, ngrid});

            Kokkos::parallel_for(
                "MortonSort::InitPermute", Kokkos::RangePolicy<Kokkos::Cuda>(0, n),
                KOKKOS_LAMBDA(size_type i) { permute(i) = i; });
            Kokkos::fence();

            // Step 2: query temp storage, grow buffer if needed, then sort
            auto keys_out = bufs.keysOut();
            auto perm_out = bufs.permOut();

            size_t temp_bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys.data(), keys_out.data(),
                                            permute.data(), perm_out.data(), static_cast<int>(n));

            bufs.ensureTempStorage(temp_bytes);

            cub::DeviceRadixSort::SortPairs(bufs.tempStorage().data(), temp_bytes, keys.data(),
                                            keys_out.data(), permute.data(), perm_out.data(),
                                            static_cast<int>(n));
            Kokkos::fence();

            return bufs.permOut();
        }
#endif

#ifdef KOKKOS_ENABLE_HIP
        /**
         * @brief Sort particles on HIP using rocPRIM radix_sort_pairs with buffered temporaries
         *
         * @return Subview of the buffered permOut array containing sorted particle indices
         */
        template <unsigned Dim, typename T>
        Kokkos::View<size_t*, Kokkos::HIPSpace> sortParticlesHip(
            Kokkos::View<Vector<T, Dim>*, Kokkos::HIPSpace> positions, const Vector<T, Dim>& origin,
            const Vector<T, Dim>& invdx, const Vector<size_t, Dim>& ngrid, size_t n) {
            using size_type = size_t;
            auto& bufs      = ippl::detail::getDefaultBinSortBuffers<Kokkos::HIPSpace>();
            bufs.ensureCapacity(n, /*n_bins_p1=*/1);

            auto keys    = bufs.binKeys();
            auto permute = bufs.permute();

            Kokkos::parallel_for(
                "MortonSort::ComputeKeys", Kokkos::RangePolicy<Kokkos::HIP>(0, n),
                ComputeMortonCodesFunctor<Dim, decltype(positions), T, decltype(keys)>{
                    positions, keys, origin, invdx, ngrid});

            Kokkos::parallel_for(
                "MortonSort::InitPermute", Kokkos::RangePolicy<Kokkos::HIP>(0, n),
                KOKKOS_LAMBDA(size_type i) { permute(i) = i; });
            Kokkos::fence();

            auto keys_out = bufs.keysOut();
            auto perm_out = bufs.permOut();

            size_t temp_bytes = 0;
            std::ignore = rocprim::radix_sort_pairs(nullptr, temp_bytes, keys.data(), keys_out.data(),
                                      permute.data(), perm_out.data(), n, 0, sizeof(uint64_t) * 8,
                                      Kokkos::HIP().hip_stream());

            bufs.ensureTempStorage(temp_bytes);

            std::ignore = rocprim::radix_sort_pairs(bufs.tempStorage().data(), temp_bytes, keys.data(),
                                      keys_out.data(), permute.data(), perm_out.data(), n, 0,
                                      sizeof(uint64_t) * 8, Kokkos::HIP().hip_stream());
            Kokkos::fence();

            return bufs.permOut();
        }
#endif

        /**
         * @brief Generic sort dispatcher — selects CUDA/HIP/host implementation
         */
        template <unsigned Dim, typename ExecSpace, typename T>
        Kokkos::View<size_t*, typename ExecSpace::memory_space> sortParticles(
            Kokkos::View<Vector<T, Dim>*, typename ExecSpace::memory_space> positions,
            const Vector<T, Dim>& origin, const Vector<T, Dim>& invdx,
            const Vector<size_t, Dim>& ngrid, size_t n) {
            using memory_space = typename ExecSpace::memory_space;

#ifdef KOKKOS_ENABLE_CUDA
            if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
                return sortParticlesCuda<Dim, T>(positions, origin, invdx, ngrid, n);
            }
#endif
#ifdef KOKKOS_ENABLE_HIP
            if constexpr (std::is_same_v<ExecSpace, Kokkos::HIP>) {
                return sortParticlesHip<Dim, T>(positions, origin, invdx, ngrid, n);
            }
#endif
            if constexpr (std::is_same_v<memory_space, Kokkos::HostSpace>) {
                return sortParticlesHost<Dim, T>(positions, origin, invdx, ngrid, n);
            } else {
                // Generic fallback: sort on host, copy result to device buffer
                auto positions_host =
                    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, positions);

                Vector<size_t, Dim> ngrid_host;
                for (unsigned d = 0; d < Dim; ++d)
                    ngrid_host[d] = ngrid[d];

                auto permute_host =
                    sortParticlesHost<Dim, T>(positions_host, origin, invdx, ngrid_host, n);

                auto& bufs = ippl::detail::getDefaultBinSortBuffers<memory_space>();
                bufs.ensureCapacity(n, /*n_bins_p1=*/1);

                Kokkos::deep_copy(Kokkos::subview(bufs.permute(), std::make_pair(size_t(0), n)),
                                  Kokkos::subview(permute_host, std::make_pair(size_t(0), n)));

                return bufs.permute();
            }
        }

        // -------------------------------------------------------------------
        // Permutation application helpers
        // -------------------------------------------------------------------

        template <typename SrcView, typename DstView, typename IndexView>
        struct ApplyPermutationFunctor {
            SrcView src;
            DstView dst;
            IndexView permute;
            KOKKOS_INLINE_FUNCTION void operator()(size_t i) const { dst(i) = src(permute(i)); }
        };

        template <typename SrcView, typename DstView, typename IndexView>
        struct ApplyInversePermutationFunctor {
            SrcView src;
            DstView dst;
            IndexView permute;
            KOKKOS_INLINE_FUNCTION void operator()(size_t i) const { dst(permute(i)) = src(i); }
        };

        /**
         * @brief Reorder data in-place according to permutation
         */
        template <typename ExecSpace, typename DataView, typename IndexView>
        void applyPermutation(DataView& data, const IndexView& permute, size_t n) {
            DataView temp("temp", n);
            Kokkos::parallel_for(
                "apply_permutation", Kokkos::RangePolicy<ExecSpace>(0, n),
                ApplyPermutationFunctor<DataView, DataView, IndexView>{data, temp, permute});
            Kokkos::deep_copy(Kokkos::subview(data, Kokkos::make_pair(size_t(0), n)), temp);
        }

        /**
         * @brief Scatter src → dst using inverse permutation (dst[permute[i]] = src[i])
         */
        template <typename ExecSpace, typename DataView, typename IndexView>
        void applyInversePermutation(const DataView& src, DataView& dst, const IndexView& permute,
                                     size_t n) {
            Kokkos::parallel_for(
                "apply_inverse_permutation", Kokkos::RangePolicy<ExecSpace>(0, n),
                ApplyInversePermutationFunctor<DataView, DataView, IndexView>{src, dst, permute});
            Kokkos::fence();
        }

    }  // namespace detail
}  // namespace ippl

#endif  // IPPL_PARTICLE_SORT_H