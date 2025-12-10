//
// Class ParticleSort
//   Utilities for sorting particles based on spatial locality using Morton codes.
//   Uses CUB for CUDA, rocPRIM for HIP, and Kokkos::BinSort as fallback.
//
#ifndef IPPL_PARTICLE_SORT_H
#define IPPL_PARTICLE_SORT_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>

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
    KOKKOS_INLINE_FUNCTION
    uint64_t computeMortonCode(const Vector<T, Dim>& position,
                                const Vector<T, Dim>& origin,
                                const Vector<T, Dim>& invdx,
                                const Vector<IndexType, Dim>& ngrid) {
        uint64_t morton = 0;
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
    template <unsigned Dim, typename PositionView, typename T>
    struct ComputeMortonCodesFunctor {
        using memory_space = typename PositionView::memory_space;
        using size_type = typename memory_space::size_type;

        PositionView positions;
        Kokkos::View<uint64_t*, memory_space> keys;
        Vector<T, Dim> origin;
        Vector<T, Dim> invdx;
        Vector<size_type, Dim> ngrid;

        KOKKOS_INLINE_FUNCTION
        void operator()(size_type i) const {
            keys(i) = computeMortonCode<Dim, T, size_type>(
                positions(i), origin, invdx, ngrid);
        }
    };

    /**
     * @brief Sort particles on host using std::sort
     */
    template <unsigned Dim, typename T>
    void sortParticlesHost(
        Kokkos::View<Vector<T, Dim>*, Kokkos::HostSpace> positions,
        Kokkos::View<size_t*, Kokkos::HostSpace> permute,
        const Vector<T, Dim>& origin,
        const Vector<T, Dim>& invdx,
        const Vector<size_t, Dim>& ngrid,
        size_t n) {

        // Compute Morton codes
        std::vector<std::pair<uint64_t, size_t>> key_index_pairs(n);

        for (size_t i = 0; i < n; ++i) {
            key_index_pairs[i] = {
                computeMortonCode<Dim, T, size_t>(positions(i), origin, invdx, ngrid),
                i
            };
        }

        // Sort by Morton code
        std::sort(key_index_pairs.begin(), key_index_pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Extract permutation
        for (size_t i = 0; i < n; ++i) {
            permute(i) = key_index_pairs[i].second;
        }
    }

#ifdef KOKKOS_ENABLE_CUDA
    /**
     * @brief Sort particles on CUDA using CUB RadixSort
     */
    template <unsigned Dim, typename T, typename PermuteViewType>
    void sortParticlesCuda(
        Kokkos::View<Vector<T, Dim>*, Kokkos::CudaSpace> positions,
        PermuteViewType permute,
        const Vector<T, Dim>& origin,
        const Vector<T, Dim>& invdx,
        const Vector<size_t, Dim>& ngrid,
        size_t n) {

        using memory_space = Kokkos::CudaSpace;
        using size_type = size_t;

        // Allocate temporary arrays
        //Kokkos::View<uint64_t*, memory_space> keys("morton_keys", n);
        //Kokkos::View<uint64_t*, memory_space> keys_sorted("morton_keys_sorted", n);
        //Kokkos::View<size_type*, memory_space> indices("indices", n);
        //Kokkos::View<size_type*, memory_space> indices_sorted("indices_sorted", n);

        auto size = computeBufferSize<uint64_t, uint64_t, size_type, size_type>(n, n, n, n);
        MultiViewBuffer<memory_space> sortBuf(size);

        auto keys = sortBuf.template getView<uint64_t>(n);
        auto keys_sorted = sortBuf.template getView<uint64_t>(n);
        auto indices = sortBuf.template getView<size_type>(n);
        auto indices_sorted = sortBuf.template getView<size_type>(n);


        // Compute Morton codes
        Kokkos::parallel_for("compute_morton_codes",
            Kokkos::RangePolicy<Kokkos::Cuda>(0, n),
            ComputeMortonCodesFunctor<Dim, decltype(positions), T>{
                positions, keys, origin, invdx, ngrid
            });

        // Initialize indices
        Kokkos::parallel_for("init_indices",
            Kokkos::RangePolicy<Kokkos::Cuda>(0, n),
            KOKKOS_LAMBDA(size_type i) { indices(i) = i; });

        Kokkos::fence();

        // Determine temporary storage requirements
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            keys.data(), keys_sorted.data(),
            indices.data(), indices_sorted.data(),
            n);

        // Allocate temporary storage
        Kokkos::View<char*, memory_space> d_temp_storage("temp_storage", temp_storage_bytes);

        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage.data(), temp_storage_bytes,
            keys.data(), keys_sorted.data(),
            indices.data(), indices_sorted.data(),
            n);

        Kokkos::fence();

        // Copy sorted indices to permute array
        Kokkos::deep_copy(permute, indices_sorted);
    }
#endif

#ifdef KOKKOS_ENABLE_HIP
    /**
     * @brief Sort particles on HIP using rocPRIM RadixSort
     */
    template <unsigned Dim, typename T>
    void sortParticlesHip(
        Kokkos::View<Vector<T, Dim>*, Kokkos::HIPSpace> positions,
        Kokkos::View<size_t*, Kokkos::HIPSpace> permute,
        const Vector<T, Dim>& origin,
        const Vector<T, Dim>& invdx,
        const Vector<size_t, Dim>& ngrid,
        size_t n) {

        using memory_space = Kokkos::HIPSpace;
        using size_type = size_t;

        // Allocate temporary arrays
        Kokkos::View<uint64_t*, memory_space> keys("morton_keys", n);
        Kokkos::View<uint64_t*, memory_space> keys_sorted("morton_keys_sorted", n);
        Kokkos::View<size_type*, memory_space> indices("indices", n);
        Kokkos::View<size_type*, memory_space> indices_sorted("indices_sorted", n);

        // Compute Morton codes
        Kokkos::parallel_for("compute_morton_codes",
            Kokkos::RangePolicy<Kokkos::HIP>(0, n),
            ComputeMortonCodesFunctor<Dim, decltype(positions), T>{
                positions, keys, origin, invdx, ngrid
            });

        // Initialize indices
        Kokkos::parallel_for("init_indices",
            Kokkos::RangePolicy<Kokkos::HIP>(0, n),
            KOKKOS_LAMBDA(size_type i) { indices(i) = i; });

        Kokkos::fence();

        // Determine temporary storage requirements
        size_t temp_storage_bytes = 0;
        rocprim::radix_sort_pairs(
            nullptr, temp_storage_bytes,
            keys.data(), keys_sorted.data(),
            indices.data(), indices_sorted.data(),
            n,
            0, sizeof(uint64_t) * 8,
            Kokkos::HIP().hip_stream());

        // Allocate temporary storage
        Kokkos::View<char*, memory_space> d_temp_storage("temp_storage", temp_storage_bytes);

        // Run sorting operation
        rocprim::radix_sort_pairs(
            d_temp_storage.data(), temp_storage_bytes,
            keys.data(), keys_sorted.data(),
            indices.data(), indices_sorted.data(),
            n,
            0, sizeof(uint64_t) * 8,
            Kokkos::HIP().hip_stream());

        Kokkos::fence();

        // Copy sorted indices to permute array
        Kokkos::deep_copy(permute, indices_sorted);
    }
#endif

    /**
     * @brief Generic sort dispatcher based on execution space
     *
     * @tparam Dim Number of dimensions
     * @tparam ExecSpace Kokkos execution space
     * @tparam T Floating point type
     */
    template <unsigned Dim, typename ExecSpace, typename T, typename PermuteViewType>
    void sortParticles(
        Kokkos::View<Vector<T, Dim>*, typename ExecSpace::memory_space> positions,
        PermuteViewType permute,
        const Vector<T, Dim>& origin,
        const Vector<T, Dim>& invdx,
        const Vector<size_t, Dim>& ngrid,
        size_t n) {

        using memory_space = typename ExecSpace::memory_space;

#ifdef KOKKOS_ENABLE_CUDA
        if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
            sortParticlesCuda<Dim, T>(positions, permute, origin, invdx, ngrid, n);
            return;
        }
#endif

#ifdef KOKKOS_ENABLE_HIP
        if constexpr (std::is_same_v<ExecSpace, Kokkos::HIP>) {
            sortParticlesHip<Dim, T>(positions, permute, origin, invdx, ngrid, n);
            return;
        }
#endif

        // Host fallback
        if constexpr (std::is_same_v<memory_space, Kokkos::HostSpace>) {
            sortParticlesHost<Dim, T>(positions, permute, origin, invdx, ngrid, n);
        } else {
            // For other device spaces without specialized sort, copy to host
            auto positions_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, positions);
            Kokkos::View<size_t*, Kokkos::HostSpace> permute_host("permute_host", n);

            Vector<size_t, Dim> ngrid_host;
            for (unsigned d = 0; d < Dim; ++d) {
                ngrid_host[d] = ngrid[d];
            }

            sortParticlesHost<Dim, T>(positions_host, permute_host, origin, invdx, ngrid_host, n);

            Kokkos::deep_copy(permute, permute_host);
        }
    }

    /**
     * @brief Functor to apply permutation to particle data
     */
    template <typename SrcView, typename DstView, typename IndexView>
    struct ApplyPermutationFunctor {
        SrcView src;
        DstView dst;
        IndexView permute;

        KOKKOS_INLINE_FUNCTION
        void operator()(size_t i) const {
            dst(i) = src(permute(i));
        }
    };

    /**
     * @brief Functor to apply inverse permutation to results
     */
    template <typename SrcView, typename DstView, typename IndexView>
    struct ApplyInversePermutationFunctor {
        SrcView src;
        DstView dst;
        IndexView permute;

        KOKKOS_INLINE_FUNCTION
        void operator()(size_t i) const {
            dst(permute(i)) = src(i);
        }
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
        using value_type = typename DataView::value_type;
        using memory_space = typename DataView::memory_space;

        DataView temp("temp", n);

        using policy_type = Kokkos::RangePolicy<ExecSpace>;
        Kokkos::parallel_for("apply_permutation", policy_type(0, n),
            ApplyPermutationFunctor<DataView, DataView, IndexView>{data, temp, permute});

        Kokkos::deep_copy(Kokkos::subview(data, Kokkos::make_pair(size_t(0), n)), temp);
    }

    /**
     * @brief Apply inverse permutation to restore original ordering
     */
    template <typename ExecSpace, typename DataView, typename IndexView>
    void applyInversePermutation(const DataView& src, DataView& dst, const IndexView& permute, size_t n) {
        using policy_type = Kokkos::RangePolicy<ExecSpace>;
        Kokkos::parallel_for("apply_inverse_permutation", policy_type(0, n),
            ApplyInversePermutationFunctor<DataView, DataView, IndexView>{src, dst, permute});
        Kokkos::fence();
    }

}  // namespace detail
}  // namespace ippl

#endif  // IPPL_PARTICLE_SORT_H
