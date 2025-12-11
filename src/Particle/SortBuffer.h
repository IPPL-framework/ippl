//
// Class SortBufferManager
//   Manages temporary memory buffers for particle sorting operations.
//   Allocates once and reuses across multiple sort calls to avoid
//   repeated allocation overhead on GPU.
//
#ifndef IPPL_SORT_BUFFER_MANAGER_H
#define IPPL_SORT_BUFFER_MANAGER_H

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>

namespace ippl {
    namespace detail {

        /**
         * @brief Buffer manager for particle sorting operations
         *
         * This class manages temporary memory allocations needed for sorting
         * particles by Morton codes. It grows buffers as needed but never
         * shrinks them, avoiding repeated allocations for repeated sorts
         * of similar-sized particle sets.
         *
         * @tparam MemorySpace Kokkos memory space (e.g., CudaSpace, HIPSpace, HostSpace)
         */
        template <typename MemorySpace>
        class SortBufferManager {
        public:
            using memory_space = MemorySpace;
            using size_type    = size_t;

            SortBufferManager() = default;

            // Non-copyable, movable
            SortBufferManager(const SortBufferManager&)            = delete;
            SortBufferManager& operator=(const SortBufferManager&) = delete;
            SortBufferManager(SortBufferManager&&)                 = default;
            SortBufferManager& operator=(SortBufferManager&&)      = default;

            /**
             * @brief Ensure buffers are large enough for n particles
             *
             * Allocates or reallocates buffers if the current capacity is
             * insufficient. Uses a growth factor to reduce future reallocations.
             *
             * @param n Number of particles to accommodate
             * @param growth_factor Factor by which to over-allocate (default 1.2)
             */
            void ensureCapacity(size_type n, double growth_factor = 1.2) {
                if (n <= capacity_) {
                    return;
                }

                // Apply growth factor to reduce future reallocations
                size_type new_capacity = static_cast<size_type>(n * growth_factor);
                new_capacity           = std::max(new_capacity, n);  // Safety check

                // Reallocate all buffers
                morton_keys_ =
                    Kokkos::View<uint64_t*, memory_space>("sort_buffer_morton_keys", new_capacity);
                morton_keys_sorted_ = Kokkos::View<uint64_t*, memory_space>(
                    "sort_buffer_morton_keys_sorted", new_capacity);
                indices_ =
                    Kokkos::View<size_type*, memory_space>("sort_buffer_indices", new_capacity);
                indices_sorted_ = Kokkos::View<size_type*, memory_space>(
                    "sort_buffer_indices_sorted", new_capacity);

                capacity_ = new_capacity;
            }

            /**
             * @brief Ensure temporary storage buffer is large enough
             *
             * Separate from particle buffers as CUB/rocPRIM may need
             * varying amounts of temp storage.
             *
             * @param bytes Number of bytes needed
             */
            void ensureTempStorageCapacity(size_t bytes) {
                if (bytes <= temp_storage_capacity_) {
                    return;
                }

                // Add some headroom for temp storage too
                size_t new_capacity = static_cast<size_t>(bytes * 1.1);
                temp_storage_ =
                    Kokkos::View<char*, memory_space>("sort_buffer_temp_storage", new_capacity);
                temp_storage_capacity_ = new_capacity;
            }

            // Accessors for the buffers (return subviews of actual size when needed)
            Kokkos::View<uint64_t*, memory_space>& mortonKeys() { return morton_keys_; }
            Kokkos::View<uint64_t*, memory_space>& mortonKeysSorted() {
                return morton_keys_sorted_;
            }
            Kokkos::View<size_type*, memory_space>& indices() { return indices_; }
            Kokkos::View<size_type*, memory_space>& indicesSorted() { return indices_sorted_; }
            Kokkos::View<char*, memory_space>& tempStorage() { return temp_storage_; }

            // Const accessors
            const Kokkos::View<uint64_t*, memory_space>& mortonKeys() const { return morton_keys_; }
            const Kokkos::View<uint64_t*, memory_space>& mortonKeysSorted() const {
                return morton_keys_sorted_;
            }
            const Kokkos::View<size_type*, memory_space>& indices() const { return indices_; }
            const Kokkos::View<size_type*, memory_space>& indicesSorted() const {
                return indices_sorted_;
            }
            const Kokkos::View<char*, memory_space>& tempStorage() const { return temp_storage_; }

            size_type capacity() const { return capacity_; }
            size_t tempStorageCapacity() const { return temp_storage_capacity_; }

            /**
             * @brief Release all allocated memory
             */
            void clear() {
                morton_keys_           = Kokkos::View<uint64_t*, memory_space>();
                morton_keys_sorted_    = Kokkos::View<uint64_t*, memory_space>();
                indices_               = Kokkos::View<size_type*, memory_space>();
                indices_sorted_        = Kokkos::View<size_type*, memory_space>();
                temp_storage_          = Kokkos::View<char*, memory_space>();
                capacity_              = 0;
                temp_storage_capacity_ = 0;
            }

            /**
             * @brief Get approximate memory usage in bytes
             */
            size_t memoryUsage() const {
                return capacity_ * (2 * sizeof(uint64_t) + 2 * sizeof(size_type))
                       + temp_storage_capacity_;
            }

        private:
            Kokkos::View<uint64_t*, memory_space> morton_keys_;
            Kokkos::View<uint64_t*, memory_space> morton_keys_sorted_;
            Kokkos::View<size_type*, memory_space> indices_;
            Kokkos::View<size_type*, memory_space> indices_sorted_;
            Kokkos::View<char*, memory_space> temp_storage_;

            size_type capacity_           = 0;
            size_t temp_storage_capacity_ = 0;
        };

        /**
         * @brief Extended buffer manager that also handles permutation and bin offset arrays
         *
         * This is useful when you need persistent storage for permute arrays
         * and bin offsets across gather/scatter operations.
         *
         * @tparam MemorySpace Kokkos memory space
         */
        template <typename MemorySpace>
        class ParticleSortBufferManager : public SortBufferManager<MemorySpace> {
        public:
            using Base         = SortBufferManager<MemorySpace>;
            using memory_space = MemorySpace;
            using size_type    = size_t;

            ParticleSortBufferManager() = default;

            /**
             * @brief Ensure all buffers including permute and bin_offsets are sized
             *
             * @param n_particles Number of particles
             * @param n_bins Number of bins (for bin_offsets, typically n_bins + 1)
             * @param growth_factor Over-allocation factor
             */
            void ensureCapacity(size_type n_particles, size_type n_bins = 0,
                                double growth_factor = 1.2) {
                Base::ensureCapacity(n_particles, growth_factor);

                if (n_particles > permute_capacity_) {
                    size_type new_capacity = static_cast<size_type>(n_particles * growth_factor);
                    permute_ =
                        Kokkos::View<size_type*, memory_space>("sort_buffer_permute", new_capacity);
                    permute_capacity_ = new_capacity;
                }

                if (n_bins > 0 && n_bins > bin_offsets_capacity_) {
                    size_type new_capacity = static_cast<size_type>(n_bins * growth_factor);
                    bin_offsets_ = Kokkos::View<size_type*, memory_space>("sort_buffer_bin_offsets",
                                                                          new_capacity);
                    bin_offsets_capacity_ = new_capacity;
                }
            }

            Kokkos::View<size_type*, memory_space>& permute() { return permute_; }
            Kokkos::View<size_type*, memory_space>& binOffsets() { return bin_offsets_; }

            const Kokkos::View<size_type*, memory_space>& permute() const { return permute_; }
            const Kokkos::View<size_type*, memory_space>& binOffsets() const {
                return bin_offsets_;
            }

            size_type permuteCapacity() const { return permute_capacity_; }
            size_type binOffsetsCapacity() const { return bin_offsets_capacity_; }

            void clear() {
                Base::clear();
                permute_              = Kokkos::View<size_type*, memory_space>();
                bin_offsets_          = Kokkos::View<size_type*, memory_space>();
                permute_capacity_     = 0;
                bin_offsets_capacity_ = 0;
            }

            size_t memoryUsage() const {
                return Base::memoryUsage() + permute_capacity_ * sizeof(size_type)
                       + bin_offsets_capacity_ * sizeof(size_type);
            }

        private:
            Kokkos::View<size_type*, memory_space> permute_;
            Kokkos::View<size_type*, memory_space> bin_offsets_;
            size_type permute_capacity_     = 0;
            size_type bin_offsets_capacity_ = 0;
        };

        /**
         * @brief Thread-local or per-instance buffer manager accessor
         *
         * Provides a convenient way to get a buffer manager that persists
         * across calls. Can be used as a static local or as a member variable.
         */
        template <typename MemorySpace>
        ParticleSortBufferManager<MemorySpace>& getDefaultSortBufferManager() {
            static ParticleSortBufferManager<MemorySpace> manager;
            return manager;
        }

    }  // namespace detail
}  // namespace ippl

#endif  // IPPL_SORT_BUFFER_MANAGER_H