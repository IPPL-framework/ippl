#ifndef IPPL_SORT_BUFFER_H
#define IPPL_SORT_BUFFER_H

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>

namespace ippl {
    namespace detail {

        /**
         * @brief Persistent scratch buffers for bin-sort-based particle binning.
         *
         * Grows on demand, never shrinks. Call finalizeBinSortBuffers() before
         * Kokkos::finalize() to release device memory safely.
         *
         * Buffer layout
         *   bin_keys      uint64_t[n_particles]   sort keys (bin index per particle)
         *   permute       size_t  [n_particles]   identity → sorted particle order
         *   bin_offsets   size_t  [n_bins + 1]    start of each bin in permute
         *   keys_out      uint64_t[n_particles]   CUB radix-sort output (CUDA only)
         *   perm_out      size_t  [n_particles]   CUB radix-sort output (CUDA only)
         *   temp_storage  char    [temp_bytes]    CUB device-temp (CUDA only)
         */
        template <typename MemorySpace>
        class BinSortBuffers {
        public:
            using memory_space = MemorySpace;
            using size_type    = size_t;

            BinSortBuffers()                                 = default;
            BinSortBuffers(const BinSortBuffers&)            = delete;
            BinSortBuffers& operator=(const BinSortBuffers&) = delete;
            BinSortBuffers(BinSortBuffers&&)                 = default;
            BinSortBuffers& operator=(BinSortBuffers&&)      = default;

            /**
             * @brief Ensure particle-sized buffers are large enough.
             *
             * @param n_particles  Number of particles.
             * @param n_bins_p1    Number of bins + 1 (size needed for bin_offsets).
             * @param growth       Over-allocation factor (default 1.2).
             */
            void ensureCapacity(size_type n_particles, size_type n_bins_p1, double growth = 1.2) {
                if (n_particles > particle_capacity_) {
                    const size_type cap =
                        std::max(static_cast<size_type>(n_particles * growth), n_particles);

                    bin_keys_ = Kokkos::View<uint64_t*, memory_space>("bin_keys", cap);
                    permute_  = Kokkos::View<size_type*, memory_space>("permute", cap);
                    keys_out_ = Kokkos::View<uint64_t*, memory_space>("keys_out", cap);
                    perm_out_ = Kokkos::View<size_type*, memory_space>("perm_out", cap);

                    particle_capacity_ = cap;
                }

                if (n_bins_p1 > bin_offset_capacity_) {
                    const size_type cap =
                        std::max(static_cast<size_type>(n_bins_p1 * growth), n_bins_p1);

                    bin_offsets_ = Kokkos::View<size_type*, memory_space>("bin_offsets", cap);
                    bin_offset_capacity_ = cap;
                }
            }

            /**
             * @brief Ensure CUB/rocPRIM temp-storage buffer is large enough.
             * @param bytes  Required bytes.
             */
            void ensureTempStorage(size_t bytes) {
                if (bytes <= temp_capacity_)
                    return;
                const size_t cap = static_cast<size_t>(bytes * 1.1);
                temp_storage_    = Kokkos::View<char*, memory_space>("temp_storage", cap);
                temp_capacity_   = cap;
            }

            // --- accessors -------------------------------------------------------
            Kokkos::View<uint64_t*, memory_space>& binKeys() { return bin_keys_; }
            Kokkos::View<size_type*, memory_space>& permute() { return permute_; }
            Kokkos::View<size_type*, memory_space>& binOffsets() { return bin_offsets_; }
            Kokkos::View<uint64_t*, memory_space>& keysOut() { return keys_out_; }
            Kokkos::View<size_type*, memory_space>& permOut() { return perm_out_; }
            Kokkos::View<char*, memory_space>& tempStorage() { return temp_storage_; }

            size_type particleCapacity() const { return particle_capacity_; }
            size_type binOffsetCapacity() const { return bin_offset_capacity_; }
            size_t tempCapacity() const { return temp_capacity_; }

            size_t memoryUsage() const {
                // Sum the bytes actually held by each view rather than
                // assuming all of them are populated — keys_out_ / perm_out_
                // are only allocated on the CUDA fast path.
                return bin_keys_.span() * sizeof(uint64_t)
                       + permute_.span() * sizeof(size_type)
                       + bin_offsets_.span() * sizeof(size_type)
                       + keys_out_.span() * sizeof(uint64_t)
                       + perm_out_.span() * sizeof(size_type)
                       + temp_storage_.span();
            }

            void clear() {
                bin_keys_            = {};
                permute_             = {};
                bin_offsets_         = {};
                keys_out_            = {};
                perm_out_            = {};
                temp_storage_        = {};
                particle_capacity_   = 0;
                bin_offset_capacity_ = 0;
                temp_capacity_       = 0;
            }

        private:
            Kokkos::View<uint64_t*, memory_space> bin_keys_;
            Kokkos::View<size_type*, memory_space> permute_;
            Kokkos::View<size_type*, memory_space> bin_offsets_;
            Kokkos::View<uint64_t*, memory_space> keys_out_;
            Kokkos::View<size_type*, memory_space> perm_out_;
            Kokkos::View<char*, memory_space> temp_storage_;

            size_type particle_capacity_   = 0;
            size_type bin_offset_capacity_ = 0;
            size_t temp_capacity_          = 0;
        };

        // ---------------------------------------------------------------------------
        // Singleton access
        // ---------------------------------------------------------------------------

        template <typename MemorySpace>
        class BinSortBuffersHolder {
        public:
            static BinSortBuffersHolder& instance() {
                static BinSortBuffersHolder h;
                return h;
            }

            BinSortBuffers<MemorySpace>& get() {
                if (!buffers_)
                    buffers_ = std::make_unique<BinSortBuffers<MemorySpace>>();
                return *buffers_;
            }

            void finalize() {
                if (buffers_) {
                    buffers_->clear();
                    buffers_.reset();
                }
            }

        private:
            BinSortBuffersHolder()                                       = default;
            ~BinSortBuffersHolder()                                      = default;
            BinSortBuffersHolder(const BinSortBuffersHolder&)            = delete;
            BinSortBuffersHolder& operator=(const BinSortBuffersHolder&) = delete;

            std::unique_ptr<BinSortBuffers<MemorySpace>> buffers_;
        };

        template <typename MemorySpace>
        BinSortBuffers<MemorySpace>& getDefaultBinSortBuffers() {
            return BinSortBuffersHolder<MemorySpace>::instance().get();
        }

        /**
         * @brief Release all bin-sort buffers for known memory spaces.
         *        Call this before Kokkos::finalize().
         */
        inline void finalizeBinSortBuffers() {
#ifdef KOKKOS_ENABLE_CUDA
            BinSortBuffersHolder<Kokkos::CudaSpace>::instance().finalize();
#endif
#ifdef KOKKOS_ENABLE_HIP
            BinSortBuffersHolder<Kokkos::HIPSpace>::instance().finalize();
#endif
            BinSortBuffersHolder<Kokkos::HostSpace>::instance().finalize();
        }

    }  // namespace detail
}  // namespace ippl

#endif  // IPPL_SORT_BUFFER_H