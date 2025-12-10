#ifndef IPPL_BUFFERVIEW_H
#define IPPL_BUFFERVIEW_H

#include "Ippl.h"

#include <memory>
#include <mpi.h>

#include "Utility/TypeUtils.h"

#include "Communicate/Archive.h"
#include "Communicate/BufferHandler.h"
#include "Communicate/LoggingBufferHandler.h"

namespace ippl {
    /**
     * RAII wrapper that automatically frees the buffer back to the communicator
     */
    template <typename T, typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
    class BufferView {
    public:
        using buffer_type = typename mpi::Communicator::buffer_type<MemorySpace>;
        using view_type   = Kokkos::View<T*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        BufferView(size_t count, double overallocation = 1.5)
            : count_m(count) {
            size_t byteSize = count * sizeof(T);
            buffer_m        = Comm->getBuffer<MemorySpace, T>(byteSize, overallocation);
            view_m          = view_type(reinterpret_cast<T*>(buffer_m->getBuffer()), count);
        }

        ~BufferView() { Comm->freeBuffer<MemorySpace>(buffer_m); }

        // Non-copyable
        BufferView(const BufferView&)            = delete;
        BufferView& operator=(const BufferView&) = delete;

        // Movable
        BufferView(BufferView&& other) noexcept
            : buffer_m(std::move(other.buffer_m))
            , view_m(other.view_m)
            , count_m(other.count_m) {
            other.buffer_m = nullptr;
        }

        view_type& getView() { return view_m; }
        const view_type& getView() const { return view_m; }

        buffer_type& getBuffer() { return buffer_m; }
        size_t getCount() const { return count_m; }

        // Convenient access
        T* data() { return view_m.data(); }
        size_t size() const { return count_m; }

    private:
        buffer_type buffer_m;
        view_type view_m;
        size_t count_m;
    };

    /**
     * @brief Helper to compute aligned offset
     */
    inline constexpr size_t alignUp(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    /**
     * @brief Allocates a single buffer and provides multiple aligned views into it.
     *
     * This avoids multiple buffer allocations when you need several temporary arrays
     * simultaneously. Each view is properly aligned for its element type.
     *
     * @tparam MemorySpace The Kokkos memory space for the buffer
     */
    template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
    class MultiViewBuffer {
    public:
        using buffer_type = typename mpi::Communicator::buffer_type<MemorySpace>;

        /**
         * @brief Construct without allocation. Call allocate() before getting views.
         */
        explicit MultiViewBuffer()
            : buffer_m(nullptr)
            , totalSize_m(0)
            , currentOffset_m(0) {}

        MultiViewBuffer(size_t totalBytes, double overallocation = 1.5)
            : currentOffset_m(0) {
            allocate(totalBytes, overallocation);
        }

        ~MultiViewBuffer() {
            if (buffer_m) {
                Comm->freeBuffer<MemorySpace>(buffer_m);
            }
        }

        // Non-copyable
        MultiViewBuffer(const MultiViewBuffer&)            = delete;
        MultiViewBuffer& operator=(const MultiViewBuffer&) = delete;

        // Movable
        MultiViewBuffer(MultiViewBuffer&& other) noexcept
            : buffer_m(std::move(other.buffer_m))
            , totalSize_m(other.totalSize_m)
            , currentOffset_m(other.currentOffset_m) {
            other.buffer_m        = nullptr;
            other.totalSize_m     = 0;
            other.currentOffset_m = 0;
        }

        void allocate(size_t totalBytes, double overallocation = 1.0) {
            totalSize_m     = totalBytes;
            buffer_m        = Comm->getBuffer<MemorySpace, char>(totalBytes, overallocation);
            currentOffset_m = 0;
        }

        template <typename T>
        Kokkos::View<T*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> getView(
            size_t count) {
            using view_type =
                Kokkos::View<T*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

            constexpr size_t alignment = 256;
            size_t alignedOffset       = alignUp(currentOffset_m, alignment);
            size_t requiredSize        = alignedOffset + count * sizeof(T);

            if (requiredSize > buffer_m->getBufferSize()) {
                throw std::runtime_error(
                    "MultiViewBuffer: insufficient space. Required: " + std::to_string(requiredSize)
                    + ", Available: " + std::to_string(buffer_m->getBufferSize()));
            }

            char* basePtr = reinterpret_cast<char*>(buffer_m->getBuffer());
            T* viewPtr    = reinterpret_cast<T*>(basePtr + alignedOffset);

            currentOffset_m = alignedOffset + count * sizeof(T);

            return view_type(viewPtr, count);
        }

        void reset() { currentOffset_m = 0; }

        /**
         * @brief Get the current used size in bytes.
         */
        size_t getUsedSize() const { return currentOffset_m; }

        /**
         * @brief Get the total available size in bytes.
         */
        size_t getTotalSize() const { return buffer_m ? buffer_m->getBufferSize() : 0; }

        /**
         * @brief Get remaining available bytes.
         */
        size_t getRemainingSize() const { return getTotalSize() - currentOffset_m; }

        /**
         * @brief Access the underlying buffer (e.g., for MPI operations).
         */
        buffer_type& getBuffer() { return buffer_m; }

    private:
        buffer_type buffer_m;
        size_t totalSize_m;
        size_t currentOffset_m;
    };

    /**
     * @brief Compute total buffer size needed for multiple views with alignment.
     * e.g. auto size = computeBufferSize<double, int, float>(100, 200, 50);
     */
    /**
     * @brief Helper struct for recursive buffer size computation
     */
    template <typename... Ts>
    struct BufferSizeComputer;

    // Base case: single type
    template <typename T>
    struct BufferSizeComputer<T> {
        static constexpr size_t compute(size_t offset, size_t count) {
            return alignUp(offset, 256) + count * sizeof(T);
        }
    };

    // Recursive case: multiple types
    template <typename T, typename... Rest>
    struct BufferSizeComputer<T, Rest...> {
        template <typename... Counts>
        static constexpr size_t compute(size_t offset, size_t count, Counts... counts) {
            static_assert(sizeof...(Rest) == sizeof...(Counts),
                          "Number of remaining types must match remaining counts");
            size_t alignedOffset = alignUp(offset, 256);
            size_t newOffset     = alignedOffset + count * sizeof(T);
            return BufferSizeComputer<Rest...>::compute(newOffset, counts...);
        }
    };

    template <typename... Ts, typename... Counts>
    constexpr size_t computeBufferSize(Counts... counts) {
        static_assert(sizeof...(Ts) == sizeof...(Counts),
                      "Number of types must match number of counts");
        return BufferSizeComputer<Ts...>::compute(0, static_cast<size_t>(counts)...);
    }
}  // namespace ippl

#endif  // IPPL_BUFFERVIEW_H
