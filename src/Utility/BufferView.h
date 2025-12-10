#ifndef IPPL_BUFFERVIEW_H
#define IPPL_BUFFERVIEW_H

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "Communicate/Archive.h"
#include "Communicate/BufferHandler.h"

namespace ippl {

    /**
     * @brief Helper to compute aligned offset
     */
    inline constexpr size_t alignUp(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    /**
     * @brief Singleton access to compute buffer handler for each memory space.
     *        Separate from MPI buffer handler to avoid interference.
     */
    template <typename MemorySpace>
    DefaultBufferHandler<MemorySpace>& getComputeBufferHandler() {
        static DefaultBufferHandler<MemorySpace> handler;
        return handler;
    }

    /**
     * @brief Clean up compute buffer handlers before Kokkos::finalize().
     *        Call this for each memory space you used.
     */
    template <typename MemorySpace>
    void finalizeComputeBufferHandler() {
        getComputeBufferHandler<MemorySpace>().deleteAllBuffers();
    }

    /**
     * @brief Clean up all common compute buffer handlers.
     *        Call this before Kokkos::finalize().
     */
    inline void finalizeComputeBufferHandlers() {
        // Host space
        finalizeComputeBufferHandler<Kokkos::HostSpace>();

#ifdef KOKKOS_ENABLE_CUDA
        finalizeComputeBufferHandler<Kokkos::CudaSpace>();
#endif

#ifdef KOKKOS_ENABLE_HIP
        finalizeComputeBufferHandler<Kokkos::HIPSpace>();
#endif

#ifdef KOKKOS_ENABLE_SYCL
        finalizeComputeBufferHandler<Kokkos::Experimental::SYCLDeviceUSMSpace>();
#endif

        // Default execution space memory (in case it's different)
        finalizeComputeBufferHandler<Kokkos::DefaultExecutionSpace::memory_space>();
    }

    /**
     * @brief Allocates a single buffer and provides multiple aligned views into it.
     *
     * Uses a separate buffer handler from MPI to avoid interference with
     * CUDA IPC / GPUDirect RDMA memory registration.
     *
     * @tparam MemorySpace The Kokkos memory space for the buffer
     */
    template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
    class MultiViewBuffer {
    public:
        using handler_type = DefaultBufferHandler<MemorySpace>;
        using buffer_type  = typename handler_type::buffer_type;

        static constexpr size_t DEFAULT_ALIGNMENT = 256;

        /**
         * @brief Construct without allocation. Call allocate() before getting views.
         */
        MultiViewBuffer()
            : buffer_m(nullptr)
            , currentOffset_m(0) {}

        /**
         * @brief Construct and allocate buffer of specified size.
         */
        explicit MultiViewBuffer(size_t totalBytes, double overallocation = 1.0)
            : currentOffset_m(0) {
            allocate(totalBytes, overallocation);
        }

        ~MultiViewBuffer() {
            if (buffer_m) {
                getComputeBufferHandler<MemorySpace>().freeBuffer(buffer_m);
            }
        }

        // Non-copyable
        MultiViewBuffer(const MultiViewBuffer&)            = delete;
        MultiViewBuffer& operator=(const MultiViewBuffer&) = delete;

        // Movable
        MultiViewBuffer(MultiViewBuffer&& other) noexcept
            : buffer_m(std::move(other.buffer_m))
            , currentOffset_m(other.currentOffset_m) {
            other.buffer_m        = nullptr;
            other.currentOffset_m = 0;
        }

        MultiViewBuffer& operator=(MultiViewBuffer&& other) noexcept {
            if (this != &other) {
                if (buffer_m) {
                    getComputeBufferHandler<MemorySpace>().freeBuffer(buffer_m);
                }
                buffer_m              = std::move(other.buffer_m);
                currentOffset_m       = other.currentOffset_m;
                other.buffer_m        = nullptr;
                other.currentOffset_m = 0;
            }
            return *this;
        }

        /**
         * @brief Allocate or reallocate the underlying buffer from the pool.
         */
        void allocate(size_t totalBytes, double overallocation = 1.0) {
            if (buffer_m) {
                getComputeBufferHandler<MemorySpace>().freeBuffer(buffer_m);
            }
            buffer_m        = getComputeBufferHandler<MemorySpace>().getBuffer(totalBytes, overallocation);
            currentOffset_m = 0;
        }

        /**
         * @brief Get an aligned view of count elements of type T from the buffer.
         */
        template <typename T>
        Kokkos::View<T*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        getView(size_t count) {
            using view_type = Kokkos::View<T*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

            size_t alignedOffset = alignUp(currentOffset_m, DEFAULT_ALIGNMENT);
            size_t requiredSize  = alignedOffset + count * sizeof(T);

            if (!buffer_m || requiredSize > buffer_m->getBufferSize()) {
                throw std::runtime_error(
                    "MultiViewBuffer: insufficient space. Required: "
                    + std::to_string(requiredSize)
                    + ", Available: " + std::to_string(buffer_m ? buffer_m->getBufferSize() : 0));
            }

            T* viewPtr      = reinterpret_cast<T*>(buffer_m->getBuffer() + alignedOffset);
            currentOffset_m = alignedOffset + count * sizeof(T);

            return view_type(viewPtr, count);
        }

        void reset() { currentOffset_m = 0; }

        bool isAllocated() const { return buffer_m != nullptr; }
        size_t getUsedSize() const { return currentOffset_m; }
        size_t getTotalSize() const { return buffer_m ? buffer_m->getBufferSize() : 0; }
        size_t getRemainingSize() const { return getTotalSize() - currentOffset_m; }

    private:
        buffer_type buffer_m;
        size_t currentOffset_m;
    };

    /**
     * @brief RAII wrapper for a single typed view from the compute buffer pool.
     */
    template <typename T, typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
    class BufferView {
    public:
        using handler_type = DefaultBufferHandler<MemorySpace>;
        using buffer_type  = typename handler_type::buffer_type;
        using view_type    = Kokkos::View<T*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        explicit BufferView(size_t count, double overallocation = 1.0)
            : count_m(count) {
            size_t byteSize = count * sizeof(T);
            buffer_m        = getComputeBufferHandler<MemorySpace>().getBuffer(byteSize, overallocation);
            view_m          = view_type(reinterpret_cast<T*>(buffer_m->getBuffer()), count);
        }

        ~BufferView() {
            if (buffer_m) {
                getComputeBufferHandler<MemorySpace>().freeBuffer(buffer_m);
            }
        }

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

        T* data() { return view_m.data(); }
        const T* data() const { return view_m.data(); }
        size_t size() const { return count_m; }

    private:
        buffer_type buffer_m;
        view_type view_m;
        size_t count_m;
    };

    /**
     * @brief Helper struct for recursive buffer size computation
     */
    template <typename... Ts>
    struct BufferSizeComputer;

    template <typename T>
    struct BufferSizeComputer<T> {
        static constexpr size_t compute(size_t offset, size_t count) {
            return alignUp(offset, MultiViewBuffer<>::DEFAULT_ALIGNMENT) + count * sizeof(T);
        }
    };

    template <typename T, typename... Rest>
    struct BufferSizeComputer<T, Rest...> {
        template <typename... Counts>
        static constexpr size_t compute(size_t offset, size_t count, Counts... counts) {
            static_assert(sizeof...(Rest) == sizeof...(Counts),
                          "Number of remaining types must match remaining counts");
            size_t alignedOffset = alignUp(offset, MultiViewBuffer<>::DEFAULT_ALIGNMENT);
            size_t newOffset     = alignedOffset + count * sizeof(T);
            return BufferSizeComputer<Rest...>::compute(newOffset, counts...);
        }
    };

    /**
     * @brief Compute total buffer size needed for multiple views with alignment.
     *
     * Usage: auto size = computeBufferSize<double, int, float>(100, 200, 50);
     */
    template <typename... Ts, typename... Counts>
    constexpr size_t computeBufferSize(Counts... counts) {
        static_assert(sizeof...(Ts) == sizeof...(Counts),
                      "Number of types must match number of counts");
        return BufferSizeComputer<Ts...>::compute(0, static_cast<size_t>(counts)...);
    }

}  // namespace ippl

#endif  // IPPL_BUFFERVIEW_H