#ifndef IPPL_BUFFERVIEW_H
#define IPPL_BUFFERVIEW_H

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ippl {

    /**
     * @brief Helper to compute aligned offset
     */
    inline constexpr size_t alignUp(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    /**
     *
     * @tparam T Element type
     * @tparam MemorySpace Kokkos memory space
     */
    template <typename T, typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
    class BufferView {
    public:
        using view_type = Kokkos::View<T*, MemorySpace>;

        explicit BufferView(size_t count)
            : view_m("BufferView", count) {}

        ~BufferView() = default;

        // Non-copyable
        BufferView(const BufferView&)            = delete;
        BufferView& operator=(const BufferView&) = delete;

        // Movable
        BufferView(BufferView&&) = default;
        BufferView& operator=(BufferView&&) = default;

        view_type& getView() { return view_m; }
        const view_type& getView() const { return view_m; }

        T* data() { return view_m.data(); }
        const T* data() const { return view_m.data(); }
        size_t size() const { return view_m.extent(0); }

    private:
        view_type view_m;
    };

    /**
     * @brief Allocates a single buffer and provides multiple aligned views into it.
     *
     * This class manages its own memory via a Kokkos View, completely independent
     * from MPI buffer management. Use this for temporary computation buffers
     * that don't need MPI communication.
     *
     * @tparam MemorySpace The Kokkos memory space for the buffer
     */
    template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
    class MultiViewBuffer {
    public:
        using buffer_view_type = Kokkos::View<char*, MemorySpace>;

        static constexpr size_t DEFAULT_ALIGNMENT = 256;

        /**
         * @brief Construct without allocation. Call allocate() before getting views.
         */
        MultiViewBuffer()
            : currentOffset_m(0) {}

        /**
         * @brief Construct and allocate buffer of specified size.
         * @param totalBytes Total size in bytes
         * @param label Optional Kokkos label for debugging
         */
        explicit MultiViewBuffer(size_t totalBytes, const std::string& label = "MultiViewBuffer")
            : buffer_m(label, totalBytes)
            , currentOffset_m(0) {}

        ~MultiViewBuffer() = default;

        // Non-copyable
        MultiViewBuffer(const MultiViewBuffer&)            = delete;
        MultiViewBuffer& operator=(const MultiViewBuffer&) = delete;

        // Movable
        MultiViewBuffer(MultiViewBuffer&& other) noexcept
            : buffer_m(std::move(other.buffer_m))
            , currentOffset_m(other.currentOffset_m) {
            other.currentOffset_m = 0;
        }

        MultiViewBuffer& operator=(MultiViewBuffer&& other) noexcept {
            if (this != &other) {
                buffer_m              = std::move(other.buffer_m);
                currentOffset_m       = other.currentOffset_m;
                other.currentOffset_m = 0;
            }
            return *this;
        }

        /**
         * @brief Allocate or reallocate the underlying buffer.
         * @param totalBytes Total size in bytes
         * @param label Optional Kokkos label for debugging
         */
        void allocate(size_t totalBytes, const std::string& label = "MultiViewBuffer") {
            buffer_m        = buffer_view_type(label, totalBytes);
            currentOffset_m = 0;
        }

        /**
         * @brief Get an aligned view of count elements of type T from the buffer.
         *
         * Views are allocated sequentially from the buffer. Each view starts at
         * an address aligned to DEFAULT_ALIGNMENT (256 bytes).
         *
         * @tparam T Element type for the view
         * @param count Number of elements
         * @return Unmanaged Kokkos::View pointing into the buffer
         */
        template <typename T>
        Kokkos::View<T*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        getView(size_t count) {
            using view_type = Kokkos::View<T*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

            size_t alignedOffset = alignUp(currentOffset_m, DEFAULT_ALIGNMENT);
            size_t requiredSize  = alignedOffset + count * sizeof(T);

            if (requiredSize > buffer_m.size()) {
                throw std::runtime_error(
                    "MultiViewBuffer: insufficient space. Required: "
                    + std::to_string(requiredSize)
                    + ", Available: " + std::to_string(buffer_m.size()));
            }

            T* viewPtr = reinterpret_cast<T*>(buffer_m.data() + alignedOffset);
            currentOffset_m = alignedOffset + count * sizeof(T);

            return view_type(viewPtr, count);
        }

        /**
         * @brief Reset the allocation offset to reuse the buffer for new views.
         *
         * Previously obtained views become invalid after this call.
         */
        void reset() { currentOffset_m = 0; }

        /**
         * @brief Check if buffer has been allocated.
         */
        bool isAllocated() const { return buffer_m.size() > 0; }

        /**
         * @brief Get the current used size in bytes.
         */
        size_t getUsedSize() const { return currentOffset_m; }

        /**
         * @brief Get the total available size in bytes.
         */
        size_t getTotalSize() const { return buffer_m.size(); }

        /**
         * @brief Get remaining available bytes.
         */
        size_t getRemainingSize() const { return getTotalSize() - currentOffset_m; }

        /**
         * @brief Get raw pointer to buffer start.
         */
        char* data() { return buffer_m.data(); }
        const char* data() const { return buffer_m.data(); }

    private:
        buffer_view_type buffer_m;
        size_t currentOffset_m = 0;
    };

    /**
     * @brief Helper struct for recursive buffer size computation
     */
    template <typename... Ts>
    struct BufferSizeComputer;

    // Base case: single type
    template <typename T>
    struct BufferSizeComputer<T> {
        static constexpr size_t compute(size_t offset, size_t count) {
            return alignUp(offset, MultiViewBuffer<>::DEFAULT_ALIGNMENT) + count * sizeof(T);
        }
    };

    // Recursive case: multiple types
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
     *
     * @tparam Ts Element types for each view
     * @param counts Number of elements for each view
     * @return Total required buffer size in bytes
     */
    template <typename... Ts, typename... Counts>
    constexpr size_t computeBufferSize(Counts... counts) {
        static_assert(sizeof...(Ts) == sizeof...(Counts),
                      "Number of types must match number of counts");
        return BufferSizeComputer<Ts...>::compute(0, static_cast<size_t>(counts)...);
    }

}  // namespace ippl

#endif  // IPPL_BUFFERVIEW_H