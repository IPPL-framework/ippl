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


}  // namespace ippl

#endif  // IPPL_BUFFERVIEW_H