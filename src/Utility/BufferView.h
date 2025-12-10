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

        BufferView(size_t count, double overallocation = 1.0)
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
}  // namespace ippl

#endif  // IPPL_BUFFERVIEW_H
