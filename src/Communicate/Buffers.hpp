//
// Buffers.hpp
//   Interface for globally accessible buffer factory for communication
//
//   Data sent between MPI ranks has to be stored in a buffer for sending and receiving.
//   To reduce the number of times memory has to be allocated and freed, the buffer
//   factory interface allows buffers to be reused. This is especially relevant on
//   GPUs, as Cuda allocation calls are expensive. To avoid reallocating the buffers
//   in the case that the amount of data to be exchanged increases, when a new buffer
//   is created, an amount of memory greater than the requested size is allocated
//   for the new buffer. The factor by which memory is overallocated is determined by
//   a data member in Communicator, which can be set and queried at runtime. Only new
//   buffers are overallocated. If a buffer is requested with the same ID as a buffer
//   that has been previously allocated, the same buffer will be used. If the requested
//   size exceeds the buffer size, that buffer will be resized to have exactly
//   the requested size.
//
//   Currently, the buffer factory is used for application of periodic boundary
//   conditions; halo cell exchange along faces, edges, and vertices; as well as
//   exchanging particle data between ranks.
//

#include "Utility/Logging.h"
#include "Utility/TypeUtils.h"

namespace ippl {
    namespace mpi {

        // -----------------------------------
        template <typename BufferType, typename T>
        Communicator::buffer_type<BufferType> Communicator::getBuffer(size_type size,
                                                                      double overallocation) {
            using memory_space = BufferType::memory_space;

            auto& buffer_handler = buffer_handlers_m->get<memory_space>();

            auto b = buffer_handler.getBuffer(size * sizeof(T),
                                              std::max(overallocation, defaultOveralloc_m));
            SPDLOG_TRACE("{}, getBuffer {}, buf, {}, size {}", (void*)this,
                         ippl::debug::print_type<memory_space>(), (void*)(b->getBuffer()),
                         size * sizeof(T));
            return b;
        }

        template <typename BufferType>
        void Communicator::freeBuffer(Communicator::buffer_type<BufferType> buffer) {
            using memory_space   = BufferType::memory_space;
            auto& buffer_handler = buffer_handlers_m->get<memory_space>();
            SPDLOG_TRACE("freeBuffer buf, {}", (void*)(buffer->getBuffer()));
            buffer_handler.freeBuffer(buffer);
        }

    }  // namespace mpi

}  // namespace ippl
