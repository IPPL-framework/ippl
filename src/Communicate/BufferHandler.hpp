#ifndef IPPL_BUFFER_HANDLER_HPP
#define IPPL_BUFFER_HANDLER_HPP

namespace ippl {

    template <typename MemorySpace>
    BufferHandler<MemorySpace>::~BufferHandler() {}

    template <typename MemorySpace>
    typename BufferHandler<MemorySpace>::buffer_type BufferHandler<MemorySpace>::getBuffer(
        size_type size, double overallocation) {
        size_type requiredSize = static_cast<size_type>(size * overallocation);

        auto freeBuffer = findFreeBuffer(requiredSize);
        if (freeBuffer != nullptr) {
            return getFreeBuffer(freeBuffer);
        }

        if (!free_buffers.empty()) {
            return reallocateLargestFreeBuffer(requiredSize);
        }

        return allocateNewBuffer(requiredSize);
    }

    template <typename MemorySpace>
    void BufferHandler<MemorySpace>::freeBuffer(buffer_type buffer) {
        if (isBufferUsed(buffer)) {
            releaseUsedBuffer(buffer);
        }
    }

    template <typename MemorySpace>
    void BufferHandler<MemorySpace>::freeAllBuffers() {
        free_buffers.insert(used_buffers.begin(), used_buffers.end());
        used_buffers.clear();

        freeSize_m += usedSize_m;
        usedSize_m = 0;
    }

    template <typename MemorySpace>
    void BufferHandler<MemorySpace>::deleteAllBuffers() {
        freeSize_m      = 0;
        usedSize_m = 0;

        used_buffers.clear();
        free_buffers.clear();
    }

    template <typename MemorySpace>
    typename BufferHandler<MemorySpace>::size_type BufferHandler<MemorySpace>::getUsedSize()
        const {
        return usedSize_m;
    }

    template <typename MemorySpace>
    typename BufferHandler<MemorySpace>::size_type BufferHandler<MemorySpace>::getFreeSize() const {
        return freeSize_m;
    }

    template <typename MemorySpace>
    bool BufferHandler<MemorySpace>::bufferSizeComparator(const buffer_type& lhs,
                                                          const buffer_type& rhs) {
        if (lhs->getBufferSize() != rhs->getBufferSize()) {
            return lhs->getBufferSize() < rhs->getBufferSize();
        }

        // Use memory address as a tie-breaker to enforce total ordering of buffers.
        return lhs < rhs;
    }

    template <typename MemorySpace>
    bool BufferHandler<MemorySpace>::isBufferUsed(buffer_type buffer) const {
        return used_buffers.find(buffer) != used_buffers.end();
    }

    template <typename MemorySpace>
    void BufferHandler<MemorySpace>::releaseUsedBuffer(buffer_type buffer) {
        auto it = used_buffers.find(buffer);

        usedSize_m -= buffer->getBufferSize();
        freeSize_m += buffer->getBufferSize();

        used_buffers.erase(it);
        free_buffers.insert(buffer);
    }

    template <typename MemorySpace>
    typename BufferHandler<MemorySpace>::buffer_type BufferHandler<MemorySpace>::findFreeBuffer(
        size_type requiredSize) {
        auto it = findSmallestSufficientBuffer(requiredSize);
        if (it != free_buffers.end()) {
            return *it;
        }
        return nullptr;
    }

    template <typename MemorySpace>
    typename BufferHandler<MemorySpace>::buffer_set_type::iterator
    BufferHandler<MemorySpace>::findSmallestSufficientBuffer(size_type requiredSize) {
        return std::find_if(free_buffers.begin(), free_buffers.end(),
                            [requiredSize](const buffer_type& buffer) {
                                return buffer->getBufferSize() >= requiredSize;
                            });
    }

    template <typename MemorySpace>
    typename BufferHandler<MemorySpace>::buffer_type
    BufferHandler<MemorySpace>::getFreeBuffer(buffer_type buffer) {
        freeSize_m -= buffer->getBufferSize();
        usedSize_m += buffer->getBufferSize();

        free_buffers.erase(buffer);
        used_buffers.insert(buffer);
        return buffer;
    }

    template <typename MemorySpace>
    typename BufferHandler<MemorySpace>::buffer_type
    BufferHandler<MemorySpace>::reallocateLargestFreeBuffer(size_type requiredSize) {
        auto largest_it    = std::prev(free_buffers.end());
        buffer_type buffer = *largest_it;

        freeSize_m -= buffer->getBufferSize();
        usedSize_m += requiredSize;

        free_buffers.erase(buffer);
        buffer->reallocBuffer(requiredSize);

        used_buffers.insert(buffer);
        return buffer;
    }

    template <typename MemorySpace>
    typename BufferHandler<MemorySpace>::buffer_type BufferHandler<MemorySpace>::allocateNewBuffer(
        size_type requiredSize) {
        buffer_type newBuffer = std::make_shared<archive_type>(requiredSize);

        usedSize_m += newBuffer->getBufferSize();
        used_buffers.insert(newBuffer);
        return newBuffer;
    }

}  // namespace ippl

#endif
