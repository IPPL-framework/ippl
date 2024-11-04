#ifndef IPPL_BUFFER_HANDLER_HPP
#define IPPL_BUFFER_HANDLER_HPP


template <typename MemorySpace>
BufferHandler<MemorySpace>::~BufferHandler() {}

template <typename MemorySpace>
typename BufferHandler<MemorySpace>::buffer_type 
BufferHandler<MemorySpace>::getBuffer(size_type size, double overallocation) {
    size_type requiredSize = static_cast<size_type>(size * overallocation);

    auto freeBuffer = findFreeBuffer(requiredSize);
    if (freeBuffer != nullptr) {
        return allocateFromFreeBuffer(freeBuffer);
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

    freeSize += allocatedSize;
    allocatedSize = 0;
}

template <typename MemorySpace>
void BufferHandler<MemorySpace>::deleteAllBuffers() {
    freeSize      = 0;
    allocatedSize = 0;

    used_buffers.clear();
    free_buffers.clear();
}

template <typename MemorySpace>
typename BufferHandler<MemorySpace>::size_type 
BufferHandler<MemorySpace>::getAllocatedSize() const {
    return allocatedSize;
}

template <typename MemorySpace>
typename BufferHandler<MemorySpace>::size_type 
BufferHandler<MemorySpace>::getFreeSize() const {
    return freeSize;
}

template <typename MemorySpace>
bool BufferHandler<MemorySpace>::bufferSizeComparator(const buffer_type& lhs, const buffer_type& rhs) {
    if (lhs->getBufferSize() != rhs->getBufferSize()) {
        return lhs->getBufferSize() < rhs->getBufferSize();
    }
    return lhs < rhs;
}

template <typename MemorySpace>
bool BufferHandler<MemorySpace>::isBufferUsed(buffer_type buffer) const {
    return used_buffers.find(buffer) != used_buffers.end();
}

template <typename MemorySpace>
void BufferHandler<MemorySpace>::releaseUsedBuffer(buffer_type buffer) {
    auto it = used_buffers.find(buffer);

    allocatedSize -= buffer->getBufferSize();
    freeSize += buffer->getBufferSize();

    used_buffers.erase(it);
    free_buffers.insert(buffer);
}

template <typename MemorySpace>
typename BufferHandler<MemorySpace>::buffer_type 
BufferHandler<MemorySpace>::findFreeBuffer(size_type requiredSize) {
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
BufferHandler<MemorySpace>::allocateFromFreeBuffer(buffer_type buffer) {
    freeSize -= buffer->getBufferSize();
    allocatedSize += buffer->getBufferSize();

    free_buffers.erase(buffer);
    used_buffers.insert(buffer);
    return buffer;
}

template <typename MemorySpace>
typename BufferHandler<MemorySpace>::buffer_type 
BufferHandler<MemorySpace>::reallocateLargestFreeBuffer(size_type requiredSize) {
    auto largest_it    = std::prev(free_buffers.end());
    buffer_type buffer = *largest_it;

    freeSize -= buffer->getBufferSize();
    allocatedSize += requiredSize;

    free_buffers.erase(buffer);
    buffer->reallocBuffer(requiredSize);

    used_buffers.insert(buffer);
    return buffer;
}

template <typename MemorySpace>
typename BufferHandler<MemorySpace>::buffer_type 
BufferHandler<MemorySpace>::allocateNewBuffer(size_type requiredSize) {
    buffer_type newBuffer = std::make_shared<archive_type>(requiredSize);

    allocatedSize += newBuffer->getBufferSize();
    used_buffers.insert(newBuffer);
    return newBuffer;
}

#endif
