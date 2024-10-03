#ifndef IPPL_BUFFER_HANDLER_H
#define IPPL_BUFFER_HANDLER_H

#include <set>

#include "Communicate/Archive.h"


template <typename MemorySpace>
class IBufferHandler {
public:
    using archive_type = ippl::detail::Archive<MemorySpace>;
    using buffer_type  = std::shared_ptr<archive_type>;
    using size_type    = ippl::detail::size_type;

    virtual ~IBufferHandler() {}

    virtual buffer_type getBuffer(size_type size, double overallocation) = 0;
    virtual void freeBuffer(buffer_type buffer) = 0;
    virtual void freeAllBuffers() = 0;
    virtual void deleteAllBuffers() = 0;

    virtual size_type getAllocatedSize() const = 0;
    virtual size_type getFreeSize() const = 0;
};

template <typename MemorySpace>
class BufferHandler : public IBufferHandler<MemorySpace> {
public:
    using archive_type = ippl::detail::Archive<MemorySpace>;
    using buffer_type  = std::shared_ptr<archive_type>;
    using size_type    = ippl::detail::size_type;

    ~BufferHandler() {}

    buffer_type getBuffer(size_type size, double overallocation) override {
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

    void freeBuffer(buffer_type buffer) override {
        if (isBufferUsed(buffer)) {
            releaseUsedBuffer(buffer);
        }
    }

    void freeAllBuffers() override {
        free_buffers.insert(used_buffers.begin(), used_buffers.end());
        used_buffers.clear();

        freeSize += allocatedSize;
        allocatedSize = 0;
    }

    void deleteAllBuffers() override {
        freeSize = 0;
        allocatedSize = 0;

        used_buffers.clear();
        free_buffers.clear();
    }

    size_type getAllocatedSize() const override {
      return allocatedSize;
    }

    size_type getFreeSize() const override {
      return freeSize;
    }

private:
    using buffer_comparator_type = bool (*)(const buffer_type&, const buffer_type&);

    static bool bufferSizeComparator(const buffer_type& lhs, const buffer_type& rhs) {
        // Compare by size first, then by memory address
        if (lhs->getBufferSize() != rhs->getBufferSize()) {
            return lhs->getBufferSize() < rhs->getBufferSize();
        }
        return lhs < rhs;
    }

    bool isBufferUsed(buffer_type buffer) const {
        return used_buffers.find(buffer) != used_buffers.end();
    }

    void releaseUsedBuffer(buffer_type buffer) {
        auto it = used_buffers.find(buffer);
        
        allocatedSize -= buffer->getBufferSize();
        freeSize += buffer->getBufferSize();

        used_buffers.erase(it);
        free_buffers.insert(buffer);
    }

    buffer_type findFreeBuffer(size_type requiredSize) {
        auto it = findSmallestSufficientBuffer(requiredSize);
        if (it != free_buffers.end()) {
            return *it;
        }
        return nullptr;
    }

    typename std::set<buffer_type, buffer_comparator_type>::iterator findSmallestSufficientBuffer(
        size_type requiredSize) {
        return std::find_if(free_buffers.begin(), free_buffers.end(),
                            [requiredSize](const buffer_type& buffer) {
                                return buffer->getBufferSize() >= requiredSize;
                            });
    }


    buffer_type allocateFromFreeBuffer(buffer_type buffer) {
        freeSize -= buffer->getBufferSize();
        allocatedSize += buffer->getBufferSize();

        free_buffers.erase(buffer);
        used_buffers.insert(buffer);
        return buffer;
    }

    buffer_type reallocateLargestFreeBuffer(size_type requiredSize) {
        auto largest_it = std::prev(free_buffers.end());
        buffer_type buffer = *largest_it;

        freeSize -= buffer->getBufferSize();
        allocatedSize += requiredSize;

        free_buffers.erase(buffer);
        buffer->reallocBuffer(requiredSize);

        used_buffers.insert(buffer);
        return buffer;
    }

    buffer_type allocateNewBuffer(size_type requiredSize) {
        buffer_type newBuffer = std::make_shared<archive_type>(requiredSize);

        allocatedSize += newBuffer->getBufferSize();
        used_buffers.insert(newBuffer);
        return newBuffer;
    }
    
  size_type allocatedSize;
  size_type freeSize;


protected:
    std::set<buffer_type, buffer_comparator_type> used_buffers{&BufferHandler::bufferSizeComparator};
    std::set<buffer_type, buffer_comparator_type> free_buffers{&BufferHandler::bufferSizeComparator};

};

#endif
