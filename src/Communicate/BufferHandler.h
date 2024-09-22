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
protected:
    virtual void insertBuffer(buffer_type buffer, bool isUsed) = 0;
    virtual void eraseBuffer(buffer_type buffer, bool isUsed) = 0;
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

        auto it = findSmallestSufficientBuffer(requiredSize);

        if (it != free_buffers.end()) {
            buffer_type buffer = *it;
            eraseBuffer(buffer, false);
            insertBuffer(buffer, true);
            return buffer;
        }

        if (!free_buffers.empty()) {
            auto largest_it    = std::prev(free_buffers.end());
            buffer_type buffer = *largest_it;
            eraseBuffer(buffer, false);
            buffer->reallocBuffer(requiredSize);

            insertBuffer(buffer, true);
            return buffer;
        }

        buffer_type newBuffer = std::make_shared<archive_type>(requiredSize);
        insertBuffer(newBuffer, true);
        return newBuffer;
    }

    void freeBuffer(buffer_type buffer) override {
        auto it = used_buffers.find(buffer);
        if (it != used_buffers.end()) {
            eraseBuffer(*it, true);
            insertBuffer(*it, false);
        }
    }

    void freeAllBuffers() override {
        std::vector<buffer_type> buffersToMove(used_buffers.begin(), used_buffers.end());

        for (auto& buffer : buffersToMove) {
            eraseBuffer(buffer, true);
            insertBuffer(buffer, false);
        }
    }

    void deleteAllBuffers() override {
        used_buffers.clear();
        free_buffers.clear();
    }
protected:

    void insertBuffer(buffer_type buffer, bool isUsed) override {
        if (isUsed) {
            used_buffers.insert(buffer);
        } else {
            free_buffers.insert(buffer);
        }
    }

    void eraseBuffer(buffer_type buffer, bool isUsed) override {
        if (isUsed) {
            used_buffers.erase(buffer);
        } else {
            free_buffers.erase(buffer);
        }
    }

private:
    struct BufferComparator {
        bool operator()(const buffer_type& lhs, const buffer_type& rhs) const {
            // First compare by size
            if (lhs->getBufferSize() != rhs->getBufferSize()) {
                return lhs->getBufferSize() < rhs->getBufferSize();
            }
            // If sizes are equal, compare by address
            return lhs < rhs;
        }
    };

    typename std::set<buffer_type, BufferComparator>::iterator findSmallestSufficientBuffer(
        size_type requiredSize) {
        return std::find_if(free_buffers.begin(), free_buffers.end(),
                            [requiredSize](const buffer_type& buffer) {
                                return buffer->getBufferSize() >= requiredSize;
                            });
    }

protected:
    std::set<buffer_type, BufferComparator> used_buffers;
    std::set<buffer_type, BufferComparator> free_buffers;
};

#endif
