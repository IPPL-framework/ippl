#ifndef IPPL_BUFFER_HANDLER_H
#define IPPL_BUFFER_HANDLER_H

#include <set>

template <typename MemorySpace>
class BufferHandler {
public:
    using archive_type = ippl::detail::Archive<MemorySpace>;
    using buffer_type  = std::shared_ptr<archive_type>;
    using size_type    = ippl::detail::size_type;

    ~BufferHandler() {
        deleteAllBuffers();
    }

    buffer_type getBuffer(size_type size, double overallocation) {
        size_type requiredSize = static_cast<size_type>(size * overallocation);

        auto it = findSmallestSufficientBuffer(requiredSize);

        if (it != free_buffers.end()) {
            buffer_type buffer = *it;
            free_buffers.erase(it);
            used_buffers.insert(buffer);
            return buffer;
        }

        if (!free_buffers.empty()) {
            auto largest_it    = std::prev(free_buffers.end());
            buffer_type buffer = *largest_it;
            free_buffers.erase(largest_it);

            buffer->reallocBuffer(requiredSize);

            used_buffers.insert(buffer);
            return buffer;
        }

        buffer_type newBuffer = std::make_shared<archive_type>(requiredSize);
        used_buffers.insert(newBuffer);
        return newBuffer;
    }

    void deleteBuffer(buffer_type buffer) {
        auto it = used_buffers.find(buffer);
        if (it != used_buffers.end()) {
            used_buffers.erase(it);
            free_buffers.insert(buffer);
        }
    }

    void freeAllBuffers() {
        free_buffers.insert(used_buffers.begin(), used_buffers.end());
        used_buffers.clear();
    }

    void deleteAllBuffers() {
        used_buffers.clear();
        free_buffers.clear();
    }

protected:
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

    std::set<buffer_type, BufferComparator> used_buffers;
    std::set<buffer_type, BufferComparator> free_buffers;
};

#endif
