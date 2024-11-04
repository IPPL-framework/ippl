#ifndef IPPL_BUFFER_HANDLER_H
#define IPPL_BUFFER_HANDLER_H

#include <set>
#include <memory>
#include "Communicate/Archive.h"

template <typename MemorySpace>
class IBufferHandler {
public:
    using archive_type = ippl::detail::Archive<MemorySpace>;
    using buffer_type  = std::shared_ptr<archive_type>;
    using size_type    = ippl::detail::size_type;

    virtual ~IBufferHandler() {}

    virtual buffer_type getBuffer(size_type size, double overallocation) = 0;
    virtual void freeBuffer(buffer_type buffer)                          = 0;
    virtual void freeAllBuffers()                                        = 0;
    virtual void deleteAllBuffers()                                      = 0;

    virtual size_type getAllocatedSize() const = 0;
    virtual size_type getFreeSize() const      = 0;
};

/**
 * @class BufferHandler
 * @brief Concrete implementation of IBufferHandler for managing memory buffers.
 * 
 * This class implements the IBufferHandler interface, providing concrete behavior for
 * buffer allocation, freeing, and memory management. It maintains a pool of allocated
 * and free buffers to optimize memory usage.
 *
 * @tparam MemorySpace The memory space type for the buffer (e.g., `Kokkos::HostSpace`).
 */
template <typename MemorySpace>
class BufferHandler : public IBufferHandler<MemorySpace> {
public:
    using archive_type = ippl::detail::Archive<MemorySpace>;
    using buffer_type  = std::shared_ptr<archive_type>;
    using size_type    = ippl::detail::size_type;

    ~BufferHandler() override;

    /**
     * @brief Retrieves a buffer of the specified size, or creates a new one if needed.
     *
     * This function first searches for a free buffer of the requested size or larger.
     * If none is found, it allocates a new buffer or reallocates an existing one.
     * 
     * @param size The required size of the buffer.
     * @param overallocation A multiplier to determine additional buffer space.
     * @return A shared pointer to the allocated buffer.
     */
    buffer_type getBuffer(size_type size, double overallocation) override;

    /**
     * @brief Frees a specific buffer, returning it to the free buffer pool without deallocating the actual memory region.
     *
     * @param buffer The buffer to free.
     */
    void freeBuffer(buffer_type buffer) override;

    /**
     * @brief Frees all allocated buffers and adds them to the free pool without deallocating the actual memory region.
     */
    void freeAllBuffers() override;

    /**
     * @brief Deletes all buffers, thereby freeing the memory.
     */
    void deleteAllBuffers() override;

    /**
     * @brief Retrieves the total allocated size, i.e. the size of allocated memory currently in use.
     * 
     * @return The total size of allocated buffers in use.
     */
    size_type getAllocatedSize() const override;

    /**
     * @brief Retrieves the total size of free buffers, i.e. size of allocated memory currently not in use.
     * 
     * @return The total size of free buffers.
     */
    size_type getFreeSize() const override;

private:
    using buffer_comparator_type = bool (*)(const buffer_type&, const buffer_type&);
    using buffer_set_type        = std::set<buffer_type, buffer_comparator_type>;

    static bool bufferSizeComparator(const buffer_type& lhs, const buffer_type& rhs);

    bool isBufferUsed(buffer_type buffer) const;
    void releaseUsedBuffer(buffer_type buffer);
    buffer_type findFreeBuffer(size_type requiredSize);
    buffer_set_type::iterator findSmallestSufficientBuffer(size_type requiredSize);
    buffer_type allocateFromFreeBuffer(buffer_type buffer);
    buffer_type reallocateLargestFreeBuffer(size_type requiredSize);
    buffer_type allocateNewBuffer(size_type requiredSize);

    size_type allocatedSize; ///< Total size of all allocated buffers
    size_type freeSize; ///< Total size of all free buffers

protected:
    buffer_set_type used_buffers{&BufferHandler::bufferSizeComparator}; ///< Set of used buffers
    buffer_set_type free_buffers{&BufferHandler::bufferSizeComparator}; ///< Set of free buffers
};

#include "BufferHandler.hpp"

#endif

