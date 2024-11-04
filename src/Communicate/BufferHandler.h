#ifndef IPPL_BUFFER_HANDLER_H
#define IPPL_BUFFER_HANDLER_H

#include <set>
#include <memory>
#include "Communicate/Archive.h"

/**
 * @brief Interface for memory buffer handling.
 *
 * Defines methods for acquiring, freeing, and managing memory buffers.
 * Implementations are responsible for managing buffers efficiently, 
 * ensuring that allocated buffers are reused where possible.
 *
 * @tparam MemorySpace The memory space type used for buffer allocation.
 */
template <typename MemorySpace>
class IBufferHandler {
public:
    using archive_type = ippl::detail::Archive<MemorySpace>;
    using buffer_type  = std::shared_ptr<archive_type>;
    using size_type    = ippl::detail::size_type;

    virtual ~IBufferHandler() {}

    /**
     * @brief Requests a memory buffer of a specified size.
     *
     * Provides a buffer of at least the specified size, with the option
     * to allocate additional space based on an overallocation multiplier.
     * This function attempts to reuse available buffers if possible.
     *
     * @param size The required size of the buffer, in bytes.
     * @param overallocation A multiplier to allocate extra space, which may help
     *                       avoid frequent reallocation in some use cases.
     * @return A shared pointer to the allocated buffer.
     */
    virtual buffer_type getBuffer(size_type size, double overallocation) = 0;

    /**
     * @brief Frees a specified buffer.
     *
     * Moves the specified buffer to a free state, making it available
     * for reuse in future buffer requests.
     *
     * @param buffer The buffer to be freed.
     */
    virtual void freeBuffer(buffer_type buffer)                          = 0;

    /**
     * @brief Frees all currently used buffers.
     *
     * Transfers all used buffers to the free state, making them available
     * for reuse. This does not deallocate memory but resets buffer usage.
     */
    virtual void freeAllBuffers()                                        = 0;

    /**
     * @brief Deletes all buffers.
     *
     * Releases all allocated memory buffers, both used and free.
     * After this call, no buffers are available until new allocations.
     */
    virtual void deleteAllBuffers()                                      = 0;

    /**
     * @brief Gets the size of all allocated buffers.
     *
     * @return Total size of allocated buffers in bytes.
     */
    virtual size_type getAllocatedSize() const = 0;

    /**
     * @brief Gets the size of all free buffers.
     *
     * @return Total size of free buffers in bytes.
     */
    virtual size_type getFreeSize() const = 0;
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
     * @brief Acquires a buffer of at least the specified size.
     *
     * Requests a memory buffer of the specified size, with the option
     * to request a buffer larger than the base size by an overallocation
     * multiplier. Implementations should attempt to reuse existing
     * buffers if possible.
     *
     * @param size The required buffer size.
     * @param overallocation A multiplier to allocate additional buffer space.
     * @return A shared pointer to the allocated buffer.
     */
    buffer_type getBuffer(size_type size, double overallocation) override;

    /**
     * @brief Frees a specified buffer.
     *
     * Moves the specified buffer to a free state, making it available
     * for reuse in future buffer requests.
     *
     * @param buffer The buffer to be freed.
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

