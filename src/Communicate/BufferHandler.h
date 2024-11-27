#ifndef IPPL_BUFFER_HANDLER_H
#define IPPL_BUFFER_HANDLER_H

#include <memory>
#include <set>

#include "Communicate/Archive.h"

namespace ippl {

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
    class BufferHandler {
    public:
        using archive_type = ippl::detail::Archive<MemorySpace>;
        using buffer_type  = std::shared_ptr<archive_type>;
        using size_type    = ippl::detail::size_type;

        virtual ~BufferHandler() {}

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
        virtual void freeBuffer(buffer_type buffer) = 0;

        /**
         * @brief Frees all currently used buffers.
         *
         * Transfers all used buffers to the free state, making them available
         * for reuse. This does not deallocate memory but resets buffer usage.
         */
        virtual void freeAllBuffers() = 0;

        /**
         * @brief Deletes all buffers.
         *
         * Releases all allocated memory buffers, both used and free.
         * After this call, no buffers are available until new allocations.
         */
        virtual void deleteAllBuffers() = 0;

        /**
         * @brief Gets the size of all buffers in use.
         *
         * @return Total size of buffers that are in use in bytes.
         */
        virtual size_type getUsedSize() const = 0;

        /**
         * @brief Gets the size of all free buffers.
         *
         * @return Total size of free buffers in bytes.
         */
        virtual size_type getFreeSize() const = 0;
    };

    /**
     * @class DefaultBufferHandler
     * @brief Concrete implementation of BufferHandler for managing memory buffers.
     *
     * This class implements the BufferHandler interface, providing concrete behavior for
     * buffer allocation, freeing, and memory management. It maintains two sorted sets of free and
     * in-use buffers to allow for efficient queries.
     *
     * @tparam MemorySpace The memory space type for the buffer (e.g., `Kokkos::HostSpace`).
     */
    template <typename MemorySpace>
    class DefaultBufferHandler : public BufferHandler<MemorySpace> {
    public:
        using typename BufferHandler<MemorySpace>::archive_type;
        using typename BufferHandler<MemorySpace>::buffer_type;
        using typename BufferHandler<MemorySpace>::size_type;

        ~DefaultBufferHandler() override;

        /**
         * @brief Acquires a buffer of at least the specified size.
         *
         * Requests a memory buffer of the specified size, with the option
         * to request a buffer larger than the base size by an overallocation
         * multiplier. If a sufficiently large buffer is available, it is returned. If not, the
         * largest free buffer is reallocated. If there are no free buffers available, only then a
         * new buffer is allocated.
         *
         * @param size The required buffer size.
         * @param overallocation A multiplier to allocate additional buffer space.
         * @return A shared pointer to the allocated buffer.
         */
        buffer_type getBuffer(size_type size, double overallocation) override;

        /**
         * @copydoc BufferHandler::freeBuffer
         */
        void freeBuffer(buffer_type buffer) override;

        /**
         * @copydoc BufferHandler::freeBuffer
         */
        void freeAllBuffers() override;

        /**
         * @copydoc BufferHandler::freeBuffer
         */
        void deleteAllBuffers() override;

        /**
         * @copydoc BufferHandler::freeBuffer
         */
        size_type getUsedSize() const override;

        /**
         * @copydoc BufferHandler::freeBuffer
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
        buffer_type getFreeBuffer(buffer_type buffer);
        buffer_type reallocateLargestFreeBuffer(size_type requiredSize);
        buffer_type allocateNewBuffer(size_type requiredSize);

        size_type usedSize_m;  ///< Total size of all allocated buffers
        size_type freeSize_m;  ///< Total size of all free buffers

    protected:
        buffer_set_type used_buffers{
            &DefaultBufferHandler::bufferSizeComparator};  ///< Set of used buffers
        buffer_set_type free_buffers{
            &DefaultBufferHandler::bufferSizeComparator};  ///< Set of free buffers
    };
}  // namespace ippl

#include "Communicate/BufferHandler.hpp"

#endif
