#ifndef IPPL_BUFFER_HANDLER_H
#define IPPL_BUFFER_HANDLER_H

#include <memory>
#include <set>

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

#include "Utility/Logging.h"
#include "Utility/TypeUtils.h"

#include "Communicate/Archive.h"

namespace ippl::comms {

#ifdef IPPL_SIMPLE_VIEW_STORAGE
    template <typename... Properties>
    using communicator_storage =
        ippl::detail::ViewType<char, 1, Properties...,
                               Kokkos::MemoryTraits<Kokkos::Aligned>>::view_type;
#else
    template <typename... Properties>
    using communicator_storage = ippl::detail::ViewType<
        char, 1, Properties...,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::view_type;
#endif

#define ALIGMNEMT 1024

    // make number a multiple of the alignment
    inline std::int64_t to_multiple(std::int64_t num) {
        return ((2 * num + (ALIGMNEMT - 1)) & (-ALIGMNEMT));
    }

    struct AlignedCudaBuffer {
        void* ptrOriginal;
        void* ptrAligned;
        detail::size_type space;
        //
        AlignedCudaBuffer()
            : ptrOriginal{nullptr}
            , ptrAligned{nullptr}
            , space{0} {}
        //
        AlignedCudaBuffer(std::size_t size) {
            void* original;
            space = to_multiple(size);
            cudaMalloc(&original, space);
            ptrOriginal = original;
            ptrAligned  = std::align(ALIGMNEMT, size, original, space);
            SPDLOG_TRACE("AlignedCudaBuffer: original {}, aligned {}, size {}, space {}",
                         (void*)(ptrOriginal), (void*)(ptrAligned), size, space);
            // sanity check should always be true when std::align used
            assert(space >= size);
        }
        // don't delete internal memory until explicitly told to
        ~AlignedCudaBuffer() {
            if (ptrOriginal) {
                ptrOriginal = 0;
            }
        }
        // the function that really releases memory
        void destroy_buffer() {
            if (ptrOriginal) {
                cudaFree(ptrOriginal);
            } else {
                throw std::runtime_error("Destroying cuda buffer after reset");
            }
        }
    };

    template <typename MemorySpace, typename... Properties>
    struct comm_storage_wrapper {
        using memory_space = MemorySpace;
        using buffer_type  = communicator_storage<MemorySpace, Properties...>;
        using pointer_type = typename buffer_type::pointer_type;
        using size_type    = detail::size_type;
        //
        comm_storage_wrapper(const std::string& /*name*/, size_type size)
            : view()    // we will construct the view manually
            , buffer()  //
        {
            reallocBuffer(size);
            SPDLOG_TRACE("Construct: view  origin {}, aligned {}", (void*)(view.data()),
                         (void*)(buffer.ptrAligned));
            auto aligned_ptr_check = std::align(ALIGMNEMT, size, buffer.ptrAligned, buffer.space);
            SPDLOG_TRACE("Construct: view    data {}, align_check {}", (void*)(view.data()),
                         (void*)(aligned_ptr_check));
            assert(view.data() == buffer.ptrAligned);
        }
        //
        size_type size() const { return buffer.space; }
        //
        pointer_type data() { return view.data(); }
        //
        void reallocBuffer(size_type newsize) {
            auto old_pointer = buffer.ptrOriginal;
            buffer           = AlignedCudaBuffer(newsize);
            view             = buffer_type((pointer_type)buffer.ptrAligned, newsize);
            SPDLOG_DEBUG("Realloc  : view {}, aligned {}, size {}, space {}", (void*)(view.data()),
                         (void*)(buffer.ptrAligned), newsize, buffer.space);
            if (old_pointer)
                cudaFree(old_pointer);
        }
        //
        AlignedCudaBuffer buffer;
        buffer_type view;
    };

    // ---------------------------------------------
    // archive wrapper around some arbitrary buffer
    template <typename BufferType>
    struct rma_archive {
        using type = detail::Archive<BufferType>;
    };

    template <typename BufferType>
    using rma_archive_type = rma_archive<BufferType>::type;

#ifdef IPPL_SIMPLE_VIEW_STORAGE
    template <typename... Properties>
    using archive_buffer = rma_archive_type<communicator_storage<Properties...>>;
#else
    template <typename... Properties>
    using archive_buffer = rma_archive_type<comm_storage_wrapper<Properties...>>;
#endif

    /**
     * @brief Interface for memory buffer handling.
     *
     * Defines methods for acquiring, freeing, and managing memory buffers.
     * Implementations are responsible for managing buffers efficiently,
     * ensuring that allocated buffers are reused where possible.
     *
     * @tparam MemorySpace The memory space type used for buffer allocation.
     */
    template <typename Buffer, typename MemorySpace>
    class BufferHandler {
    public:
        using archive_type = Buffer;
        using buffer_type  = std::shared_ptr<Buffer>;
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
    class DefaultBufferHandler : public BufferHandler<archive_buffer<MemorySpace>, MemorySpace> {
    public:
        using buffer_type =
            typename BufferHandler<archive_buffer<MemorySpace>, MemorySpace>::buffer_type;
        using typename BufferHandler<archive_buffer<MemorySpace>, MemorySpace>::archive_type;
        using typename BufferHandler<archive_buffer<MemorySpace>, MemorySpace>::size_type;

        ~DefaultBufferHandler() override;

        /**
         * @brief Acquires a buffer of at least the specified size.
         *
         * Requests a memory buffer of the specified size, with the option
         * to request a buffer larger than the base size by an overallocation
         * multiplier. If a sufficiently large buffer is available, it is returned. If not, the
         * largest free buffer is reallocated. If there are no free buffers available, only then
         * a new buffer is allocated.
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
        typename buffer_set_type::iterator findSmallestSufficientBuffer(size_type requiredSize);
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
}  // namespace ippl::comms

#include "Communicate/BufferHandler.hpp"

#endif
