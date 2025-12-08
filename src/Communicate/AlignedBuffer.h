#ifndef IPPL_ALIGNED_BUFFER_H
#define IPPL_ALIGNED_BUFFER_H

#include "Types/ViewTypes.h"

#include "Utility/Logging.h"
#include "Utility/TypeUtils.h"
//
#include "Communicate/Archive.h"

namespace ippl::comms {

#define DEFAULT_BUFFER_ALIGNMENT 1024
    // Here's a simple class that provides an aligned buffer, by default on the host
    // but we can specialize the constructor/destructor for other memory spaces
    template <typename MemorySpace = Kokkos::HostSpace>
    struct AlignedBuffer {
        using memory_space = MemorySpace;
        void* ptrOriginal{nullptr};
        void* ptrAligned{nullptr};
        detail::size_type space{0};
        //
        AlignedBuffer() {}
        //
        AlignedBuffer& operator=(AlignedBuffer&& other) {
            ptrOriginal       = other.ptrOriginal;
            ptrAligned        = other.ptrAligned;
            space             = other.space;
            other.ptrOriginal = nullptr;
            other.ptrAligned  = nullptr;
            other.space       = 0;
            return *this;
        }
        //
        AlignedBuffer(std::size_t size) {
            ptrOriginal = std::aligned_alloc(DEFAULT_BUFFER_ALIGNMENT, size);
            ptrAligned  = ptrOriginal;
            space       = size;
            SPDLOG_TRACE("AlignedBuffer: original {}, aligned {}, size {}, space {}",
                         (void*)(ptrOriginal), (void*)(ptrAligned), size, space);
            // sanity check should always be true when std::align used
            assert(space >= size);
        }
        //
        ~AlignedBuffer() {
            if (ptrOriginal) {
                SPDLOG_DEBUG("Destroying host buffer {}", ptrOriginal);
                std::free(ptrOriginal);
            }
        }
    };

    // ---------------------------------------------------------------------
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
    // make number a multiple of the alignment
    inline std::int64_t to_multiple(std::int64_t num) {
        return ((2 * num + (DEFAULT_BUFFER_ALIGNMENT - 1)) & (-DEFAULT_BUFFER_ALIGNMENT));
    }
#endif

    // ---------------------------------------------------------------------
#ifdef KOKKOS_ENABLE_CUDA
    // Specialize buffer allocation/free for cuda
    template <>
    inline AlignedBuffer<Kokkos::CudaSpace>::AlignedBuffer(std::size_t size) {
        void* original;
        space = to_multiple(size);
        cudaMalloc(&original, space);
        if (!original) {
            throw std::runtime_error("Error allocating cuda memory in AlignedBuffer");
        }
        ptrOriginal = original;
        ptrAligned  = std::align(DEFAULT_BUFFER_ALIGNMENT, size, original, space);
        SPDLOG_TRACE("AlignedBuffer: original {}, aligned {}, size {}, space {}",
                     (void*)(ptrOriginal), (void*)(ptrAligned), size, space);
        // sanity check should always be true when std::align used
        assert(space >= size);
    }
    //
    template <>
    inline AlignedBuffer<Kokkos::CudaSpace>::~AlignedBuffer() {
        if (ptrOriginal) {
            SPDLOG_DEBUG("Destroying cuda buffer {}", ptrOriginal);
            cudaFree(ptrOriginal);
        }
    }
#endif

    // ---------------------------------------------------------------------
#ifdef KOKKOS_ENABLE_HIP
#define HIP_CHECK(expression)                                                                  \
    {                                                                                          \
        const hipError_t status = expression;                                                  \
        if (status != hipSuccess) {                                                            \
            std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;                             \
        }                                                                                      \
    }

    // Specialize buffer allocation/free for HIP
    template <>
    inline AlignedBuffer<Kokkos::HIPSpace>::AlignedBuffer(std::size_t size) {
        void* original;
        space = to_multiple(size);
        HIP_CHECK(hipMalloc(&original, space));
        if (!original) {
            throw std::runtime_error("Error allocating HIP memory in AlignedBuffer");
        }
        ptrOriginal = original;
        ptrAligned  = std::align(DEFAULT_BUFFER_ALIGNMENT, size, original, space);
        SPDLOG_TRACE("AlignedBuffer: original {}, aligned {}, size {}, space {}",
                     (void*)(ptrOriginal), (void*)(ptrAligned), size, space);
        // sanity check should always be true when std::align used
        assert(space >= size);
    }
    //
    template <>
    inline AlignedBuffer<Kokkos::HIPSpace>::~AlignedBuffer() {
        if (ptrOriginal) {
            SPDLOG_DEBUG("Destroying HIP buffer {}", ptrOriginal);
            HIP_CHECK(hipFree(ptrOriginal));
        }
    }
#endif

    // ---------------------------------------------------------------------
    template <typename MemorySpace, typename... Properties>
    struct aligned_storage_wrapper {
        //
        using memory_space = MemorySpace;
        using buffer_type =
            ippl::detail::ViewType<char, 1, MemorySpace, Properties...,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>::view_type;
        using pointer_type = typename buffer_type::pointer_type;
        using size_type    = detail::size_type;
        //
        aligned_storage_wrapper(const std::string& /*name*/, size_type size)
            : view()        // we will construct the view manually
            , buffer(size)  //
        {
            SPDLOG_TRACE("Construct: view  origin {}, aligned {}", (void*)(view.data()),
                         (void*)(buffer.ptrAligned));
            view = buffer_type((pointer_type)buffer.ptrAligned, size);
            assert(view.data() == buffer.ptrAligned);
        }
        //
        size_type size() const { return buffer.space; }
        //
        pointer_type data() { return view.data(); }

        // Note that this makes no effort to preserve any existing data
        void reallocBuffer(size_type newsize) {
            // wipe the old memory, before allocating new, (help prevent out-of-space errors)
            buffer = AlignedBuffer<memory_space>();
            // allocate new
            buffer = AlignedBuffer<memory_space>(newsize);
            view   = buffer_type((pointer_type)buffer.ptrAligned, newsize);
            SPDLOG_DEBUG("Realloc  : view {}, aligned {}, size {}, space {}", (void*)(view.data()),
                         (void*)(buffer.ptrAligned), newsize, buffer.space);
        }
        //
        buffer_type view;
        AlignedBuffer<memory_space> buffer;
    };

}  // namespace ippl::comms

#endif
