//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
#include "Archive.h"
#include "Utility/IpplException.h"

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace ippl {
    namespace detail {
        KOKKOS_INLINE_FUNCTION void copyBytes(char* dst, const char* src, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                dst[i] = src[i];
            }
        }

#if defined(KOKKOS_ENABLE_CUDA)
        inline void* archiveDeviceAlloc(size_t size) {
            void* ptr        = nullptr;
            cudaError_t rc   = cudaMalloc(&ptr, size);
            if (rc != cudaSuccess) {
                throw IpplException(
                    "Archive::gpuAlloc",
                    std::string("cudaMalloc(") + std::to_string(size)
                        + " bytes) failed: " + cudaGetErrorString(rc));
            }
            return ptr;
        }
        inline void archiveDeviceFree(void* ptr) {
            if (!ptr) return;
            cudaError_t rc = cudaFree(ptr);
            if (rc != cudaSuccess) {
                throw IpplException("Archive::gpuFree",
                                    std::string("cudaFree failed: ") + cudaGetErrorString(rc));
            }
        }
        inline void archiveDeviceCopy(void* dst, const void* src, size_t bytes) {
            cudaError_t rc = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
            if (rc != cudaSuccess) {
                throw IpplException(
                    "Archive::resizeBuffer",
                    std::string("cudaMemcpy(D2D) failed: ") + cudaGetErrorString(rc));
            }
        }
#elif defined(KOKKOS_ENABLE_HIP)
        inline void* archiveDeviceAlloc(size_t size) {
            void* ptr      = nullptr;
            hipError_t rc  = hipMalloc(&ptr, size);
            if (rc != hipSuccess) {
                throw IpplException(
                    "Archive::gpuAlloc",
                    std::string("hipMalloc(") + std::to_string(size)
                        + " bytes) failed: " + hipGetErrorString(rc));
            }
            return ptr;
        }
        inline void archiveDeviceFree(void* ptr) {
            if (!ptr) return;
            hipError_t rc = hipFree(ptr);
            if (rc != hipSuccess) {
                throw IpplException("Archive::gpuFree",
                                    std::string("hipFree failed: ") + hipGetErrorString(rc));
            }
        }
        inline void archiveDeviceCopy(void* dst, const void* src, size_t bytes) {
            hipError_t rc = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
            if (rc != hipSuccess) {
                throw IpplException(
                    "Archive::resizeBuffer",
                    std::string("hipMemcpy(D2D) failed: ") + hipGetErrorString(rc));
            }
        }
#endif

        template <typename T, typename HashView, typename BufferPtr>
        struct SerializeHashFunctor {
            const T* view_data;
            HashView hash;
            BufferPtr buf;
            size_t elem_size;
            size_t wpos;

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                // detail::copyBytes (PR #532): byte-loop avoids invoking
                // std::memcpy from a Kokkos device kernel.
                const char* src = reinterpret_cast<const char*>(view_data + hash(i));
                char* dst       = reinterpret_cast<char*>(buf) + i * elem_size + wpos;
                copyBytes(dst, src, elem_size);
            }
        };

        template <typename T, unsigned Dim, typename HashView, typename BufferPtr>
        struct SerializeHashVectorFunctor {
            const Vector<T, Dim>* view_data;
            HashView hash;
            BufferPtr buf;
            size_t elem_size;
            size_t wpos;

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i, const size_t d) const {
                const Vector<T, Dim>* vec = view_data + hash(i);
                const T* elem             = reinterpret_cast<const T*>(vec) + d;
                char* dst                 = reinterpret_cast<char*>(buf) + (Dim * i + d) * elem_size + wpos;
                copyBytes(dst, reinterpret_cast<const char*>(elem), elem_size);
            }
        };

        // =================================================================
        // Buffer management
        // =================================================================
        //
        // Two storage paths:
        //   * Host-accessible memory spaces (HostSpace, OpenMP, Serial, ...):
        //     a regular Kokkos::View<char*, MemorySpace> in `buffer_m`.
        //   * Device memory spaces (CudaSpace, HIPSpace): raw cuda/hipMalloc
        //     in `buffer_ptr_m`.

        template <class... Properties>
        void Archive<Properties...>::gpuAlloc(size_type size) {
            if (!uses_raw_device_alloc || size == 0) return;
#if defined(KOKKOS_ENABLE_HIP)
            // HSA IPC likes allocation sizes to be multiples of the GPU page
            // granularity (64 KB on MI250X / MI300X).
            static constexpr size_type kGranularity = 65536;
            size = ((size + kGranularity - 1) / kGranularity) * kGranularity;
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            buffer_ptr_m  = static_cast<pointer_type>(archiveDeviceAlloc(size));
            buffer_size_m = size;
#endif
        }

        template <class... Properties>
        void Archive<Properties...>::gpuFree() {
            if (!uses_raw_device_alloc || !buffer_ptr_m) return;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            archiveDeviceFree(buffer_ptr_m);
#endif
            buffer_ptr_m  = nullptr;
            buffer_size_m = 0;
        }

        template <class... Properties>
        Archive<Properties...>::Archive(size_type size)
            : writepos_m(0)
            , readpos_m(0) {
            if constexpr (uses_raw_device_alloc) {
                gpuAlloc(size);
            } else {
                buffer_m = buffer_type("buffer", size);
            }
        }

        template <class... Properties>
        Archive<Properties...>::~Archive() {
            if constexpr (uses_raw_device_alloc) {
                gpuFree();
            }
        }

        template <class... Properties>
        void Archive<Properties...>::resizeBuffer(size_type size) {
            if constexpr (uses_raw_device_alloc) {
                if (size <= buffer_size_m) return;
#if defined(KOKKOS_ENABLE_HIP)
                static constexpr size_type kGranularity = 65536;
                size = ((size + kGranularity - 1) / kGranularity) * kGranularity;
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                pointer_type new_ptr =
                    static_cast<pointer_type>(archiveDeviceAlloc(size));

                if (buffer_ptr_m && buffer_size_m > 0) {
                    archiveDeviceCopy(new_ptr, buffer_ptr_m, buffer_size_m);
                    archiveDeviceFree(buffer_ptr_m);
                }

                buffer_ptr_m  = new_ptr;
                buffer_size_m = size;
#endif
            } else {
                Kokkos::resize(buffer_m, size);
            }
        }

        template <class... Properties>
        void Archive<Properties...>::reallocBuffer(size_type size) {
            // Reallocation discards any data that may have been written into
            // the buffer; reset read/write positions so the next caller sees
            // a fresh archive.
            writepos_m = 0;
            readpos_m  = 0;
            if constexpr (uses_raw_device_alloc) {
                gpuFree();
                gpuAlloc(size);
            } else {
                Kokkos::realloc(buffer_m, size);
            }
        }

        // =================================================================
        // Serialize -- scalar
        // =================================================================

        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::serialize(const Kokkos::View<T*, ViewArgs...>& view,
                                               size_type nsends) {
            // Take main/master's Kokkos::deep_copy-over-Unmanaged-View idiom
            // (PR #532) so no std::memcpy is called from a KOKKOS_LAMBDA. The
            // *destination* pointer still has to go through bufferData() /
            // bufferSize() because this branch keeps the raw cuda/hipMalloc
            // path for device-only Archives (the buffer_m view is empty when
            // uses_raw_device_alloc is true).
            constexpr size_t size = sizeof(T);
            char* dst_ptr         = reinterpret_cast<char*>(bufferData()) + writepos_m;
            char* src_ptr         = reinterpret_cast<char*>(const_cast<T*>(view.data()));
            assert(writepos_m + (nsends * size) <= bufferSize());
            // construct temp views of the src/dst buffers of the correct size (bytes)
            using src_view_type =
                Kokkos::View<char*, typename Kokkos::View<T*, ViewArgs...>::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            using dst_view_type =
                Kokkos::View<char*, memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            src_view_type src_view(src_ptr, size * nsends);
            dst_view_type dst_view(dst_ptr, size * nsends);
            Kokkos::deep_copy(dst_view, src_view);
            Kokkos::fence();
            writepos_m += size * nsends;
        }

        // =================================================================
        // Serialize -- scalar with hash
        // =================================================================

        template <class... Properties>
        template <typename T, class... ViewArgs, typename HashView>
        void Archive<Properties...>::serialize(const Kokkos::View<T*, ViewArgs...>& view,
                                               const HashView& hash, size_type nsends) {
            using exec_space  = HashView::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;
            using BufferPtr   = pointer_type;

            SerializeHashFunctor<T, HashView, BufferPtr> f{view.data(), hash, bufferData(),
                                                           sizeof(T), writepos_m};

            Kokkos::parallel_for("Archive::serialize(hash)", policy_type(0, nsends), f);
            Kokkos::fence();
            writepos_m += sizeof(T) * nsends;
        }

        // =================================================================
        // Serialize -- vector
        // =================================================================

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::serialize(
            const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view, size_type nsends) {
            using exec_space =
                typename Kokkos::View<Vector<T, Dim>*, ViewArgs...>::execution_space;

            // Capture raw pointers + bufferData() so the kernel calls
            // detail::copyBytes (PR #532) instead of std::memcpy while still
            // honouring the raw-device-alloc path that pif-pr added.
            constexpr size_t size           = sizeof(T);
            char* dst_ptr                   = reinterpret_cast<char*>(bufferData());
            ippl::Vector<T, Dim>* src_ptr   = const_cast<ippl::Vector<T, Dim>*>(view.data());
            const size_type wp              = writepos_m;
            // Default index type for range policies is int64,
            // so we have to explicitly specify size_type (uint64)
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            Kokkos::parallel_for(
                "Archive::serialize()",
                // The constructor for Kokkos range policies always
                // expects int64 regardless of index type provided
                // by template parameters, so the typecast is necessary
                // to avoid compiler warnings
                mdrange_t({0, 0}, {static_cast<long>(nsends), Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    const char* src = reinterpret_cast<const char*>(&src_ptr[i][d]);
                    char* dst       = dst_ptr + (Dim * i + d) * size + wp;
                    copyBytes(dst, src, size);
                });
            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        // =================================================================
        // Serialize -- vector with hash
        // =================================================================

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs, typename HashView>
        void Archive<Properties...>::serialize(
            const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view, const HashView& hash,
            size_type nsends) {
            using exec_space = typename HashView::execution_space;
            size_t size      = sizeof(T);
            using BufferPtr  = pointer_type;
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;

            SerializeHashVectorFunctor<T, Dim, HashView, BufferPtr> f{
                view.data(), hash, bufferData(), size, writepos_m};

            Kokkos::parallel_for("Archive::serialize(hash, vector)",
                                 mdrange_t({0, 0}, {static_cast<long>(nsends), Dim}), f);
            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        // =================================================================
        // Deserialize -- scalar
        // =================================================================

        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<T*, ViewArgs...>& view,
                                                 size_type nrecvs) {
            constexpr size_t size = sizeof(T);
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            // Same Kokkos::deep_copy-over-Unmanaged-View pattern as serialize()
            // (PR #532), going through bufferData() / bufferSize() so the
            // raw-device-alloc Archive variant works.
            char* src_ptr         = reinterpret_cast<char*>(bufferData()) + readpos_m;
            char* dst_ptr         = reinterpret_cast<char*>(view.data());
            assert(readpos_m + (nrecvs * size) <= bufferSize());
            // construct temp views of the src/dst buffers of the correct size (bytes)
            using src_view_type =
                Kokkos::View<char*, memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            using dst_view_type =
                Kokkos::View<char*, typename Kokkos::View<T*, ViewArgs...>::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            src_view_type src_view(src_ptr, size * nrecvs);
            dst_view_type dst_view(dst_ptr, size * nrecvs);
            Kokkos::deep_copy(dst_view, src_view);
            Kokkos::fence();
            readpos_m += size * nrecvs;
        }

        // =================================================================
        // Deserialize -- vector
        // =================================================================

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                                 size_type nrecvs) {
            using exec_space =
                typename Kokkos::View<Vector<T, Dim>*, ViewArgs...>::execution_space;

            constexpr size_t size = sizeof(T);
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            char* src_ptr                  = reinterpret_cast<char*>(bufferData());
            ippl::Vector<T, Dim>* dst_ptr  = view.data();
            const size_type rp             = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize()", mdrange_t({0, 0}, {static_cast<long>(nrecvs), Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    const char* src = src_ptr + (Dim * i + d) * size + rp;
                    char* dst       = reinterpret_cast<char*>(&dst_ptr[i][d]);
                    copyBytes(dst, src, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }

        // =================================================================
        // Deserialize -- scalar with offset
        // =================================================================
        //
        // Offset variants are kept on the pif-pr branch -- ParticleAttrib uses
        // them to deserialize into a sub-range of dview_m for incremental
        // particle migration buffer reads. main/master removed them in PR #532
        // because its ParticleBase deserialises particles in one shot, but the
        // pif-pr Particle refactor (2878b90b) still needs the offset path.

        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<T*, ViewArgs...>& view,
                                                 size_type offset, size_type nrecvs) {
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;
            constexpr size_t size = sizeof(T);
            if (offset + nrecvs > view.extent(0)) {
                Kokkos::resize(view, offset + nrecvs);
            }
            char* src_ptr   = reinterpret_cast<char*>(bufferData());
            T* dst_ptr      = view.data() + offset;
            const size_type rp = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize(offset)", policy_type(0, nrecvs),
                KOKKOS_LAMBDA(const size_type i) {
                    const char* src = src_ptr + i * size + rp;
                    char* dst       = reinterpret_cast<char*>(dst_ptr + i);
                    copyBytes(dst, src, size);
                });
            Kokkos::fence();
            readpos_m += size * nrecvs;
        }

        // =================================================================
        // Deserialize -- vector with offset
        // =================================================================

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                                 size_type offset, size_type nrecvs) {
            using exec_space =
                typename Kokkos::View<Vector<T, Dim>*, ViewArgs...>::execution_space;
            constexpr size_t size = sizeof(T);
            if (offset + nrecvs > view.extent(0)) {
                Kokkos::resize(view, offset + nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            char* src_ptr                  = reinterpret_cast<char*>(bufferData());
            ippl::Vector<T, Dim>* dst_ptr  = view.data() + offset;
            const size_type rp             = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize(offset, vector)", mdrange_t({0, 0}, {static_cast<long>(nrecvs), Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    const char* src = src_ptr + (Dim * i + d) * size + rp;
                    char* dst       = reinterpret_cast<char*>(&dst_ptr[i][d]);
                    copyBytes(dst, src, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }
    }  // namespace detail
}  // namespace ippl
