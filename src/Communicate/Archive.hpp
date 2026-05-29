//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
#include <cstring>

#include "Archive.h"
#include "Utility/IpplException.h"

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace ippl {
    namespace detail {

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
                std::memcpy(buf + i * elem_size + wpos, view_data + hash(i), elem_size);
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
                std::memcpy(buf + (Dim * i + d) * elem_size + wpos, elem, elem_size);
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
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            size_t size    = sizeof(T);
            auto base      = bufferData();
            auto writepos  = writepos_m;
            Kokkos::parallel_for(
                "Archive::serialize()", policy_type(0, nsends), KOKKOS_LAMBDA(const size_type i) {
                    std::memcpy(base + i * size + writepos, view.data() + i, size);
                });
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

            size_t size    = sizeof(T);
            auto base      = bufferData();
            auto writepos  = writepos_m;
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
                    std::memcpy(base + (Dim * i + d) * size + writepos,
                                &(*(view.data() + i))[d], size);
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
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            size_t size = sizeof(T);
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            auto base    = bufferData();
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize()", policy_type(0, nrecvs), KOKKOS_LAMBDA(const size_type i) {
                    std::memcpy(view.data() + i, base + i * size + readpos, size);
                });
            // Wait for deserialization kernel to complete
            // (as with serialization kernels)
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

            size_t size = sizeof(T);
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            auto base    = bufferData();
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize()", mdrange_t({0, 0}, {static_cast<long>(nrecvs), Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    std::memcpy(&(*(view.data() + i))[d],
                                base + (Dim * i + d) * size + readpos, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }

        // =================================================================
        // Deserialize -- scalar with offset
        // =================================================================

        template <class... Properties>
        template <typename T, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<T*, ViewArgs...>& view,
                                                 size_type offset, size_type nrecvs) {
            using exec_space  = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;
            size_t size       = sizeof(T);
            if (offset + nrecvs > view.extent(0)) {
                Kokkos::resize(view, offset + nrecvs);
            }
            auto base    = bufferData();
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize(offset)", policy_type(0, nrecvs),
                KOKKOS_LAMBDA(const size_type i) {
                    std::memcpy(view.data() + offset + i, base + i * size + readpos, size);
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
            size_t size      = sizeof(T);
            if (offset + nrecvs > view.extent(0)) {
                Kokkos::resize(view, offset + nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            auto base    = bufferData();
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize(offset, vector)", mdrange_t({0, 0}, {static_cast<long>(nrecvs), Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    std::memcpy(&(*(view.data() + offset + i))[d],
                                base + (Dim * i + d) * size + readpos, size);
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }
    }  // namespace detail
}  // namespace ippl
