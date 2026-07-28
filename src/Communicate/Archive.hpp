//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
#include "Archive.h"

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

        /*
         * Packs a sparse selection of scalar particle-attribute entries into
         * the archive buffer. `hash(i)` gives the source index in `view_data`
         * for the i-th item being sent; the functor copies that value as raw
         * bytes to `buf + wpos + i * sizeof(T)`. This is used by
         * Archive::serialize(view, hash, nsends) during particle migration,
         * where only particles listed in the send hash are serialized.
         */
        template <typename T, typename HashView, typename BufferPtr>
        struct SerializeHashFunctor {
            const T* view_data;
            HashView hash;
            BufferPtr buf;
            size_t elem_size;
            size_t wpos;

            KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
                const char* src = reinterpret_cast<const char*>(view_data + hash(i));
                char* dst       = buf + i * elem_size + wpos;
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
                const T value             = (*vec)[d];
                const char* src           = reinterpret_cast<const char*>(&value);
                char* dst                 = buf + (Dim * i + d) * elem_size + wpos;
                copyBytes(dst, src, elem_size);
            }
        };

        // =================================================================
        // GPU buffer management
        // =================================================================

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

        template <class... Properties>
        void Archive<Properties...>::gpuAlloc(size_type size) {
            if (size == 0) return;
#if defined(KOKKOS_ENABLE_HIP)
            // HSA IPC (used by Cray MPICH for large-message GPU transfers)
            // requires allocation sizes to be multiples of the GPU page
            // granularity (64 KB on MI250X / MI300X).  Without this,
            // hsa_amd_ipc_memory_attach fails with INVALID_ARGUMENT.
            static constexpr size_type kGranularity = 65536;
            size = ((size + kGranularity - 1) / kGranularity) * kGranularity;
#endif
            void* ptr = nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
            (void)cudaMalloc(&ptr, size);
#else
            (void)hipMalloc(&ptr, size);
#endif
            buffer_ptr_m  = static_cast<pointer_type>(ptr);
            buffer_size_m = size;
        }

        template <class... Properties>
        void Archive<Properties...>::gpuFree() {
            if (buffer_ptr_m) {
#if defined(KOKKOS_ENABLE_CUDA)
                (void)cudaFree(buffer_ptr_m);
#else
                (void)hipFree(buffer_ptr_m);
#endif
                buffer_ptr_m  = nullptr;
                buffer_size_m = 0;
            }
        }

        template <class... Properties>
        Archive<Properties...>::Archive(size_type size)
            : writepos_m(0)
            , readpos_m(0) {
            if constexpr (useRawGpuBuffer()) {
                gpuAlloc(size);
            } else {
                buffer_m = buffer_type("buffer", size);
            }
        }

        template <class... Properties>
        Archive<Properties...>::~Archive() {
            if constexpr (useRawGpuBuffer()) {
                gpuFree();
            }
        }

        template <class... Properties>
        void Archive<Properties...>::resizeBuffer(size_type size) {
            if constexpr (useRawGpuBuffer()) {
                if (size <= buffer_size_m) return;

#if defined(KOKKOS_ENABLE_HIP)
                static constexpr size_type kGranularity = 65536;
                size = ((size + kGranularity - 1) / kGranularity) * kGranularity;
#endif
                pointer_type new_ptr = nullptr;
                void* vptr           = nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
                (void)cudaMalloc(&vptr, size);
#else
                (void)hipMalloc(&vptr, size);
#endif
                new_ptr = static_cast<pointer_type>(vptr);

                if (buffer_ptr_m && buffer_size_m > 0) {
#if defined(KOKKOS_ENABLE_CUDA)
                    (void)cudaMemcpy(new_ptr, buffer_ptr_m, buffer_size_m, cudaMemcpyDeviceToDevice);
                    (void)cudaFree(buffer_ptr_m);
#else
                    (void)hipMemcpy(new_ptr, buffer_ptr_m, buffer_size_m, hipMemcpyDeviceToDevice);
                    (void)hipFree(buffer_ptr_m);
#endif
                }

                buffer_ptr_m  = new_ptr;
                buffer_size_m = size;
            } else {
                Kokkos::resize(buffer_m, size);
            }
        }

        template <class... Properties>
        void Archive<Properties...>::reallocBuffer(size_type size) {
            if constexpr (useRawGpuBuffer()) {
                gpuFree();
                gpuAlloc(size);
            } else {
                Kokkos::realloc(buffer_m, size);
            }
        }

#else  // CPU path

        template <class... Properties>
        Archive<Properties...>::Archive(size_type size)
            : writepos_m(0)
            , readpos_m(0)
            , buffer_m("buffer", size) {}

        template <class... Properties>
        Archive<Properties...>::~Archive() = default;

        template <class... Properties>
        void Archive<Properties...>::resizeBuffer(size_type size) {
            Kokkos::resize(buffer_m, size);
        }

        template <class... Properties>
        void Archive<Properties...>::reallocBuffer(size_type size) {
            Kokkos::realloc(buffer_m, size);
        }

#endif  // KOKKOS_ENABLE_CUDA || KOKKOS_ENABLE_HIP

        // =================================================================
        // Serialize — scalar
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
                    const char* src = reinterpret_cast<const char*>(view.data() + i);
                    char* dst       = base + i * size + writepos;
                    copyBytes(dst, src, size);
                });
            Kokkos::fence();
            writepos_m += size * nsends;
        }

        // =================================================================
        // Serialize — scalar with hash
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
        // Serialize — vector
        // =================================================================

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::serialize(
            const Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view, size_type nsends) {
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;

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
                mdrange_t({0, 0}, {(long int)nsends, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    const T value   = view.data()[i][d];
                    const char* src = reinterpret_cast<const char*>(&value);
                    char* dst       = base + (Dim * i + d) * size + writepos;
                    copyBytes(dst, src, size);
                });
            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        // =================================================================
        // Serialize — vector with hash
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
                                 mdrange_t({0, 0}, {(long int)nsends, Dim}), f);
            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        // =================================================================
        // Deserialize — scalar
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
                    const char* src = base + i * size + readpos;
                    char* dst       = reinterpret_cast<char*>(view.data() + i);
                    copyBytes(dst, src, size);
                });
            // Wait for deserialization kernel to complete
            // (as with serialization kernels)
            Kokkos::fence();
            readpos_m += size * nrecvs;
        }

        // =================================================================
        // Deserialize — vector
        // =================================================================

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                                 size_type nrecvs) {
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;

            size_t size = sizeof(T);
            if (nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            auto base    = bufferData();
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize()", mdrange_t({0, 0}, {(long int)nrecvs, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    const char* src = base + (Dim * i + d) * size + readpos;
                    T value;
                    char* dst = reinterpret_cast<char*>(&value);
                    copyBytes(dst, src, size);
                    view.data()[i](d) = value;
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }

        // =================================================================
        // Deserialize — scalar with offset
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
                    const char* src = base + i * size + readpos;
                    char* dst       = reinterpret_cast<char*>(view.data() + offset + i);
                    copyBytes(dst, src, size);
                });
            Kokkos::fence();
            readpos_m += size * nrecvs;
        }

        // =================================================================
        // Deserialize — vector with offset
        // =================================================================

        template <class... Properties>
        template <typename T, unsigned Dim, class... ViewArgs>
        void Archive<Properties...>::deserialize(Kokkos::View<Vector<T, Dim>*, ViewArgs...>& view,
                                                 size_type offset, size_type nrecvs) {
            using exec_space = typename Kokkos::View<T*, ViewArgs...>::execution_space;
            size_t size      = sizeof(T);
            if (offset + nrecvs > view.extent(0)) {
                Kokkos::resize(view, offset + nrecvs);
            }
            using mdrange_t =
                Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<size_type>, exec_space>;
            auto base    = bufferData();
            auto readpos = readpos_m;
            Kokkos::parallel_for(
                "Archive::deserialize(offset, vector)", mdrange_t({0, 0}, {(long int)nrecvs, Dim}),
                KOKKOS_LAMBDA(const size_type i, const size_t d) {
                    const char* src = base + (Dim * i + d) * size + readpos;
                    T value;
                    char* dst = reinterpret_cast<char*>(&value);
                    copyBytes(dst, src, size);
                    view.data()[offset + i](d) = value;
                });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }
    }  // namespace detail
}  // namespace ippl
