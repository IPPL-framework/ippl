/*!
 * @file KokkosDLPack.h
 * @brief Zero-copy export of Kokkos::View memory as DLPack capsules.
 *
 * Exposes Kokkos::View memory to any DLPack consumer (PyTorch, CuPy, JAX,
 * NumPy, ...) as a py::capsule, without copying.
 * Tested by unit_tests/Interoperability/DLPackExport.
 */
#ifndef IPPL_KOKKOS_DLPACK_H
#define IPPL_KOKKOS_DLPACK_H

#include "dlpack/dlpack.h"
#include <pybind11/pybind11.h>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace ippl {
    namespace Interoperability {

        namespace py = pybind11;

        /*!
         * @brief Whether the consumer may write into the exported memory.
         *
         * Only DLPackVersion::Versioned capsules can carry the read-only flag. For
         * DLPackVersion::Legacy capsules it is ignored and consumers always treat the memory 
         * as writable, so it is not enforced there.
         */
        enum class ReadOnly { No, Yes };

        /*!
         * @brief Which DLPack capsule format to produce.
         *
         * Legacy produces a "dltensor" capsule (DLManagedTensor), accepted by all consumers, but
         * is deprecated. 
         * Versioned produces a "dltensor_versioned" capsule
         * (DLManagedTensorVersioned, DLPack >= 1.0), which carries the version and the
         * read-only flag, but is only accepted by newer consumers.
         */
        enum class DLPackVersion { Legacy, Versioned };

        namespace detail {

            // static_assert(false) is ill-formed in C++20
            template <typename T>
            struct always_false : std::false_type {};

            // maps a C++ scalar type to its DLDataType
            template <typename T>
            inline DLDataType dtypeOf() {
                DLDataType dt;
                dt.lanes = 1;
                dt.bits  = static_cast<uint8_t>(sizeof(T) * 8);
                if constexpr (std::is_same_v<T, bool>) {
                    dt.code = kDLBool;
                } else if constexpr (std::is_floating_point_v<T>) {
                    dt.code = kDLFloat;
                } else if constexpr (std::is_signed_v<T>) {
                    dt.code = kDLInt;
                } else if constexpr (std::is_unsigned_v<T>) {
                    dt.code = kDLUInt;
                } else {
                    static_assert(always_false<T>::value, "unsupported scalar type for DLPack export");
                }
                return dt;
            }

            // map Kokkos memory space to DLDevice
            //
            // Kokkos::device_id() gives us either the device that the DefaultExecutionSpace is
            // bound to, or -1 if only host backends are enabled.  By querying, the binding is
            // whatever Kokkos' map_device_id_by strategy chose from the MPI local rank.
            template <typename MemorySpace>
            inline DLDevice deviceOf() {
                DLDevice dev;
                if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
                    dev.device_type = kDLCPU;
                    dev.device_id   = 0;
                }
#ifdef KOKKOS_ENABLE_CUDA
                else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>) {
                    dev.device_type = kDLCUDA;
                    dev.device_id   = Kokkos::device_id();
                } else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaUVMSpace>) {
                    dev.device_type = kDLCUDAManaged;
                    dev.device_id   = Kokkos::device_id();
                } else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaHostPinnedSpace>) {
                    dev.device_type = kDLCUDAHost;
                    dev.device_id   = 0;
                }
#endif
#ifdef KOKKOS_ENABLE_HIP
                else if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>) {
                    dev.device_type = kDLROCM;
                    dev.device_id   = Kokkos::device_id();
                }
#endif
                else {
                    static_assert(always_false<MemorySpace>::value,
                              "KokkosDLPack: no DLPack device mapping for this Kokkos memory space");
                }
                return dev;
            }

            // The owning context carried by the DLManagedTensor or DLManagedTensorVersioned.
            // If it holds a copy of the View (not the raw pointer) it bumps the Kokkos
            // reference count, so the allocation cant be freed while Python holds
            // the capsule.  shape/strides must outlive the DLTensor too, which is why the
            // vectors live here and not on the stack.
            template <typename ViewType>
            struct ManagerCtx {
                ViewType view_m;
                std::vector<int64_t> shape_m;
                std::vector<int64_t> strides_m;  // in elements, not bytes
            };

            // Deletes the view copy, allowing it to be freed (if not used elsewhere).
            template <typename ManagedTensorType, typename ViewType>
            void ctxDeleter(ManagedTensorType* self) {
                delete static_cast<ManagerCtx<ViewType>*>(self->manager_ctx);
                delete self;
            }

            // builds the managed tensor and its capsule
            template <typename ManagedT, typename ScalarT, typename ViewType>
            py::capsule exportRaw(ViewType view, ScalarT* scalarPtr,
                                std::vector<int64_t> shape, std::vector<int64_t> strides,
                                ReadOnly readOnly) {
                using MemorySpace = typename ViewType::memory_space;

                DLDevice device  = deviceOf<MemorySpace>();
                DLDataType dtype = dtypeOf<ScalarT>();

                auto* ctx     = new ManagerCtx<ViewType>{view, std::move(shape), std::move(strides)};
                auto* managed = new ManagedT();

                managed->dl_tensor.data        = static_cast<void*>(scalarPtr);
                managed->dl_tensor.device      = device;
                managed->dl_tensor.ndim        = static_cast<int32_t>(ctx->shape_m.size());
                managed->dl_tensor.dtype       = dtype;
                managed->dl_tensor.shape       = ctx->shape_m.data();
                managed->dl_tensor.strides     = ctx->strides_m.data();
                managed->dl_tensor.byte_offset = 0;
                managed->manager_ctx           = ctx;
                managed->deleter               = &ctxDeleter<ManagedT, ViewType>;

                if constexpr (std::is_same_v<ManagedT, DLManagedTensorVersioned>) {
                    managed->version.major = DLPACK_MAJOR_VERSION;
                    managed->version.minor = DLPACK_MINOR_VERSION;
                    managed->flags = (readOnly == ReadOnly::Yes) ? DLPACK_FLAG_BITMASK_READ_ONLY : 0;
                }

                constexpr const char* NAME = std::is_same_v<ManagedT, DLManagedTensorVersioned>
                                                ? "dltensor_versioned" : "dltensor";

                //capsule name is "dltensor" or "dltensor_versioned", consumers check it, then rename to
                // "used_<name>" once they have taken ownership.  The capsule
                // destructor below only fires if Python never consumed it.
                return py::capsule(managed, NAME, [](PyObject* obj) {
                    if (PyCapsule_IsValid(obj, NAME)) {
                        auto* p = static_cast<ManagedT*>(PyCapsule_GetPointer(obj, NAME));
                        if (p && p->deleter) {
                            p->deleter(p);
                        }
                    }
                });
            }

        }  // namespace detail

        /*!
         * @brief Exports Kokkos View memory as a DLPack capsule with caller-defined layout.
         *
         * Describes the memory starting at scalarPtr as a tensor of type ScalarT with the given
         * shape and strides, in the View's memory space (host or device), without copying.
         * This way a View of structs (e.g. ippl::Vector<double, 3>) can be exported as a
         * dense array of its scalars. The capsule holds a copy of the View, so the allocation
         * stays alive until the consumer releases the tensor.
         *
         * The call is rank-local. It does not fence, and kernels writing the memory must
         * finish before the consumer reads it.
         *
         * @tparam ScalarT element type of the tensor (bool, integer or floating point)
         * @tparam ViewType Kokkos::View owning the memory; its memory space sets the device
         * @param view the View to keep alive; only its reference count is used
         * @param scalarPtr first element of the tensor, inside the View's allocation
         * @param shape extent of each tensor dimension
         * @param strides stride of each dimension, in elements
         * @param readOnly marks the tensor read-only, only honoured by Versioned capsules
         * @param version capsule format to produce
         * @returns a "dltensor" or "dltensor_versioned" capsule for e.g. torch.from_dlpack
         */
        template <typename ScalarT, typename ViewType>
        py::capsule exportTensor(ViewType view, ScalarT* scalarPtr,
                                std::vector<int64_t> shape, std::vector<int64_t> strides,
                                ReadOnly readOnly, DLPackVersion version) {
            if (version == DLPackVersion::Versioned) {
                return detail::exportRaw<DLManagedTensorVersioned>(view, scalarPtr, std::move(shape),
                                                    std::move(strides), readOnly);
            } else {
                return detail::exportRaw<DLManagedTensor>(view, scalarPtr, std::move(shape),
                                        std::move(strides), readOnly);
            }
        }

        /*!
         * @brief Exports a Kokkos View of plain scalars as a DLPack capsul eas-is.
         *
         * The tensor has the View's rank, extents and strides, and aliases its memory in the
         * View's memory space (host or device). Has the same lifetime and synchronization rules
         * as exportTensor() (rank-local, no fence).
         *
         * @tparam ViewType Kokkos::View whose value_type is bool, integer or floating point
         * @param view the View to export
         * @param readOnly marks the tensor read-only; only honoured by Versioned capsules
         * @param version capsule format to produce
         * @returns a "dltensor" or "dltensor_versioned" capsule for e.g. torch.from_dlpack
         */
        template <typename ViewType>
        py::capsule toDLPack(ViewType view, ReadOnly readOnly, DLPackVersion version) {
            using ScalarT             = typename ViewType::value_type;
            constexpr int RANK        = static_cast<int>(ViewType::rank);
            std::vector<int64_t> shape(RANK), strides(RANK);
            for (int i = 0; i < RANK; ++i) {
                shape[i]   = static_cast<int64_t>(view.extent(i));
                strides[i] = static_cast<int64_t>(view.stride(i));
            }
            return exportTensor<ScalarT>(view, view.data(), std::move(shape), std::move(strides),
                                        readOnly, version);
        }

    }  // namespace Interoperability
}  // namespace ippl

#endif
