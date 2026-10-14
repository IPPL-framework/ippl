// Generic bridge that exposes Kokkos::View memory to any
//DLPack consumer as a py::capsule
//
// Independent of IPPL: includes no IPPL headers, only shares the ippl namespace.
//
// Requires dlpack/dlpack.h and pybind/pybind11 (https://github.com/dmlc/dlpack)

#ifndef KOKKOS_DLPACK_H
#define KOKKOS_DLPACK_H

#include "dlpack/dlpack.h"
#include <pybind11/pybind11.h>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ippl {
    namespace Interoperability {

        namespace py = pybind11;

        enum class ReadOnly { No, Yes };
        enum class DLPackVersion { Legacy, Versioned };
        //enum class ComponentAxis { Last, First }; //is it shape (N,Dim) or (Dim,N)

        namespace detail {

            //static_assert(false) is ill-formed in C++20
            template <typename T>
            struct always_false : std::false_type {};

            // map C++ scalar type to DLDataType
            // code - uint8_t type code of base types, value should be one of DLDataTypeCode enum values
            // bits - number of bits of datatype
            // lanes - number of lanes in the type, for vector types (like float4)
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
                ViewType view_m; // this holds the refcount
                std::vector<int64_t> shape_m;
                std::vector<int64_t> strides_m;  // strides are in ELEMENTS, not bytes (DLPack convention)!!
            };

            // Only when this is called is the view copy deleted, and the view itself able to
            // potentially be freeed (if not used elsewhere).
            template <typename ManagedTensorType, typename ViewType>
            void ctxDeleter(ManagedTensorType* self) {
                delete static_cast<ManagerCtx<ViewType>*>(self->manager_ctx);
                delete self;
            }

            // The core export.
            //
            // scalarPtr, shape and strides are given so callers can
            // reinterpret a View of small structs (like ippl::Vector<double,3>) as a
            // dense array of the underlying scalar. view` is only used to keep the
            // allocation alive (refcount bump).
            // Strides are in elements, not bytes.
            template <typename ManagedT, typename ScalarT, typename ViewType>
            py::capsule exportRaw(ViewType view, ScalarT* scalarPtr,
                                std::vector<int64_t> shape, std::vector<int64_t> strides,
                                ReadOnly readOnly) {
                using MemorySpace = typename ViewType::memory_space;

                DLDevice device  = deviceOf<MemorySpace>();
                DLDataType dtype = dtypeOf<ScalarT>();

                auto* ctx     = new ManagerCtx<ViewType>{view, std::move(shape), std::move(strides)};
                auto* managed = new ManagedT();

                // DLTensor is the plain C Tensor object
                managed->dl_tensor.data        = static_cast<void*>(scalarPtr); // Data ptr to allocated data
                managed->dl_tensor.device      = device;
                managed->dl_tensor.ndim        = static_cast<int32_t>(ctx->shape_m.size());
                managed->dl_tensor.dtype       = dtype; //Data type of the pointer
                managed->dl_tensor.shape       = ctx->shape_m.data();
                managed->dl_tensor.strides     = ctx->strides_m.data();
                managed->dl_tensor.byte_offset = 0; // offset in struct to the beginning pointer to data in bytes
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

        // the single public entry point, defers down to version or legacy dltensor depending on
        // what is supported
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

        // exports a kokkos view of plain scalars as is
        template <typename ViewType>
        py::capsule toDLPack(ViewType view, ReadOnly readOnly, DLPackVersion version) {
            using ScalarT             = typename ViewType::value_type;
            constexpr int RANK        = static_cast<int>(ViewType::rank);
            std::vector<int64_t> shape(RANK), strides(RANK);
            for (int i = 0; i < RANK; ++i) {
                shape[i]   = static_cast<int64_t>(view.extent(i));
                strides[i] = static_cast<int64_t>(view.stride(i));  // Kokkos strides are in elements
            }
            return exportTensor<ScalarT>(view, view.data(), std::move(shape), std::move(strides),
                                        readOnly, version);
        }

    }  // namespace Interoperability
}  // namespace ippl

#endif
