/*!
 * @file IpplDLPack.h
 * @brief Zero-copy export of ippl::ParticleAttrib data as DLPack capsules.
 *
 * Lets Python frameworks (e.g. PyTorch) read/write particle attributes in
 * place, on the attribute's device.
 *
 * Exports cover this rank's live particles. They do not fence, and the
 * exported tensor stays valid only until the particle container is next
 * updated, created into, destroyed from or resized - a re-export is necessary after that.
 * Python must release all the exported tensors before ippl::finalize().
 * Tested by unit_tests/Interoperability/DLPackExport.
 */
#ifndef IPPL_DLPACK_H
#define IPPL_DLPACK_H

#include "KokkosDLPack.h"

namespace ippl {
    namespace Interoperability {

        /*!
         * @brief Exports a vector-valued ParticleAttrib as a DLPack capsule.
         *
         * The tensor is of shape (localNum, Dim) and row-major strides,
         * and aliases the attribute's memory in its memory space
         * (host or device) without copying. Only this rank's live particles are exported,
         * and the call is rank-local, not collective.
         *
         * The export does not fence: kernels that write the attribute must have finished
         * before the consumer reads it, and the consumer must have
         * synchronized before IPPL uses the attribute again.
         *
         * If the element type is not a dense array of scalars, the data is repacked.
         *
         * @tparam Attrib ParticleAttrib whose value_type is an ippl::Vector<T, Dim>
         * @param attrib the attribute to export
         * @param readOnly marks the tensor read-only; only honoured by Versioned capsules
         * @param version capsule format
         * @returns a capsule for e.g. torch.from_dlpack
         */
        template <typename Attrib>
        py::capsule attribToDLPackVec(Attrib& attrib, ReadOnly readOnly, DLPackVersion version) {
            // retrieves the shape and type of the bare scalar type underlying a ParticleAttrib:
            using view_type       = typename Attrib::view_type;
            using vector_type     = typename view_type::value_type;
            using ScalarT         = typename vector_type::value_type;
            constexpr int64_t Dim = static_cast<int64_t>(vector_type::dim);

            static_assert(view_type::rank == 1, "ParticleAttrib is always rank-1");

            // gets attribute Kokkos View, trimmed to live particles and their count
            view_type v            = attrib.getView();
            const int64_t localNum = static_cast<int64_t>(v.extent(0));

            // checks to see if zero-copy is possible
            if constexpr (std::is_standard_layout_v<vector_type>
                          && sizeof(vector_type) == Dim * sizeof(ScalarT)) {
                // if yes, simply reinterpret as bare scalar type
                auto* scalarPtr = reinterpret_cast<ScalarT*>(v.data());

                std::vector<int64_t> shape, strides;
                shape   = {localNum, Dim};
                strides = {static_cast<int64_t>(v.stride(0)) * Dim, 1};

                return exportTensor<ScalarT>(v, scalarPtr, std::move(shape), std::move(strides),
                                             readOnly, version);
            } else {
                // else repack into a flat (localNum * Dim) scalar array
                // in the attribute's memory space
                Kokkos::View<ScalarT*, typename view_type::memory_space> flat(
                    Kokkos::view_alloc("dlpack_repack", Kokkos::WithoutInitializing),
                    localNum * Dim);

                Kokkos::parallel_for(
                    "ippl::Interoperability::repack",
                    Kokkos::RangePolicy<typename view_type::execution_space>(0, localNum),
                    KOKKOS_LAMBDA(const int64_t i) {
                        for (int64_t d = 0; d < Dim; ++d) {
                            flat(i * Dim + d) = v(i)[d];
                        }
                    });
                Kokkos::fence();

                return exportTensor<ScalarT>(flat, flat.data(), {localNum, Dim}, {Dim, 1}, readOnly,
                                             version);
            }
        }

        /*!
         * @brief Exports a scalar ParticleAttrib as a DLPack capsule.
         *
         * The tensor has shape (localNum) and aliases the attribute's memory in its memory
         * space (host or device) without copying. Same rules as attribToDLPackVec():
         * rank-local, no fence, valid until next particle update.
         *
         * @tparam Attrib ParticleAttrib whose value_type is bool, integer or floating point
         * @param attrib the attribute to export
         * @param readOnly marks the tensor read-only; only honoured by Versioned capsules
         * @param version capsule format; PyTorch <= 2.8 requires DLPackVersion::Legacy
         * @returns a capsule for e.g. torch.from_dlpack
         */
        template <typename Attrib>
        py::capsule attribToDLPackScalar(Attrib& attrib, ReadOnly readOnly, DLPackVersion version) {
            return toDLPack(attrib.getView(), readOnly, version);
        }

    }  // namespace Interoperability
}  // namespace ippl

#endif
