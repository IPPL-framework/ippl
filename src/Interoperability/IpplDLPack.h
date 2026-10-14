/*!
TODO: Doxygen stuff
*/

#ifndef IPPL_DLPACK_H
#define IPPL_DLPACK_H

#include "KokkosDLPack.h"


namespace ippl {
    namespace Interoperability {

        template <typename Attrib>
        py::capsule attribToDLPackVec(Attrib& attrib, ReadOnly readOnly, DLPackVersion version) {
            // peeling the onion - say we are given a ippl::ParticleAttrib<ippl::Vector<double,3>>:
            using view_type = typename Attrib::view_type; // the underlying kokkos view, e.g. Kokkos::View<Vector<double,3>*>
            using vector_type = typename view_type::value_type; // elements of the kokkos view, e.g. ippl::Vector<double,3>
            using ScalarT = typename vector_type::value_type; // the scalar elements themselves, e.g. double
            constexpr int64_t Dim = static_cast<int64_t>(vector_type::dim);  // the shape of the data, e.g. 3

            static_assert(view_type::rank == 1, "ParticleAttrib is always rank-1");

            // gets the attribute Kokkos View, trimmed to live particles
            view_type v = attrib.getView();
            // ...and their count, cast from size_t to int64_t, which DLPack uses
            // (ParticleAttrib is always rank-1 View, so extent(0) gets the count)
            const int64_t localNum = static_cast<int64_t>(v.extent(0));

            // Checks to see if zero-copy is possible
            // check if layout is standard, not reordered and no padding...
            if constexpr (std::is_standard_layout_v<vector_type>
                && sizeof(vector_type) == Dim * sizeof(ScalarT)) {
                // ...so if it is contiguous and there is no padding, then i can simply
                // reinterpret the pointer, e.g. here ScalarT would be ippl::Vector<double,3>*
                // and i would reinterpret to double*
                auto* scalarPtr = reinterpret_cast<ScalarT*>(v.data());

                std::vector<int64_t> shape, strides;
                shape   = {localNum, Dim}; // (localNum, Dim)
                strides = {static_cast<int64_t>(v.stride(0)) * Dim, 1};

                return exportTensor<ScalarT>(
                    v, scalarPtr,
                    std::move(shape), std::move(strides),
                    readOnly, version);
            } else {
                // otherwise i cannot reinterpret the raw pointer.
                // Instead repack into a flat (localNum * Dim) scalar array
                // on the device (no host transfer). operator[] is
                // a KOKKOS_INLINE_FUNCTION and works regardless of layout

                // a cuda malloc per call... will move scratch buffer to manager if this actually get used
                Kokkos::View<ScalarT*, typename view_type::memory_space> flat(
                    Kokkos::view_alloc("dlpack_repack", Kokkos::WithoutInitializing),
                    localNum * Dim);

                Kokkos::parallel_for(
                    "ippl::Interoperability::repack", localNum, KOKKOS_LAMBDA(const int64_t i) {
                        for (int64_t d = 0; d < Dim; ++d) {
                            flat(i * Dim + d) = v(i)[d];
                        }
                    });
                Kokkos::fence();

                return exportTensor<ScalarT>(
                    flat, flat.data(),
                    {localNum, Dim},
                    {Dim, 1},
                    readOnly, version);
            }
        }

        // for bare scalar views, which are already in correct form
        template <typename Attrib>
        py::capsule attribToDLPackScalar(Attrib& attrib, ReadOnly readOnly, DLPackVersion version) {
            return toDLPack(attrib.getView(), readOnly, version);
        }

    }  // namespace Interoperability
}  // namespace ippl

#endif
