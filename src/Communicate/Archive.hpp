//
// Class Archive
//   Class to (de-)serialize in MPI communication.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Archive.h"

#include <cstring>

namespace ippl {
    namespace detail {

        template <class... Properties>
        Archive<Properties...>::Archive(size_type size)
        : writepos_m(0)
        , readpos_m(0)
        , buffer_m("buffer", size)
        { }

        template <class... Properties>
        template <typename T>
        void Archive<Properties...>::serialize(const Kokkos::View<T*>& view, count_type nsends) {
            size_t size = sizeof(T);
            Kokkos::parallel_for(
                "Archive::serialize()", nsends,
                KOKKOS_CLASS_LAMBDA(const size_t i) {
                    std::memcpy(buffer_m.data() + i * size + writepos_m,
                                view.data() + i,
                                size);
            });
            Kokkos::fence();
            writepos_m += size * nsends;
        }

        template <class... Properties>
        template <typename T, unsigned Dim>
        void Archive<Properties...>::operator<<(const Kokkos::View<Vector<T, Dim>*>& view) {
            size_t size = sizeof(T);
            using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_for(
                "Archive::serialize()",
                mdrange_t({0, 0}, {(long int)view.extent(0), Dim}),
                KOKKOS_CLASS_LAMBDA(const size_t i, const size_t d) {
                    std::memcpy(buffer_m.data() + (Dim * i + d) * size + writepos_m,
                                &(*(view.data() + i))[d],
                                size);
                });
            writepos_m += Dim * size * view.size();
            Kokkos::fence();
        }

        template <class... Properties>
        template <typename T, unsigned Dim>
        void Archive<Properties...>::serialize(const Kokkos::View<Vector<T, Dim>*>& view, count_type nsends) {
            size_t size = sizeof(T);
            using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_for(
                "Archive::serialize()",
                mdrange_t({0, 0}, {nsends, Dim}),
                KOKKOS_CLASS_LAMBDA(const size_t i, const size_t d) {
                    std::memcpy(buffer_m.data() + (Dim * i + d) * size + writepos_m,
                                &(*(view.data() + i))[d],
                                size);
                });
            Kokkos::fence();
            writepos_m += Dim * size * nsends;
        }

        template <class... Properties>
        template <typename T>
        void Archive<Properties...>::deserialize(Kokkos::View<T*>& view, count_type nrecvs) {
            size_t size = sizeof(T);
            if(nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            Kokkos::parallel_for(
                "Archive::deserialize()", nrecvs,
                KOKKOS_CLASS_LAMBDA(const size_t i) {
                    std::memcpy(view.data() + i,
                                buffer_m.data() + i * size + readpos_m,
                                size);
            });
            // Wait for deserialization kernel to complete
            // (as with serialization kernels)
            Kokkos::fence();
            readpos_m += size * nrecvs;
        }

        template <class... Properties>
        template <typename T, unsigned Dim>
        void Archive<Properties...>::deserialize(Kokkos::View<Vector<T, Dim>*>& view, count_type nrecvs) {
            size_t size = sizeof(T);
            if(nrecvs > view.extent(0)) {
                Kokkos::realloc(view, nrecvs);
            }
            using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_for(
                "Archive::deserialize()",
                mdrange_t({0, 0}, {nrecvs, Dim}),
                KOKKOS_CLASS_LAMBDA(const size_t i, const size_t d) {
                    std::memcpy(&(*(view.data() + i))[d],
                                buffer_m.data() + (Dim * i + d) * size + readpos_m,
                                size);
            });
            Kokkos::fence();
            readpos_m += Dim * size * nrecvs;
        }
    }
}
