//
// Class Ippl
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
        Archive<Properties...>::Archive()
        : writepos_m(0)
        , readpos_m(0)
        , buffer_m("buffer")
        { }


        template <class... Properties>
        template <typename T>
        void Archive<Properties...>::operator<<(const ViewType<T, 1, Properties...>& view) {
            size_t size = sizeof(T);
            Kokkos::resize(buffer_m, buffer_m.size() + size * view.size());
            Kokkos::parallel_for("Archive::serialize()", view.size(),
                                 KOKKOS_CLASS_LAMBDA(const int i) {
                                     std::memcpy(buffer_m.data() + i * size + writepos_m,
                                                 view.data() + i,
                                                 size);
                                });
                writepos_m += size * view.size();
        }


        template <class... Properties>
        template <typename T>
        void Archive<Properties...>::operator>>(const ViewType<T, 1, Properties...>& view) {
            size_t size = sizeof(T);
            Kokkos::resize(buffer_m, buffer_m.size() + size * view.size());
            Kokkos::parallel_for("Archive::deserialize()", view.size(),
                                 KOKKOS_CLASS_LAMBDA(const int i) {
                                     std::memcpy(view.data() + i,
                                                 buffer_m.data() + i * size + readpos_m,
                                                 size);
                                });
            readpos_m += size * view.size();
        }
    }
}