//
// Struct
//   Kokkos::Views of different dimensions.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_VIEW_TYPES_H
#define IPPL_VIEW_TYPES_H

#include <Kokkos_Core.hpp>

template <typename T, unsigned Dim, class... Properties>
struct ViewType { };


template <typename T, class... Properties>
struct ViewType<T, 1, Properties...> {
    typedef Kokkos::View<T*, Properties...> view_type;
};


template <typename T, class... Properties>
struct ViewType<T, 2, Properties...> {
    typedef Kokkos::View<T**, Properties...> view_type;
};


template <typename T, class... Properties>
struct ViewType<T, 3, Properties...> {
    typedef Kokkos::View<T***, Properties...> view_type;
};


/*
 * write functions
 */
template <typename T, unsigned Dim, class... Properties>
void write_(const typename ViewType<T, Dim, Properties...>::view_type& view,
           std::ostream& out = std::cout);


template <typename T, class... Properties>
void write_(const typename ViewType<T, 1, Properties...>::view_type& view,
           std::ostream& out = std::cout)
{
    typename ViewType<T, 1, Properties...>::view_type::HostMirror hview = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hview, view);
    for (std::size_t i = 0; i < view.extent(0); ++i) {
        out << view(i) << " ";
    }
    out << std::endl;
}


template <typename T, class... Properties>
void write_(const typename ViewType<T, 2, Properties...>::view_type& view,
           std::ostream& out = std::cout)
{
    typename ViewType<T, 2, Properties...>::view_type::HostMirror hview = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hview, view);
    for (std::size_t j = 0; j < view.extent(1); ++j) {
        for (std::size_t i = 0; i < view.extent(0); ++i) {
            out << view(i, j) << " ";
        }
        out << std::endl;
    }
}

template <typename T, class... Properties>
void write_(const typename ViewType<T, 3, Properties...>::view_type& view,
           std::ostream& out = std::cout)
{
    typename ViewType<T, 3, Properties...>::view_type::HostMirror hview = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hview, view);
    for (std::size_t k = 0; k < view.extent(2); ++k) {
        for (std::size_t j = 0; j < view.extent(1); ++j) {
            for (std::size_t i = 0; i < view.extent(0); ++i) {
                out << view(i, j, k) << " ";
            }
            out << std::endl;
        }
        if (k < view.extent(2) - 1)
            out << std::endl;
    }
}


#endif