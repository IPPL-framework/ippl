//
// View Utilities
//   Utility functions relating to Kokkos views
//
// Copyright (c) 2023, Paul Scherrer Institut, Villigen PSI, Switzerland
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

#ifndef IPPL_VIEW_UTILS_H
#define IPPL_VIEW_UTILS_H

#include <Kokkos_Core.hpp>

#include "Types/ViewTypes.h"

namespace ippl {
    namespace detail {
        template <unsigned Dim, unsigned Current = 0, typename View, typename... Args>
        static constexpr void printLoop(const View& view, std::ostream& out, Args&&... args) {
            for (size_t i = 0; i < view.extent(Dim - Current - 1); ++i) {
                if constexpr (Dim - 1 == Current) {
                    out << view(i, args...) << " ";
                } else {
                    printLoop<Dim, Current + 1>(view, out, i, args...);
                }
            }
            if (Current + 1 >= 2 || Current == 0) {
                out << std::endl;
            }
        }

        /*!
         * Writes a view to an output stream
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         *
         * @param view to write
         * @param out stream
         */
        template <typename T, unsigned Dim, class... Properties>
        void write(const typename ViewType<T, Dim, Properties...>::view_type& view,
                   std::ostream& out = std::cout) {
            using view_type = typename ViewType<T, Dim, Properties...>::view_type;
            typename view_type::HostMirror hview = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hview, view);

            printLoop<Dim>(hview, out);
        }

        /*!
         * Utility function for shrinkView
         */
        template <unsigned Dim, typename T, size_t... Idx>
        decltype(auto) shrinkView_impl(std::string label,
                                       const typename ViewType<T, Dim>::view_type& view, int nghost,
                                       const std::index_sequence<Idx...>&) {
            return Kokkos::View<typename NPtr<T, Dim>::type, Kokkos::LayoutLeft>(
                label, (view.extent(Idx) - 2 * nghost)...);
        }

        /*!
         * Constructs a new view with size equal to that of the given view, minus the ghost cells
         * @tparam Dim the view's rank
         * @tparam T the view's value type
         * @param label the new view's name
         * @param view the view to shrink
         * @param nghost the number of ghost cells on the view's boundary
         * @return The shrunken view
         */
        template <unsigned Dim, typename T>
        decltype(auto) shrinkView(std::string label,
                                  const typename ViewType<T, Dim>::view_type& view, int nghost) {
            return shrinkView_impl<Dim, T>(label, view, nghost, std::make_index_sequence<Dim>{});
        }
    }  // namespace detail
}  // namespace ippl

#endif
