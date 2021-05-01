//
// Struct ViewType
//   Kokkos::Views of different dimensions.
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
#ifndef IPPL_VIEW_TYPES_H
#define IPPL_VIEW_TYPES_H

#include <Kokkos_Core.hpp>

namespace ippl {
    /**
     * @file ViewTypes.h
     * This file defines multi-dimensional arrays to store mesh and particle attributes.
     * It provides specialized versions for 1, 2 and 3 dimensions. The file further
     * provides write functions for the different view types.
     */
    namespace detail {
        /*!
         * Empty struct for the specialized view types.
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         */
        template <typename T, unsigned Dim, class... Properties>
        struct ViewType { };

        /*!
         * Specialized view type for one dimension.
         */
        template <typename T, class... Properties>
        struct ViewType<T, 1, Properties...> {
            typedef Kokkos::View<T*, Properties...> view_type;
        };

        /*!
         * Specialized view type for two dimensions.
         */
        template <typename T, class... Properties>
        struct ViewType<T, 2, Properties...> {
            typedef Kokkos::View<T**, Properties...> view_type;
        };

        /*!
         * Specialized view type for thee dimensions.
         */
        template <typename T, class... Properties>
        struct ViewType<T, 3, Properties...> {
            typedef Kokkos::View<T***, Properties...> view_type;
        };


        /*!
         * Muldimensional range policies.
         */
        template <unsigned Dim>
        struct RangePolicy {
            typedef Kokkos::MDRangePolicy<Kokkos::Rank<Dim>> range_policy_type;
        };

        /*!
         * Specialized range policy for one dimension.
         */
        template <>
        struct RangePolicy<1> {
            typedef Kokkos::RangePolicy<> range_policy_type;
        };


        /*!
         * Empty function for general write.
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         *
         * @param view to write
         * @param out stream
         */
        template <typename T, unsigned Dim, class... Properties>
        void write(const typename ViewType<T, Dim, Properties...>::view_type& view,
                std::ostream& out = std::cout);


        /*!
         * Specialized write function for one-dimensional views.
         */
        template <typename T, class... Properties>
        void write(const typename ViewType<T, 1, Properties...>::view_type& view,
                std::ostream& out = std::cout)
        {
            using view_type = typename ViewType<T, 1, Properties...>::view_type;
            typename view_type::HostMirror hview = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hview, view);
            for (std::size_t i = 0; i < hview.extent(0); ++i) {
                out << hview(i) << " ";
            }
            out << std::endl;
        }


        /*!
         * Specialized write function for two-dimensional views.
         */
        template <typename T, class... Properties>
        void write(const typename ViewType<T, 2, Properties...>::view_type& view,
                std::ostream& out = std::cout)
        {
            using view_type = typename ViewType<T, 2, Properties...>::view_type;
            typename view_type::HostMirror hview = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hview, view);
            for (std::size_t j = 0; j < hview.extent(1); ++j) {
                for (std::size_t i = 0; i < hview.extent(0); ++i) {
                    out << hview(i, j) << " ";
                }
                out << std::endl;
            }
        }

        /*!
         * Specialized write function for three-dimensional views.
         */
        template <typename T, class... Properties>
        void write(const typename ViewType<T, 3, Properties...>::view_type& view,
                std::ostream& out = std::cout)
        {
            using view_type = typename ViewType<T, 3, Properties...>::view_type;
            typename view_type::HostMirror hview = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hview, view);
            for (std::size_t k = 0; k < hview.extent(2); ++k) {
                for (std::size_t j = 0; j < hview.extent(1); ++j) {
                    for (std::size_t i = 0; i < hview.extent(0); ++i) {
                        out << hview(i, j, k) << " ";
                    }
                    out << std::endl;
                }
                if (k < view.extent(2) - 1)
                    out << std::endl;
            }
        }
    }
}


#endif
