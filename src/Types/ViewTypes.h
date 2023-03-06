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

#include <tuple>

namespace ippl {
    /**
     * @file ViewTypes.h
     * This file defines multi-dimensional arrays to store mesh and particle attributes.
     * It provides specialized versions for 1, 2 and 3 dimensions. The file further
     * provides write functions for the different view types.
     */
    namespace detail {
        /*!
         * Recursively templated struct for defining pointers with arbitrary
         * indirection depth.
         * @tparam T data type
         * @tparam N indirection level
         */
        template <typename T, int N>
        struct NPtr {
            typedef typename NPtr<T, N - 1>::type* type;
        };

        /*!
         * Base case template specialization for a simple pointer.
         */
        template <typename T>
        struct NPtr<T, 1> {
            typedef T* type;
        };

        template <unsigned Dim, typename T = size_t>
        struct Coords {
            // https://stackoverflow.com/a/53398815/2773311
            // https://en.cppreference.com/w/cpp/utility/declval
            typedef decltype(std::tuple_cat(
                std::declval<typename Coords<1, T>::type>(),
                std::declval<typename Coords<Dim - 1, T>::type>())) type;
        };

        template <typename T>
        struct Coords<1, T> {
            typedef std::tuple<T> type;
        };

        /*!
         * View type for an arbitrary number of dimensions.
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         */
        template <typename T, unsigned Dim, class... Properties>
        struct ViewType {
            typedef Kokkos::View<typename NPtr<T, Dim>::type, Properties...> view_type;
        };

        /*!
         * Multidimensional range policies.
         */
        template <unsigned Dim, typename Tag = void>
        struct RangePolicy {
            typedef Kokkos::MDRangePolicy<Tag, Kokkos::Rank<Dim>> policy_type;
        };

        /*!
         * Specialized range policy for one dimension.
         */
        template <typename Tag>
        struct RangePolicy<1, Tag> {
            typedef Kokkos::RangePolicy<Tag> policy_type;
        };

        /*!
         * Create a range policy that spans an entire Kokkos view, excluding
         * a specifiable number of ghost cells at the extremes.
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         *
         * @param view to span
         * @param shift number of ghost cells
         *
         * @return A (MD)RangePolicy that spans the desired elements of the given view
         */
        template <unsigned Dim, typename Tag = void, typename View>
        typename RangePolicy<Dim, Tag>::policy_type getRangePolicy(const View& view,
                                                                   int shift = 0) {
            using policy_type = typename RangePolicy<Dim, Tag>::policy_type;
            using index_type  = typename policy_type::array_index_type;
            Kokkos::Array<index_type, Dim> begin, end;
            for (unsigned int d = 0; d < Dim; d++) {
                begin[d] = shift;
                end[d]   = view.extent(d) - shift;
            }
            return policy_type(begin, end);
        }

#define CreateFunctor(contents)                                                          \
    template <typename, typename>                                                        \
    struct Functor;                                                                      \
    template <typename... Idx, typename Acc>                                             \
    struct Functor<std::tuple<Idx...>, Acc> {                                            \
        static constexpr unsigned FunctorDim = sizeof...(Idx);                           \
        using view_type = typename ::ippl::detail::ViewType<Acc, FunctorDim>::view_type; \
        view_type view_m;                                                                \
        KOKKOS_FUNCTION                                                                  \
        Functor(view_type v)                                                             \
            : view_m(v) {}                                                               \
        contents                                                                         \
    };

#define CreateTaggedFunctor(tag) \
    KOKKOS_INLINE_FUNCTION void operator()(const tag&, const Idx... args, Acc& acc) const;

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
                   std::ostream& out = std::cout) {
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
                   std::ostream& out = std::cout) {
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
                   std::ostream& out = std::cout) {
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
    }  // namespace detail
}  // namespace ippl

#endif
