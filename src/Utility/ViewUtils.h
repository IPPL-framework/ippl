//
// View Utilities
//   Utility functions relating to Kokkos views
//

#ifndef IPPL_VIEW_UTILS_H
#define IPPL_VIEW_UTILS_H

#include <Kokkos_Core.hpp>

#include <iostream>

#include "Types/ViewTypes.h"

namespace ippl {
    namespace detail {
        /*!
         * Expands into a nested loop via templating
         * Source:
         * https://stackoverflow.com/questions/34535795/n-dimensionally-nested-metaloops-with-templates
         * @tparam Dim the number of nested levels
         * @tparam BeginFunctor functor type for determining the start index of each loop
         * @tparam EndFunctor functor type for determining the end index of each loop
         * @tparam Functor functor type for the loop body
         * @tparam Check functor type for loop check
         * @param begin a functor that returns the starting index for each level of the loop
         * @param end a functor that returns the ending index (exclusive) for each level of the loop
         * @param body a functor to be called in each iteration of the loop with the indices as
         * arguments
         * @param check a check function to be run after each loop; takes the current axis as an
         * argument (default none)
         */
        template <unsigned Dim, unsigned Current = 0, class BeginFunctor, class EndFunctor,
                  class Functor, typename Check  = std::nullptr_t>
        constexpr void nestedLoop(BeginFunctor&& begin, EndFunctor&& end, Functor&& body,
                                  Check&& check = nullptr) {
            for (size_t i = begin(Current); i < end(Current); ++i) {
                if constexpr (Dim - 1 == Current) {
                    body(i);
                } else {
                    auto inner = [i, &body](auto... args) {
                        body(i, args...);
                    };
                    nestedLoop<Dim, Current + 1>(begin, end, inner, check);
                }
            }
            if constexpr (!std::is_null_pointer_v<std::decay_t<Check>>) {
                check(Current);
            }
        }

        /*!
         * Convenience function for nested looping through a view
         * @tparam View the view type
         * @tparam Functor the loop body functor type
         * @tparam Check functor type for loop check
         * @param view the view
         * @param shift the number of ghost cells
         * @param body the functor to be called in each iteration
         * @param check a check function to be run after each loop; takes the current axis as an
         * argument (default none)
         */
        template <typename View, class Functor, typename Check = std::nullptr_t>
        constexpr void nestedViewLoop(View& view, int shift, Functor&& body,
                                      Check&& check = nullptr) {
            nestedLoop<View::rank>(
                [&](unsigned) {
                    return shift;
                },
                [&](unsigned d) {
                    return view.extent(d) - shift;
                },
                body, check);
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
            typename view_type::host_mirror_type hview = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(hview, view);

            nestedViewLoop(
                view, 0,
                [&]<typename... Args>(Args&&... args) {
                    out << hview(args...) << " ";
                },
                [&](unsigned axis) {
                    if (axis + 1 >= 2 || axis == 0) {
                        out << std::endl;
                    }
                });
        }

        /*!
         * Recursive implementation to write a view in list format to output stream
         * @tparam Dim
         *
         * @param view
         * @param out stream
         */
        template <unsigned Dim, typename View>
        void write_as_list_impl(const View& view, std::ostream& out = std::cout) {
            auto N = view.extent(0);
            out << "[";
            for (std::size_t i = 0; i < N; ++i) {
                if constexpr (Dim == 1) {
                    out << view(i);
                } else {
                    auto make_subview = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                        return Kokkos::subview(view, i, (static_cast<void>(Is), Kokkos::ALL)...);
                    };

                    auto subview = make_subview(std::make_index_sequence<Dim - 1>());
                    write_as_list_impl<Dim - 1>(subview, out);
                }
                if (i != N - 1)
                    out << ", ";
            }
            out << "]";
        };

        /*!
         * Writes a view to an output stream in folded list format
         * @tparam T view data type
         * @tparam Dim view dimension
         * @tparam Properties further template parameters of Kokkos
         *
         * @param view to write
         * @param out stream
         */
        template <typename T, unsigned Dim, class... Properties>
        void write_as_list(const typename ViewType<T, Dim, Properties...>::view_type& view,
                           std::ostream& out = std::cout) {
            auto hview =
                Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), view);

            write_as_list_impl<Dim>(hview, out);
        }

        /*!
         * Utility function for shrinkView
         */
        template <typename View, size_t... Idx>
        decltype(auto) shrinkView_impl(std::string label, const View& view, int nghost,
                                       const std::index_sequence<Idx...>&) {
            using view_type = typename Kokkos::View<typename View::data_type, Kokkos::LayoutLeft,
                                                    typename View::memory_space>::uniform_type;
            return view_type(label, (view.extent(Idx) - 2 * nghost)...);
        }

        /*!
         * Constructs a new view with size equal to that of the given view, minus the ghost cells
         * (used for heFFTe, which expects the data to have a certain layout and no ghost cells)
         * @param label the new view's name
         * @param view the view to shrink
         * @param nghost the number of ghost cells on the view's boundary
         * @return The shrunken view
         */
        template <typename View>
        decltype(auto) shrinkView(std::string label, const View& view, int nghost) {
            return shrinkView_impl(label, view, nghost, std::make_index_sequence<View::rank>{});
        }
    }  // namespace detail
}  // namespace ippl

#endif
