//
// View Utilities
//   Utility functions relating to Kokkos views
//

#ifndef IPPL_VIEW_UTILS_H
#define IPPL_VIEW_UTILS_H

#include <Kokkos_Core.hpp>

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
            typename view_type::HostMirror hview = Kokkos::create_mirror_view(view);
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
