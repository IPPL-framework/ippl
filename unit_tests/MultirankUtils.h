//
// Utility class for rank independent unit testing
//   Provides a framework with which unit tests can easily work with
//   objects of different dimensionalities.
//
// Copyright (c) 2023 Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef MULTIRANK_UTILS_H
#define MULTIRANK_UTILS_H

#include <algorithm>
#include <array>
#include <tuple>

template <unsigned... Dims>
class MultirankUtils {
    // Checking for specialization; inherits from true_type
    // if one template parameter is a specialization of the other
    // https://stackoverflow.com/a/28796458
    template <typename, template <typename...> class>
    struct is_specialization : std::false_type {};

    template <template <typename...> class Ref, typename... Args>
    struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

    /*!
     * Utility function for apply. If there are no arguments, run the function
     * once for each dimension. If the arguments are tuples, run the function
     * with the tuple elements as arguments for each tuple provided. Otherwise,
     * assume a single argument and run the function with that argument for each dimension.
     * @param f the function
     * @param args the arguments for the function for each dimension
     */
    template <typename Functor, typename... Args, unsigned long... Idx>
    static void apply_impl(Functor& f, std::tuple<Args...>& args,
                           const std::index_sequence<Idx...>&) {
        if constexpr (sizeof...(Args) == 0) {
            // Dim == Idx + 1
            (f.template operator()<Idx + 1>(), ...);
        } else if constexpr (is_specialization<std::tuple_element_t<0, std::tuple<Args...>>,
                                               std::tuple>::value) {
            (std::apply(f, std::get<Idx>(args)), ...);
        } else {
            (f(std::get<Idx>(args)), ...);
        }
    }

    /*!
     * Utility function for setting up the testing environment
     */
    template <typename Tester, size_t... Idx>
    void setup_impl(Tester&& t, const std::index_sequence<Idx...>&) {
        ((t->template setupDim<Idx, Dims>()), ...);
    }

    // Tuple zipping
    // https://stackoverflow.com/a/47127033
    template <size_t Idx, typename... Tuples>
    using zipped_element = std::tuple<std::tuple_element_t<Idx, std::decay_t<Tuples>>&...>;

    /*!
     * Constructs a new tuple consisting of the elements of other tuples at a given index
     * @tparam Idx the index at which to zip
     * @tparam Tuples.. the tuple types
     * @param ts... the tuples to zip
     * @return A new tuple containing references to the original tuples' elements at the given index
     */
    template <size_t Idx, typename... Tuples>
    static zipped_element<Idx, Tuples...> zip_at(Tuples&&... ts) {
        return {std::get<Idx>(std::forward<Tuples>(ts))...};
    }

    /*!
     * Utility function for tuple zipping
     */
    template <typename... Tuples, size_t... Idx>
    static std::tuple<zipped_element<Idx, Tuples...>...> zip_impl(
        Tuples&&... ts, const std::index_sequence<Idx...>&) {
        return {zip_at<Idx>(std::forward<Tuples>(ts)...)...};
    }

protected:
    /*!
     * Defines a type alias for a collection of constructs
     * with the desired ranks
     */
    template <template <unsigned Dim> class Type>
    using Collection = std::tuple<Type<Dims>...>;

    /*!
     * Defines a type alias for a collection of pointers to
     * constructs with the desired ranks
     */
    template <template <typename> class Pointer, template <unsigned Dim> class Type>
    using PtrCollection = std::tuple<Pointer<Type<Dims>>...>;

    constexpr static unsigned MaxDim = std::max({Dims...});

    constexpr static unsigned dimToIndex(unsigned dim) {
        constexpr std::array<unsigned, sizeof...(Dims)> dims = {Dims...};
        return std::distance(dims.begin(), std::find(dims.begin(), dims.end(), dim));
    }

    /*!
     * Set up the testing environment using a given class. Requires that the tester
     * has a function named setupDim with two unsigned integers as template arguments.
     * These represent the index of each element in the collections and its rank.
     * @tparam Tester the testing class
     * @param t the instance of the testing class
     */
    template <typename Tester>
    void setup(Tester&& t) {
        setup_impl(t, std::make_index_sequence<sizeof...(Dims)>{});
    }

public:
    /*!
     * Runs a function with some arguments
     * @tparam Functor the functor type
     * @tparam Args the argument types
     * @param f the function
     * @param args arguments for the function
     */
    template <typename Functor, typename... Args>
    static auto apply(Functor& f, std::tuple<Args...>& args) {
        apply_impl(f, args, std::make_index_sequence<sizeof...(Dims)>{});
    }

    /*!
     * Runs a function with no arguments for each rank
     */
    template <typename Functor>
    static auto apply(Functor& f) {
        auto args = std::tuple<>{};
        apply_impl(f, args, std::make_index_sequence<sizeof...(Dims)>{});
    }

    /*!
     * Zips a set of tuples together
     * @tparam Tuple the first tuple type
     * @tparam Tuples... the remaining tuple types
     * @param t0 the first tuple
     * @param ts the other tuples
     * @return A new tuple containing references to the original tuple elements
     */
    template <typename Tuple, typename... Tuples>
    static auto zip(Tuple&& t0, Tuples&&... ts) {
        constexpr size_t size = std::tuple_size_v<std::decay_t<Tuple>>;
        static_assert(((std::tuple_size_v<std::decay_t<Tuples>> == size) && ...),
                      "Mismatched tuple sizes");

        return zip_impl<Tuple, Tuples...>(std::forward<Tuple>(t0), std::forward<Tuples>(ts)...,
                                          std::make_index_sequence<size>{});
    }

    /*!
     * Expands into a nested loop via templating
     * Source:
     * https://stackoverflow.com/questions/34535795/n-dimensionally-nested-metaloops-with-templates
     * @tparam Dim the number of nested levels
     * @tparam BeginFunctor functor type for determining the start index of each loop
     * @tparam EndFunctor functor type for determining the end index of each loop
     * @tparam Functor functor type for the loop body
     * @param begin a functor that returns the starting index for each level of the loop
     * @param end a functor that returns the ending index (exclusive) for each level of the loop
     * @param c a functor to be called in each iteration of the loop with the indices as arguments
     */
    template <unsigned Dim, class BeginFunctor, class EndFunctor, class Functor>
    static constexpr void nestedLoop(BeginFunctor&& begin, EndFunctor&& end, Functor&& c) {
        for (size_t i = begin(Dim); i < end(Dim); ++i) {
            if constexpr (Dim == 1) {
                c(i);
            } else {
                auto next = [i, &c](auto... args) {
                    c(i, args...);
                };
                nestedLoop<Dim - 1>(begin, end, next);
            }
        }
    }

    /*!
     * Convenience function for nested looping through a view
     * @tparam Dim the view's rank
     * @tparam View the view type
     * @tparam Functor the loop body functor type
     * @param view the view
     * @param shift the number of ghost cells
     * @param c the functor to be called in each iteration
     */
    template <unsigned Dim, typename View, class Functor>
    static constexpr void nestedViewLoop(View& view, int shift, Functor&& c) {
        nestedLoop<Dim>(
            [&](unsigned) {
                return shift;
            },
            [&](unsigned d) {
                return view.extent(Dim - d) - shift;
            },
            c);
    }
};

#endif
