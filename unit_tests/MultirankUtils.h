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
    /*!
     * Utility function for apply. Calls the function with one argument stemming from each argument
     * tuple.
     * @tparam Idx the index of the elements to pass
     * @param f the function
     * @param args the argument tuples
     */
    template <unsigned long Idx, typename Functor, typename... Args>
    static void arg_impl(Functor& f, Args&&... args) {
        f(std::get<Idx>(args)...);
    }

    /*!
     * Utility function for apply. If there are no arguments, run the function
     * once for each dimension. Otherwise, assume the arguments are tuples and run the function
     * with the tuple elements as arguments for each tuple provided.
     * @param f the function
     * @param args the arguments for the function for each dimension
     */
    template <typename Functor, unsigned long... Idx, typename... Args>
    static void apply_impl(const std::index_sequence<Idx...>&, Functor& f, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            // Dim == Idx + 1
            (f.template operator()<indexToDim(Idx)>(), ...);
        } else {
            (arg_impl<Idx>(f, args...), ...);
        }
    }

    /*!
     * Utility function for setting up the testing environment
     */
    template <typename Tester, size_t... Idx>
    void setup_impl(Tester&& t, const std::index_sequence<Idx...>&) {
        ((t->template setupDim<Idx, Dims>()), ...);
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

    constexpr static unsigned MaxDim   = std::max({Dims...});
    constexpr static unsigned DimCount = sizeof...(Dims);

    constexpr static unsigned dimToIndex(unsigned dim) {
        constexpr std::array<unsigned, sizeof...(Dims)> dims = {Dims...};
        return std::distance(dims.begin(), std::find(dims.begin(), dims.end(), dim));
    }

    constexpr static unsigned indexToDim(unsigned idx) {
        constexpr std::array<unsigned, sizeof...(Dims)> dims = {Dims...};
        return dims[idx];
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

    /*!
     * Computes axis lengths for heterogeneous mesh dimensions
     * @param nr Array in which to store the axis lengths (must be of length DimCount or greater)
     */
    void computeGridSizes(size_t nr[]) {
        for (unsigned d = 0; d < DimCount; d++)
            if (DimCount > 1 + d)
                nr[d] = 1 << (DimCount - 1 - d);
            else
                nr[d] = 2;
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
    static auto apply(Functor& f, Args&&... args) {
        apply_impl(std::make_index_sequence<sizeof...(Dims)>{}, f, args...);
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
    template <unsigned Dim, unsigned Current = 0, class BeginFunctor, class EndFunctor,
              class Functor>
    static constexpr void nestedLoop(BeginFunctor&& begin, EndFunctor&& end, Functor&& c) {
        for (size_t i = begin(Current); i < end(Current); ++i) {
            if constexpr (Dim - 1 == Current) {
                c(i);
            } else {
                auto next = [i, &c](auto... args) {
                    c(i, args...);
                };
                nestedLoop<Dim, Current + 1>(begin, end, next);
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
                return view.extent(d) - shift;
            },
            c);
    }
};

#endif
