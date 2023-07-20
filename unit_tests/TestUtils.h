//
// Utilities for versatile unit testing
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

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <tuple>
#include <type_traits>

#include "Utility/TypeUtils.h"

#include "MultirankUtils.h"
#include "gtest/gtest.h"

template <typename... Ts>
struct Parameters {
    using flat_type = Parameters<Ts...>;
};

template <typename... Ts, typename U>
struct Parameters<Parameters<Ts...>, U> {
    using flat_type = Parameters<Ts..., U>;
};

template <typename...>
struct CreateCombinations;
template <typename, typename>
struct CombineTuples;
template <typename, typename>
struct AddType;

template <typename... Ts, typename U>
struct AddType<std::tuple<Ts...>, U> {
    using type = std::tuple<typename Parameters<Ts, U>::flat_type...>;
};

template <typename PackA, typename... TypesB>
struct CombineTuples<PackA, std::tuple<TypesB...>> {
    using type = decltype(std::tuple_cat(std::declval<typename AddType<PackA, TypesB>::type>()...));
};

template <typename Tuple1, typename Tuple2>
struct CreateCombinations<Tuple1, Tuple2> {
    using type = typename CombineTuples<Tuple1, Tuple2>::type;
};

template <typename Tuple1, typename Tuple2, typename... Tuples>
struct CreateCombinations<Tuple1, Tuple2, Tuples...> {
    using first = typename CombineTuples<Tuple1, Tuple2>::type;
    using type  = typename CreateCombinations<first, Tuples...>::type;
};

template <typename>
struct TestForTypes;

template <typename... Types>
struct TestForTypes<std::tuple<Types...>> {
    using type = ::testing::Types<Types...>;
};

template <typename T>
void assertEqual(T valA, T valB) {
    if constexpr (std::is_same_v<T, double>) {
        ASSERT_DOUBLE_EQ(valA, valB);
    } else {
        ASSERT_FLOAT_EQ(valA, valB);
    }
};

template <unsigned Dim>
constexpr std::array<size_t, Dim> getGridSizes() {
    constexpr unsigned max = std::max(6U, Dim);
    std::array<size_t, Dim> nr{};
    for (unsigned d = 0; d < Dim; d++) {
        if (max > 1 + d) {
            nr[d] = 1U << (max - 1 - d);
        } else {
            nr[d] = 2;
        }
    }
    return nr;
}

template <unsigned>
struct Rank;

struct TestParams {
    using Spaces     = ippl::detail::TypeForAllSpaces<std::tuple>::exec_spaces_type;
    using Precisions = std::tuple<double, float>;
    using Combos     = CreateCombinations<Precisions, Spaces>::type;

    template <unsigned... Dims>
    using Ranks = std::tuple<Rank<Dims>...>;

    template <unsigned... Dims>
    using CombosWithRanks = typename CreateCombinations<Precisions, Spaces, Ranks<Dims...>>::type;

    template <unsigned... Dims>
    using tests = typename TestForTypes<
        std::conditional_t<sizeof...(Dims) == 0, Combos, CombosWithRanks<Dims...>>>::type;

    static bool skipSerialTests;

    static void checkArgs([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
        skipSerialTests = true;
#ifdef KOKKOS_ENABLE_SERIAL
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--run-serial") == 0) {
                skipSerialTests = false;
            }
        }
#endif
    }
};

bool TestParams::skipSerialTests = true;

#ifdef KOKKOS_ENABLE_SERIAL
#define CHECK_SKIP_SERIAL                                                           \
    if (std::is_same_v<ExecSpace, Kokkos::Serial> && TestParams::skipSerialTests) { \
        GTEST_SKIP();                                                               \
    }

#define CHECK_SKIP_SERIAL_CONSTRUCTOR                                               \
    if (std::is_same_v<ExecSpace, Kokkos::Serial> && TestParams::skipSerialTests) { \
        return;                                                                     \
    }
#else
#define CHECK_SKIP_SERIAL \
    {}
#define CHECK_SKIP_SERIAL_CONSTRUCTOR \
    {}
#endif

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
template <unsigned Dim, unsigned Current = 0, class BeginFunctor, class EndFunctor, class Functor>
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
 * @tparam View the view type
 * @tparam Functor the loop body functor type
 * @param view the view
 * @param shift the number of ghost cells
 * @param c the functor to be called in each iteration
 */
template <typename View, class Functor>
static constexpr void nestedViewLoop(View& view, int shift, Functor&& c) {
    nestedLoop<View::rank>(
        [&](unsigned) {
            return shift;
        },
        [&](unsigned d) {
            return view.extent(d) - shift;
        },
        c);
}

#endif
