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
#include "Utility/ViewUtils.h"

#include "gtest/gtest.h"

/*!
 * Utility struct for holding a set of template parameters. It also defines
 * a type alias to ensure that this struct is never nested in itself.
 */
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

/*!
 * Generates parameter packs containing all combinations of a new type
 * and some existing types; used for generating combinations of types
 * @tparam Ts the existing parameters
 * @tparam U the next parameter to add
 */
template <typename... Ts, typename U>
struct AddType<std::tuple<Ts...>, U> {
    using type = std::tuple<typename Parameters<Ts, U>::flat_type...>;
};

/*!
 * Generates parameter packs containing all combinations of types in
 * two sets of parameters; used for generating combinations of types
 * @tparam PackA a parameter pack
 * @tparam TypesB a set of types to add to the existing types
 */
template <typename PackA, typename... TypesB>
struct CombineTuples<PackA, std::tuple<TypesB...>> {
    using type = decltype(std::tuple_cat(std::declval<typename AddType<PackA, TypesB>::type>()...));
};

template <typename Tuple1, typename Tuple2>
struct CreateCombinations<Tuple1, Tuple2> {
    using type = typename CombineTuples<Tuple1, Tuple2>::type;
};

/*!
 * Generates all possible combinations of types contained in the provided tuples
 */
template <typename Tuple1, typename Tuple2, typename... Tuples>
struct CreateCombinations<Tuple1, Tuple2, Tuples...> {
    using first = typename CombineTuples<Tuple1, Tuple2>::type;
    using type  = typename CreateCombinations<first, Tuples...>::type;
};

template <typename>
struct TestForTypes;

/*!
 * Create a set of gtest types for all types in a given tuple
 */
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

/*!
 * Utility struct for defining relevant parameters for IPPL unit tests
 */
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

using ippl::detail::nestedViewLoop, ippl::detail::nestedLoop;

// Allow the user to skip serial execution tests, since they could be slow and don't test anything
// different from OpenMP tests, given that both execution spaces use host memory
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

#endif
