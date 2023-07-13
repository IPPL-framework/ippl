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

template <typename...>
struct CreateCombinations;
template <typename, typename>
struct CombineTuples;
template <typename, typename>
struct AddType;

template <typename... Ts, typename U>
struct AddType<std::tuple<Ts...>, U> {
    using type = std::tuple<std::tuple<Ts, U>...>;
};

template <typename... TypesA, typename... TypesB>
struct CombineTuples<std::tuple<TypesA...>, std::tuple<TypesB...>> {
    using type = decltype(std::tuple_cat(
        std::declval<typename AddType<std::tuple<TypesA...>, TypesB>::type>()...));
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
void assertTypeParam(T valA, T valB) {
    if constexpr (std::is_same_v<T, double>) {
        ASSERT_DOUBLE_EQ(valA, valB);
    } else {
        ASSERT_FLOAT_EQ(valA, valB);
    }
};

struct MixedPrecisionAndSpaces {
    using Spaces     = ippl::detail::TypeForAllSpaces<std::tuple>::type;
    using Precisions = std::tuple<double, float>;
    using Combos     = CreateCombinations<Precisions, Spaces>::type;
    using tests      = TestForTypes<Combos>::type;
};

#endif
