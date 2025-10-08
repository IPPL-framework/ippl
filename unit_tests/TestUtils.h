//
// Utilities for versatile unit testing
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

/*!
 * Numerical tolerance for equality checks for computed results
 * @tparam T precision
 */
template <typename T>
constexpr T tolerance = 10.*Kokkos::Experimental::epsilon_v<T>; 

/*!
 * Verifies that two values are equal to the correct level of precision
 * @tparam T data type
 */
template <typename T>
void assertEqual(T valA, T valB) {
    if constexpr (std::is_same_v<T, double>) {
        ASSERT_DOUBLE_EQ(valA, valB);
    } else {
        ASSERT_FLOAT_EQ(valA, valB);
    }
};

/*!
 * Generates the mesh refinement for unit tests such that the refinement and domain
 * are heterogeneous along all axes
 * @tparam Dim number of dimensions
 * @return Mesh refinement
 */
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

/*!
 * Dummy struct for holding an unsigned number as a template parameter
 * @param _ number of dimensions
 */
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
};

using ippl::detail::nestedViewLoop, ippl::detail::nestedLoop;

#endif
