//
// Type Utilities
//   Metaprogramming utility functions for type manipulation
//
// Copyright (c) 2023, Paul Scherrer Institut, Villigen PSI, Switzerland
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

#ifndef IPPL_TYPE_UTILS_H
#define IPPL_TYPE_UTILS_H

#include <variant>

namespace ippl {
    namespace detail {

        /*!
         * Checks whether a given type is present in a parameter pack
         * @tparam Check the type for whose presence to check
         * @tparam Collection... a collection of types
         */
        template <typename Check, typename... Collection>
        struct IsPresent {
            constexpr static bool value = std::disjunction_v<std::is_same<Check, Collection>...>;
        };

        /*!
         * Base struct declaration (see full declaration below for details)
         */
        template <typename, typename>
        struct ConstructVariant;

        /*!
         * Base case for variant construction with no types to add
         */
        template <>
        struct ConstructVariant<std::variant<>, std::variant<>> {
            typedef std::variant<> type;
        };

        /*!
         * Base case for a fully constructed variant
         * @tparam T... the types to be included in the variant
         */
        template <typename... T>
        struct ConstructVariant<std::variant<>, std::variant<T...>> {
            typedef std::variant<T...> type;
        };

        /*!
         * Constructs a variant type that allows all of the provided types. This is useful
         * if the provided types include duplicates, e.g. if they are type aliases that
         * can sometimes refer to the same type. In particular, Kokkos memory spaces can
         * sometimes have different names but refer to the same memory space. Variants do
         * not allow duplicate types to appear in their parameter packs; each type may only
         * appear once.
         *
         * The struct works by recursively adding types to a variant, but only if the type
         * has not already been added.
         *
         * @tparam Next the next type to add to the variant
         * @tparam ToAdd... the remaining types waiting to be added to the variant
         * @tparam Added... the types that have already been added to the variant
         */
        template <typename Next, typename... ToAdd, typename... Added>
        struct ConstructVariant<std::variant<Next, ToAdd...>, std::variant<Added...>> {
            // Convenience aliases
            template <bool B, class T, class F>
            using cond = std::conditional_t<B, T, F>;
            template <typename... T>
            using variant = std::variant<T...>;

            typedef cond<
                IsPresent<Next, Added...>::value,
                // Type is already present, don't add it
                typename ConstructVariant<variant<ToAdd...>, variant<Added...>>::type,
                // Add the type
                typename ConstructVariant<variant<ToAdd...>, variant<Next, Added...>>::type>
                type;
        };

        /*!
         * Constructs a variant type that allows all of the provided types, but wrapped in
         * another templated type. This is a convenient shorthand for a variant of pointers to
         * different types, for example.
         *
         * Example:
         * ```
         * WrapUnique<shared_ptr, int, float, int>::type == std::variant<shared_ptr<int>,
         * shared_ptr<float>>
         * ```
         *
         * @tparam Wrapper the wrapper type
         * @tparam Types... the types to be wrapped and included in the variant
         */
        template <template <typename> class Wrapper, typename... Types>
        struct WrapUnique {
            typedef typename ConstructVariant<std::variant<Wrapper<Types>...>, std::variant<>>::type
                type;
        };
    }  // namespace detail
}  // namespace ippl

#endif
