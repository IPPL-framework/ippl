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

#include <Kokkos_Core.hpp>

#include <variant>

namespace ippl {
    namespace detail {

        /*!
         * Instantiates a parameter pack with all the available Kokkos memory spaces
         */
        template <template <typename...> class Type>
        using TypesForAllSpaces =
            Type<Kokkos::HostSpace, Kokkos::SharedSpace, Kokkos::SharedHostPinnedSpace
#ifdef KOKKOS_ENABLE_CUDA
                 ,
                 Kokkos::CudaSpace, Kokkos::CudaHostPinnedSpace, Kokkos::CudaUVMSpace
#endif
                 >;

        /*!
         * Variant verification struct
         * Checks whether a given type is present in a parameter pack
         * @tparam Check the type for whose presence to check
         * @tparam Collection... a collection of types
         */
        template <typename Check, typename... Collection>
        struct IsPresent {
            constexpr static bool enable = !std::disjunction_v<std::is_same<Check, Collection>...>;
            typedef Check type;
        };

        /*!
         * Convenience alias for types that should or should not be included
         * in variants constructed with ConstructVariant (defined below) based
         * on some compile-time constant
         * @tparam B whether the type should be enabled
         * @tparam T the type
         */
        template <bool B, typename T>
        using ConditionalType = std::conditional_t<B, T, void>;

        /*!
         * Variant verification struct
         * Enables the type if it is not void (intended for use with std::conditional_t
         * where the user passes void if the type should not be included)
         * @tparam Type the type that should be added, or void
         * @tparam ... dummy parameter to ensure compatibility with IsPresent
         */
        template <typename Type, typename...>
        struct IsEnabled {
            constexpr static bool enable = !std::is_void_v<Type>;
            typedef Type type;
        };

        /*!
         * Base struct declaration (see full declaration below for details)
         */
        template <typename, typename, template <typename...> class Verifier = IsPresent>
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

        /*!Constructs a variant type containing all the provided types that fulfill a certain
         * condition. This is done by recursively adding types to the variant based on the
         * inclusion criteria.
         *
         * The default verification struct is IsPresent defined above, which includes the type
         * if it has not already been added to the variant before. This is useful
         * if the provided types include duplicates, e.g. if they are type aliases that
         * can sometimes refer to the same type. In particular, Kokkos memory spaces can
         * sometimes have different names but refer to the same memory space. Variants do
         * not allow duplicate types to appear in their parameter packs; each type may only
         * appear once.
         *
         * The verification struct can be user defined as long as it conforms to the variant
         * verification struct interface. The struct must accept at least a parameter pack; the
         * first parameter is the type currently being checked and the rest are the types already
         * added to the variant. The struct must expose a boolean `enable` that indicates whether
         * the next type should be included in the variant. The struct must also expose a type
         * `type`, which is the type to be added to the variant
         *
         * @tparam Next the next type to add to the variant
         * @tparam ToAdd... the remaining types waiting to be added to the variant
         * @tparam Added... the types that have already been added to the variant
         * @tparam Verifier the variant verification struct
         */
        template <typename Next, typename... ToAdd, typename... Added,
                  template <typename...> class Verifier>
        struct ConstructVariant<std::variant<Next, ToAdd...>, std::variant<Added...>, Verifier> {
            // Convenience aliases
            template <bool B, class T, class F>
            using cond = std::conditional_t<B, T, F>;
            template <typename... T>
            using variant = std::variant<T...>;

            using Check = Verifier<Next, Added...>;

            typedef cond<Check::enable,
                         // The verifier has indicated that this type should be added
                         typename ConstructVariant<variant<ToAdd...>,
                                                   variant<typename Check::type, Added...>>::type,
                         // The verifier has indicated that the type should not be added
                         typename ConstructVariant<variant<ToAdd...>, variant<Added...>>::type>
                type;
        };

        template <typename... Types>
        using VariantFromConditionalTypes =
            typename ConstructVariant<std::variant<Types...>, std::variant<>, IsEnabled>::type;

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
