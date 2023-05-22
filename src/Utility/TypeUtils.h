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
        using TypeForAllSpaces =
            Type<Kokkos::HostSpace, Kokkos::SharedSpace, Kokkos::SharedHostPinnedSpace
#ifdef KOKKOS_ENABLE_CUDA
                 ,
                 Kokkos::CudaSpace, Kokkos::CudaHostPinnedSpace, Kokkos::CudaUVMSpace
#endif
                 >;

        /*!
         * Variant verification struct
         * Checks that a given type has not already been added to a variant
         * @tparam Check the type for whose presence to check
         * @tparam Collection... a collection of types
         */
        template <typename Check, typename... Collection>
        struct IsUnique {
            constexpr static bool enable = !std::disjunction_v<std::is_same<Check, Collection>...>;
            typedef Check type;
        };

        /*!
         * Defines a variant verification struct
         * Performs the same check as IsUnique, but instead of using the provided types
         * directly, the types are wrapped in another provided type. For example, if the
         * wrapper type is std::shared_ptr and the types are <int, float, int>, then
         * the final variant will allow std::shared_ptr<int> and std::shared_ptr<float>.
         * @tparam Wrapper the wrapper type
         */
        template <template <typename> class Wrapper>
        struct WrapUnique {
            template <typename Check, typename... Collection>
            struct Verifier {
                typedef Wrapper<Check> type;
                constexpr static bool enable =
                    !std::disjunction_v<std::is_same<type, Collection>...>;
            };
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
        template <typename, typename, template <typename...> class Verifier = IsUnique>
        struct ConstructVariant;

        /*!
         * Base case for variant construction with no types to add
         */
        template <template <typename...> class Verifier>
        struct ConstructVariant<std::variant<>, std::variant<>, Verifier> {
            typedef std::variant<> type;
        };

        /*!
         * Base case for a fully constructed variant
         * @tparam T... the types to be included in the variant
         */
        template <typename... T, template <typename...> class Verifier>
        struct ConstructVariant<std::variant<>, std::variant<T...>, Verifier> {
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

            typedef cond<
                Check::enable,
                // The verifier has indicated that this type should be added
                typename ConstructVariant<variant<ToAdd...>,
                                          variant<typename Check::type, Added...>, Verifier>::type,
                // The verifier has indicated that the type should not be added
                typename ConstructVariant<variant<ToAdd...>, variant<Added...>, Verifier>::type>
                type;
        };

        template <typename... Types>
        using VariantFromConditionalTypes =
            typename ConstructVariant<std::variant<Types...>, std::variant<>, IsEnabled>::type;

        template <typename... Types>
        using VariantFromUniqueTypes =
            typename ConstructVariant<std::variant<Types...>, std::variant<>, IsUnique>::type;

        template <template <typename...> class Verifier, typename... Types>
        using VariantWithVerifier =
            typename ConstructVariant<std::variant<Types...>, std::variant<>, Verifier>::type;

        template <template <typename> class Type, typename... Spaces>
        struct MultispaceContainer {
            template <typename T, typename... Ts>
            using Verifier = typename WrapUnique<Type>::template Verifier<T, Ts...>;

            using Types = VariantWithVerifier<Verifier, Spaces...>;

            std::array<Types, sizeof...(Spaces)> elements_m;

            template <typename Space, unsigned Idx = 0>
            constexpr unsigned spaceToIndex() {
                static_assert(Idx < sizeof...(Spaces));
                if constexpr (std::is_same_v<Space,
                                             std::tuple_element_t<Idx, std::tuple<Spaces...>>>) {
                    return Idx;
                } else {
                    return spaceToIndex<Space, Idx + 1>();
                }
            }

        public:
            template <typename Space>
            Type<Space>& get() {
                return std::get<Type<Space>>(elements_m[spaceToIndex<Space>()]);
            }

            template <typename Functor>
            void forAll(Functor&& f) {
                (f(get<Spaces>()), ...);
            }
        };

        template <template <typename> class Type>
        struct ContainerForAllSpaces {
            template <typename... Spaces>
            using container_type = MultispaceContainer<Type, Spaces...>;

            using type = TypeForAllSpaces<container_type>;
        };
    }  // namespace detail
}  // namespace ippl

#endif
