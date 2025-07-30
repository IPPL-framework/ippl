//
// Type Utilities
//   Metaprogramming utility functions for type manipulation
//

#ifndef IPPL_TYPE_UTILS_H
#define IPPL_TYPE_UTILS_H

#include <Kokkos_Core.hpp>

#include "Types/Variant.h"

#include "Utility/IpplException.h"

namespace ippl {
    namespace detail {

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

        /*!
         * Constructs a variant type containing all the provided types that fulfill a certain
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

        /*!
         * A variant containing all the enabled types,
         * where "enabled" types are assumed to be void
         * when disabled (i.e. std::conditional_t<B, T, void>)
         */
        template <typename... Types>
        using VariantFromConditionalTypes =
            typename ConstructVariant<std::variant<Types...>, std::variant<>, IsEnabled>::type;

        /*!
         * A variant containing just the unique types
         * from the pack
         */
        template <typename... Types>
        using VariantFromUniqueTypes =
            typename ConstructVariant<std::variant<Types...>, std::variant<>, IsUnique>::type;

        /*!
         * A variant containing the types enabled by a custom
         * verifier; to implement a custom verifier, provide the following:
         * - template <typename Next, typename... Added>
         *   Next: the next input type to check
         *   Added: the types that have already been added
         * - bool enable: whether the type should be added
         * - typename type: the output type to be added
         */
        template <template <typename...> class Verifier, typename... Types>
        using VariantWithVerifier =
            typename ConstructVariant<std::variant<Types...>, std::variant<>, Verifier>::type;

        /*!
         * Utility struct for forwarding parameter packs
         * (see specializations)
         * @tparam Type the templated type
         * @tparam Pack a type containing the parameters to forward
         */
        template <template <typename...> class Type, typename Pack>
        struct Forward;

        /*!
         * Forwards the types in a variant to another type
         */
        template <template <typename...> class Type, typename... Spaces>
        struct Forward<Type, std::variant<Spaces...>> {
            using type = Type<Spaces...>;
        };

        /*!
         * Forwards the properties of a Kokkos view to another type
         */
        template <template <typename...> class Type, typename T, typename... Properties>
        struct Forward<Type, Kokkos::View<T, Properties...>> {
            using type = Type<Properties...>;
        };

        /*!
         * Constructs a uniform type based on Kokkos views' uniform
         * types (i.e. a type where all optional template parameters
         * are explicitly specified)
         * @tparam Type the type to specialize
         * @tparam View the view type
         */
        template <template <typename...> class Type, typename View>
        struct CreateUniformType {
            using view_type = typename View::uniform_type;
            using type      = typename Forward<Type, view_type>::type;
        };

        /*!
         * Instantiates a parameter pack with all the available Kokkos memory spaces
         */
        template <template <typename...> class Type>
        struct TypeForAllSpaces {
            using unique_memory_spaces = VariantFromUniqueTypes<
                Kokkos::HostSpace, Kokkos::SharedSpace, Kokkos::SharedHostPinnedSpace
#ifdef KOKKOS_ENABLE_CUDA
                ,
                Kokkos::CudaSpace, Kokkos::CudaHostPinnedSpace, Kokkos::CudaUVMSpace
#endif
#ifdef KOKKOS_ENABLE_HIP
                ,
                Kokkos::HIPSpace, Kokkos::HIPHostPinnedSpace, Kokkos::HIPManagedSpace
#endif
#ifdef KOKKOS_ENABLE_SYCL
                ,
                Kokkos::Experimental::SYCLDeviceUSMSpace, Kokkos::Experimental::SYCLHostUSMSpace,
                Kokkos::Experimental::SYCLSharedUSMSpace
#endif
                >;

            using unique_exec_spaces = VariantFromUniqueTypes<Kokkos::DefaultExecutionSpace
#ifdef KOKKOS_ENABLE_OPENMP
                                                              ,
                                                              Kokkos::OpenMP
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
                                                              ,
                                                              Kokkos::OpenMPTarget
#endif
#ifdef KOKKOS_ENABLE_THREADS
                                                              ,
                                                              Kokkos::Threads
#endif
#ifdef KOKKOS_ENABLE_SERIAL
                                                              ,
                                                              Kokkos::Serial
#endif
#ifdef KOKKOS_ENABLE_CUDA
                                                              ,
                                                              Kokkos::Cuda
#endif
#ifdef KOKKOS_ENABLE_HIP
                                                              ,
                                                              Kokkos::HIP
#endif
#ifdef KOKKOS_ENABLE_SYCL
                                                              ,
                                                              Kokkos::Experimental::SYCL
#endif
#ifdef KOKKOS_ENABLE_HPX
                                                              ,
                                                              Kokkos::HPX
#endif
                                                              >;

            using memory_spaces_type = typename Forward<Type, unique_memory_spaces>::type;
            using exec_spaces_type   = typename Forward<Type, unique_exec_spaces>::type;
        };

        /*!
         * A container indexed by type instead of by numerical indices;
         * designed for storing elements associated with Kokkos memory spaces
         * @tparam Type the element type
         * @tparam Spaces... the memory spaces of interest
         */
        template <template <typename> class Type, typename... Spaces>
        class MultispaceContainer {
            template <typename T, typename... Ts>
            using Verifier = typename WrapUnique<Type>::template Verifier<T, Ts...>;

            using Types = VariantWithVerifier<Verifier, Spaces...>;

            std::array<Types, sizeof...(Spaces)> elements_m;

            /*!
             * Locates an element associated with a space
             * @tparam Space the memory space
             * @return The numerical index for that space's element
             */
            template <typename Space, unsigned Idx = 0>
            constexpr static unsigned spaceToIndex() {
                static_assert(Idx < sizeof...(Spaces));
                if constexpr (std::is_same_v<Space,
                                             std::tuple_element_t<Idx, std::tuple<Spaces...>>>) {
                    return Idx;
                } else {
                    return spaceToIndex<Space, Idx + 1>();
                }
                // Silences incorrect nvcc warning: missing return statement at end of non-void
                // function
                throw IpplException("detail::MultispaceContainer::spaceToIndex",
                                    "Unreachable state");
            }

            /*!
             * Initializes the element for a space
             */
            template <typename Space>
            void initElements() {
                elements_m[spaceToIndex<Space>()] = Type<Space>{};
            }

            /*!
             * Determine whether the element for a space should be initialized,
             * possibly based on a predicate functor
             */
            template <typename MemorySpace, typename Filter,
                      std::enable_if_t<std::is_null_pointer_v<std::decay_t<Filter>>, int> = 0>
            constexpr bool copyToSpace(Filter&&) {
                return true;
            }

            template <typename MemorySpace, typename Filter,
                      std::enable_if_t<!std::is_null_pointer_v<std::decay_t<Filter>>, int> = 0>
            bool copyToSpace(Filter&& predicate) {
                return predicate.template operator()<MemorySpace>();
            }

        public:
            MultispaceContainer() { (initElements<Spaces>(), ...); }

            /*!
             * Constructs a container where all spaces have a mirror with
             * the same data as the provided data structure; a predicate
             * functor can be provided to skip any undesired memory spaces
             * @tparam DataType the type of the provided element
             * @tparam Filter the predicate type, or nullptr_t if there is no predicate
             * @param data the original data
             * @param predicate an optional functor that determines which memory spaces need a copy
             * of the data
             */
            template <typename DataType, typename Filter = std::nullptr_t>
            MultispaceContainer(const DataType& data, Filter&& predicate = nullptr)
                : MultispaceContainer() {
                using space = typename DataType::memory_space;
                static_assert(std::is_same_v<DataType, Type<space>>);

                elements_m[spaceToIndex<space>()] = data;
                copyToOtherSpaces<space>(predicate);
            }

            /*!
             * Copies the data from one memory space to all other memory spaces
             * @tparam Space the source space
             * @tparam Filter the predicate type
             * @param predicate an optional functor that determines which memory spaces need a copy
             * of the data
             */
            template <typename Space, typename Filter = std::nullptr_t>
            void copyToOtherSpaces(Filter&& predicate = nullptr) {
                forAll([&]<typename DataType>(DataType& dst) {
                    using memory_space = typename DataType::memory_space;
                    if constexpr (!std::is_same_v<Space, memory_space>) {
                        if (copyToSpace<memory_space>(predicate)) {
                            dst = Kokkos::create_mirror_view_and_copy(
                                Kokkos::view_alloc(memory_space{}, Kokkos::WithoutInitializing),
                                get<Space>());
                        }
                    }
                });
            }

            /*!
             * Accessor for a space's element
             * @tparam Space the memory space
             * @return The element associated with that space
             */
            template <typename Space>
            const Type<Space>& get() const {
                return std::get<Type<Space>>(elements_m[spaceToIndex<Space>()]);
            }

            template <typename Space>
            Type<Space>& get() {
                return std::get<Type<Space>>(elements_m[spaceToIndex<Space>()]);
            }

            /*!
             * Performs an action for each element
             * @tparam Functor the functor type
             * @param f a functor taking an element for a given space
             */
            template <typename Functor>
            void forAll(Functor&& f) const {
                (f(get<Spaces>()), ...);
            }

            template <typename Functor>
            void forAll(Functor&& f) {
                (f(get<Spaces>()), ...);
            }
        };

        /*!
         * Constructs a MultispaceContainer for all the available Kokkos memory spaces
         * @tparam Type the element type
         */
        template <template <typename> class Type>
        struct ContainerForAllSpaces {
            template <typename... Spaces>
            using container_type = MultispaceContainer<Type, Spaces...>;

            using type = typename TypeForAllSpaces<container_type>::memory_spaces_type;

            // Static factory function that takes a lambda to initialize each memory space
            template <typename Functor>
            static type createContainer(Functor&& initFunc) {
                return type{std::forward<Functor>(initFunc)};
            }
        };

        /*!
         * Performs an action for all memory spaces
         * @tparam Functor the functor type
         * @param f a functor object whose call operator takes a memory space as a template
         * parameter
         */
        template <typename Functor>
        void runForAllSpaces(Functor&& f) {
            using all_spaces = typename TypeForAllSpaces<std::variant>::memory_spaces_type;
            auto runner      = [&]<typename... Spaces>(const std::variant<Spaces...>&) {
                (f.template operator()<Spaces>(), ...);
            };
            runner(all_spaces{});
        }
    }  // namespace detail
}  // namespace ippl

#endif
