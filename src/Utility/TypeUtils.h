#include <tuple>
#include <variant>

namespace ippl {
    namespace detail {

        template <typename Check, typename... Collection>
        struct IsPresent {
            constexpr static bool value = std::disjunction_v<std::is_same<Check, Collection>...>;
        };

        template <typename, typename>
        struct CollapseTypes;

        template <>
        struct CollapseTypes<std::tuple<>, std::tuple<>> {
            typedef std::tuple<> type;
        };

        template <typename... T>
        struct CollapseTypes<std::tuple<>, std::tuple<T...>> {
            typedef std::tuple<T...> type;
        };

        template <typename Next, typename... ToAdd, typename... Added>
        struct CollapseTypes<std::tuple<Next, ToAdd...>, std::tuple<Added...>> {
            // Convenience aliases
            template <bool B, class T, class F>
            using cond = std::conditional_t<B, T, F>;
            template <typename... T>
            using tuple = std::tuple<T...>;

            typedef cond<IsPresent<Next, Added...>::value,
                         // Type is already present
                         typename CollapseTypes<tuple<ToAdd...>, tuple<Added...>>::type,
                         // Add the type
                         typename CollapseTypes<tuple<ToAdd...>, tuple<Next, Added...>>::type>
                type;
        };

        template <typename>
        struct TupleToVariant {};

        template <typename... T>
        struct TupleToVariant<std::tuple<T...>> {
            typedef std::variant<T...> type;
        };

        template <template <typename> class Wrapper, typename... Types>
        struct WrapUnique {
            typedef typename CollapseTypes<std::tuple<Wrapper<Types>...>, std::tuple<>>::type
                UniqueTypes;
            typedef typename TupleToVariant<UniqueTypes>::type type;
        };
    }  // namespace detail
}  // namespace ippl
