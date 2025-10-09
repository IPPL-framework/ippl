#ifndef VISREGISTRY2_H
#define VISREGISTRY2_H

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>

#include "Stream/Registry/RegistryHelper.h"

// namespace ippl {

/**
 * @brief A type-erased registry that can store arbitrary types with string labels
 * and provide type-safe iteration over stored entries.
 * 
 * This class uses variadic templates and std::tuple to store heterogeneous types
 * while maintaining type safety during iteration.
 */
template<typename... Types>
class VisRegistry_mini {
public:
    using storage_type = std::tuple<std::pair<std::string, Types>...>;

private:
    storage_type storage_;

    // Helper to apply a function to each element in the tuple
    template<std::size_t I = 0, typename Func>
    void for_each_impl(Func&& func) const {
        if constexpr (I < sizeof...(Types)) {
            const auto& pair = std::get<I>(storage_);
            func(pair.first, pair.second);
            for_each_impl<I + 1>(std::forward<Func>(func));
        }
    }

public:
    /**
     * @brief Constructor that takes pairs of labels and objects
     */
    explicit VisRegistry_mini(std::pair<std::string, Types>... entries)
        : storage_(std::move(entries)...) {}

    /**
     * @brief Iterate over all stored entries with a callback function
     * 
     * The callback function should accept (const std::string& label, const auto& entry)
     * and will be called for each stored entry with the correct type.
     * 
     * @param func Callback function that will be called for each entry
     */
    template<typename Func>
    void for_each(Func&& func) const {
        for_each_impl(std::forward<Func>(func));
    }

    /**
     * @brief Get the number of stored entries
     */
    static constexpr std::size_t size() {
        return sizeof...(Types);
    }

    /**
     * @brief Check if the registry is empty
     */
    static constexpr bool empty() {
        return sizeof...(Types) == 0;
    }
};

namespace detail {
    // Helper to extract types from alternating label-object pattern
    template<typename... Args>
    struct extract_object_types;
    
    template<>
    struct extract_object_types<> {
        using type = std::tuple<>;
    };
    
    template<typename Label, typename Object, typename... Rest>
    struct extract_object_types<Label, Object, Rest...> {
        using rest_types = typename extract_object_types<Rest...>::type;
        using type = decltype(std::tuple_cat(
            std::tuple<std::decay_t<Object>>{},
            rest_types{}
        ));
    };
    
    template<typename... Args>
    using extract_object_types_t = typename extract_object_types<Args...>::type;
    
    // Helper to convert tuple of types to VisRegistry_mini template
    template<typename Tuple>
    struct tuple_to_registry;
    
    template<typename... Types>
    struct tuple_to_registry<std::tuple<Types...>> {
        using type = VisRegistry_mini<Types...>;
    };
    
    template<typename Tuple>
    using tuple_to_registry_t = typename tuple_to_registry<Tuple>::type;
    
    // Helper implementation for building pairs (forward declaration needed)
    template<typename Label, typename Object, typename... Rest>
    auto build_pairs_impl(Label&& label, Object&& obj, Rest&&... rest) {
        auto pair = std::make_pair(std::string(std::forward<Label>(label)), std::forward<Object>(obj));
        if constexpr (sizeof...(rest) == 0) {
            return std::make_tuple(std::move(pair));
        } else {
            auto rest_pairs = build_pairs_impl(std::forward<Rest>(rest)...);
            return std::tuple_cat(std::make_tuple(std::move(pair)), std::move(rest_pairs));
        }
    }
    
    // Helper to build pairs from alternating args
    template<typename... Args>
    auto build_pairs(Args&&... args) {
        static_assert(sizeof...(args) % 2 == 0, "Arguments must come in label-object pairs");
        return build_pairs_impl(std::forward<Args>(args)...);
    }
    
    // Helper to construct registry from tuple of pairs
    template<typename Registry, typename PairsTuple>
    struct construct_registry;
    
    template<typename Registry, typename... Pairs>
    struct construct_registry<Registry, std::tuple<Pairs...>> {
        static auto create(std::tuple<Pairs...>&& pairs) {
            return std::make_shared<Registry>(std::apply([](auto&&... ps) {
                return Registry{std::forward<decltype(ps)>(ps)...};
            }, std::move(pairs)));
        }
    };
}

/**
 * @brief Factory function to create a VisRegistry_mini with automatic type deduction
 * 
 * Usage: auto registry = VisRegistry_mini("field1", field_obj, "particles", particle_obj);
 * 
 * @param args Alternating label-object pairs
 * @return std::shared_ptr to the created VisRegistry_mini
 */
template<typename... Args>
auto MakeVisRegistry_mini(Args&&... args) {
    static_assert(sizeof...(args) % 2 == 0, "Arguments must come in label-object pairs");
    static_assert(sizeof...(args) > 0, "At least one label-object pair is required");
    
    using object_types = detail::extract_object_types_t<Args...>;
    using registry_type = detail::tuple_to_registry_t<object_types>;
    
    auto pairs = detail::build_pairs(std::forward<Args>(args)...);
    return detail::construct_registry<registry_type, decltype(pairs)>::create(std::move(pairs));
}

/**
 * @brief Alternative factory function with compile-time string labels
 * 
 * Usage: auto registry = MakeVisRegistryNamed<"field1", "particles">(field_obj, particle_obj);
 * Note: Requires C++20 for template string parameters
 */
#if __cplusplus >= 202002L
template<fixed_string Label1, fixed_string... Labels, typename... Types>
auto MakeVisRegistryNamed(Types&&... objects) {
    static_assert(sizeof...(Labels) + 1 == sizeof...(Types), 
                  "Number of labels must match number of objects");
    
    using registry_type = VisRegistry_mini<std::decay_t<Types>...>;
    return std::make_shared<registry_type>(
        std::make_pair(std::string(Label1.sv()), std::forward<Types>(objects))...
    );
}
#endif

// } // namespace ippl

#endif // VISREGISTRY2_H