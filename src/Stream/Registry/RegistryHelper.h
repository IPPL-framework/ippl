#pragma once
// Fluent variant: forward declarations, utilities, and public includes
// This mirrors src_static's sectioned layout, with notes on fluent-specific changes.
//
// Differences vs src_static:
// - Uses RegistryFluent instead of RegistryTyped (no m_names tracking).
// - Provides id_tag/id<> tag-based API utilities.
// - VisBaseAdaptor supports fluent add<"ID"> that returns a new type (builder pattern).
// - Shared_ptr-based registry ownership and construction from existing registries.

// === Standard library includes (shared across components) ===
#include <any>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <initializer_list>



// === Compile-time ID utilities ===
// fixed_string: C++20 NTTP string literal wrapper used to name registry slots at compile time.
// operator== enables comparing IDs at compile time.

template <std::size_t N>
struct fixed_string {
    char data[N]{};
    constexpr fixed_string(const char (&str)[N]) {
        for (std::size_t i = 0; i < N; ++i) data[i] = str[i];
    }
    constexpr std::string_view sv() const { return std::string_view{data, N - 1}; }
};

template <std::size_t N, std::size_t M>
constexpr bool operator==(const fixed_string<N>& a, const fixed_string<M>& b) {
    if constexpr (N != M) return false;
    for (std::size_t i = 0; i < N; ++i) if (a.data[i] != b.data[i]) return false;
    return true;
}

// Tag type and inline variable to reference IDs without angle brackets
// Fluent-only: avoids needing 'template' at call sites in dependent contexts.
// Usage: vis.get(id<"density">)

template <fixed_string Id>
struct id_tag { static constexpr auto value = Id; };

template <fixed_string Id>
inline constexpr id_tag<Id> id{};

// Slot<"id", T> declares one compile-time entry.
template <fixed_string IdV, typename T>
struct Slot {
    static constexpr auto Id = IdV;
    using type = T;
};

// === Registry metaprogramming (compile-time Slots and lookup) ===
// nth<I, Ts...> selects the I-th type from a parameter pack.
// find_index_rec locates the index of a Slot by its compile-time ID (or -1 if not found).

template <std::size_t I, typename First, typename... Rest>
struct nth { using type = typename nth<I - 1, Rest...>::type; };

template <typename First, typename... Rest>
struct nth<0, First, Rest...> { using type = First; };

// index_of helper
template <auto IdV, std::size_t I, typename First, typename... Rest>
struct find_index_rec {
    static constexpr std::size_t value =
        (First::Id == IdV) ? I : find_index_rec<IdV, I + 1, Rest...>::value;
};

template <auto IdV, std::size_t I, typename Last>
struct find_index_rec<IdV, I, Last> {
    static constexpr std::size_t value = (Last::Id == IdV) ? I : static_cast<std::size_t>(-1);
};

// meta: ensure no duplicate IDs in Slots
template <typename...>
struct ids_unique : std::true_type {};

template <typename S, typename... Rest>
struct ids_unique<S, Rest...>
    : std::bool_constant<((!(Rest::Id == S::Id)) && ...) && ids_unique<Rest...>::value> {};

// Helper dependent false
template <auto>
struct always_false_id : std::false_type {};

// === Forward declarations to break circular dependencies ===

template <typename... Slots>
class RegistryFluent;

template <typename... Slots>
class VisBaseAdaptor;

class RegistryBase;

template<typename T, unsigned Dim>
class ParticleBase;

template<typename T, unsigned Dim>
class Field;

// === Aggregate public headers ===
// Including public project headers here allows end-users to include only this file.
#include "VisRegistry.h"
#include "VisBase.h"


