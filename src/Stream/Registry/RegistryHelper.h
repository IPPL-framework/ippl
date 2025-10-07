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

/*  NOT NEEDED UP TO 07.10

// ======================================
// // Helper trait to extract T and Dim from ParticleBase types
// template<typename T>
// struct ParticleTraits;

// template<typename T, unsigned Dim, typename... PositionProperties, typename... IDProperties>
// struct ParticleTraits<ippl::ParticleBase<ippl::detail::ParticleLayout<T, Dim, PositionProperties...>, IDProperties...>> {
//     using value_type = T;
//     static constexpr unsigned dimension = Dim;
// };

// // --- Shorthands for ParticleTraits ---
// template<typename ParticleBaseT>
// using particle_value_t = typename ParticleTraits<std::decay_t<ParticleBaseT>>::value_type;

// template<typename ParticleBaseT>
// constexpr unsigned particle_dim_v = ParticleTraits<std::decay_t<ParticleBaseT>>::dimension;

// template<typename ParticleBaseT>
// void foo(const ParticleBaseT& pc) {
//     using T = particle_value_t<ParticleBaseT>;
//     constexpr unsigned Dim = particle_dim_v<ParticleBaseT>;
//     // Now you can use T and Dim directly
// }auto& pc = some_particle_container;
// constexpr unsigned Dim = particle_dim_v<decltype(pc)>; // Correct!
// ======================================

*/

// Helper: matches any Layout<T, Dim, ...>
template<typename Layout>
struct ExtractTypeDim;

template<template<typename, unsigned, typename...> class Layout, typename T, unsigned Dim, typename... Rest>
struct ExtractTypeDim<Layout<T, Dim, Rest...>> {
    using value_type = T;
    static constexpr unsigned dimension = Dim;
};

// General ParticleTraits: matches any class whose first template parameter is a layout
template<typename T>
struct ParticleTraits;

template<   template<typename, typename...> class C,
            typename Layout, typename... OtherArgs>
struct ParticleTraits<C<Layout, OtherArgs...>> : ExtractTypeDim<Layout> {};

// Generic: matches any template class with <T, unsigned Dim>
template<template<typename, unsigned> class PContainer, typename T, unsigned Dim>
struct ParticleTraits<PContainer<T, Dim>> {
    using value_type = T;
    static constexpr unsigned dimension = Dim;
};


template<typename ParticleLikeT>
using particle_value_t = typename ParticleTraits<std::decay_t<ParticleLikeT>>::value_type;

template<typename ParticleLikeT>
constexpr unsigned particle_dim_v = ParticleTraits<std::decay_t<ParticleLikeT>>::dimension;

/*  NOT NEEDED UP TO 07.10
// ======================================
// Helper trait to remove shared_ptr wrapper and get underlying type
template<typename T>
struct UnwrapType {
    using type = T;
};

template<typename T>
struct UnwrapType<std::shared_ptr<T>> {
    using type = T;
};

template<typename T>
using UnwrapType_t = typename UnwrapType<T>::type;
// ======================================

// ======================================
// Helper to check if type is shared_ptr
template<typename T>
struct IsSharedPtr : std::false_type {};

template<typename T>
struct IsSharedPtr<std::shared_ptr<T>> : std::true_type {};

template<typename T>
constexpr bool IsSharedPtr_v = IsSharedPtr<T>::value;
// ======================================
*/
