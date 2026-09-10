/**
 * @file RegistryHelper.h
 * @brief Compile-time utilities and traits visualization registries.
 */
#pragma once
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

namespace ippl{



// Helper: matches any Layout<T, Dim, ...>
template<typename Layout>
struct ExtractTypeDim;

template<template<typename, unsigned, typename...> class Layout, typename T, unsigned Dim, typename... Rest>
struct ExtractTypeDim<Layout<T, Dim, Rest...>> {
    using value_type = T;
    static constexpr unsigned DIMENSION = Dim;
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
    static constexpr unsigned DIMENSION = Dim;
};


template<typename ParticleLikeT>
using particle_value_t = typename ParticleTraits<std::decay_t<ParticleLikeT>>::value_type;

template<typename ParticleLikeT>
constexpr unsigned particle_dim_v = ParticleTraits<std::decay_t<ParticleLikeT>>::DIMENSION;


}