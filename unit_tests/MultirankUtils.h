//
// Utility class for rank independent unit testing
//   Provides a framework with which unit tests can easily work with
//   objects of different dimensionalities.
//
// Copyright (c) 2023 Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef MULTIRANK_UTILS_H
#define MULTIRANK_UTILS_H

#include <array>
#include <tuple>

template <unsigned... Dims>
class MultirankUtils {
    // Checking for specialization
    // https://stackoverflow.com/a/28796458
    template <typename, template <typename...> class>
    struct is_specialization : std::false_type {};

    template <template <typename...> class Ref, typename... Args>
    struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

    template <typename Functor, typename... Args, unsigned long... Idx>
    void apply_impl(Functor& f, std::tuple<Args...>& args, const std::index_sequence<Idx...>&) {
        if constexpr (sizeof...(Args) == 0) {
            // Dim == Idx + 1
            (f.template operator()<Idx + 1>(), ...);
        } else if constexpr (is_specialization<std::tuple_element_t<0, std::tuple<Args...>>,
                                               std::tuple>::value) {
            (std::apply(f, std::get<Idx>(args)), ...);
        } else {
            (f(std::get<Idx>(args)), ...);
        }
    }

    template <typename Tester, size_t... Idx>
    void setup_impl(Tester&& t, const std::index_sequence<Idx...>&) {
        ((t->template setupDim<Idx, Dims>()), ...);
    }

    // Tuple zipping
    // https://stackoverflow.com/a/47127033
    template <size_t Idx, typename... Tuples>
    using zipped_element = std::tuple<std::tuple_element_t<Idx, std::decay_t<Tuples>>&...>;

    template <size_t Idx, typename... Tuples>
    zipped_element<Idx, Tuples...> zip_at(Tuples&&... ts) {
        return {std::get<Idx>(std::forward<Tuples>(ts))...};
    }

    template <typename... Tuples, size_t... Idx>
    std::tuple<zipped_element<Idx, Tuples...>...> zip_impl(Tuples&&... ts,
                                                           const std::index_sequence<Idx...>&) {
        return {zip_at<Idx>(std::forward<Tuples>(ts)...)...};
    }

protected:
    template <template <unsigned Dim> class Type>
    using Collection = std::tuple<Type<Dims>...>;

    template <template <typename> class Pointer, template <unsigned Dim> class Type>
    using PtrCollection = std::tuple<Pointer<Type<Dims>>...>;

public:
    template <typename Tester>
    void setup(Tester&& t) {
        setup_impl(t, std::make_index_sequence<sizeof...(Dims)>{});
    }

    template <typename Functor, typename... Args>
    auto apply(Functor& f, std::tuple<Args...>& args) {
        return apply_impl(f, args, std::make_index_sequence<sizeof...(Dims)>{});
    }

    template <typename Functor>
    auto apply(Functor& f) {
        auto args = std::tuple<>{};
        return apply_impl(f, args, std::make_index_sequence<sizeof...(Dims)>{});
    }

    template <typename Tuple, typename... Tuples>
    auto zip(Tuple&& t0, Tuples&&... ts) {
        constexpr size_t size = std::tuple_size_v<std::decay_t<Tuple>>;
        static_assert(((std::tuple_size_v<std::decay_t<Tuples>> == size) && ...),
                      "Mismatched tuple sizes");

        return zip_impl<Tuple, Tuples...>(std::forward<Tuple>(t0), std::forward<Tuples>(ts)...,
                                          std::make_index_sequence<size>{});
    }

    // https://stackoverflow.com/questions/34535795/n-dimensionally-nested-metaloops-with-templates
    template <unsigned Dim, class BeginFunctor, class EndFunctor, class Functor>
    constexpr void nestedLoop(BeginFunctor&& begin, EndFunctor&& end, Functor&& c) {
        for (size_t i = begin(Dim); i < end(Dim); ++i) {
            if constexpr (Dim == 1) {
                c(i);
            } else {
                auto next = [i, &c](auto... args) {
                    c(i, args...);
                };
                nestedLoop<Dim - 1>(begin, end, next);
            }
        }
    }

    template <unsigned Dim, typename View, class Functor>
    constexpr void nestedViewLoop(View& view, int shift, Functor&& c) {
        nestedLoop<Dim>(
            [&](unsigned) {
                return shift;
            },
            [&](unsigned d) {
                return view.extent(Dim - d) - shift;
            },
            c);
    }
};

#endif
