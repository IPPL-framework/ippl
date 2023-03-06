#include <tuple>
#include <Kokkos_Core.hpp>

#include "nptr.h"

template <typename T, unsigned Dim>
struct Coords {
	// https://stackoverflow.com/a/53398815/2773311
	// https://en.cppreference.com/w/cpp/utility/declval
	using type = decltype(std::tuple_cat(
				std::declval<typename Coords<T, 1>::type>(),
				std::declval<typename Coords<T, Dim - 1>::type>()
			));
};

template <typename T>
struct Coords<T, 1> {
	using type = std::tuple<T>;
};

template <typename View, typename Coords, size_t... Idx>
KOKKOS_INLINE_FUNCTION
constexpr decltype(auto)
apply_impl(const View& view, const Coords& coords, std::index_sequence<Idx...>) {
    return view(coords[Idx]...);
}

template <unsigned Dim, typename View, typename Coords>
KOKKOS_INLINE_FUNCTION
constexpr decltype(auto)
apply(const View& view, const Coords& coords) {
    using Indices = std::make_index_sequence<Dim>;
    return apply_impl(view, coords, Indices{});
}

template <typename, unsigned, typename> struct Reducer;

template <typename... T, unsigned Dim, typename R>
struct Reducer<std::tuple<T...>, Dim, R> {
	using view_type = Kokkos::View<typename NPtr<R, Dim>::type>;
	view_type view_m;

	KOKKOS_INLINE_FUNCTION
	Reducer(view_type v) : view_m(v) {}

	KOKKOS_INLINE_FUNCTION
	auto operator()(const T&... xs, R& res) const {
		using T1 = std::tuple_element_t<0, std::tuple<T...>>;
		T1 args[sizeof...(T)] = {xs...};

		//res += view_m(args[Idx]...);
        res += apply<Dim>(view_m, args);
	}
};

template <unsigned Dim, typename R, typename C=unsigned int>
using ConvenientReducer = Reducer<typename Coords<C, Dim>::type, Dim, R>;
