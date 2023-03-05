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

// stdlib specialization (unclear whether this is needed)
/*template <class T, std::size_t Size> struct std::tuple_size<Kokkos::Array<T, Size>>
    : public integral_constant<std::size_t, Size> {};*/

template <typename, typename, unsigned, typename> struct Reducer;

template <typename... T, size_t... Idx, unsigned Dim, typename R>
struct Reducer<std::tuple<T...>, std::index_sequence<Idx...>, Dim, R> {
	using view_type = Kokkos::View<typename NPtr<R, Dim>::type>;
	view_type view_m;

	KOKKOS_INLINE_FUNCTION
	Reducer(view_type v) : view_m(v) {}

	KOKKOS_INLINE_FUNCTION
	auto operator()(const T&... xs, R& res) const {
		using T1 = std::tuple_element_t<0, std::tuple<T...>>;
		T1 args[sizeof...(T)] = {xs...};

		res += view_m(args[Idx]...);
	}
};

template <unsigned Dim, typename R, typename C=unsigned int>
using ConvenientReducer = Reducer<typename Coords<C, Dim>::type, std::make_index_sequence<Dim>, Dim, R>;
