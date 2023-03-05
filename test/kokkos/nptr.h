template <typename T, int N>
struct NPtr {
	using type = typename NPtr<T, N - 1>::type*;
};

template <typename T>
struct NPtr<T, 1> {
	using type = T*;
};
