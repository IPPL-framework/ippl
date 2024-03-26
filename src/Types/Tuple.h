//
// Class Tuple
//   Tuple class used for fixed-size containers containing objects of different types, e.g. different KOKKOS_LAMBDAs
//
#ifndef IPPL_Tuple_H
#define IPPL_Tuple_H
#include <Kokkos_Macros.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace ippl {
    /*!
     * @file Tuple.h
     */

    /*!
     * @struct TupleImpl
     * @brief Implementation details for the Tuple class.
     * @tparam i Current index in the Tuple.
     * @tparam N Total number of elements in the Tuple.
     * @tparam Ts REMAINING Types of subsequent elements in the Tuple going forward from i.
     */
    template <std::size_t i, std::size_t N, typename... Ts>
    struct TupleImpl;
    /*!
     * @struct TupleImpl<i, N, T, R, Ts...>
     * @brief Partial specialization of TupleImpl for handling non-terminal elements.
     * @tparam i Current index in the Tuple.
     * @tparam N Total number of elements in the Tuple.
     * @tparam T Type of the current element.
     * @tparam R Type of the next element.
     * @tparam Ts REMAINING Types of subsequent elements in the Tuple going forward from i.
     */
    template <std::size_t i, std::size_t N, typename T, typename R, typename... Ts>
    struct TupleImpl<i, N, T, R, Ts...> /* : TupleImpl<i + 1, N, R, Ts...> */{
        T val;
        /**
         * @brief Remaining tuple elements:
         * next doesn't have typename T anymore.
         */
        TupleImpl<i + 1, N, R, Ts...> next;
        template <std::size_t Idx>
            requires(Idx < N)
        KOKKOS_INLINE_FUNCTION auto& get() noexcept {
            if constexpr (Idx == i) {
                return val;
            } else {
                return next.template get<Idx>();
            }
        }
        template <std::size_t Idx>
            requires(Idx < N)
        KOKKOS_INLINE_FUNCTION const auto& get() const noexcept {
            if constexpr (Idx == i) {
                return val;
            } else {
                return next.template get<Idx>();
            }
        }
        TupleImpl& operator=(const TupleImpl<i, N, T, R, Ts...>& t)
            requires(std::is_copy_assignable_v<T> && std::is_copy_assignable_v<R>
                     && (std::is_copy_assignable_v<Ts> && ...))
        = default;
        TupleImpl& operator=(TupleImpl<i, N, T, R, Ts...>&& t)
            requires(std::is_move_assignable_v<T> && std::is_move_assignable_v<R>
                     && (std::is_move_assignable_v<Ts> && ...))
        = default;
        TupleImpl(const TupleImpl<i, N, T, R, Ts...>& t)
            requires(std::is_copy_constructible_v<T> && std::is_copy_constructible_v<R>
                     && (std::is_copy_constructible_v<Ts> && ...))
        = default;
        TupleImpl(TupleImpl<i, N, T, R, Ts...>&& t)
            requires(std::is_move_constructible_v<T> && std::is_move_constructible_v<R>
                     && (std::is_move_constructible_v<Ts> && ...))
        = default;

        TupleImpl()
            requires(std::is_default_constructible_v<T> && std::is_default_constructible_v<R>
                     && (std::is_default_constructible_v<Ts> && ...))
        = default;
        template <typename CtorT, typename CtorR, typename... CtorTs>
        KOKKOS_INLINE_FUNCTION TupleImpl(CtorT&& t, CtorR&& r, CtorTs&&... ts)
            : val(std::forward<T>(t)), next(std::forward<CtorR>(r), std::forward<CtorTs>(ts)...) {}
    };
    /*!
     * @struct TupleImpl<i, N, T>
     * @brief Partial specialization of TupleImpl for handling the terminal element.
     * @tparam i Current index in the Tuple.
     * @tparam N Total number of elements in the Tuple.
     * @tparam T Type of the terminal element.
     */
    template <std::size_t i, std::size_t N, typename T>
    struct TupleImpl<i, N, T> {
        T val;
        template <std::size_t Idx>
            requires(Idx == N - 1)
        KOKKOS_INLINE_FUNCTION auto& get() noexcept {
            return val;
        }
        template <std::size_t Idx>
            requires(Idx == N - 1)
        KOKKOS_INLINE_FUNCTION const auto& get() const noexcept {
            return val;
        }
        TupleImpl()
            requires(std::is_default_constructible_v<T>)
         = default;        
        KOKKOS_INLINE_FUNCTION
        TupleImpl(const T& t) : val(t){}
        KOKKOS_INLINE_FUNCTION
        TupleImpl(T&& t) : val(std::forward<T>(t)){}
    };
    /*!
     * @class Tuple
     * @brief Generic tuple class with various operations.
     * @tparam Ts Types of elements in the Tuple.
     */
    template <typename... Ts>
    struct Tuple {
        private:
        TupleImpl<0, sizeof...(Ts), Ts...> tupleImpl_m;
        public:
        constexpr static std::size_t dim  = sizeof...(Ts);
        constexpr static std::size_t size = sizeof...(Ts);
        template <std::size_t Idx>
            requires(Idx < sizeof...(Ts))
        KOKKOS_INLINE_FUNCTION auto&& get() noexcept {
            return tupleImpl_m.template get<Idx>();
        }
        template <std::size_t Idx>
            requires(Idx < sizeof...(Ts))
        KOKKOS_INLINE_FUNCTION auto&& get() const noexcept {
            return tupleImpl_m.template get<Idx>();
        }
        template <typename Functor, std::size_t Idx, typename... OtherTs>
        KOKKOS_INLINE_FUNCTION int applySingle(Functor func, const Tuple<OtherTs...>& other)
            requires(std::is_copy_assignable_v<Ts> && ...)
        {
            func(get<Idx>(), other.template get<Idx>());
            return Idx;  // Dummy
        }
        template <typename Functor, typename... OtherTs, std::size_t... Idx>
        KOKKOS_INLINE_FUNCTION void applySequence(Functor func, const Tuple<OtherTs...>& other,
                                                  const std::index_sequence<Idx...>&) {
            int val = (applySingle<Functor, Idx, OtherTs...>(func, other) ^ ...);
            (void)val;  // Dummy for ^
        }
        template <std::size_t Idx, typename... OtherTs>
        KOKKOS_INLINE_FUNCTION int assignToSingle(const Tuple<OtherTs...>& other)
            requires(std::is_copy_assignable_v<Ts> && ...)
        {
            (get<Idx>() = other.template get<Idx>());
            return Idx;  // Dummy
        }
        template <typename... OtherTs, std::size_t... Idx>
        KOKKOS_INLINE_FUNCTION void assignToSequence(const Tuple<OtherTs...>& other,
                                                     const std::index_sequence<Idx...>&)
            requires(std::is_copy_assignable_v<Ts> && ...)
        {
            int val = (assignToSingle<Idx, OtherTs...>(other) ^ ...);
            (void)val;  // Dummy for ^
        }
        template <std::size_t Idx, typename... OtherTs>
        KOKKOS_INLINE_FUNCTION int assignToSingle(Tuple<OtherTs...>&& other)
            requires(std::is_move_assignable_v<Ts> && ...)
        {
            (get<Idx>() = std::move(other.template get<Idx>()));
            return Idx;  // Dummy
        }
        template <typename... OtherTs, std::size_t... Idx>
        KOKKOS_INLINE_FUNCTION void assignToSequence(Tuple<OtherTs...>&& other,
                                                     const std::index_sequence<Idx...>&)
            requires(std::is_move_assignable_v<Ts> && ...)
        {
            int val = (assignToSingle<Idx, OtherTs...>(std::move(other)) ^ ...);
            (void)val;  // Dummy for ^
        }
        template <typename... OtherTs>
        KOKKOS_INLINE_FUNCTION Tuple& operator=(const Tuple<OtherTs...>& other)
            requires(std::is_copy_assignable_v<Ts> && ...)
        {
            assignToSequence(other, std::make_index_sequence<sizeof...(Ts)>{});
            return *this;
        }

        template <typename... OtherTs>
        KOKKOS_INLINE_FUNCTION Tuple& operator=(Tuple<OtherTs...>&& other)
            requires(std::is_move_assignable_v<Ts> && ...)
        {
            assignToSequence(std::move(other), std::make_index_sequence<sizeof...(Ts)>{});
            return *this;
        }
        KOKKOS_INLINE_FUNCTION Tuple& operator+=(const Tuple& other)
            requires(std::is_arithmetic_v<Ts> && ...)
        {
            applySequence(
                KOKKOS_LAMBDA(auto& x, const auto& y) { x += y; }, other,
                std::make_index_sequence<sizeof...(Ts)>{});
            return *this;
        }
        KOKKOS_INLINE_FUNCTION Tuple& operator-=(const Tuple& other)
            requires(std::is_arithmetic_v<Ts> && ...)
        {
            applySequence(
                KOKKOS_LAMBDA(auto& x, const auto& y) { x -= y; }, other,
                std::make_index_sequence<sizeof...(Ts)>{});
            return *this;
        }
        KOKKOS_INLINE_FUNCTION Tuple& operator*=(const Tuple& other)
            requires(std::is_arithmetic_v<Ts> && ...)
        {
            applySequence(
                KOKKOS_LAMBDA(auto& x, const auto& y) { x *= y; }, other,
                std::make_index_sequence<sizeof...(Ts)>{});
            return *this;
        }
        KOKKOS_INLINE_FUNCTION Tuple& operator/=(const Tuple& other)
            requires(std::is_arithmetic_v<Ts> && ...)
        {
            applySequence(
                KOKKOS_LAMBDA(auto& x, const auto& y) { x /= y; }, other,
                std::make_index_sequence<sizeof...(Ts)>{});
            return *this;
        }
        KOKKOS_INLINE_FUNCTION Tuple operator+(const Tuple& other) const
            requires(std::is_arithmetic_v<Ts> && ...)
        {
            return Tuple(Tuple(*this) += other);
        }
        KOKKOS_INLINE_FUNCTION Tuple operator-(const Tuple& other) const
            requires(std::is_arithmetic_v<Ts> && ...)
        {
            return Tuple(Tuple(*this) -= other);
        }
        KOKKOS_INLINE_FUNCTION Tuple operator*(const Tuple& other) const
            requires(std::is_arithmetic_v<Ts> && ...)
        {
            return Tuple(Tuple(*this) *= other);
        }
        KOKKOS_INLINE_FUNCTION Tuple operator/(const Tuple& other) const
            requires(std::is_arithmetic_v<Ts> && ...)
        {
            return Tuple(Tuple(*this) /= other);
        }
        template <std::size_t Idx, std::size_t N, typename... OtherTs>
        KOKKOS_INLINE_FUNCTION
        bool lexicographicalLess(const Tuple& other) const {
            if constexpr (Idx == N) {
                return false;
            } else {
                return (get<Idx>() < other.template get<Idx>())
                           ? true
                           : lexicographicalLess<Idx + 1, N, OtherTs...>(other);
            }
        }
        template <std::size_t Idx, std::size_t N, typename... OtherTs>
        KOKKOS_INLINE_FUNCTION
        bool lexicographicalEquals(const Tuple& other) const {
            if constexpr (Idx == N) {
                return true;
            } else {
                return (get<Idx>() == other.template get<Idx>())
                       && lexicographicalEquals<Idx + 1, N, OtherTs...>(other);
            }
        }
        template <typename... OtherTs>
        KOKKOS_INLINE_FUNCTION bool operator<(const Tuple<OtherTs...>& other) const
            requires((sizeof...(Ts) == sizeof...(OtherTs)) && (std::totally_ordered<Ts> && ...))
        {
            return lexicographicalLess<0, sizeof...(Ts), OtherTs...>(other);
        }
        template <typename... OtherTs>
        KOKKOS_INLINE_FUNCTION bool operator==(const Tuple<OtherTs...>& other) const
            requires((sizeof...(Ts) == sizeof...(OtherTs)) && (std::totally_ordered<Ts> && ...))
        {
            return lexicographicalEquals<0, sizeof...(Ts)>(other);
        }
        Tuple& operator=(const Tuple<Ts...>& other)
            requires(std::is_copy_assignable_v<Ts> && ...)
        = default;

        Tuple& operator=(Tuple<Ts...>&& other)
            requires(std::is_move_assignable_v<Ts> && ...)
        = default;

        Tuple(const Tuple<Ts...>& t)
            requires(std::is_copy_constructible_v<Ts> && ...)
        = default;
        Tuple(Tuple<Ts...>&& t)
            requires(std::is_move_constructible_v<Ts> && ...)
        = default;

        Tuple()
            requires(std::is_default_constructible_v<Ts> && ...)
        = default;
        template <typename... CtorTs>
            requires(std::constructible_from<Ts, CtorTs> && ...)
        KOKKOS_INLINE_FUNCTION Tuple(CtorTs&&... args)
            : tupleImpl_m(std::forward<Ts>(args)...) {}
    };
    /*!
     * @brief Accessor function to get an element mutable reference at a specific index from a
     * Tuple.
     * @tparam Idx Index of the element to retrieve.
     * @tparam Ts Types of elements in the Tuple.
     * @param t Tuple from which to retrieve the element.
     * @return Reference to the specified element.
     */
    template <std::size_t Idx, typename... Ts>
    KOKKOS_INLINE_FUNCTION auto& get(Tuple<Ts...>& t) {
        return t.template get<Idx>();
    }

    /*!
     * @brief Accessor function to get a element const reference at a specific index from a Tuple.
     * @tparam Idx Index of the element to retrieve.
     * @tparam Ts Types of elements in the Tuple.
     * @param t Tuple from which to retrieve the element.
     * @return Reference to the specified element.
     */
    template <std::size_t Idx, typename... Ts>
    KOKKOS_INLINE_FUNCTION const auto& get(const Tuple<Ts...>& t) {
        return t.template get<Idx>();
    }

    /*!
     * @brief Function to create a Tuple with specified elements.
     * @tparam Ts Types of elements in the Tuple.
     * @param args Elements to initialize the Tuple.
     * @return Newly created Tuple.
     */
    template <typename... Ts>
    KOKKOS_INLINE_FUNCTION Tuple<Ts...> makeTuple(Ts&&... args) {
        return Tuple<Ts...>(std::forward<Ts>(args)...);
    }
    template <std::size_t Idx, typename T, typename... Ts>
    struct TupleTypeImpl {
        using type = TupleTypeImpl<Idx - 1, Ts...>;
    };
    template <typename T, typename... Ts>
    struct TupleTypeImpl<0, T, Ts...> {
        using type = T;
    };
    template <std::size_t Idx, typename... Ts>
    using TupleType = typename TupleTypeImpl<Idx, Ts...>::type;
}  // namespace ippl

namespace std {
    template <typename... Ts> struct tuple_size<::ippl::Tuple<Ts...>> : std::integral_constant<size_t, sizeof...(Ts)> { };

    template <size_t Idx, typename... Ts> struct tuple_element<Idx, ::ippl::Tuple<Ts...>> { using type = typename ::ippl::TupleType<Idx, Ts...>; };

    template<size_t Idx, typename... Ts>
    KOKKOS_INLINE_FUNCTION auto& get(::ippl::Tuple<Ts...>& t){
        return t.template get<Idx>();
    }
    template<size_t Idx, typename... Ts>
    KOKKOS_INLINE_FUNCTION const auto& get(const ::ippl::Tuple<Ts...>& t){
        return t.template get<Idx>();
    }
}
#endif
