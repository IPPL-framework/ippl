//
// Parallel dispatch
//   Utility functions relating to parallel dispatch in IPPL
//

#ifndef IPPL_PARALLEL_DISPATCH_H
#define IPPL_PARALLEL_DISPATCH_H

#include <Kokkos_Core.hpp>

#include <tuple>

#include "Types/Vector.h"

#include "Utility/IpplException.h"

namespace ippl {
    /*!
     * Wrapper type for Kokkos range policies with some convenience aliases
     * @tparam Dim range policy rank
     * @tparam PolicyArgs... additional template parameters for the range policy
     */
    template <unsigned Dim, class... PolicyArgs>
    struct RangePolicy {
        // The range policy type
        using policy_type = Kokkos::MDRangePolicy<PolicyArgs..., Kokkos::Rank<Dim>>;
        // The index type used by the range policy
        using index_type = typename policy_type::array_index_type;
        // A vector type containing the index type
        using index_array_type = ::ippl::Vector<index_type, Dim>;
    };

    /*!
     * Specialized range policy for one dimension.
     */
    template <class... PolicyArgs>
    struct RangePolicy<1, PolicyArgs...> {
        using policy_type      = Kokkos::RangePolicy<PolicyArgs...>;
        using index_type       = typename policy_type::index_type;
        using index_array_type = ::ippl::Vector<index_type, 1>;
    };

    /*!
     * Create a range policy that spans an entire Kokkos view, excluding
     * a specifiable number of ghost cells at the extremes.
     * @tparam Tag range policy tag
     * @tparam View the view type
     *
     * @param view to span
     * @param shift number of ghost cells
     *
     * @return A (MD)RangePolicy that spans the desired elements of the given view
     */
    template <class... PolicyArgs, typename View>
    typename RangePolicy<View::rank, typename View::execution_space, PolicyArgs...>::policy_type
    getRangePolicy(const View& view, int shift = 0) {
        constexpr unsigned Dim = View::rank;
        using exec_space       = typename View::execution_space;
        using policy_type      = typename RangePolicy<Dim, exec_space, PolicyArgs...>::policy_type;
        if constexpr (Dim == 1) {
            return policy_type(shift, view.size() - shift);
        } else {
            using index_type = typename RangePolicy<Dim, exec_space, PolicyArgs...>::index_type;
            Kokkos::Array<index_type, Dim> begin, end;
            for (unsigned int d = 0; d < Dim; d++) {
                begin[d] = shift;
                end[d]   = view.extent(d) - shift;
            }
            return policy_type(begin, end);
        }
        // Silences incorrect nvcc warning: missing return statement at end of non-void function
        throw IpplException("detail::getRangePolicy", "Unreachable state");
    }

    /*!
     * Create a range policy for an index range given in the form of arrays
     * (required because Kokkos doesn't allow the initialization of 1D range
     * policies using arrays)
     * @tparam Dim the dimension of the range
     * @tparam PolicyArgs... additional template parameters for the range policy
     *
     * @param begin the starting indices
     * @param end the ending indices
     *
     * @return A (MD)RangePolicy spanning the given range
     */
    template <size_t Dim, class... PolicyArgs>
    typename RangePolicy<Dim, PolicyArgs...>::policy_type createRangePolicy(
        const Kokkos::Array<typename RangePolicy<Dim, PolicyArgs...>::index_type, Dim>& begin,
        const Kokkos::Array<typename RangePolicy<Dim, PolicyArgs...>::index_type, Dim>& end) {
        using policy_type = typename RangePolicy<Dim, PolicyArgs...>::policy_type;
        if constexpr (Dim == 1) {
            return policy_type(begin[0], end[0]);
        } else {
            return policy_type(begin, end);
        }
        // Silences incorrect nvcc warning: missing return statement at end of non-void function
        throw IpplException("detail::createRangePolicy", "Unreachable state");
    }

    namespace detail {
        /*!
         * Recursively templated struct for defining tuples with arbitrary
         * length
         * @tparam Dim the length of the tuple
         * @tparam T the data type to repeat (default size_t)
         */
        template <unsigned Dim, typename T = size_t>
        struct Coords {
            // https://stackoverflow.com/a/53398815/2773311
            // https://en.cppreference.com/w/cpp/utility/declval
            using type =
                decltype(std::tuple_cat(std::declval<typename Coords<1, T>::type>(),
                                        std::declval<typename Coords<Dim - 1, T>::type>()));
        };

        template <typename T>
        struct Coords<1, T> {
            using type = std::tuple<T>;
        };

        enum e_functor_type {
            FOR,
            REDUCE,
            SCAN
        };

        template <e_functor_type, typename, typename, typename, typename...>
        struct FunctorWrapper;

        /*!
         * Wrapper struct for reduction kernels
         * Source:
         * https://stackoverflow.com/questions/50713214/familiar-template-syntax-for-generic-lambdas
         * @tparam Functor functor type
         * @tparam Policy range policy type
         * @tparam T... index types
         * @tparam Acc accumulator data type
         */
        template <typename Functor, typename Policy, typename... T, typename... Acc>
        struct FunctorWrapper<REDUCE, Functor, Policy, std::tuple<T...>, Acc...> {
            Functor f;

            /*!
             * Inline operator forwarding to a specialized instantiation
             * of the functor's own operator()
             * @param x... the indices
             * @param res the accumulator variable
             * @return The functor's return value
             */
            KOKKOS_INLINE_FUNCTION void operator()(T... x, Acc&... res) const {
                using index_type                       = typename Policy::index_type;
                typename Policy::index_array_type args = {static_cast<index_type>(x)...};
                f(args, res...);
            }
        };

        template <typename Functor, typename Policy, typename... T>
        struct FunctorWrapper<FOR, Functor, Policy, std::tuple<T...>> {
            Functor f;

            KOKKOS_INLINE_FUNCTION void operator()(T... x) const {
                using index_type                       = typename Policy::index_type;
                typename Policy::index_array_type args = {static_cast<index_type>(x)...};
                f(args);
            }
        };

        // Extracts the rank of a Kokkos range policy
        template <typename>
        struct ExtractRank;

        template <typename... T>
        struct ExtractRank<Kokkos::RangePolicy<T...>> {
            static constexpr int rank = 1;
        };
        template <typename... T>
        struct ExtractRank<Kokkos::MDRangePolicy<T...>> {
            static constexpr int rank = Kokkos::MDRangePolicy<T...>::rank;
        };
        template<typename T>
        concept HasMemberValueType = requires(){
            {typename T::value_type() };
        };
        template<typename T>
        struct ExtractReducerReturnType {
            using type = T;
        };
        template<HasMemberValueType T>
        struct ExtractReducerReturnType<T> {
            using type = typename T::value_type;
        };

        /*!
         * Convenience function for wrapping a functor with the wrapper struct.
         * @tparam Functor the functor type
         * @tparam Type the parallel dispatch type
         * @tparam Policy the range policy type
         * @tparam Acc... the accumulator type(s)
         * @return A wrapper containing the given functor
         */
        template <e_functor_type Type, typename Policy, typename... Acc, typename Functor>
        auto functorize(const Functor& f) {
            constexpr unsigned Dim = ExtractRank<Policy>::rank;
            using PolicyProperties = RangePolicy<Dim, typename Policy::execution_space>;
            using index_type       = typename PolicyProperties::index_type;
            return FunctorWrapper<Type, Functor, PolicyProperties,
                                  typename Coords<Dim, index_type>::type, Acc...>{f};
        }
    }  // namespace detail

    // Wrappers for Kokkos' parallel dispatch functions that use
    // the IPPL functor wrapper
    template <class ExecPolicy, class FunctorType>
    void parallel_for(const std::string& name, const ExecPolicy& policy,
                      const FunctorType& functor) {
        Kokkos::parallel_for(name, policy, detail::functorize<detail::FOR, ExecPolicy>(functor));
    }

    template <class ExecPolicy, class FunctorType, class... ReducerArgument>
    void parallel_reduce(const std::string& name, const ExecPolicy& policy,
                         const FunctorType& functor, ReducerArgument&&... reducer) {
        Kokkos::parallel_reduce(
            name, policy,
            detail::functorize<detail::REDUCE, ExecPolicy, typename detail::ExtractReducerReturnType<ReducerArgument>::type...>(
                functor),
            std::forward<ReducerArgument>(reducer)...);
    }
}  // namespace ippl

#endif
