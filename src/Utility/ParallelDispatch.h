//
// Parallel dispatch
//   Utility functions relating to parallel dispatch in IPPL
//

#ifndef IPPL_PARALLEL_DISPATCH_H
#define IPPL_PARALLEL_DISPATCH_H

#include <Kokkos_Core.hpp>
#include "Ippl.h"

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
        // Silences nvcc "missing return" at end of exhaustive if constexpr.
        __builtin_unreachable();
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
        // Silences nvcc "missing return" at end of exhaustive if constexpr.
        __builtin_unreachable();
    }

    /*!
     * @brief Compile-time loop: invoke @c f.template operator()<Is>() for each Is.
     *
     * Equivalent to a hand-unrolled @c for loop where the index is a non-type
     * template parameter, useful for generating per-axis specialized code.
     */
    template <int... Is, typename F>
    KOKKOS_FORCEINLINE_FUNCTION void for_constexpr(std::integer_sequence<int, Is...>, F&& f) {
        (f.template operator()<Is>(), ...);
    }

    /*!
     * @brief Compile-time fold over a product:
     *        returns @c (f<I0>() * f<I1>() * ... * f<IN>()).
     */
    template <int... Is, typename F>
    KOKKOS_FORCEINLINE_FUNCTION auto product_over(std::integer_sequence<int, Is...>, F&& f) {
        return (f.template operator()<Is>() * ...);
    }

    /*!
     * @brief Convenience wrapper that calls product_over over [0, N).
     */
    template <int N, typename F>
    KOKKOS_FORCEINLINE_FUNCTION auto product_over(F&& f) {
        return product_over(std::make_integer_sequence<int, N>{}, std::forward<F>(f));
    }

    /*!
     * @brief Multi-dimensional ThreadVectorRange parallel-for inside a Kokkos team.
     *
     * @tparam Dim     Number of dimensions to range over.
     * @tparam Team    Kokkos team policy member type.
     * @tparam Extents Indexable type with @p Dim integer entries.
     * @tparam F       Callable @c void(int, int, ..., int) (Dim ints).
     */
    template <int Dim, typename Team, typename Extents, typename F>
    KOKKOS_FORCEINLINE_FUNCTION void thread_vector_md_for(const Team& team, const Extents& extents,
                                                          F&& f) {
        [&]<int... Is>(std::integer_sequence<int, Is...>) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorMDRange<Kokkos::Rank<Dim>, Team>(team, extents[Is]...),
                std::forward<F>(f));
        }(std::make_integer_sequence<int, Dim>{});
    }

    /*!
     * @brief Iterate a W^Dim stencil with one ThreadVectorRange element per cell.
     *
     * @tparam Dim Spatial dimension.
     * @tparam W   Compile-time per-axis stencil width.
     */
    template <int Dim, int W, typename Team, typename F>
    KOKKOS_FORCEINLINE_FUNCTION void thread_vector_stencil_for(const Team& team, F&& f) {
        constexpr int TotalCells = [] {
            int r = 1;
            for (int i = 0; i < Dim; ++i)
                r *= W;
            return r;
        }();

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, TotalCells), [&](const int flat_idx) {
            // Compile-time unrollable index decomposition
            Kokkos::Array<int, Dim> idx;
            int tmp = flat_idx;
            for (int d = Dim - 1; d >= 0; --d) {
                idx[d] = tmp % W;
                tmp /= W;
            }

            // Call with index pack
            [&]<int... Is>(std::integer_sequence<int, Is...>) {
                f(idx[Is]...);
            }(std::make_integer_sequence<int, Dim>{});
        });
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
            REDUCE
        };

        template <e_functor_type, typename, typename, typename, typename...>
        struct FunctorWrapper;

        template <typename ExecSpace>
        constexpr bool inline isGPUSpace = false;

#ifdef KOKKOS_ENABLE_CUDA
        template <>
        constexpr bool inline isGPUSpace<Kokkos::Cuda> = true;
#endif
#ifdef KOKKOS_ENABLE_HIP
        template <>
        constexpr bool inline isGPUSpace<Kokkos::HIP> = true;
#endif

        // Dispatches F(i) for i in [0, n) either in parallel (OpenMP host)
        // when MPI_THREAD_MULTIPLE is available, or serially otherwise.
        // `f(i)` must be synchronous w.r.t. its own work -- this dispatcher
        // fences after the parallel_for so the caller's next operation sees
        // a consistent state.
        template <typename F>
        void parallelForMPI(size_t n, F&& f) {
            constexpr bool useGPU = isGPUSpace<Kokkos::DefaultExecutionSpace>;
            const bool threadSafe = Env->threadMultiple();

            if constexpr (useGPU) {
                if (threadSafe) {
                    Kokkos::parallel_for(
                        "Parallel dispatch",
                        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n), [&](int i) {
                            f(i);
                        });
                    Kokkos::fence();
                    return;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                f(i);
            }
        }

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
        template <typename T>
        concept HasMemberValueType = requires { typename T::value_type; };
        template <typename T>
        struct ExtractReducerReturnType {
            using type = T;
        };
        template <HasMemberValueType T>
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
            detail::functorize<detail::REDUCE, ExecPolicy,
                               typename detail::ExtractReducerReturnType<ReducerArgument>::type...>(
                functor),
            std::forward<ReducerArgument>(reducer)...);
    }

    template <std::size_t I, typename T, typename... Rest>
    KOKKOS_FORCEINLINE_FUNCTION auto get_arg(T first, Rest... rest) {
        if constexpr (I == 0) {
            return first;
        } else {
            return get_arg<I - 1>(rest...);
        }
    }

    template <typename Grid, typename Base, typename... Stencils>
    KOKKOS_FORCEINLINE_FUNCTION decltype(auto) grid_at(Grid& grid, const Base& base, int team_rank,
                                                       Stencils... stencil_idx) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> decltype(auto) {
            return grid((base(team_rank, Is) + get_arg<Is>(stencil_idx...))...);
        }(std::index_sequence_for<Stencils...>{});
    }

    template <int D, typename Grid, typename Base, typename Stencils>
    KOKKOS_FORCEINLINE_FUNCTION decltype(auto) grid_at_t(Grid& grid, const Base& base,
                                                         int team_rank, Stencils& stencil_idx) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> decltype(auto) {
            return grid((base(team_rank, Is) + stencil_idx[Is])...);
        }(std::make_index_sequence<D>{});
    }
}  // namespace ippl

#endif
