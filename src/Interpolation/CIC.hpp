//
// Class CIC
//   First order/cloud-in-cell grid interpolation. Currently implemented as
//   global functions, but in order to support higher or lower order interpolation,
//   these should be moved into structs.
//

namespace ippl {
    namespace detail {
        template <unsigned long Point, unsigned long Index, typename Weights>
        KOKKOS_INLINE_FUNCTION constexpr typename Weights::value_type interpolationWeight(
            const Weights& wlo, const Weights& whi) {
            if constexpr (Point & (1 << Index)) {
                return wlo[Index];
            } else {
                return whi[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
            return 0;
        }

        template <unsigned long Point, unsigned long Index, typename Indices>
        KOKKOS_INLINE_FUNCTION constexpr typename Indices::value_type interpolationIndex(
            const Indices& args) {
            if constexpr (Point & (1 << Index)) {
                return args[Index] - 1;
            } else {
                return args[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
            return 0;
        }

        template <unsigned long ScatterPoint, unsigned long... Index, typename View, typename T,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToPoint(
            const std::index_sequence<Index...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args, const T& val) {
            Kokkos::atomic_add(&view(interpolationIndex<ScatterPoint, Index>(args)...),
                               val * (interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...));
        }

        template <unsigned long... ScatterPoint, typename View, typename T, typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToField(
            const std::index_sequence<ScatterPoint...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args, T val) {
            // The number of indices is equal to the view rank
            (scatterToPoint<ScatterPoint>(std::make_index_sequence<View::rank>{}, view, wlo, whi,
                                          args, val),
             ...);
        }

        template <unsigned long GatherPoint, unsigned long... Index, typename View, typename T,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr typename View::value_type gatherFromPoint(
            const std::index_sequence<Index...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args) {
            return (interpolationWeight<GatherPoint, Index>(wlo, whi) * ...)
                   * view(interpolationIndex<GatherPoint, Index>(args)...);
        }

        template <unsigned long... GatherPoint, typename View, typename T, typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr typename View::value_type gatherFromField(
            const std::index_sequence<GatherPoint...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args) {
            // The number of indices is equal to the view rank
            return (gatherFromPoint<GatherPoint>(std::make_index_sequence<View::rank>{}, view, wlo,
                                                 whi, args)
                    + ...);
        }
    }  // namespace detail
}  // namespace ippl
