//
// Class CIC
//   First order/cloud-in-cell grid interpolation. Currently implemented as
//   global functions, but in order to support higher or lower order interpolation,
//   these should be moved into structs.
//
// Copyright (c) 2023, Paul Scherrer Institut, Villigen PSI, Switzerland
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

namespace ippl {
    namespace detail {
        template <unsigned long Point, unsigned long Index, typename T, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr T interpolationWeight(const Vector<T, Dim>& wlo,
                                                               const Vector<T, Dim>& whi) {
            if constexpr (Point & (1 << Index)) {
                return wlo[Index];
            } else {
                return whi[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
            return 0;
        }

        template <unsigned long Point, unsigned long Index, typename IndexType, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr IndexType interpolationIndex(
            const Vector<IndexType, Dim>& args) {
            if constexpr (Point & (1 << Index)) {
                return args[Index] - 1;
            } else {
                return args[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
            return 0;
        }

        template <unsigned long ScatterPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr int scatterToPoint(
            const std::index_sequence<Index...>&,
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args, const T& val) {
            Kokkos::atomic_add(&view(interpolationIndex<ScatterPoint, Index>(args)...),
                               val * (interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...));
            return 0;
        }

        template <unsigned long... ScatterPoint, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToField(
            const std::index_sequence<ScatterPoint...>&,
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args, T val) {
            // The number of indices is Dim
            [[maybe_unused]] auto _ = (scatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{},
                                                                    view, wlo, whi, args, val)
                                       ^ ...);
        }

        template <unsigned long GatherPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr T gatherFromPoint(
            const std::index_sequence<Index...>&,
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args) {
            return (interpolationWeight<GatherPoint, Index>(wlo, whi) * ...)
                   * view(interpolationIndex<GatherPoint, Index>(args)...);
        }

        template <unsigned long... GatherPoint, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr T gatherFromField(
            const std::index_sequence<GatherPoint...>&,
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args) {
            // The number of indices is Dim
            return (
                gatherFromPoint<GatherPoint>(std::make_index_sequence<Dim>{}, view, wlo, whi, args)
                + ...);
        }
    }  // namespace detail
}  // namespace ippl
