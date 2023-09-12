//
// Class CIC
//   First order/cloud-in-cell grid interpolation. Currently implemented as
//   global functions, but in order to support higher or lower order interpolation,
//   these should be moved into structs.
//
#ifndef CIC_INTERPOLATION_H
#define CIC_INTERPOLATION_H

namespace ippl {
    namespace detail {
        /*!
         * Computes the weight for a given point for a given axial direction
         * @tparam Point index of the point
         * @tparam Index index of the axis
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         */
        template <unsigned long Point, unsigned long Index, typename T, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr T interpolationWeight(const Vector<T, Dim>& wlo,
                                                               const Vector<T, Dim>& whi);
        /*!
         * Computes the index for a given point for a given axis
         * @tparam Point index of the point
         * @tparam Index index of the axis
         * @param args the indices of the source point
         */
        template <unsigned long Point, unsigned long Index, typename IndexType, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr IndexType interpolationIndex(
            const Vector<IndexType, Dim>& args);

        /*!
         * Scatters to a field at a single point
         * @tparam ScatterPoint the index of the point to which we are scattering
         * @tparam Index the sequence 0...Dim - 1
         * @tparam View the field view type
         * @tparam T the field data type
         * @tparam Dim the number of dimensions
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         * @param val the value to interpolate
         * @return An unused dummy value (required to allow use of a more performant fold
         * expression)
         */
        template <unsigned long ScatterPoint, unsigned long... Index, typename View, typename T,
                  unsigned Dim, typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr int scatterToPoint(
            const std::index_sequence<Index...>&, const View& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args, const T& val);

        /*!
         * Scatters the particle attribute to the field.
         *
         * The coordinates to which an attribute must be scattered is given by 2^n,
         * where n is the number of dimensions. Example: the point (x, y) is scattered
         * to (x, y), (x - 1, y), (x, y - 1), and (x - 1, y - 1). In other words,
         * for each coordinate, we choose between the unchanged coordinate and a neighboring
         * value. We can identify each point to which the attribute is scattered by
         * interpreting this set of choices as a binary number.
         * @tparam ScatterPoint... the indices of the points to which to scatter (sequence 0 to
         * 2^Dim)
         * @tparam View the field view type
         * @tparam T the field data type
         * @tparam Dim the number of dimensions
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         * @param val the value to interpolate
         */
        template <unsigned long... ScatterPoint, typename View, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToField(
            const std::index_sequence<ScatterPoint...>&, const View& view,
            const Vector<T, Dim>& wlo, const Vector<T, Dim>& whi,
            const Vector<IndexType, Dim>& args, T val = 1);

        /*!
         * Gathers from a field at a single point
         * @tparam GatherPoint the index of the point from which data is gathered
         * @tparam Index the sequence 0...Dim - 1
         * @tparam View the field view type
         * @tparam T the field data type
         * @tparam Dim the number of dimensions
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         * @return The gathered value
         */
        template <unsigned long GatherPoint, unsigned long... Index, typename View, typename T,
                  unsigned Dim, typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr T gatherFromPoint(const std::index_sequence<Index...>&,
                                                           const View& view,
                                                           const Vector<T, Dim>& wlo,
                                                           const Vector<T, Dim>& whi,
                                                           const Vector<IndexType, Dim>& args);

        /*!
         * Gathers the particle attribute from a field (see scatter_field for more details)
         * @tparam GatherPoint... the indices of the points from which to gather (sequence 0 to
         * 2^Dim)
         * @tparam View the field view type
         * @tparam T the field data type
         * @tparam Dim the number of dimensions
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         */
        template <unsigned long... GatherPoint, typename View, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr T gatherFromField(
            const std::index_sequence<GatherPoint...>&, const View& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args);
    }  // namespace detail
}  // namespace ippl

#include "Interpolation/CIC.hpp"

#endif
