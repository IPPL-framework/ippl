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
         * @tparam Weights the weight vector type
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @return Interpolation weight for the given point's displacement along the given axis
         */
        template <unsigned long Point, unsigned long Index, typename Weights>
        KOKKOS_INLINE_FUNCTION constexpr typename Weights::value_type interpolationWeight(
            const Weights& wlo, const Weights& whi);

        /*!
         * Computes the index for a given point for a given axis
         * @tparam Point index of the point
         * @tparam Index index of the axis
         * @tparam Indices the index vector type
         * @param args the indices of the source point
         * @return The index along the given axis for the displaced point
         */
        template <unsigned long Point, unsigned long Index, typename Indices>
        KOKKOS_INLINE_FUNCTION constexpr typename Indices::value_type interpolationIndex(
            const Indices& args);

        /*!
         * Scatters to a field at a single point
         * @tparam ScatterPoint the index of the point to which we are scattering
         * @tparam Index the sequence 0...Dim - 1
         * @tparam View the field view type
         * @tparam T the field data type
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         * @param val the value to interpolate
         */
        template <unsigned long ScatterPoint, unsigned long... Index, typename View, typename T,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToPoint(
            const std::index_sequence<Index...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args, const T& val);

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
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         * @param val the value to interpolate
         */
        template <unsigned long... ScatterPoint, typename View, typename T,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToField(
            const std::index_sequence<ScatterPoint...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args, T val = 1);

        /*!
         * Gathers from a field at a single point
         * @tparam GatherPoint the index of the point from which data is gathered
         * @tparam Index the sequence 0...Dim - 1
         * @tparam View the field view type
         * @tparam T the field data type
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         * @return The gathered value
         */
        template <unsigned long GatherPoint, unsigned long... Index, typename View, typename T,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr typename View::value_type gatherFromPoint(
            const std::index_sequence<Index...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args);

        /*!
         * Gathers the particle attribute from a field (see scatter_field for more details)
         * @tparam GatherPoint... the indices of the points from which to gather (sequence 0 to
         * 2^Dim)
         * @tparam View the field view type
         * @tparam T the field data type
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         */
        template <unsigned long... GatherPoint, typename View, typename T,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr typename View::value_type gatherFromField(
            const std::index_sequence<GatherPoint...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args);

        template <unsigned long ScatterPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr int ZigzagScatterToPoint(
            const std::index_sequence<Index...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, 3>, Dim>::view_type& view,
            const Vector<T, Dim>& wlo, const Vector<T, Dim>& whi,
            const Vector<IndexType, Dim>& args, const Vector<T, Dim>& val, T scale, const Vector<T, Dim>& hr, const NDIndex<Dim> lDom, int nghost, const Vector<T, Dim>& source);
        /**
         * @brief Scatter particles moving from 'from' to 'to' into a Kokkos View using zigzag
         * interpolation.
         *
         *
         * @tparam ScatterPoint A list of unsigned long values representing the scatter points.
         * @tparam T The data type of the values being scattered.
         * @tparam Dim The dimension of the scatter operation.
         * @tparam IndexType The data type used for indexing.
         * @param[in] view The Kokkos View to scatter data into.
         * @param[in] from The starting point for scattering.
         * @param[in] to The ending point for scattering.
         * @param[in] hr Grid spacing for scattering.
         * @param[in] scale The scaling factor for scattering.
         *
         * @details This function performs zigzag interpolation to scatter data points from 'from'
         * to 'to' into a Kokkos View using specified grid spacing and scaling factor. It utilizes
         * Kokkos functions for efficient parallelization.
         */
        template <unsigned long... ScatterPoint, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION void ZigzagScatterToField(
            const std::index_sequence<ScatterPoint...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, 3>, Dim>::view_type& view,
            Vector<T, Dim> from, Vector<T, Dim> to,
            const Vector<T, Dim> hr /*Grid Spacing*/, T scale, NDIndex<Dim> lDom, int nghost);
        
    }  // namespace detail
}  // namespace ippl

#include "Interpolation/CIC.hpp"

#endif
