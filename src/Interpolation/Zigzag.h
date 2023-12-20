#ifndef ZIGZAG_INTERPOLATION_H
#define ZIGZAG_INTERPOLATION_H
#include "Interpolation/CIC.h"
namespace ippl{
    namespace detail {
        /**
         * @brief Scatter particles moving from 'from' to 'to' into a Kokkos View using zigzag
         * interpolation.
         *
         *
         * @tparam ScatterPoint An unsigned long value representing the scatter point.
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
        template <unsigned long ScatterPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr int ZigzagScatterToPoint(
            const std::index_sequence<Index...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, Dim>, Dim>::view_type& view,
            const Vector<T, Dim>& wlo, const Vector<T, Dim>& whi,
            const Vector<IndexType, Dim>& args, const Vector<T, Dim>& val, T scale,
            const Vector<T, Dim>& hr, const NDIndex<Dim> lDom, int nghost,
            const Vector<T, Dim>& source);
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
            const typename ippl::detail::ViewType<ippl::Vector<T, Dim>, Dim>::view_type& view,
            Vector<T, Dim> from, Vector<T, Dim> to, const Vector<T, Dim> hr,
            T scale, NDIndex<Dim> lDom, int nghost);

    }  // namespace detail
}
#include "Interpolation/Zigzag.hpp"
#endif