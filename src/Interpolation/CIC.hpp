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
        template <unsigned long ScatterPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr int ZigzagScatterToPoint(
            const std::index_sequence<Index...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, 3>, Dim>::view_type& view,
            const Vector<T, Dim>& wlo, const Vector<T, Dim>& whi,
            const Vector<IndexType, Dim>& args, const Vector<T, Dim>& val, T scale, const Vector<T, Dim>& hr, const NDIndex<Dim> lDom, int nghost, const Vector<T, Dim>& source) {
            (void)hr;
            (void)lDom;
            (void)nghost;
            (void)source;
            bool isinbound = true;
            ippl::Vector<T, Dim> depot = scale * val * (interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...);
            ippl::Vector<IndexType, Dim> index3{interpolationIndex<ScatterPoint, Index>(args)...};
            for(unsigned int d = 0; d < Dim;d++){
                isinbound &= (index3[d] < view.extent(d));
            }
            if(!isinbound){
                return 0;
            }
            for(unsigned int d = 0;d < Dim;d++){
                typename ippl::detail::ViewType<ippl::Vector<T, Dim>,
                                            Dim>::view_type::value_type::value_type* destptr =
                    &(view(interpolationIndex<ScatterPoint, Index>(args)...)[d]);
                Kokkos::atomic_add(
                    destptr,
                    scale * val[d] * (interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...));
            }

            return 0;
        }

        /**
         * This function performs a zigzag scatter operation from a field to a view.
         * 
         * @tparam ScatterPoint The scatter points to be used.
         * @tparam T The data type of the field and view.
         * @tparam Dim The dimensionality of the field and view.
         * @tparam IndexType The index type used for indexing.
         * @param view The view to scatter into.
         * @param from The starting position of the scatter operation.
         * @param to The ending position of the scatter operation.
         * @param hr The grid spacing.
         * @param scale The scaling factor to apply during the scatter operation. This is usually set to the inverse of the timestep.
         */
        template<typename T>
        KOKKOS_INLINE_FUNCTION T fractional_part(T x){
            using Kokkos::floor;
            return x - floor(x);
        }
        template <unsigned long... ScatterPoint, typename T, unsigned Dim, typename IndexType>
        KOKKOS_INLINE_FUNCTION void ZigzagScatterToField(
            const std::index_sequence<ScatterPoint...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, 3>, Dim>::view_type& view,
            Vector<T, Dim> from, Vector<T, Dim> to,
            const Vector<T, Dim> hr, T scale, const NDIndex<Dim> lDom, int nghost) {
            
            // Use utility functions
            
            using Kokkos::max;
            using Kokkos::min;
            using Kokkos::floor;

            Vector<T, Dim> from_in_grid_coordinates;
            Vector<T, Dim> to_in_grid_coordinates;
            // Calculate the indices for the scatter operation
            ippl::Vector<IndexType, Dim> fromi, toi;
            for (unsigned int i = 0; i < Dim; i++) {
                from[i] += hr[i] * T(0.5); //Centering offset
                to[i]   += hr[i] * T(0.5); //Centering offset
                from_in_grid_coordinates[i] = from[i] / hr[i];
                to_in_grid_coordinates  [i] = to[i] / hr[i];
                fromi[i] = floor(from_in_grid_coordinates[i]) + nghost;
                toi[i]   = floor(to_in_grid_coordinates[i])   + nghost;
            }
            ippl::Vector<IndexType, Dim> fromi_local = fromi - lDom.first();
            ippl::Vector<IndexType, Dim> toi_local = toi - lDom.first();
            
            // Calculate the relay point for each dimension
            ippl::Vector<T, Dim> relay;
            for (unsigned int i = 0; i < Dim; i++) {
                relay[i] =
                min(
                    min(fromi[i] - nghost, toi[i] - nghost) * hr[i] + hr[i],
                    max(
                        max(fromi[i] - nghost, toi[i] - nghost) * hr[i],
                        T(0.5) * (to[i] + from[i])
                    )
                );
            }
            
            // Calculate jcfrom and jcto (deposited split currents)
            ippl::Vector<T, Dim> jcfrom, jcto;
            jcfrom = relay;
            jcfrom -= from;
            
            jcto = to;
            jcto -= relay;
            
            // Calculate wlo and whi
            Vector<T, Dim> wlo, whi;
            Vector<T, Dim> source1, source2;
            for (unsigned i = 0; i < Dim; i++) {
                wlo[i] = T(1.0) - fractional_part((from[i] + relay[i]) * T(0.5) / hr[i]);
                whi[i] = fractional_part((from[i] + relay[i]) * T(0.5) / hr[i]);
                source1[i] = (from[i] + relay[i]) * T(0.5);
                source2[i] = (to[i]   + relay[i]) * T(0.5);
            }
            
            // Perform the scatter operation for the first midpoint
            auto _ =
                (ZigzagScatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{}, view, wlo,
                                                     whi, fromi_local, jcfrom, scale, hr, lDom, nghost, source1)
                 ^ ...);
            
            for (unsigned i = 0; i < Dim; i++) {
                wlo[i] = T(1.0) - fractional_part((to[i] + relay[i]) * T(0.5) / hr[i]);
                whi[i] = fractional_part((to[i] + relay[i]) * T(0.5) / hr[i]);
            }

            auto __ =
                (ZigzagScatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{}, view, wlo,
                                                     whi, toi_local, jcto, scale, hr, lDom, nghost, source2)
                 ^ ...);

            (void)_;
            (void)__; // [[maybe_unused]] causes issues on certain compilers
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
