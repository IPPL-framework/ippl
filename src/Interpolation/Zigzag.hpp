namespace ippl {
    namespace detail {
        template <unsigned long ScatterPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr int ZigzagScatterToPoint(
            const std::index_sequence<Index...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, Dim>, Dim>::view_type& view,
            const Vector<T, Dim>& wlo, const Vector<T, Dim>& whi,
            const Vector<IndexType, Dim>& args, const Vector<T, Dim>& val, T scale,
            const Vector<T, Dim>&, const NDIndex<Dim>, int, const Vector<T, Dim>&) {
            bool isinbound = true;
            ippl::Vector<T, Dim> depot =
                scale * val * (interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...);
            ippl::Vector<IndexType, Dim> index3{interpolationIndex<ScatterPoint, Index>(args)...};
            for (unsigned int d = 0; d < Dim; d++) {
                isinbound &= (index3[d] < view.extent(d));
            }
            if (!isinbound) {
                return 0;
            }
            for (unsigned int d = 0; d < Dim; d++) {
                Kokkos::atomic_add(
                    &(view(interpolationIndex<ScatterPoint, Index>(args)...)[d]),
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
         * @param scale The scaling factor to apply during the scatter operation. This is usually
         * set to the inverse of the timestep.
         */
        template <typename T>
        KOKKOS_INLINE_FUNCTION T fractional_part(T x) {
            using Kokkos::floor;
            return x - floor(x);
        }
        template <unsigned long... ScatterPoint, typename T, unsigned Dim, typename IndexType>
        KOKKOS_INLINE_FUNCTION void ZigzagScatterToField(
            const std::index_sequence<ScatterPoint...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, Dim>, Dim>::view_type& view,
            Vector<T, Dim> from, Vector<T, Dim> to, const Vector<T, Dim> hr, T scale,
            const NDIndex<Dim> lDom, int nghost) {
            // Use utility functions

            using Kokkos::floor;
            using Kokkos::max;
            using Kokkos::min;

            Vector<T, Dim> fromInGridCoordinates;
            Vector<T, Dim> toInGridCoordinates;
            // Calculate the indices for the scatter operation
            ippl::Vector<IndexType, Dim> fromi, toi;
            for (unsigned int i = 0; i < Dim; i++) {
                from[i] += hr[i] * T(0.5);  // Centering offset
                to[i] += hr[i] * T(0.5);    // Centering offset
                fromInGridCoordinates[i] = from[i] / hr[i];
                toInGridCoordinates[i]   = to[i] / hr[i];
                fromi[i]                 = floor(fromInGridCoordinates[i]) + nghost;
                toi[i]                   = floor(toInGridCoordinates[i]) + nghost;
            }
            ippl::Vector<IndexType, Dim> fromiLocal = fromi - lDom.first();
            ippl::Vector<IndexType, Dim> toiLocal   = toi - lDom.first();

            // Calculate the relay point for each dimension
            ippl::Vector<T, Dim> relay;
            for (unsigned int i = 0; i < Dim; i++) {
                relay[i] = min(min(fromi[i] - nghost, toi[i] - nghost) * hr[i] + hr[i],
                               max(max(fromi[i] - nghost, toi[i] - nghost) * hr[i],
                                   T(0.5) * (to[i] + from[i])));
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
                wlo[i]     = T(1.0) - fractional_part((from[i] + relay[i]) * T(0.5) / hr[i]);
                whi[i]     = fractional_part((from[i] + relay[i]) * T(0.5) / hr[i]);
                source1[i] = (from[i] + relay[i]) * T(0.5);
                source2[i] = (to[i] + relay[i]) * T(0.5);
            }

            // Perform the scatter operation for the first midpoint
            auto dummy_ = (ZigzagScatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{}, view,
                                                              wlo, whi, fromiLocal, jcfrom, scale,
                                                              hr, lDom, nghost, source1)
                           ^ ...);

            for (unsigned i = 0; i < Dim; i++) {
                wlo[i] = T(1.0) - fractional_part((to[i] + relay[i]) * T(0.5) / hr[i]);
                whi[i] = fractional_part((to[i] + relay[i]) * T(0.5) / hr[i]);
            }

            dummy_ = (ZigzagScatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{}, view, wlo,
                                                         whi, toiLocal, jcto, scale, hr, lDom,
                                                         nghost, source2)
                      ^ ...);

            (void)dummy_;  // [[maybe_unused]] causes issues on certain compilers
        }
    }  // namespace detail
}  // namespace ippl