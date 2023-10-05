//
// Functor ParticleBC
//   Functors specifying particle boundary conditions.
//
#ifndef IPPL_PARTICLE_BC_H
#define IPPL_PARTICLE_BC_H

#include "Region/NDRegion.h"

namespace ippl {
    enum BC {
        PERIODIC,
        REFLECTIVE,
        SINK,
        NO
    };

    namespace detail {

        template <typename T, unsigned Dim, class ViewType>
        struct ParticleBC {
            using value_type = typename ViewType::value_type::value_type;

            //! Kokkos view containing the field data
            ViewType view_m;
            //! The dimension along which this boundary condition
            //  is applied
            size_t dim_m;
            //! Minimum and maximum coordinates of the domain along the given dimension
            double minval_m;
            double maxval_m;
            //! Whether the boundary conditions are being applied for an upper
            //  face (i.e. with greater coordinate values)
            bool isUpper_m;

            //! The length of the domain along the given dimension
            double extent_m;
            //! The coordinate of the midpoint of the domain along the given dimension
            double middle_m;

            KOKKOS_DEFAULTED_FUNCTION
            ParticleBC() = default;

            KOKKOS_INLINE_FUNCTION ParticleBC(const ViewType& view, const NDRegion<T, Dim>& nr,
                                              const unsigned& dim, const bool& isUpper)
                : view_m(view)
                , dim_m(dim)
                , minval_m(nr[dim].min())
                , maxval_m(nr[dim].max())
                , isUpper_m(isUpper) {
                extent_m = nr[dim].length();
                middle_m = (minval_m + maxval_m) / 2;
            }

            KOKKOS_DEFAULTED_FUNCTION
            ~ParticleBC() = default;
        };

        template <typename T, unsigned Dim, class ViewType>
        struct PeriodicBC : public ParticleBC<T, Dim, ViewType> {
            using value_type = typename ParticleBC<T, Dim, ViewType>::value_type;

            using ParticleBC<T, Dim, ViewType>::extent_m;
            using ParticleBC<T, Dim, ViewType>::middle_m;

            KOKKOS_DEFAULTED_FUNCTION
            PeriodicBC() = default;

            KOKKOS_INLINE_FUNCTION PeriodicBC(const ViewType& view, const NDRegion<T, Dim>& nr,
                                              const unsigned& dim, const bool& isUpper)
                : ParticleBC<T, Dim, ViewType>(view, nr, dim, isUpper) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_t& i) const {
                value_type& value = this->view_m(i)[this->dim_m];
                value             = value - extent_m * (int)((value - middle_m) * 2 / extent_m);
            }

            KOKKOS_DEFAULTED_FUNCTION
            ~PeriodicBC() = default;
        };

        template <typename T, unsigned Dim, class ViewType>
        struct ReflectiveBC : public ParticleBC<T, Dim, ViewType> {
            using value_type = typename ParticleBC<T, Dim, ViewType>::value_type;

            using ParticleBC<T, Dim, ViewType>::maxval_m;
            using ParticleBC<T, Dim, ViewType>::minval_m;
            using ParticleBC<T, Dim, ViewType>::isUpper_m;

            KOKKOS_DEFAULTED_FUNCTION
            ReflectiveBC() = default;

            KOKKOS_INLINE_FUNCTION ReflectiveBC(const ViewType& view, const NDRegion<T, Dim>& nr,
                                                const unsigned& dim, const bool& isUpper)
                : ParticleBC<T, Dim, ViewType>(view, nr, dim, isUpper) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_t& i) const {
                value_type& value = this->view_m(i)[this->dim_m];
                bool tooHigh      = value >= maxval_m;
                bool tooLow       = value < minval_m;
                value += 2
                         * ((tooHigh && isUpper_m) * (maxval_m - value)
                            + (tooLow && !isUpper_m) * (minval_m - value));
            }

            KOKKOS_DEFAULTED_FUNCTION
            ~ReflectiveBC() = default;
        };

        template <typename T, unsigned Dim, class ViewType>
        struct SinkBC : public ParticleBC<T, Dim, ViewType> {
            using value_type = typename ParticleBC<T, Dim, ViewType>::value_type;

            using ParticleBC<T, Dim, ViewType>::maxval_m;
            using ParticleBC<T, Dim, ViewType>::minval_m;
            using ParticleBC<T, Dim, ViewType>::isUpper_m;

            KOKKOS_DEFAULTED_FUNCTION
            SinkBC() = default;

            KOKKOS_INLINE_FUNCTION SinkBC(const ViewType& view, const NDRegion<T, Dim>& nr,
                                          const unsigned& dim, const bool& isUpper)
                : ParticleBC<T, Dim, ViewType>(view, nr, dim, isUpper) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_t& i) const {
                value_type& value = this->view_m(i)[this->dim_m];
                bool tooHigh      = value >= maxval_m;
                bool tooLow       = value < minval_m;
                value += (tooHigh && isUpper_m) * (maxval_m - value)
                         + (tooLow && !isUpper_m) * (minval_m - value);
            }

            KOKKOS_DEFAULTED_FUNCTION
            ~SinkBC() = default;
        };

    }  // namespace detail
}  // namespace ippl

#endif
