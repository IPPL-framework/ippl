//
// Class ParticleAttrib
//   Templated class for all particle attribute classes.
//
//   This templated class is used to represent a single particle attribute.
//   An attribute is one data element within a particle object, and is
//   stored as a Kokkos::View. This class stores the type information for the
//   attribute, and provides methods to create and destroy new items, and
//   to perform operations involving this attribute with others.
//
//   ParticleAttrib is the primary element involved in expressions for
//   particles (just as LField is the primary element there).  This file
//   defines the necessary templated classes and functions to make
//   ParticleAttrib a capable expression-template participant.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef IPPL_PARTICLE_ATTRIB_H
#define IPPL_PARTICLE_ATTRIB_H

#include "Expression/IpplExpressions.h"

#include "Particle/ParticleAttribBase.h"

namespace ippl {

    // ParticleAttrib class definition
    template <typename T, class... Properties>
    class ParticleAttrib : public detail::ParticleAttribBase<Properties...>,
                           public detail::Expression<
                               ParticleAttrib<T, Properties...>,
                               sizeof(typename detail::ViewType<T, 1, Properties...>::view_type)> {
    public:
        typedef T value_type;
        using boolean_view_type =
            typename detail::ParticleAttribBase<Properties...>::boolean_view_type;
        using view_type  = typename detail::ViewType<T, 1, Properties...>::view_type;
        using HostMirror = typename view_type::host_mirror_type;

        using size_type = detail::size_type;

        // Create storage for M particle attributes.  The storage is uninitialized.
        // New items are appended to the end of the array.
        void create(size_type) override;

        /*!
         * Particle deletion function. Partition the particles into a valid region
         * and an invalid region.
         * @param deleteIndex List of indices of invalid particles in the valid region
         * @param keepIndex List of indices of valid particles in the invalid region
         * @param invalidCount Number of invalid particles in the valid region
         */
        void destroy(const Kokkos::View<int*>& deleteIndex, const Kokkos::View<int*>& keepIndex,
                     size_type invalidCount) override;

        void pack(void*, const Kokkos::View<int*>&) const override;

        void unpack(void*, size_type) override;

        void serialize(detail::Archive<Properties...>& ar, size_type nsends) override {
            ar.serialize(dview_m, nsends);
        }

        void deserialize(detail::Archive<Properties...>& ar, size_type nrecvs) override {
            ar.deserialize(dview_m, nrecvs);
        }

        virtual ~ParticleAttrib() = default;

        size_type size() const override { return dview_m.extent(0); }

        size_type packedSize(const size_type count) const override {
            return count * sizeof(value_type);
        }

        void resize(size_type n) { Kokkos::resize(dview_m, n); }

        void realloc(size_type n) { Kokkos::realloc(dview_m, n); }

        void print() {
            HostMirror hview = Kokkos::create_mirror_view(dview_m);
            Kokkos::deep_copy(hview, dview_m);
            for (size_type i = 0; i < *(this->localNum_mp); ++i) {
                std::cout << hview(i) << std::endl;
            }
        }

        KOKKOS_INLINE_FUNCTION T& operator()(const size_t i) const { return dview_m(i); }

        view_type& getView() { return dview_m; }

        const view_type& getView() const { return dview_m; }

        HostMirror getHostMirror() { return Kokkos::create_mirror(dview_m); }

        /*!
         * Assign the same value to the whole attribute.
         */
        // KOKKOS_INLINE_FUNCTION
        ParticleAttrib<T, Properties...>& operator=(T x);

        /*!
         * Assign an arbitrary particle attribute expression
         * @tparam E expression type
         * @tparam N size of the expression, this is necessary for running on the
         * device since otherwise it does not allocate enough memory
         * @param expr is the expression
         */
        template <typename E, size_t N>
        // KOKKOS_INLINE_FUNCTION
        ParticleAttrib<T, Properties...>& operator=(detail::Expression<E, N> const& expr);

        //     // scatter the data from this attribute onto the given Field, using
        //     // the given Position attribute
        template <unsigned Dim, class M, class C, typename P2>
        void scatter(Field<T, Dim, M, C>& f,
                     const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp) const;

        template <unsigned Dim, class M, class C, typename P2>
        void gather(Field<T, Dim, M, C>& f,
                    const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp);

        T sum();
        T max();
        T min();
        T prod();

    private:
        view_type dview_m;
    };

    namespace detail {
        /*!
         * Computes the weight for a given point for a given axial direction
         * @tparam Point index of the point
         * @tparam Index index of the axis
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         */
        template <unsigned long Point, unsigned long Index, typename Vector>
        KOKKOS_INLINE_FUNCTION constexpr auto scattergather_weight(const Vector& wlo,
                                                                   const Vector& whi) {
            if constexpr (Point & (1 << Index))
                return wlo[Index];
            else
                return whi[Index];
        }

        /*!
         * Computes the index for a given point for a given axis
         * @tparam Point index of the point
         * @tparam Index index of the axis
         * @param args the indices of the source point
         */
        template <unsigned long Point, unsigned long Index, typename Vector>
        KOKKOS_INLINE_FUNCTION constexpr auto scattergather_arg(const Vector& args) {
            if constexpr (Point & (1 << Index))
                return args[Index] - 1;
            else
                return args[Index];
        }

        /*!
         * Scatters to a field at a single point
         * @tparam ScatterPoint the index of the point to which we are scattering
         * @tparam Index the sequence 0...Dim - 1
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
        template <unsigned long ScatterPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr int scatter_point(
            const std::index_sequence<Index...>&,
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args, const T& val) {
            Kokkos::atomic_add(&view(scattergather_arg<ScatterPoint, Index>(args)...),
                               val * (scattergather_weight<ScatterPoint, Index>(wlo, whi) * ...));
            return 0;
        }

        /*!
         * Utility function for scattering. Scatters to all neighboring points in the field
         * via fold expression.
         */
        template <unsigned long... ScatterPoint, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr void scatter_impl(
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const std::index_sequence<ScatterPoint...>&,
            const Vector<IndexType, Dim>& args, const T& val) {
            // The number of indices is Dim
            [[maybe_unused]] auto _ = (scatter_point<ScatterPoint>(std::make_index_sequence<Dim>{},
                                                                   view, wlo, whi, args, val)
                                       ^ ...);
        }

        /*!
         * Scatters the particle attribute to the field.
         *
         * The coordinates to which an attribute must be scattered is given by 2^n,
         * where n is the number of dimensions. Example: the point (x, y) is scattered
         * to (x, y), (x - 1, y), (x, y - 1), and (x - 1, y - 1). In other words,
         * for each coordinate, we choose between the unchanged coordinate and a neighboring
         * value. We can identify each point to which the attribute is scattered by
         * interpreting this set of choices as a binary number.
         * @tparam Dim the number of dimensions
         * @tparam T the field data type
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         * @param val the value to interpolate
         */
        template <unsigned Dim, typename T, typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr void scatter_field(
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args, T val = 1) {
            // The number of points to which we scatter values is 2^Dim
            constexpr unsigned count = 1 << Dim;
            scatter_impl(view, wlo, whi, std::make_index_sequence<count>{}, args, val);
        }

        /*!
         * Gathers from a field at a single point
         * @tparam GatherPoint the index of the point from which data is gathered
         * @tparam Index the sequence 0...Dim - 1
         * @tparam T the field data type
         * @tparam Dim the number of dimensions
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         * @return The gathered value
         */
        template <unsigned long GatherPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr T gather_point(
            const std::index_sequence<Index...>&,
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args) {
            return (scattergather_weight<GatherPoint, Index>(wlo, whi) * ...)
                   * view(scattergather_arg<GatherPoint, Index>(args)...);
        }

        /*!
         * Utility function for gathering. Gathers from all neighboring points in the field
         * via fold expression.
         * @tparam GatherPoint the indices representing the points from which data is gathered
         */
        template <unsigned long... GatherPoint, typename T, unsigned Dim,
                  typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr T gather_impl(
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const std::index_sequence<GatherPoint...>&,
            const Vector<IndexType, Dim>& args) {
            // The number of indices is Dim
            return (gather_point<GatherPoint>(std::make_index_sequence<Dim>{}, view, wlo, whi, args)
                    + ...);
        }

        /*!
         * Gathers the particle attribute from a field (see scatter_field for more details)
         * @tparam Dim the number of dimensions
         * @tparam T the field data type
         * @tparam IndexType the index type for accessing the field (default size_t)
         * @param view the field view on which to scatter
         * @param wlo lower weights for interpolation
         * @param whi upper weights for interpolation
         * @param args the indices at which to access the field
         */
        template <unsigned Dim, typename T, typename IndexType = size_t>
        KOKKOS_INLINE_FUNCTION constexpr T gather_field(
            const typename detail::ViewType<T, Dim>::view_type& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args) {
            // The number of points from which we gather field values is 2^Dim
            constexpr unsigned count = 1 << Dim;
            return gather_impl(view, wlo, whi, std::make_index_sequence<count>{}, args);
        }
    }  // namespace detail
}  // namespace ippl

#include "Particle/ParticleAttrib.hpp"

#endif
