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

    template <unsigned long Idx, typename T, unsigned Dim, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION constexpr void scatter_point(
        const typename detail::ViewType<T, Dim>::view_type& v, const Vector<T, Dim>& wlo,
        const Vector<T, Dim>& whi, Vector<IndexType, Dim> args, T val = 1) {
        for (unsigned d = 0; d < Dim; d++) {
            if (Idx & (1 << d)) {
                args[d]--;
                val *= wlo[d];
            } else {
                val *= whi[d];
            }
        }

        Kokkos::atomic_add(&apply<Dim>(v, args), val);
    }

    template <unsigned long... Idx, typename T, unsigned Dim, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION constexpr void scatter_impl(
        const typename detail::ViewType<T, Dim>::view_type& v, const Vector<T, Dim>& wlo,
        const Vector<T, Dim>& whi, std::index_sequence<Idx...>, const Vector<IndexType, Dim>& args,
        T val = 1) {
        (scatter_point<Idx>(v, wlo, whi, args, val), ...);
    }

    template <unsigned Dim, typename T, typename... Args, typename IndexType = size_t>
    KOKKOS_INLINE_FUNCTION constexpr void scatter_field(
        const typename detail::ViewType<T, Dim>::view_type& v, const Vector<T, Dim>& wlo,
        const Vector<T, Dim>& whi, const Vector<IndexType, Dim> args, T val = 1) {
        constexpr unsigned count = 1 << Dim;
        scatter_impl(v, wlo, whi, std::make_index_sequence<count>{}, args, val);
    }
}  // namespace ippl

#include "Particle/ParticleAttrib.hpp"

#endif
