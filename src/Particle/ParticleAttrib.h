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
    class ParticleAttrib : public detail::ParticleAttribBase<Properties...>
                         , public detail::ViewType<T, 1, Properties...>::view_type
                         , public Expression<ParticleAttrib<T, Properties...>,
                                             sizeof(typename detail::ViewType<T, 1, Properties...>::view_type)>
    {
    public:
        typedef T value_type;
        using boolean_view_type = typename detail::ParticleAttribBase<Properties...>::boolean_view_type;
        using view_type = typename detail::ViewType<T, 1, Properties...>::view_type;

        // Create storage for M particle attributes.  The storage is uninitialized.
        // New items are appended to the end of the array.
        virtual void create(size_t);

        virtual void destroy(boolean_view_type, Kokkos::View<int*> cc, size_t);


        virtual ~ParticleAttrib() = default;
       
        size_t size() const {
            return this->extent(0);
        }

        void resize(size_t n) {
            Kokkos::resize(*this, n);
        }

        void print() {
            typename view_type::HostMirror hview = Kokkos::create_mirror_view(*this);
            Kokkos::deep_copy(hview, *this);
            for (size_t i = 0; i < this->size(); ++i) {
                std::cout << hview(i) << std::endl;
            }
        }


        /*!
         * Assign the same value to the whole attribute.
         */
        ParticleAttrib<T, Properties...>& operator=(T x);

        /*!
         * Assign an arbitrary particle attribute expression
         * @tparam E expression type
         * @tparam N size of the expression, this is necessary for running on the
         * device since otherwise it does not allocate enough memory
         * @param expr is the expression
         */
        template <typename E, size_t N>
        ParticleAttrib<T, Properties...>& operator=(Expression<E, N> const& expr);


        //     // scatter the data from this attribute onto the given Field, using
//     // the given Position attribute
        template <unsigned Dim, class M, class C, typename P2>
        void
        scatter(Field<T, Dim, M, C>& f,
                const ParticleAttrib<Vector<P2, Dim>, Properties... >& pp) const;


        template <unsigned Dim, class M, class C, typename P2>
        void
        gather(const Field<T, Dim, M, C>& f,
               const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp);
    };
}

#include "Particle/ParticleAttrib.hpp"

#endif
