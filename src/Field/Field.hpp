//
// Class Field
//   BareField with a mesh and configurable boundary conditions
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
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
//

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, class M, class C>
        struct isExpression<Field<T, Dim, M, C>> : std::true_type {};
    }

    //////////////////////////////////////////////////////////////////////////
    // A default constructor, which should be used only if the user calls the
    // 'initialize' function before doing anything else.  There are no special
    // checks in the rest of the Field methods to check that the Field has
    // been properly initialized
    template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field()
    : BareField<T, Dim>()
    , mesh_m(nullptr)
    , bc_m() 
    { }

    //////////////////////////////////////////////////////////////////////////
    // Constructors which include a Mesh object as argument
    template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field(Mesh_t& m, Layout_t& l, int nghost)
    : BareField<T,Dim>(l, nghost)
    , mesh_m(&m)
    { 
        for (unsigned int face=0; face < 2 * Dim; ++face) {
            bc_m[face] = std::make_shared<NoBcFace<T, Dim, M>>(face);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Initialize the Field, also specifying a mesh
    template<class T, unsigned Dim, class M, class C>
    void Field<T,Dim,M,C>::initialize(Mesh_t& m, Layout_t& l, int nghost) {
        BareField<T,Dim>::initialize(l, nghost);
        mesh_m = &m;
        for (unsigned int face=0; face < 2 * Dim; ++face) {
            bc_m[face] = std::make_shared<NoBcFace<T, Dim, M>>(face);
        }
    }

    template<class T, unsigned Dim, class M, class C>
    T Field<T,Dim,M,C>::getVolumeIntegral() const {
        typename M::value_type dV = mesh_m->getCellVolume();
        return this->sum() * dV;
    }

    template<class T, unsigned Dim, class M, class C>
    T Field<T,Dim,M,C>::getVolumeAverage() const {
        return getVolumeIntegral() / mesh_m->getMeshVolume();
    }

    template<class T, unsigned Dim, class M, class C>
    void Field<T,Dim,M,C>::updateLayout(Layout_t& l, int nghost) {
        BareField<T,Dim>::updateLayout(l, nghost);
    }

}

