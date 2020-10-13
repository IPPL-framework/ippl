//
// Class Mesh
//   The Mesh base class. Right now, this mainly acts as a standard base
//   class for all meshes so that other objects can register as users of
//   the mesh and can be notified if the mesh changes (e.g., it is rescaled
//   or restructured entirely).
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
namespace ippl {
    template<typename T, unsigned Dim>
    typename Mesh<T, Dim>::vector_type Mesh<T, Dim>::getOrigin() const {
        return origin_m;
    }


    template<typename T, unsigned Dim>
    void Mesh<T, Dim>::setOrigin(const vector_type& origin) {
        origin_m = origin;
    }
}
