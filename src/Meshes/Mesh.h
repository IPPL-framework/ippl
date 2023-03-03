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
#ifndef IPPL_MESH_H
#define IPPL_MESH_H

#include "Types/Vector.h"

namespace ippl {
    template<typename T, unsigned Dim>
    class Mesh {

    public:
        typedef T value_type;
        enum { Dimension = Dim };

        typedef Vector<T, Dim> vector_type;
        typedef Vector<vector_type, Dim> matrix_type;

        Mesh() {};

        virtual ~Mesh() {};

        // Get the origin of mesh vertex positions
        vector_type getOrigin() const;

        // Set the origin of mesh vertex positions
        void setOrigin(const vector_type& origin);

        const vector_type& getGridsize() const;

        /*!
         * Query the cell volume of the grid
         * @return The volume of a single mesh cell
         */
        virtual T getCellVolume() const = 0;

        /*!
         * Query the volume of the represented domain
         * @return Total volume of the mesh
         */
        virtual T getMeshVolume() const = 0;

        T getGridsize(size_t dim) const;

    protected:
        vector_type origin_m;          // Origin of mesh coordinates (vertices)
        vector_type gridSizes_m;       // Sizes (number of vertices)
    };
}

#include "Meshes/Mesh.hpp"

#endif
