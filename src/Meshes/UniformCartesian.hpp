//
// Class UniformCartesian
//   UniformCartesian class - represents uniform-spacing cartesian meshes.
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
#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
#include "Field/BareField.h"
#include "Field/Field.h"

namespace ippl {

    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian()
        : Mesh<T, Dim>()
        , volume_m(0.0)
    { }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const NDIndex<Dim>& ndi,
                                               const vector_type& hx,
                                               const vector_type& origin)
    {
        this->initialize(ndi, hx, origin);
    }


    template <typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const NDIndex<Dim>& ndi,
                                              const vector_type& hx,
                                              const vector_type& origin)
    {
        meshSpacing_m = hx;

        volume_m = 1.0;
        for (unsigned d = 0; d < Dim; ++d) {
            this->gridSizes_m[d] = ndi[d].length();
            volume_m *= meshSpacing_m[d];
        }

        this->setOrigin(origin);
    }
    
    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::setMeshSpacing(const vector_type& meshSpacing) {
        meshSpacing_m = meshSpacing;
        this->updateCellVolume_m();
    }


    template<typename T, unsigned Dim>
    T UniformCartesian<T, Dim>::getMeshSpacing(unsigned dim) const {
        PAssert_LT(dim, Dim);
        return meshSpacing_m[dim];
    }


    template<typename T, unsigned Dim>
    const typename UniformCartesian<T, Dim>::vector_type&
    UniformCartesian<T, Dim>::getMeshSpacing() const
    {
        return meshSpacing_m;
    }


    template<typename T, unsigned Dim>
    T UniformCartesian<T, Dim>::getCellVolume() const {
        return volume_m;
    }
}
