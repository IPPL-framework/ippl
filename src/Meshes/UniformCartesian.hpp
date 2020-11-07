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
    {
        setup_m();
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const NDIndex<Dim>& ndi,
                                               bool evalCellVolume)
        : UniformCartesian()
    {
        for (unsigned d=0; d<Dim; d++) {
            this->gridSizes_m[d] = ndi[d].length();
            meshSpacing_m[d]     = ndi[d].stride();
            this->origin_m(d)    = ndi[d].first();
        }

        if (evalCellVolume)
            updateCellVolume_m();

        set_Dvc();
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const NDIndex<Dim>& ndi,
                                               const vector_type& hx)
        : UniformCartesian(ndi, false)
    {
        setMeshSpacing(hx);
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const NDIndex<Dim>& ndi,
                                               const vector_type& hx,
                                               const vector_type& origin)
        : UniformCartesian(ndi, hx)
    {
        this->setOrigin(origin);
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I,
                                               bool evalCellVolume)
        : UniformCartesian()
    {
        PInsist(Dim==1, "Number of Index arguments does not match mesh dimension!!");
        this->gridSizes_m[0] = I.length();
        meshSpacing_m[0]     = I.stride();
        this->origin_m[0]    = I.first();

        if (evalCellVolume)
            updateCellVolume_m();
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I,
                                               const vector_type& hx)
        : UniformCartesian(I, false)
    {
        setMeshSpacing(hx);
    }

    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I,
                                               const vector_type& hx,
                                               const vector_type& origin)
        : UniformCartesian(I, hx)
    {
        this->setOrigin(origin);
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I, const Index& J,
                                               bool evalCellVolume)
        : UniformCartesian()
    {
        PInsist(Dim==2, "Number of Index arguments does not match mesh dimension!!");

        this->gridSizes_m[0] = I.length();
        this->gridSizes_m[1] = J.length();
        meshSpacing_m[0] = I.stride();
        meshSpacing_m[1] = J.stride();
        this->origin_m(0) = I.first();
        this->origin_m(1) = J.first();

        if (evalCellVolume)
            updateCellVolume_m();

        set_Dvc();
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I,
                                               const Index& J,
                                               const vector_type& hx)
        : UniformCartesian(I, J, false)
    {
        setMeshSpacing(hx);
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I,
                                               const Index& J,
                                               const vector_type& hx,
                                               const vector_type& origin)
        : UniformCartesian(I, J, hx)
    {
        this->setOrigin(origin);
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I,
                                               const Index& J,
                                               const Index& K,
                                               bool evalCellVolume)
        : UniformCartesian()
    {
        PInsist(Dim==3, "Number of Index arguments does not match mesh dimension!!");
        this->gridSizes_m[0] = I.length();
        this->gridSizes_m[1] = J.length();
        this->gridSizes_m[2] = K.length();
        meshSpacing_m[0] = I.stride();
        meshSpacing_m[1] = J.stride();
        meshSpacing_m[2] = K.stride();
        this->origin_m(0) = I.first();
        this->origin_m(1) = J.first();
        this->origin_m(2) = K.first();

        if (evalCellVolume)
            updateCellVolume_m();

        set_Dvc();
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I,
                                               const Index& J,
                                               const Index& K,
                                               const vector_type& hx)
        : UniformCartesian(I, J, K, false)
    {
        this->setMeshSpacing(hx);
    }


    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(const Index& I,
                                               const Index& J,
                                               const Index& K,
                                               const vector_type& hx,
                                               const vector_type& orig)
        : UniformCartesian(I, J, K, hx)
    {
        this->setOrigin(orig);
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
    void UniformCartesian<T, Dim>::setMeshSpacing(const vector_type& meshSpacing) {
        meshSpacing_m = meshSpacing;
        this->updateCellVolume_m();

        if (hasSpacingFields_m)
            storeSpacingFields();
    }


    template<typename T, unsigned Dim>
    T UniformCartesian<T, Dim>::getCellVolume() const {
        return volume_m;
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::updateCellVolume_m() {
        // update cell volume
        volume_m = 1.0;
        for (unsigned i = 0; i < Dim; ++i) {
            volume_m *= meshSpacing_m[i];
        }
    }

    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::setup_m() {
        volume_m = 0.0;
        FlCell = 0;
        FlVert = 0;
        hasSpacingFields_m = false;
        VertSpacings = 0;
        CellSpacings = 0;
    }

    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const NDIndex<Dim>& ndi)
    {
        setup_m();

        for (unsigned d = 0; d < Dim; d++) {
            this->gridSizes_m[d] = ndi[d].length();
            meshSpacing_m[d] = ndi[d].stride();
            this->origin_m(d) = ndi[d].first();
        }

        updateCellVolume_m();
        set_Dvc();
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const NDIndex<Dim>& ndi,
                                              const vector_type& hx)
    {
        setup_m();
        for (unsigned d = 0; d < Dim; d++) {
            this->gridSizes_m[d] = ndi[d].length();
            this->origin_m(d) = ndi[d].first();
        }
        this->setMeshSpacing(hx);
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const NDIndex<Dim>& ndi,
                                              const vector_type& hx,
                                              const vector_type& origin)
    {
        initialize(ndi, hx);
        this->setOrigin(origin);
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I) {
        PInsist(Dim==1, "Number of Index arguments does not match mesh dimension!!");
        setup_m();
        this->gridSizes_m[0] = I.length();
        meshSpacing_m[0] = I.stride();
        this->origin_m(0) = I.first();

        updateCellVolume_m();
        set_Dvc();
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I,
                                              const vector_type& hx)
    {
        PInsist(Dim==1, "Number of Index arguments does not match mesh dimension!!");
        setup_m();
        this->gridSizes_m[0] = I.length();
        this->origin_m(0) = I.first();

        this->setMeshSpacing(hx);      // Set mesh spacings and compute cell volume_m
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I,
                                              const vector_type& hx,
                                              const vector_type& origin)
    {
        this->initialize(I, hx);
        this->sorigin_m = origin;
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I, const Index& J)
    {
        PInsist(Dim==2, "Number of Index arguments does not match mesh dimension!!");
        setup_m();
        this->gridSizes_m[0] = I.length();
        this->gridSizes_m[1] = J.length();
        meshSpacing_m[0] = I.stride();
        meshSpacing_m[1] = J.stride();
        this->origin_m(0) = I.first();      // Default this->origin_m at (I.first(),J.first())
        this->origin_m(1) = J.first();

        updateCellVolume_m();
        set_Dvc();
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I,
                                              const Index& J,
                                              const vector_type& hx)
    {
        PInsist(Dim==2, "Number of Index arguments does not match mesh dimension!!");
        setup_m();
        this->gridSizes_m[0] = I.length();
        this->gridSizes_m[1] = J.length();
        this->origin_m(0) = I.first();      // Default this->origin_m at (I.first(),J.first())
        this->origin_m(1) = J.first();

        this->setMeshSpacing(hx);      // Set mesh spacings and compute cell volume_m
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I,
                                              const Index& J,
                                              const vector_type& hx,
                                              const vector_type& origin)
    {
        this->initialize(I, J, hx);
        this->origin_m = origin;
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I,
                                              const Index& J,
                                              const Index& K)
    {
        PInsist(Dim==3, "Number of Index arguments does not match mesh dimension!!");
        setup_m();
        this->gridSizes_m[0] = I.length();
        this->gridSizes_m[1] = J.length();
        this->gridSizes_m[2] = K.length();
        meshSpacing_m[0] = I.stride();
        meshSpacing_m[1] = J.stride();
        meshSpacing_m[2] = K.stride();
        this->origin_m(0) = I.first();
        this->origin_m(1) = J.first();
        this->origin_m(2) = K.first();

        updateCellVolume_m();
        set_Dvc();
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I,
                                              const Index& J,
                                              const Index& K,
                                              const vector_type& hx)
    {
        PInsist(Dim==3, "Number of Index arguments does not match mesh dimension!!");
        setup_m();
        this->gridSizes_m[0] = I.length();
        this->gridSizes_m[1] = J.length();
        this->gridSizes_m[2] = K.length();
        this->origin_m(0) = I.first();
        this->origin_m(1) = J.first();
        this->origin_m(2) = K.first();
        this->setMeshSpacing(hx);
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(const Index& I,
                                              const Index& J,
                                              const Index& K,
                                              const vector_type& hx,
                                              const vector_type& origin)
    {
        PInsist(Dim==3, "Number of Index arguments does not match mesh dimension!!");
        this->initialize(I, J, K, hx);
        this->origin_m = origin;
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::set_Dvc()
    {
        unsigned d;
        T coef = 1.0;
        for (d = 1; d < Dim; ++d)
            coef *= 0.5;

        for (d = 0; d < Dim; ++d) {
            T dvc = coef/meshSpacing_m[d];
            for (unsigned b = 0; b < (1u << Dim); ++b) {
                int s = ( b & (1 << d) ) ? 1 : -1;
                Dvc[b][d] = s*dvc;
            }
        }
    }



    // Create BareField's of vertex and cell spacings
    // Special prototypes taking no args or FieldLayout ctor args:
    // No-arg case:
    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::storeSpacingFields() {
        // Set up default FieldLayout parameters:
        e_dim_tag et[Dim];
        for (unsigned int d = 0; d < Dim; d++)
            et[d] = PARALLEL;
        storeSpacingFields(et);
    }


    // The general storeSpacingfields() function; others invoke this internally:
    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::storeSpacingFields(e_dim_tag* et)
    {
        std::cout << "HI" << std::endl;
        // VERTEX-VERTEX SPACINGS (same as CELL-CELL SPACINGS for uniform):
        NDIndex<Dim> cells, verts;
        unsigned int d;
        for (d = 0; d < Dim; d++) {
            cells[d] = Index(this->gridSizes_m[d]-1);
            verts[d] = Index(this->gridSizes_m[d]);
        }

        if (!hasSpacingFields_m) {
            // allocate layout and spacing field
            FlCell = std::make_shared<FieldLayout<Dim>>(cells, et);
            // Note: enough guard cells only for existing Div(), etc. implementations:
            // (not really used by Div() etc for UniformCartesian); someday should make
            // this user-settable.
            VertSpacings = std::make_shared<BareField_t>(*FlCell);

            FlVert = std::make_shared<FieldLayout<Dim>>(verts, et);

            CellSpacings = std::make_shared<BareField_t>(*FlVert);
        }

        BareField_t& vertSpacings = *VertSpacings;
        vector_type vertexSpacing;
        for (d=0; d<Dim; d++)
            vertexSpacing[d] = meshSpacing_m[d];
        vertSpacings = vertexSpacing;

        // +++++++++++++++vertSpacings++++++++++++++
//         typename BareField_t::iterator_t lfield;
//         for (lfield = vertSpacings.begin();
//              lfield != vertSpacings.end(); ++lfield)
//         {
// //             LField_t& lfield_r = *(*lfield);
//
// //             const NDIndex<Dim> &owned = lfield_r.getOwned();
//             std::cout << "HI" << std::endl;
//
//         }

        // For uniform cartesian mesh, cell-cell spacings are identical to
        // vert-vert spacings:
        //12/8/98  CellSpacings = VertSpacings;
        // Added 12/8/98 --TJW:
        BareField_t& cellSpacings = *CellSpacings;
        cellSpacings = vertexSpacing;

        hasSpacingFields_m = true; // Flag this as having been done to this object.
    }

//-----------------------------------------------------------------------------
// I/O:
//-----------------------------------------------------------------------------
// Formatted output of UniformCartesian object:
template< typename T, unsigned Dim >
void
UniformCartesian<T, Dim>::
print(std::ostream& out)
{
    Inform info("", out);
    print(info);
}

template< typename T, unsigned Dim >
void
UniformCartesian<T, Dim>::
print(Inform& out)
{
    out << "======UniformCartesian<" << Dim << ",T>==begin======\n";
    unsigned int d;
    for (d=0; d < Dim; d++)
        out << "this->gridSizes_m[" << d << "] = " << this->gridSizes_m[d] << "\n";
    out << "this->origin_m = " << this->origin_m << "\n";
    for (d=0; d < Dim; d++)
        out << "meshSpacing_m[" << d << "] = " << meshSpacing_m[d] << "\n";
    for (d=0; d < (1u<<Dim); d++)
        out << "Dvc[" << d << "] = " << Dvc[d] << "\n";
    out << "cell volume_m = " << volume_m << "\n";
    out << "======UniformCartesian<" << Dim << ",T>==end========\n";
}

}
