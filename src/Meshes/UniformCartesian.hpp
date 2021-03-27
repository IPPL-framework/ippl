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


    template <typename T, unsigned Dim>
    template <class... Args,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Args>...>::value
                >
             >
    UniformCartesian<T, Dim>::UniformCartesian(const Args&... args,
                                               bool evalCellVolume)
    : UniformCartesian({args...}, evalCellVolume)
    {
        static_assert(Dim == sizeof...(args),
                      "UniformCartesian: Wrong number of arguments.");
    }


    template <typename T, unsigned Dim>
    template <class... Args,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Args>...>::value
                >
             >
    UniformCartesian<T, Dim>::UniformCartesian(const Args&... args,
                                               const vector_type& hx)
    : UniformCartesian({args...}, hx)
    {
        static_assert(Dim == sizeof...(args),
                      "UniformCartesian: Wrong number of arguments.");
    }


    template <typename T, unsigned Dim>
    template <class... Args,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Args>...>::value
                >
             >
    UniformCartesian<T, Dim>::UniformCartesian(const Args&... args,
                                               const vector_type& hx,
                                               const vector_type& origin)
    : UniformCartesian({args...}, hx, origin)
    {
        static_assert(Dim == sizeof...(args),
                      "UniformCartesian: Wrong number of arguments.");
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
    template <class... Indices,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Indices>...>::value
                >
             >
    void UniformCartesian<T, Dim>::initialize(const Indices&... indices) {

        static_assert(Dim == sizeof...(indices),
                      "UniformCartesian::initialize: Wrong number of arguments.");

        this->initialize({indices...});
    }

    template<typename T, unsigned Dim>
    template <class... Indices,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Indices>...>::value
                >
             >
    void UniformCartesian<T, Dim>::initialize(const Indices&... indices,
                                              const vector_type& hx)
    {

        static_assert(Dim == sizeof...(indices),
                      "UniformCartesian::initialize: Wrong number of arguments.");

        this->initialize({indices...}, hx);
    }


    template<typename T, unsigned Dim>
    template <class... Indices,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Indices>...>::value
                >
             >
    void UniformCartesian<T, Dim>::initialize(const Indices&... indices,
                                              const vector_type& hx,
                                              const vector_type& origin)
    {
        this->initialize({indices...}, hx);
        this->sorigin_m = origin;
    }


    template <typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(std::initializer_list<Index> indices,
                                               bool evalCellVolume)
    {
        unsigned int i = 0;
        for (auto& index : indices) {
            this->gridSizes_m[i] = index.length();
            meshSpacing_m[i] = index.stride();
            this->origin_m(i) = index.first();
            ++i;
        }
        if (evalCellVolume)
            updateCellVolume_m();

        set_Dvc();
    }


    template <typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(std::initializer_list<Index> indices,
                                               const vector_type& hx)
        : UniformCartesian(indices, false)
    {
        setMeshSpacing(hx);
    }


    template <typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian(std::initializer_list<Index> indices,
                                               const vector_type& hx,
                                               const vector_type& origin)
        : UniformCartesian(indices, hx)
    {
        this->setOrigin(origin);
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(std::initializer_list<Index> indices) {
        setup_m();

        unsigned int i = 0;
        for (auto& index : indices) {
            this->gridSizes_m[i] = index.length();
            meshSpacing_m[i] = index.stride();
            this->origin_m(i) = index.first();
            ++i;
        }

        updateCellVolume_m();
        set_Dvc();
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::initialize(std::initializer_list<Index> indices,
                                              const vector_type& hx)
    {
        setup_m();

        unsigned int i = 0;
        for (auto& index : indices) {
            this->gridSizes_m[i] = index.length();
            this->origin_m(i) = index.first();
            ++i;
        }

        this->setMeshSpacing(hx);
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
