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
#ifndef IPPL_UNIFORM_CARTESIAN_H
#define IPPL_UNIFORM_CARTESIAN_H

#include "Meshes/Mesh.h"
#include "Meshes/CartesianCentering.h"

namespace ippl {

template<typename T, unsigned Dim>
class UniformCartesian : public Mesh<T, Dim> {

public:
    typedef typename Mesh<T, Dim>::vector_type vector_type;
    typedef BareField<vector_type, Dim> BareField_t;

    typedef Cell DefaultCentering;


    UniformCartesian();

    // Non-default constructors
    UniformCartesian(const NDIndex<Dim>& ndi,
                     bool evalCellVolume = true);

    UniformCartesian(const NDIndex<Dim>& ndi,
                     const vector_type& hx);

    UniformCartesian(const NDIndex<Dim>& ndi,
                     const vector_type& hx,
                     const vector_type& origin);


    template <class... Args,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Args>...>::value
                >
             >
    UniformCartesian(const Args&... args,
                     bool evalCellVolume = true);

    template <class... Args,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Args>...>::value
                >
             >
    UniformCartesian(const Args&... args,
                     const vector_type& hx);


    template <class... Args,
              std::enable_if_t<
                std::conjunction<
                    std::is_same<Index, Args>...>::value
                >
             >
    UniformCartesian(const Args&... args,
                     const vector_type& hx,
                     const vector_type& origin);

    ~UniformCartesian() = default;


    // initialize functions
    void initialize(const NDIndex<Dim>& ndi);
    void initialize(const NDIndex<Dim>& ndi, const vector_type& hx);
    void initialize(const NDIndex<Dim>& ndi, const vector_type& hx,
                    const vector_type& origin);

    void initialize(const Index& I);
    void initialize(const Index& I, const vector_type& hx);
    void initialize(const Index& I, const vector_type& hx,
                    const vector_type& origin);

    void initialize(const Index& I, const Index& J);
    void initialize(const Index& I, const Index& J, const vector_type& hx);
    void initialize(const Index& I, const Index& J, const vector_type& hx,
                    const vector_type& origin);


    void initialize(const Index& I, const Index& J, const Index& K);
    void initialize(const Index& I, const Index& J, const Index& K,
                    const vector_type& hx);
    void initialize(const Index& I, const Index& J, const Index& K,
                    const vector_type& hx, const vector_type& origin);



    // Get the spacings of mesh vertex positions along specified direction
    T getMeshSpacing(unsigned dim) const;

    const vector_type& getMeshSpacing() const;

    // Set the spacings of mesh vertex positions (recompute Dvc, cell volume):
    void setMeshSpacing(const vector_type& meshSpacing);


    T getCellVolume() const;


    // (x,y,z) coordinates of indexed vertex:
    vector_type getVertexPosition(const NDIndex<Dim>& ndi) const {
        vector_type vertexPosition;
        for (unsigned int d = 0; d < Dim; d++)
                vertexPosition(d) = ndi[d].first() * meshSpacing_m[d] + this->origin_m(d);
        return vertexPosition;
    }

    // Vertex-vertex grid spacing of indexed cell:
    vector_type getDeltaVertex(const NDIndex<Dim>& ndi) const {
        vector_type vertexVertexSpacing;
        for (unsigned int d = 0; d < Dim; d++)
            vertexVertexSpacing[d] = meshSpacing_m[d] * ndi[d].length();
        return vertexVertexSpacing;
    }

private:
    UniformCartesian(std::initializer_list<Index> indices,
                     bool evalCellVolume);

    UniformCartesian(std::initializer_list<Index> indices,
                     const vector_type& hx);

    UniformCartesian(std::initializer_list<Index> indices,
                     const vector_type& hx,
                     const vector_type& origin);


    vector_type meshSpacing_m;     // delta-x, delta-y (>1D), delta-z (>2D)
    T volume_m;                     // Cell length(1D), area(2D), or volume (>2D)





  std::shared_ptr<FieldLayout<Dim>> FlCell;  // Layouts for BareField* CellSpacings
  std::shared_ptr<FieldLayout<Dim>> FlVert;  // Layouts for BareField* VertSpacings



  // Set only the derivative constants, using pre-set spacings:
  void set_Dvc();

  void updateCellVolume_m();

  void setup_m();

public:

  // Public member data:
  vector_type Dvc[1<<Dim]; // Constants for derivatives.
  bool hasSpacingFields_m;              // Flags allocation of the following:
  std::shared_ptr<BareField_t> VertSpacings;
  std::shared_ptr<BareField_t> CellSpacings;

  // Public member functions:

  // Create BareField's of vertex and cell spacings; allow for specifying
  // layouts via the FieldLayout e_dim_tag and vnodes parameters (these
  // get passed in to construct the FieldLayout used to construct the
  // BareField's).
  void storeSpacingFields(); // Default; will have default layout

  // Next we have one for arbitrary dimension, ala FieldLayout ctor:
  // All the others call this one internally:
  void storeSpacingFields(e_dim_tag *p);

  // Formatted output of UniformCartesian object:
  void print(std::ostream&);

  void print(Inform &);

};

// // I/O
//
// // Stream formatted output of UniformCartesian object:
// template< unsigned Dim, class T >
// inline
// std::ostream& operator<<(std::ostream& out, const UniformCartesian<Dim,T>& mesh)
// {
//   UniformCartesian<Dim,T>& ncmesh =
//     const_cast<UniformCartesian<Dim,T>&>(mesh);
//   ncmesh.print(out);
//   return out;
// }
//
// template< unsigned Dim, class T >
// inline
// Inform& operator<<(Inform& out, const UniformCartesian<Dim,T>& mesh)
// {
//   UniformCartesian<Dim,T>& ncmesh =
//     const_cast<UniformCartesian<Dim,T>&>(mesh);
//   ncmesh.print(out);
//   return out;
// }

}

#include "Meshes/UniformCartesian.hpp"

#endif
