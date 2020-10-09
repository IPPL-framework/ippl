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

#include "Meshes/Kokkos_Mesh.h"
#include "Meshes/CartesianCentering.h"
#include "AppTypes/Vector.h"

namespace ippl {

template<typename T, unsigned Dim>
class UniformCartesian : public Mesh<T, Dim> {

public:
    typedef typename Mesh<T, Dim>::MeshVector_t MeshVector_t;

    typedef Cell DefaultCentering;


    UniformCartesian();

    ~UniformCartesian();

    // Non-default constructors
    UniformCartesian(const NDIndex<Dim>& ndi,
                     bool evalCellVolume = true);

    UniformCartesian(const NDIndex<Dim>& ndi,
                     const MeshVector_t& hx);

    UniformCartesian(const NDIndex<Dim>& ndi,
                     const MeshVector_t& hx,
                     const MeshVector_t& origin);

    /*
     * Dim == 1
     */
    UniformCartesian(const Index& I,
                     bool evalCellVolume = true);

    UniformCartesian(const Index& I,
                     const MeshVector_t& hx);

    UniformCartesian(const Index& I,
                     const MeshVector_t& hx,
                     const MeshVector_t& origin);

    /*
     * Dim == 2
     */
    UniformCartesian(const Index& I,
                     const Index& J,
                     bool evalCellVolume = true);

    UniformCartesian(const Index& I,
                     const Index& J,
                     const MeshVector_t& hx);

    UniformCartesian(const Index& I,
                     const Index& J,
                     const MeshVector_t& hx,
                     const MeshVector_t& orig);

    /*
     * Dim == 3
     */
    UniformCartesian(const Index& I,
                     const Index& J,
                     const Index& K,
                     bool evalCellVolume = true);

    UniformCartesian(const Index& I,
                     const Index& J,
                     const Index& K,
                     const MeshVector_t& hx);


    UniformCartesian(const Index& I,
                     const Index& J,
                     const Index& K,
                     const MeshVector_t& hx,
                     const MeshVector_t& orig);


    // initialize functions
    void initialize(const NDIndex<Dim>& ndi);
    void initialize(const Index& I);
    void initialize(const Index& I, const Index& J);
    void initialize(const Index& I, const Index& J, const Index& K);
    // These also take a T* specifying the mesh spacings:
    void initialize(const NDIndex<Dim>& ndi, T* const delX);
    void initialize(const Index& I, T* const delX);
    void initialize(const Index& I, const Index& J, T* const delX);
    void initialize(const Index& I, const Index& J, const Index& K,
                    T* const delX);
    // These further take a MeshVector_t& specifying the origin:
    void initialize(const NDIndex<Dim>& ndi, T* const delX,
                    const MeshVector_t& orig);
    void initialize(const Index& I, T* const delX,
                    const MeshVector_t& orig);
    void initialize(const Index& I, const Index& J, T* const delX,
                    const MeshVector_t& orig);
    void initialize(const Index& I, const Index& J, const Index& K,
                    T* const delX, const MeshVector_t& orig);



    // Get the spacings of mesh vertex positions along specified direction
    T getMeshSpacing(unsigned dim) const;

    // Set the spacings of mesh vertex positions (recompute Dvc, cell volume):
    void setMeshSpacing(const MeshVector_t& meshSpacing);


    T getCellVolume() const;

private:
    MeshVector_t meshSpacing_m;     // delta-x, delta-y (>1D), delta-z (>2D)
    T volume_m;                     // Cell length(1D), area(2D), or volume (>2D)




  FieldLayout<Dim>* FlCell;  // Layouts for BareField* CellSpacings
  FieldLayout<Dim>* FlVert;  // Layouts for BareField* VertSpacings


  // Set only the derivative constants, using pre-set spacings:
  void set_Dvc();

  void updateCellVolume_m();


public:

  // Public member data:
  MeshVector_t Dvc[1<<Dim]; // Constants for derivatives.
  bool hasSpacingFields_m;              // Flags allocation of the following:
  BareField<MeshVector_t,Dim>* VertSpacings;
  BareField<MeshVector_t,Dim>* CellSpacings;

  // Public member functions:

  // Create BareField's of vertex and cell spacings; allow for specifying
  // layouts via the FieldLayout e_dim_tag and vnodes parameters (these
  // get passed in to construct the FieldLayout used to construct the
  // BareField's).
  void storeSpacingFields(); // Default; will have default layout
  // Special cases for 1-3 dimensions, ala FieldLayout ctors:
  void storeSpacingFields(e_dim_tag p1, int vnodes=-1);
  void storeSpacingFields(e_dim_tag p1, e_dim_tag p2, int vnodes=-1);
  void storeSpacingFields(e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			  int vnodes=-1);
  // Next we have one for arbitrary dimension, ala FieldLayout ctor:
  // All the others call this one internally:
  void storeSpacingFields(e_dim_tag *p, int vnodes=-1);

  // These specify both the total number of vnodes and the numbers of vnodes
  // along each dimension for the partitioning of the index space. Obviously
  // this restricts the number of vnodes to be a product of the numbers along
  // each dimension (the constructor implementation checks this): Special
  // cases for 1-3 dimensions, ala FieldLayout ctors (see FieldLayout.h for
  // more relevant comments, including definition of recurse):
  void storeSpacingFields(e_dim_tag p1,
			  unsigned vnodes1,
			  bool recurse=false,
			  int vnodes=-1);
  void storeSpacingFields(e_dim_tag p1, e_dim_tag p2,
			  unsigned vnodes1, unsigned vnodes2,
			  bool recurse=false,int vnodes=-1);
  void storeSpacingFields(e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			  unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
			  bool recurse=false, int vnodes=-1);
  // Next we have one for arbitrary dimension, ala FieldLayout ctor:
  // All the others call this one internally:
  void storeSpacingFields(e_dim_tag *p,
			  unsigned* vnodesPerDirection,
			  bool recurse=false, int vnodes=-1);


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

#include "Meshes/Kokkos_UniformCartesian.hpp"

#endif
