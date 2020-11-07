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

    /*
     * Dim == 1
     */
    UniformCartesian(const Index& I,
                     bool evalCellVolume = true);

    UniformCartesian(const Index& I,
                     const vector_type& hx);

    UniformCartesian(const Index& I,
                     const vector_type& hx,
                     const vector_type& origin);

    /*
     * Dim == 2
     */
    UniformCartesian(const Index& I,
                     const Index& J,
                     bool evalCellVolume = true);

    UniformCartesian(const Index& I,
                     const Index& J,
                     const vector_type& hx);

    UniformCartesian(const Index& I,
                     const Index& J,
                     const vector_type& hx,
                     const vector_type& origin);

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
                     const vector_type& hx);


    UniformCartesian(const Index& I,
                     const Index& J,
                     const Index& K,
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


private:
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
  // Special cases for 1-3 dimensions, ala FieldLayout ctors:
  void storeSpacingFields(e_dim_tag p1);

  void storeSpacingFields(e_dim_tag p1, e_dim_tag p2);
  void storeSpacingFields(e_dim_tag p1, e_dim_tag p2, e_dim_tag p3);
  // Next we have one for arbitrary dimension, ala FieldLayout ctor:
  // All the others call this one internally:
  void storeSpacingFields(e_dim_tag *p);

  // These specify both the total number of vnodes and the numbers of vnodes
  // along each dimension for the partitioning of the index space. Obviously
  // this restricts the number of vnodes to be a product of the numbers along
  // each dimension (the constructor implementation checks this): Special
  // cases for 1-3 dimensions, ala FieldLayout ctors (see FieldLayout.h for
  // more relevant comments, including definition of recurse):
  void storeSpacingFields(e_dim_tag p1,
			  unsigned vnodes1,
			  bool recurse=false);
  void storeSpacingFields(e_dim_tag p1, e_dim_tag p2,
			  unsigned vnodes1, unsigned vnodes2,
			  bool recurse=false);
  void storeSpacingFields(e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			  unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
			  bool recurse=false);
  // Next we have one for arbitrary dimension, ala FieldLayout ctor:
  // All the others call this one internally:
  void storeSpacingFields(e_dim_tag *p,
			  unsigned* vnodesPerDirection,
			  bool recurse=false);


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
