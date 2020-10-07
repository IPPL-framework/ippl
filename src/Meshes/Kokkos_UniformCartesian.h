// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef IPPL_UNIFORM_CARTESIAN_H
#define IPPL_UNIFORM_CARTESIAN_H

// UniformCartesian.h
// UniformCartesian class - represents uniform-spacing cartesian meshes.

// include files
#include "Meshes/Mesh.h"
#include "Meshes/Centering.h"
#include "Meshes/CartesianCentering.h"
#include "AppTypes/Vector.h"

// // forward declarations
// template<class T, unsigned D> class BareField;
// template<class T, unsigned D, class M, class C> class Field;
// template <unsigned Dim, class MFLOAT> class UniformCartesian;
// template <unsigned Dim, class MFLOAT>
// std::ostream& operator<<(std::ostream&, const UniformCartesian<Dim,MFLOAT>&);
// template <unsigned Dim, class MFLOAT>
// Inform& operator<<(Inform&, const UniformCartesian<Dim,MFLOAT>&);

namespace ippl {

template <unsigned Dim, class MFLOAT=double>
class UniformCartesian : public Mesh<Dim>
{
public:
  //# public typedefs
  typedef Cell DefaultCentering;
  typedef MFLOAT MeshValue_t;
  typedef Vector<MFLOAT,Dim> MeshVector_t;

  // Default constructor (use initialize() to make valid)
  UniformCartesian()
  {
    hasSpacingFields = false;
  };
  // Destructor
  ~UniformCartesian()
  {
    if (hasSpacingFields) {
      delete VertSpacings;
      delete CellSpacings;
      delete FlVert;
      delete FlCell;
    }
  };

  // Non-default constructors
  UniformCartesian(const NDIndex<Dim>& ndi);
  UniformCartesian(const Index& I);
  UniformCartesian(const Index& I, const Index& J);
  UniformCartesian(const Index& I, const Index& J, const Index& K);
  // These also take a MFLOAT* specifying the mesh spacings:
  UniformCartesian(const NDIndex<Dim>& ndi, MFLOAT* const delX);
  UniformCartesian(const Index& I, MFLOAT* const delX);
  UniformCartesian(const Index& I, const Index& J, MFLOAT* const delX);
  UniformCartesian(const Index& I, const Index& J, const Index& K,
                   MFLOAT* const delX);
  // These further take a Vector<MFLOAT,Dim>& specifying the origin:
  UniformCartesian(const NDIndex<Dim>& ndi, MFLOAT* const delX,
                   const Vector<MFLOAT,Dim>& orig);
  UniformCartesian(const Index& I, MFLOAT* const delX,
                   const Vector<MFLOAT,Dim>& orig);
  UniformCartesian(const Index& I, const Index& J, MFLOAT* const delX,
                   const Vector<MFLOAT,Dim>& orig);
  UniformCartesian(const Index& I, const Index& J, const Index& K,
                   MFLOAT* const delX, const Vector<MFLOAT,Dim>& orig);

  // initialize functions
  void initialize(const NDIndex<Dim>& ndi);
  void initialize(const Index& I);
  void initialize(const Index& I, const Index& J);
  void initialize(const Index& I, const Index& J, const Index& K);
  // These also take a MFLOAT* specifying the mesh spacings:
  void initialize(const NDIndex<Dim>& ndi, MFLOAT* const delX);
  void initialize(const Index& I, MFLOAT* const delX);
  void initialize(const Index& I, const Index& J, MFLOAT* const delX);
  void initialize(const Index& I, const Index& J, const Index& K,
                  MFLOAT* const delX);
  // These further take a Vector<MFLOAT,Dim>& specifying the origin:
  void initialize(const NDIndex<Dim>& ndi, MFLOAT* const delX,
                  const Vector<MFLOAT,Dim>& orig);
  void initialize(const Index& I, MFLOAT* const delX,
                  const Vector<MFLOAT,Dim>& orig);
  void initialize(const Index& I, const Index& J, MFLOAT* const delX,
                  const Vector<MFLOAT,Dim>& orig);
  void initialize(const Index& I, const Index& J, const Index& K,
                  MFLOAT* const delX, const Vector<MFLOAT,Dim>& orig);


private:

  // Private member data:
  MFLOAT meshSpacing[Dim];   // delta-x, delta-y (>1D), delta-z (>2D)
  MFLOAT volume;             // Cell length(1D), area(2D), or volume (>2D)
  Vector<MFLOAT,Dim> origin; // Origin of mesh coordinates (vertices)
  FieldLayout<Dim>* FlCell;  // Layouts for BareField* CellSpacings
  FieldLayout<Dim>* FlVert;  // Layouts for BareField* VertSpacings

  // Private member functions:
  void setup(); // Private function doing tasks common to all constructors.
  // Set only the derivative constants, using pre-set spacings:
  void set_Dvc();


public:

  // Public member data:
  unsigned gridSizes[Dim];        // Sizes (number of vertices)
  Vector<MFLOAT,Dim> Dvc[1<<Dim]; // Constants for derivatives.
  bool hasSpacingFields;              // Flags allocation of the following:
  BareField<Vector<MFLOAT,Dim>,Dim>* VertSpacings;
  BareField<Vector<MFLOAT,Dim>,Dim>* CellSpacings;

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

  // Accessor functions for member data:
  // Get the origin of mesh vertex positions:
  Vector<MFLOAT,Dim> get_origin() const;
  // Get the spacings of mesh vertex positions along specified direction:
  MFLOAT get_meshSpacing(unsigned d) const;
  // Get the cell volume:
  MFLOAT get_volume() const;

  // Set functions for member data:
  // Set the origin of mesh vertex positions:
  void set_origin(const Vector<MFLOAT,Dim>& o);
  // Set the spacings of mesh vertex positions (recompute Dvc, cell volume):
  void set_meshSpacing(MFLOAT* const del);

  // Formatted output of UniformCartesian object:
  void print(std::ostream&);

  void print(Inform &);

};

// // I/O
//
// // Stream formatted output of UniformCartesian object:
// template< unsigned Dim, class MFLOAT >
// inline
// std::ostream& operator<<(std::ostream& out, const UniformCartesian<Dim,MFLOAT>& mesh)
// {
//   UniformCartesian<Dim,MFLOAT>& ncmesh =
//     const_cast<UniformCartesian<Dim,MFLOAT>&>(mesh);
//   ncmesh.print(out);
//   return out;
// }
//
// template< unsigned Dim, class MFLOAT >
// inline
// Inform& operator<<(Inform& out, const UniformCartesian<Dim,MFLOAT>& mesh)
// {
//   UniformCartesian<Dim,MFLOAT>& ncmesh =
//     const_cast<UniformCartesian<Dim,MFLOAT>&>(mesh);
//   ncmesh.print(out);
//   return out;
// }

}

#include "Meshes/Kokkos_UniformCartesian.hpp"

#endif
