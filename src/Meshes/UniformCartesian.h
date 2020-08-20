// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef UNIFORM_CARTESIAN_H
#define UNIFORM_CARTESIAN_H

// UniformCartesian.h
// UniformCartesian class - represents uniform-spacing cartesian meshes.

// include files
#include "Meshes/Mesh.h"
#include "Meshes/Centering.h"
#include "Meshes/CartesianCentering.h"
#include "AppTypes/Vektor.h"

// forward declarations
template<class T, unsigned D> class BareField;
template<class T, unsigned D, class M, class C> class Field;
template <unsigned Dim, class MFLOAT> class UniformCartesian;
template <unsigned Dim, class MFLOAT>
std::ostream& operator<<(std::ostream&, const UniformCartesian<Dim,MFLOAT>&);
template <unsigned Dim, class MFLOAT>
Inform& operator<<(Inform&, const UniformCartesian<Dim,MFLOAT>&);

template <unsigned Dim, class MFLOAT=double>
class UniformCartesian : public Mesh<Dim>
{
public:
  //# public typedefs
  typedef Cell DefaultCentering;
  typedef MFLOAT MeshValue_t;
  typedef Vektor<MFLOAT,Dim> MeshVektor_t;

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
  // These further take a Vektor<MFLOAT,Dim>& specifying the origin:
  UniformCartesian(const NDIndex<Dim>& ndi, MFLOAT* const delX,
                   const Vektor<MFLOAT,Dim>& orig);
  UniformCartesian(const Index& I, MFLOAT* const delX,
                   const Vektor<MFLOAT,Dim>& orig);
  UniformCartesian(const Index& I, const Index& J, MFLOAT* const delX,
                   const Vektor<MFLOAT,Dim>& orig);
  UniformCartesian(const Index& I, const Index& J, const Index& K,
                   MFLOAT* const delX, const Vektor<MFLOAT,Dim>& orig);

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
  // These further take a Vektor<MFLOAT,Dim>& specifying the origin:
  void initialize(const NDIndex<Dim>& ndi, MFLOAT* const delX,
                  const Vektor<MFLOAT,Dim>& orig);
  void initialize(const Index& I, MFLOAT* const delX,
                  const Vektor<MFLOAT,Dim>& orig);
  void initialize(const Index& I, const Index& J, MFLOAT* const delX,
                  const Vektor<MFLOAT,Dim>& orig);
  void initialize(const Index& I, const Index& J, const Index& K,
                  MFLOAT* const delX, const Vektor<MFLOAT,Dim>& orig);


private:

  // Private member data:
  MFLOAT meshSpacing[Dim];   // delta-x, delta-y (>1D), delta-z (>2D)
  MFLOAT volume;             // Cell length(1D), area(2D), or volume (>2D)
  Vektor<MFLOAT,Dim> origin; // Origin of mesh coordinates (vertices)
  FieldLayout<Dim>* FlCell;  // Layouts for BareField* CellSpacings
  FieldLayout<Dim>* FlVert;  // Layouts for BareField* VertSpacings

  // Private member functions:
  void setup(); // Private function doing tasks common to all constructors.
  // Set only the derivative constants, using pre-set spacings:
  void set_Dvc();


public:

  // Public member data:
  unsigned gridSizes[Dim];        // Sizes (number of vertices)
  Vektor<MFLOAT,Dim> Dvc[1<<Dim]; // Constants for derivatives.
  bool hasSpacingFields;              // Flags allocation of the following:
  BareField<Vektor<MFLOAT,Dim>,Dim>* VertSpacings;
  BareField<Vektor<MFLOAT,Dim>,Dim>* CellSpacings;

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
  Vektor<MFLOAT,Dim> get_origin() const;
  // Get the spacings of mesh vertex positions along specified direction:
  MFLOAT get_meshSpacing(unsigned d) const;
  // Get the cell volume:
  MFLOAT get_volume() const;

  // Set functions for member data:
  // Set the origin of mesh vertex positions:
  void set_origin(const Vektor<MFLOAT,Dim>& o);
  // Set the spacings of mesh vertex positions (recompute Dvc, cell volume):
  void set_meshSpacing(MFLOAT* const del);

  // Formatted output of UniformCartesian object:
  void print(std::ostream&);

  void print(Inform &);

  //----Other UniformCartesian methods:----------------------------------------
  // Volume of single cell indexed by input NDIndex;
  MFLOAT getCellVolume(const NDIndex<Dim>&) const;
  // Field of volumes of all cells:
  Field<MFLOAT,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
  getCellVolumeField(Field<MFLOAT,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&) const;
  // Volume of range of cells bounded by verticies specified by input NDIndex:
  MFLOAT getVertRangeVolume(const NDIndex<Dim>&) const;
  // Volume of range of cells spanned by input NDIndex (index of cells):
  MFLOAT getCellRangeVolume(const NDIndex<Dim>&) const;
  // Nearest vertex index to (x,y,z):
  NDIndex<Dim> getNearestVertex(const Vektor<MFLOAT,Dim>&) const;
  // Nearest vertex index with all vertex coordinates below (x,y,z):
  NDIndex<Dim> getVertexBelow(const Vektor<MFLOAT,Dim>&) const;
  // NDIndex for cell in cell-ctrd Field containing the point (x,y,z):
  inline
  NDIndex<Dim> getCellContaining(const Vektor<MFLOAT,Dim>& x) const
  {
    return getVertexBelow(x); // I think these functions are identical. -tjw
  }
  // (x,y,z) coordinates of indexed vertex:
  Vektor<MFLOAT,Dim> getVertexPosition(const NDIndex<Dim>&) const;
  // Field of (x,y,z) coordinates of all vertices:
  Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Vert>&
  getVertexPositionField(Field<Vektor<MFLOAT,Dim>,Dim,
			 UniformCartesian<Dim,MFLOAT>,Vert>& ) const;
  // (x,y,z) coordinates of indexed cell:
  Vektor<MFLOAT,Dim> getCellPosition(const NDIndex<Dim>&) const;
  // Field of (x,y,z) coordinates of all cells:
  Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
  getCellPositionField(Field<Vektor<MFLOAT,Dim>,Dim,
		       UniformCartesian<Dim,MFLOAT>,Cell>& ) const;
  // Vertex-vertex grid spacing of indexed cell:
  Vektor<MFLOAT,Dim> getDeltaVertex(const NDIndex<Dim>&) const;
  // Field of vertex-vertex grid spacings of all cells:
  Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
  getDeltaVertexField(Field<Vektor<MFLOAT,Dim>,Dim,
		      UniformCartesian<Dim,MFLOAT>,Cell>& ) const;
  // Cell-cell grid spacing of indexed vertex:
  Vektor<MFLOAT,Dim> getDeltaCell(const NDIndex<Dim>&) const;
  // Field of cell-cell grid spacings of all vertices:
  Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Vert>&
  getDeltaCellField(Field<Vektor<MFLOAT,Dim>,Dim,
		    UniformCartesian<Dim,MFLOAT>,Vert>& ) const;
  // Array of surface normals to cells adjoining indexed cell:
  Vektor<MFLOAT,Dim>* getSurfaceNormals(const NDIndex<Dim>&) const;
  // Array of (pointers to) Fields of surface normals to all cells:
  void getSurfaceNormalFields(Field<Vektor<MFLOAT,Dim>,Dim,
			      UniformCartesian<Dim,MFLOAT>,Cell>** ) const;
  // Similar functions, but specify the surface normal to a single face, using
  // the following numbering convention: 0 means low face of 1st dim, 1 means
  // high face of 1st dim, 2 means low face of 2nd dim, 3 means high face of
  // 2nd dim, and so on:
  Vektor<MFLOAT,Dim> getSurfaceNormal(const NDIndex<Dim>&, unsigned) const;
  Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
  getSurfaceNormalField(Field<Vektor<MFLOAT,Dim>,Dim,
			UniformCartesian<Dim,MFLOAT>,Cell>&, unsigned) const;

};

// I/O

// Stream formatted output of UniformCartesian object:
template< unsigned Dim, class MFLOAT >
inline
std::ostream& operator<<(std::ostream& out, const UniformCartesian<Dim,MFLOAT>& mesh)
{
  UniformCartesian<Dim,MFLOAT>& ncmesh =
    const_cast<UniformCartesian<Dim,MFLOAT>&>(mesh);
  ncmesh.print(out);
  return out;
}

template< unsigned Dim, class MFLOAT >
inline
Inform& operator<<(Inform& out, const UniformCartesian<Dim,MFLOAT>& mesh)
{
  UniformCartesian<Dim,MFLOAT>& ncmesh =
    const_cast<UniformCartesian<Dim,MFLOAT>&>(mesh);
  ncmesh.print(out);
  return out;
}

//*****************************************************************************
// Stuff taken from old Cartesian.h, which now applies to UniformCartesian:
//*****************************************************************************

#ifndef CARTESIAN_STENCIL_SETUP_H
#include "Meshes/CartesianStencilSetup.h"
#endif

//----------------------------------------------------------------------
//
//
// Definitions of stencils.
//
// For each one we have first we have the user level function that takes
// a Field argument and returns an expression template.
// This is the thing the user code sees.
// These could use some asserts to make sure the Fields have
// enough guard cells.
//
// Then we have the 'apply' function that gets used in the inner loop
// when evaluating the expression.  This would be better done as a
// member function of the tags above, but that would require member
// function templates, which we are holding off on for now.
//
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//
// Divergence Vert->Cell
//
// First is the user function for any dimension and element types.
// it returns a unary element type.
//
//----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////
//
// Here are the old style definitions.
//
////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------
// Divergence Vektor/Vert -> Scalar/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Divergence Vektor/Cell -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Divergence Vektor/Vert -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Divergence Vektor/Edge -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Edge>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Edge>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Edge>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Divergence Vektor/Cell -> Scalar/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Divergence Tenzor/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
    Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
    Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
    Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Divergence Tenzor/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
    Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
    Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
    Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);

//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
     Field<Vektor<T,2u>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------

/// Grad operator  approximate at the border up to a term of order O(h^2)
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);

/// Old Grad operator
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Grad1Ord(Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Edge
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Edge>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Edge>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Edge>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
     Field<Vektor<T,2u>,2U,UniformCartesian<2U,MFLOAT>,Edge>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Edge>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Edge>& r);
//----------------------------------------------------------------------
// Grad Scalar/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
     Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
     Field<Vektor<T,2u>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Grad Scalar/Cell -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
     Field<Vektor<T,2u>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Grad Vektor/Vert -> Tenzor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Grad Vektor/Cell -> Tenzor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Divergence SymTenzor/Vert -> Vektor/Cell
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Div(Field<SymTenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
    Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Div(Field<SymTenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
    Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Divergence SymTenzor/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
    Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
    Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
    Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);

namespace IPPL {

//----------------------------------------------------------------------
// Weighted average Cell to Vert
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Average(Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
	Field<T2,1U,UniformCartesian<1U,MFLOAT>,Cell>& w,
	Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>& r) ;
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Average(Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
	Field<T2,2U,UniformCartesian<2U,MFLOAT>,Cell>& w,
	Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Average(Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
	Field<T2,3U,UniformCartesian<3U,MFLOAT>,Cell>& w,
	Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);

//----------------------------------------------------------------------
// Weighted average Vert to Cell
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Average(Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
	Field<T2,1U,UniformCartesian<1U,MFLOAT>,Vert>& w,
	Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>& r) ;
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Average(Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
	Field<T2,2U,UniformCartesian<2U,MFLOAT>,Vert>& w,
	Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Average(Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
	Field<T2,3U,UniformCartesian<3U,MFLOAT>,Vert>& w,
	Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);

//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Unweighted average Cell to Vert
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Average(Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
	Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>& r) ;
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Average(Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
	Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Average(Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
	Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>& r);

//----------------------------------------------------------------------
// Unweighted average Vert to Cell
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Average(Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
	Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>& r) ;
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Average(Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
	Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Average(Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
	Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>& r);

//----------------------------------------------------------------------
}

#include "Meshes/UniformCartesian.hpp"

#endif // UNIFORM_CARTESIAN_H

/***************************************************************************
 * $RCSfile: UniformCartesian.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: UniformCartesian.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $
 ***************************************************************************/
