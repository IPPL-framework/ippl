// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef CARTESIAN_H
#define CARTESIAN_H

// Cartesian.h
// Cartesian class - represents non-uniform-spacing cartesian meshes.

// include files
#include "Meshes/Mesh.h"
#include "Meshes/Centering.h"
#include "Meshes/CartesianCentering.h"
#include "AppTypes/Vektor.h"

#include <map>

// forward declarations
template<class T, unsigned D> class BareField;
template<class T, unsigned D, class M, class C> class Field;
template <unsigned Dim, class MFLOAT> class Cartesian;
template <unsigned Dim, class MFLOAT>
std::ostream& operator<<(std::ostream&, const Cartesian<Dim,MFLOAT>&);

template <unsigned Dim, class MFLOAT=double>
class Cartesian : public Mesh<Dim>
{
public: 
  //# public typedefs
  typedef Cell DefaultCentering;
  typedef MFLOAT MeshValue_t;
  typedef Vektor<MFLOAT,Dim> MeshVektor_t;

  // Default constructor (use initialize() to make valid)
  Cartesian()
  {
    hasSpacingFields = false;
  };  
  // Destructor
  ~Cartesian()
  {
    if (hasSpacingFields) {
      delete VertSpacings;
      delete CellSpacings;
      delete FlVert;
      delete FlCell;
    }
  }; 

  // Non-default constructors
  Cartesian(const NDIndex<Dim>& ndi);
  Cartesian(const Index& I);
  Cartesian(const Index& I, const Index& J);
  Cartesian(const Index& I, const Index& J, const Index& K);
  // These also take a MFLOAT** specifying the mesh spacings:
  Cartesian(const NDIndex<Dim>& ndi, MFLOAT** const delX);
  Cartesian(const Index& I, MFLOAT** const delX);
  Cartesian(const Index& I, const Index& J, MFLOAT** const delX);
  Cartesian(const Index& I, const Index& J, const Index& K,
            MFLOAT** const delX);
  // These further take a Vektor<MFLOAT,Dim>& specifying the origin:
  Cartesian(const NDIndex<Dim>& ndi, MFLOAT** const delX,
            const Vektor<MFLOAT,Dim>& orig);
  Cartesian(const Index& I, MFLOAT** const delX,
            const Vektor<MFLOAT,Dim>& orig);
  Cartesian(const Index& I, const Index& J, MFLOAT** const delX,
            const Vektor<MFLOAT,Dim>& orig);
  Cartesian(const Index& I, const Index& J, const Index& K,
            MFLOAT** const delX, const Vektor<MFLOAT,Dim>& orig);
  // These further take a MeshBC_E array specifying mesh boundary conditions.
  Cartesian(const NDIndex<Dim>& ndi, MFLOAT** const delX,
            const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc);
  Cartesian(const Index& I, MFLOAT** const delX,
            const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc);
  Cartesian(const Index& I, const Index& J, MFLOAT** const delX,
            const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc);
  Cartesian(const Index& I, const Index& J, const Index& K,
            MFLOAT** const delX, const Vektor<MFLOAT,Dim>& orig,
            MeshBC_E* const mbc);

  // initialize functions
  void initialize(const NDIndex<Dim>& ndi);
  void initialize(const Index& I);
  void initialize(const Index& I, const Index& J);
  void initialize(const Index& I, const Index& J, const Index& K);
  // These also take a MFLOAT** specifying the mesh spacings:
  void initialize(const NDIndex<Dim>& ndi, MFLOAT** const delX);
  void initialize(const Index& I, MFLOAT** const delX);
  void initialize(const Index& I, const Index& J, MFLOAT** const delX);
  void initialize(const Index& I, const Index& J, const Index& K,
                  MFLOAT** const delX);
  // These further take a Vektor<MFLOAT,Dim>& specifying the origin:
  void initialize(const NDIndex<Dim>& ndi, MFLOAT** const delX,
                  const Vektor<MFLOAT,Dim>& orig);
  void initialize(const Index& I, MFLOAT** const delX,
                  const Vektor<MFLOAT,Dim>& orig);
  void initialize(const Index& I, const Index& J, MFLOAT** const delX,
                  const Vektor<MFLOAT,Dim>& orig);
  void initialize(const Index& I, const Index& J, const Index& K,
                  MFLOAT** const delX, const Vektor<MFLOAT,Dim>& orig);
  // These further take a MeshBC_E array specifying mesh boundary conditions.
  void initialize(const NDIndex<Dim>& ndi, MFLOAT** const delX,
                  const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc);
  void initialize(const Index& I, MFLOAT** const delX,
                  const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc);
  void initialize(const Index& I, const Index& J, MFLOAT** const delX,
                  const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc);
  void initialize(const Index& I, const Index& J, const Index& K,
                  MFLOAT** const delX, const Vektor<MFLOAT,Dim>& orig,
                  MeshBC_E* const mbc);

private:
  // Private member data:
  // Vert-vert spacings along each axis - including guard cells; use STL map:
  std::map<int,MFLOAT> meshSpacing[Dim];
  // Vertex positions along each axis - including guard cells; use STL map:
  std::map<int,MFLOAT> meshPosition[Dim];
  Vektor<MFLOAT,Dim> origin; // Origin of mesh coordinates (vertices)
  MeshBC_E MeshBC[2*Dim];    // Mesh boundary conditions
  FieldLayout<Dim>* FlCell;  // Layouts for BareField* CellSpacings
  FieldLayout<Dim>* FlVert;  // Layouts for BareField* VertSpacings

  // Private member functions:
  void updateMeshSpacingGuards(int face);// Update guard layers in meshSpacings
  void setup(); // Private function doing tasks common to all constructors.

  // Set only the derivative constants, using pre-set spacings:
  void set_Dvc();


public: 

  // Public member data:
  unsigned gridSizes[Dim];            // Sizes (number of vertices)
  Vektor<MFLOAT,Dim> Dvc[1<<Dim];     // Constants for derivatives.
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
  void get_meshSpacing(unsigned d, MFLOAT* spacings) const;
  //leak  MFLOAT* get_meshSpacing(unsigned d) const;
  // Get mesh boundary conditions:
  MeshBC_E get_MeshBC(unsigned face) const; // One face at a time
  MeshBC_E* get_MeshBC() const;             // All faces at once

  // Set functions for member data:
  // Set the origin of mesh vertex positions:
  void set_origin(const Vektor<MFLOAT,Dim>& o); 
  // Set the spacings of mesh vertex positions and Dvc:
  void set_meshSpacing(MFLOAT** const del);
  // Set up mesh boundary conditions:
  // Face specifies the mesh face, following usual numbering convention.
  // MeshBC_E "type" specifies the kind of BC reflective/periodic/none.
  void set_MeshBC(unsigned face, MeshBC_E meshBCType); // One face at a time
  void set_MeshBC(MeshBC_E* meshBCTypes);              // All faces at once

  // Formatted output of Cartesian object:
  void print(std::ostream&);

  //----Other Cartesian methods:-----------------------------------------------
  // Volume of single cell indexed by input NDIndex;
  MFLOAT getCellVolume(const NDIndex<Dim>&) const;
  // Field of volumes of all cells:
  Field<MFLOAT,Dim,Cartesian<Dim,MFLOAT>,Cell>&
  getCellVolumeField(Field<MFLOAT,Dim,Cartesian<Dim,MFLOAT>,Cell>&) const;
  // Volume of range of cells bounded by verticies specified by input NDIndex:
  MFLOAT getVertRangeVolume(const NDIndex<Dim>&) const;
  // Volume of range of cells spanned by input NDIndex (index of cells):
  MFLOAT getCellRangeVolume(const NDIndex<Dim>&) const;
  // Nearest vertex index to (x,y,z):
  NDIndex<Dim> getNearestVertex(const Vektor<MFLOAT,Dim>&) const;
  // Nearest vertex index with all vertex coordinates below (x,y,z):
  NDIndex<Dim> getVertexBelow(const Vektor<MFLOAT,Dim>&) const;
  // NDIndex for cell in cell-ctrd Field containing the point (x,y,z):
  NDIndex<Dim> getCellContaining(const Vektor<MFLOAT,Dim>& x) const
  {
    return getVertexBelow(x); // I think these functions are identical. -tjw
  }
  // (x,y,z) coordinates of indexed vertex:
  Vektor<MFLOAT,Dim> getVertexPosition(const NDIndex<Dim>&) const;
  // Field of (x,y,z) coordinates of all vertices:
  Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Vert>& 
  getVertexPositionField(Field<Vektor<MFLOAT,Dim>,Dim,
			 Cartesian<Dim,MFLOAT>,Vert>& ) const;
  // (x,y,z) coordinates of indexed cell:
  Vektor<MFLOAT,Dim> getCellPosition(const NDIndex<Dim>&) const;
  // Field of (x,y,z) coordinates of all cells:
  Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Cell>& 
  getCellPositionField(Field<Vektor<MFLOAT,Dim>,Dim,
		       Cartesian<Dim,MFLOAT>,Cell>& ) const;
  // Vertex-vertex grid spacing of indexed cell:
  Vektor<MFLOAT,Dim> getDeltaVertex(const NDIndex<Dim>&) const;
  // Field of vertex-vertex grid spacings of all cells:
  Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Cell>&
  getDeltaVertexField(Field<Vektor<MFLOAT,Dim>,Dim,
		      Cartesian<Dim,MFLOAT>,Cell>& ) const;
  // Cell-cell grid spacing of indexed vertex:
  Vektor<MFLOAT,Dim> getDeltaCell(const NDIndex<Dim>&) const;
  // Field of cell-cell grid spacings of all vertices:
  Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Vert>&
  getDeltaCellField(Field<Vektor<MFLOAT,Dim>,Dim,
		    Cartesian<Dim,MFLOAT>,Vert>& ) const;
  // Array of surface normals to cells adjoining indexed cell:
  Vektor<MFLOAT,Dim>* getSurfaceNormals(const NDIndex<Dim>&) const;
  // Array of (pointers to) Fields of surface normals to all cells:
  void getSurfaceNormalFields(Field<Vektor<MFLOAT,Dim>,Dim,
			      Cartesian<Dim,MFLOAT>,Cell>** ) const;
  // Similar functions, but specify the surface normal to a single face, using
  // the following numbering convention: 0 means low face of 1st dim, 1 means
  // high face of 1st dim, 2 means low face of 2nd dim, 3 means high face of 
  // 2nd dim, and so on:
  Vektor<MFLOAT,Dim> getSurfaceNormal(const NDIndex<Dim>&, unsigned) const;
  Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Cell>& 
  getSurfaceNormalField(Field<Vektor<MFLOAT,Dim>,Dim,
			Cartesian<Dim,MFLOAT>,Cell>&, unsigned) const;

};

// I/O

// Stream formatted output of Cartesian object:
template< unsigned Dim, class MFLOAT >
inline
std::ostream& operator<<(std::ostream& out, const Cartesian<Dim,MFLOAT>& mesh)
{
  Cartesian<Dim,MFLOAT>& ncmesh = const_cast<Cartesian<Dim,MFLOAT>&>(mesh);
  ncmesh.print(out);
  return out;
}

//*****************************************************************************
// Stuff taken from old Cartesian.h, modified for new nonuniform Cartesian:
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
Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& 
Div(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
    Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& 
Div(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
    Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& 
Div(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
    Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Divergence Vektor/Cell -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& 
Div(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
    Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& 
Div(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
    Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& 
Div(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
    Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& r);

//----------------------------------------------------------------------
// Divergence Vektor/Vert -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& 
Div(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
    Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& 
Div(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
    Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& 
Div(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
    Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Divergence Vektor/Cell -> Scalar/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& 
Div(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
    Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& 
Div(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
    Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& 
Div(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
    Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Divergence Tenzor/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& 
Div(Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
    Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& 
Div(Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
    Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& 
Div(Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
    Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Divergence SymTenzor/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& 
Div(Field<SymTenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
    Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& 
Div(Field<SymTenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
    Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& 
Div(Field<SymTenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
    Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Divergence Tenzor/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& 
Div(Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
    Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& 
Div(Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
    Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& 
Div(Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
    Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Divergence SymTenzor/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& 
Div(Field<SymTenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
    Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& 
Div(Field<SymTenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
    Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& 
Div(Field<SymTenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
    Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r);


//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& 
Grad(Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
     Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& 
Grad(Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
     Field<Vektor<T,2u>,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& 
Grad(Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
     Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Grad Scalar/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& 
Grad(Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
     Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& 
Grad(Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
     Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& 
Grad(Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
     Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r);

//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& 
Grad(Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
     Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& 
Grad(Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
     Field<Vektor<T,2u>,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& 
Grad(Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
     Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
// Grad Scalar/Cell -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& 
Grad(Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
     Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& 
Grad(Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
     Field<Vektor<T,2u>,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& 
Grad(Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
     Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Grad Vektor/Vert -> Tenzor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& 
Grad(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
     Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& 
Grad(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
     Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& 
Grad(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
     Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
// Grad Vektor/Cell -> Tenzor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& 
Grad(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
     Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& 
Grad(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
     Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& 
Grad(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
     Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r);

namespace IPPL {
//----------------------------------------------------------------------
// Weighted average Cell to Vert
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& 
Average(Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
	Field<T2,1U,Cartesian<1U,MFLOAT>,Cell>& w, 
	Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& r) ;
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& 
Average(Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
	Field<T2,2U,Cartesian<2U,MFLOAT>,Cell>& w, 
	Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& 
Average(Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
	Field<T2,3U,Cartesian<3U,MFLOAT>,Cell>& w, 
	Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& r);

//----------------------------------------------------------------------
// Weighted average Vert to Cell
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& 
Average(Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
	Field<T2,1U,Cartesian<1U,MFLOAT>,Vert>& w, 
	Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& r) ;
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& 
Average(Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
	Field<T2,2U,Cartesian<2U,MFLOAT>,Vert>& w, 
	Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& 
Average(Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
	Field<T2,3U,Cartesian<3U,MFLOAT>,Vert>& w, 
	Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& r);

//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Unweighted average Cell to Vert
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& 
Average(Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& x, 
	Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& r) ;
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& 
Average(Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& x, 
	Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& r);
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& 
Average(Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& x, 
	Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& r);

//----------------------------------------------------------------------
// Unweighted average Vert to Cell
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& 
Average(Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& x, 
	Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& r) ;
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& 
Average(Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& x, 
	Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& r);
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& 
Average(Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& x, 
	Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& r);

//----------------------------------------------------------------------
}
#include "Meshes/Cartesian.hpp"

#endif // CARTESIAN_H

/***************************************************************************
 * $RCSfile: Cartesian.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: Cartesian.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
