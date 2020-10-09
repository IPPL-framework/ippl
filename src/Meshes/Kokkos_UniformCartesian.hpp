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
#include "Field/LField.h"
#include "Field/Kokkos_Field.h"

namespace ippl {

    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::UniformCartesian()
        : Mesh<T, Dim>()
        , hasSpacingFields_m(false)
    { }

    template<typename T, unsigned Dim>
    UniformCartesian<T, Dim>::~UniformCartesian() {
        if (hasSpacingFields_m) {
            delete VertSpacings;
            delete CellSpacings;
            delete FlVert;
            delete FlCell;
        }
    };



    template<typename T, unsigned Dim>
    T UniformCartesian<T, Dim>::getMeshSpacing(unsigned dim) const {
        PAssert_LT(dim, Dim);
        return meshSpacing_m[dim];
    }


    template<typename T, unsigned Dim>
    void UniformCartesian<T, Dim>::setMeshSpacing(const MeshVector_t& meshSpacing) {
        meshSpacing_m = meshSpacing;

        // update cell volume
        volume_m = 1.0;
        for (unsigned i = 0; i < Dim; ++i) {
            volume_m *= meshSpacing_m[i];
        }
    }


    template<typename T, unsigned Dim>
    T UniformCartesian<T, Dim>::getCellVolume() const {
        return volume_m;
    }




//-----------------------------------------------------------------------------
// Setup chores common to all constructors:
//-----------------------------------------------------------------------------
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
setup()
{
  hasSpacingFields_m = false;
  FlCell = 0;
  FlVert = 0;
  VertSpacings = 0;
  CellSpacings = 0;
  volume_m = 0.0;
}

//-----------------------------------------------------------------------------
// Constructors from NDIndex object:
//-----------------------------------------------------------------------------
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const NDIndex<Dim>& ndi)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    this->gridSizes_m[d] = ndi[d].length(); // Number of vertices along this dimension.
    meshSpacing_m[d] = ndi[d].stride();  // Default mesh spacing from stride()
    this->origin_m(d) = ndi[d].first();     // Default this->origin_m at ndi[d].first
  }
  volume_m = 1.0;               // Default mesh has unit cell volume_m.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const NDIndex<Dim>& ndi, T* const delX)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    this->gridSizes_m[d] = ndi[d].length(); // Number of vertices along this dimension.
    this->origin_m(d) = ndi[d].first();     // Default this->origin_m at ndi[d].first
  }
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
// Also specify mesh spacings and this->origin_m:
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const NDIndex<Dim>& ndi, T* const delX,
                 const Vector<T,Dim>& orig)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    this->gridSizes_m[d] = ndi[d].length(); // Number of vertices along this dimension.
  }
  this->setOrigin(orig);           // Set this->origin_m.
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
//-----------------------------------------------------------------------------
// Constructors from Index objects:
//-----------------------------------------------------------------------------

//===========1D============
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  meshSpacing_m[0] = I.stride();       // Default mesh spacing from stride()
  this->origin_m(0) = I.first();      // Default this->origin_m at I.first()

  volume_m = 1.0;               // Default mesh has unit cell volume_m.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I, T* const delX)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->origin_m(0) = I.first();      // Default this->origin_m at I.first()

  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
// Also specify mesh spacings and this->origin_m:
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I, T* const delX,
                 const Vector<T,Dim>& orig)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->setOrigin(orig);           // Set this->origin_m.
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}

//===========2D============
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I, const Index& J)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  meshSpacing_m[0] = I.stride();       // Default mesh spacing from stride()
  meshSpacing_m[1] = J.stride();
  this->origin_m(0) = I.first();      // Default this->origin_m at (I.first(),J.first())
  this->origin_m(1) = J.first();

  volume_m = 1.0;               // Default mesh has unit cell volume_m.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I, const Index& J, T* const delX)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->origin_m(0) = I.first();      // Default this->origin_m at (I.first(),J.first())
  this->origin_m(1) = J.first();
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
// Also specify mesh spacings and this->origin_m:
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I, const Index& J, T* const delX,
                 const Vector<T,Dim>& orig)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->setOrigin(orig);           // Set this->origin_m.
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}

//===========3D============
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I, const Index& J, const Index& K)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->gridSizes_m[2] = K.length();  // Number of vertices along this dimension.
  meshSpacing_m[0] = I.stride();       // Default mesh spacing from stride()
  meshSpacing_m[1] = J.stride();
  meshSpacing_m[2] = K.stride();
  this->origin_m(0) = I.first();   // Default this->origin_m at (I.first(),J.first(),K.first())
  this->origin_m(1) = J.first();
  this->origin_m(2) = K.first();

  volume_m = 1.0;               // Default mesh has unit cell volume_m.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I, const Index& J, const Index& K,
                 T* const delX)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->gridSizes_m[2] = K.length();  // Number of vertices along this dimension.
  this->origin_m(0) = I.first();   // Default this->origin_m at (I.first(),J.first(),K.first())
  this->origin_m(1) = J.first();
  this->origin_m(2) = K.first();
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
// Also specify mesh spacings and this->origin_m:
template<typename T, unsigned Dim>
UniformCartesian<T, Dim>::
UniformCartesian(const Index& I, const Index& J, const Index& K,
                 T* const delX, const Vector<T,Dim>& orig)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->gridSizes_m[2] = K.length();  // Number of vertices along this dimension.
  this->setOrigin(orig);           // Set this->origin_m.
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}

//-----------------------------------------------------------------------------
// initialize with NDIndex object:
//-----------------------------------------------------------------------------
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const NDIndex<Dim>& ndi)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    this->gridSizes_m[d] = ndi[d].length(); // Number of vertices along this dimension.
    meshSpacing_m[d] = ndi[d].stride();  // Default mesh spacing from stride()
    this->origin_m(d) = ndi[d].first();     // Default this->origin_m at ndi[d].first
  }
  volume_m = 1.0;               // Default mesh has unit cell volume_m.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const NDIndex<Dim>& ndi, T* const delX)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    this->gridSizes_m[d] = ndi[d].length(); // Number of vertices along this dimension.
    this->origin_m(d) = ndi[d].first();     // Default this->origin_m at ndi[d].first
  }
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
// Also specify mesh spacings and this->origin_m:
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const NDIndex<Dim>& ndi, T* const delX,
           const Vector<T,Dim>& orig)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    this->gridSizes_m[d] = ndi[d].length(); // Number of vertices along this dimension.
  }
  this->setOrigin(orig);           // Set this->origin_m.
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
//-----------------------------------------------------------------------------
// initialize from Index objects:
//-----------------------------------------------------------------------------

//===========1D============
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  meshSpacing_m[0] = I.stride();       // Default mesh spacing from stride()
  this->origin_m(0) = I.first();      // Default this->origin_m at I.first()

  volume_m = 1.0;               // Default mesh has unit cell volume_m.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I, T* const delX)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->origin_m(0) = I.first();      // Default this->origin_m at I.first()

  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
// Also specify mesh spacings and this->origin_m:
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I, T* const delX,
           const Vector<T,Dim>& orig)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->setOrigin(orig);           // Set this->origin_m.
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}

//===========2D============
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I, const Index& J)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  meshSpacing_m[0] = I.stride();       // Default mesh spacing from stride()
  meshSpacing_m[1] = J.stride();
  this->origin_m(0) = I.first();      // Default this->origin_m at (I.first(),J.first())
  this->origin_m(1) = J.first();

  volume_m = 1.0;               // Default mesh has unit cell volume_m.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I, const Index& J, T* const delX)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->origin_m(0) = I.first();      // Default this->origin_m at (I.first(),J.first())
  this->origin_m(1) = J.first();
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
// Also specify mesh spacings and this->origin_m:
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I, const Index& J, T* const delX,
           const Vector<T,Dim>& orig)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->setOrigin(orig);           // Set this->origin_m.
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}

//===========3D============
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I, const Index& J, const Index& K)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->gridSizes_m[2] = K.length();  // Number of vertices along this dimension.
  meshSpacing_m[0] = I.stride();       // Default mesh spacing from stride()
  meshSpacing_m[1] = J.stride();
  meshSpacing_m[2] = K.stride();
  this->origin_m(0) = I.first();   // Default this->origin_m at (I.first(),J.first(),K.first())
  this->origin_m(1) = J.first();
  this->origin_m(2) = K.first();

  volume_m = 1.0;               // Default mesh has unit cell volume_m.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I, const Index& J, const Index& K,
           T* const delX)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->gridSizes_m[2] = K.length();  // Number of vertices along this dimension.
  this->origin_m(0) = I.first();   // Default this->origin_m at (I.first(),J.first(),K.first())
  this->origin_m(1) = J.first();
  this->origin_m(2) = K.first();
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}
// Also specify mesh spacings and this->origin_m:
template<typename T, unsigned Dim>
void
UniformCartesian<T, Dim>::
initialize(const Index& I, const Index& J, const Index& K,
           T* const delX, const Vector<T,Dim>& orig)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  this->gridSizes_m[0] = I.length();  // Number of vertices along this dimension.
  this->gridSizes_m[1] = J.length();  // Number of vertices along this dimension.
  this->gridSizes_m[2] = K.length();  // Number of vertices along this dimension.
  this->setOrigin(orig);           // Set this->origin_m.
  this->setMeshSpacing(delX);      // Set mesh spacings and compute cell volume_m
}

//-----------------------------------------------------------------------------
// Set/accessor functions for member data:
//-----------------------------------------------------------------------------
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::set_Dvc()
{
  unsigned d;
  T coef = 1.0;
  for (d=1;d<Dim;++d) coef *= 0.5;

  for (d=0;d<Dim;++d) {
    T dvc = coef/meshSpacing_m[d];
    for (unsigned b=0; b<(1u<<Dim); ++b) {
      int s = ( b&(1<<d) ) ? 1 : -1;
      Dvc[b][d] = s*dvc;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

// Applicative templates for Mesh BC PETE_apply() functions, used
// by BrickExpression in storeSpacingFields()

// Reflective/None (all are the same for uniform, incl. Periodic).
template<class T>
struct OpUMeshExtrapolate
{
  OpUMeshExtrapolate(T& o, T& s) : Offset(o), Slope(s) {}
  T Offset, Slope;
};

template<class T>
inline void PETE_apply(OpUMeshExtrapolate<T> e, T& a, T b)
{
  a = b*e.Slope+e.Offset;
}

///////////////////////////////////////////////////////////////////////////////

// Create BareField's of vertex and cell spacings
// Special prototypes taking no args or FieldLayout ctor args:
// No-arg case:
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields()
{
  // Set up default FieldLayout parameters:
  e_dim_tag et[Dim];
  for (unsigned int d=0; d<Dim; d++) et[d] = PARALLEL;
  storeSpacingFields(et, -1);
}
// 1D
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields(e_dim_tag p1, int vnodes)
{
  e_dim_tag et[1];
  et[0] = p1;
  storeSpacingFields(et, vnodes);
}
// 2D
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields(e_dim_tag p1, e_dim_tag p2, int vnodes)
{
  e_dim_tag et[2];
  et[0] = p1;
  et[1] = p2;
  storeSpacingFields(et, vnodes);
}
// 3D
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields(e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, int vnodes)
{
  e_dim_tag et[3];
  et[0] = p1;
  et[1] = p2;
  et[2] = p3;
  storeSpacingFields(et, vnodes);
}
// The general storeSpacingfields() function; others invoke this internally:
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields(e_dim_tag* /*et*/, int /*vnodes*/)
{
    /*
  // VERTEX-VERTEX SPACINGS (same as CELL-CELL SPACINGS for uniform):
  NDIndex<Dim> cells, verts;
  unsigned int d;
  for (d=0; d<Dim; d++) {
    cells[d] = Index(this->gridSizes_m[d]-1);
    verts[d] = Index(this->gridSizes_m[d]);
  }
  if (!hasSpacingFields_m) {
    // allocate layout and spacing field
    FlCell = new FieldLayout<Dim>(cells, et, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    // (not really used by Div() etc for UniformCartesian); someday should make
    // this user-settable.
    VertSpacings =
      new BareField<Vector<T,Dim>,Dim>(*FlCell,GuardCellSizes<Dim>(1));
    // Added 12/8/98 --TJW:
    FlVert =
      new FieldLayout<Dim>(verts, et, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    CellSpacings =
      new BareField<Vector<T,Dim>,Dim>(*FlVert,GuardCellSizes<Dim>(1));
  }
  BareField<Vector<T,Dim>,Dim>& vertSpacings = *VertSpacings;
  Vector<T,Dim> vertexSpacing;
  for (d=0; d<Dim; d++)
    vertexSpacing[d] = meshSpacing_m[d];
  vertSpacings = vertexSpacing;
  //-------------------------------------------------
  // Now the hard part, filling in the guard cells:
  //-------------------------------------------------
  // The easy part of the hard part is filling so that all the internal
  // guard layers are right:
  vertSpacings.fillGuardCells();
  // The hard part of the hard part is filling the external guard layers,
  // using the mesh BC to figure out how:
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // Temporaries used in loop over faces
  Vector<T,Dim> v0,v1; v0 = 0.0; v1 = 1.0; // Used for Reflective mesh BC
  unsigned int face;
  typedef Vector<T,Dim> T;          // Used multipple places in loop below
  typename BareField<T,Dim>::iterator_if vfill_i; // Iterator used below
  int voffset;             // Pointer offsets used with LField::iterator below
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  for (face=0; face < 2*Dim; face++) {
    // NDIndex's spanning elements and guard elements:
    NDIndex<Dim> vSlab = AddGuardCells(cells,vertSpacings.getGuardCellSizes());
    // Shrink it down to be the guards along the active face:
    d = face/2;
    // The following bitwise AND logical test returns true if face is odd
    // (meaning the "high" or "right" face in the numbering convention) and
    // returns false if face is even (meaning the "low" or "left" face in
    // the numbering convention):
    if ( face & 1u ) {
      vSlab[d] = Index(cells[d].max() + 1,
		       cells[d].max() + vertSpacings.rightGuard(d));
    } else {
      vSlab[d] = Index(cells[d].min() - vertSpacings.leftGuard(d),
		       cells[d].min() - 1);
    }
    // Compute pointer offsets used with LField::iterator below:
    // Treat all as Reflective BC (see Cartesian for comparison); for
    // uniform cartesian mesh, all mesh BC's equivalent for this purpose:
    if ( face & 1u ) {
      voffset = 2*cells[d].max() + 1 - 1;
    } else {
      voffset = 2*cells[d].min() - 1 + 1;
    }

    // +++++++++++++++vertSpacings++++++++++++++
    for (vfill_i=vertSpacings.begin_if();
	 vfill_i!=vertSpacings.end_if(); ++vfill_i)
      {
	// Cache some things we will use often below.
	// Pointer to the data for the current LField (right????):
	LField<T,Dim> &fill = *(*vfill_i).second;
	// NDIndex spanning all elements in the LField, including the guards:
	const NDIndex<Dim> &fill_alloc = fill.getAllocated();
	// If the previously-created boundary guard-layer NDIndex "cSlab"
	// contains any of the elements in this LField (they will be guard
	// elements if it does), assign the values into them here by applying
	// the boundary condition:
	if ( vSlab.touches( fill_alloc ) )
	  {
	    // Find what it touches in this LField.
	    NDIndex<Dim> dest = vSlab.intersect( fill_alloc );

	    // For exrapolation boundary conditions, the boundary guard-layer
	    // elements are typically copied from interior values; the "src"
	    // NDIndex specifies the interior elements to be copied into the
	    // "dest" boundary guard-layer elements (possibly after some
	    // mathematical operations like multipplying by minus 1 later):
	    NDIndex<Dim> src = dest; // Create dest equal to src
	    // Now calculate the interior elements; the voffset variable
	    // computed above makes this right for "low" or "high" face cases:
	    src[d] = voffset - src[d];

	    // TJW: Why is there another loop over LField's here??????????
	    // Loop over the ones that src touches.
	    typename BareField<T,Dim>::iterator_if from_i;
	    for (from_i=vertSpacings.begin_if();
		 from_i!=vertSpacings.end_if(); ++from_i)
	      {
		// Cache a few things.
		LField<T,Dim> &from = *(*from_i).second;
		const NDIndex<Dim> &from_owned = from.getOwned();
		const NDIndex<Dim> &from_alloc = from.getAllocated();
		// If src touches this LField...
		if ( src.touches( from_owned ) )
		  {
		    NDIndex<Dim> from_it = src.intersect( from_alloc );
		    NDIndex<Dim> vfill_it = dest.plugBase( from_it );
		    // Build iterators for the copy...
		    typedef typename LField<T,Dim>::iterator LFI;
		    LFI lhs = fill.begin(vfill_it);
		    LFI rhs = from.begin(from_it);
		    // And do the assignment (reflective BC hardwired):
		    BrickExpression<Dim,LFI,LFI,OpUMeshExtrapolate<T> >
		      (lhs,rhs,OpUMeshExtrapolate<T>(v0,v1)).apply();
		  }
	      }
	  }
      }

  }

  // For uniform cartesian mesh, cell-cell spacings are identical to
  // vert-vert spacings:
  //12/8/98  CellSpacings = VertSpacings;
  // Added 12/8/98 --TJW:
  BareField<Vector<T,Dim>,Dim>& cellSpacings = *CellSpacings;
  cellSpacings = vertexSpacing;

  hasSpacingFields_m = true; // Flag this as having been done to this object.
  */
}

// These specify both the total number of vnodes and the numbers of vnodes
// along each dimension for the partitioning of the index space. Obviously
// this restricts the number of vnodes to be a product of the numbers along
// each dimension (the constructor implementation checks this): Special
// cases for 1-3 dimensions, ala FieldLayout ctors (see FieldLayout.h for
// more relevant comments, including definition of recurse):

// 1D
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields(e_dim_tag p1,
			unsigned vnodes1,
			bool recurse,
			int vnodes) {
  e_dim_tag et[1];
  et[0] = p1;
  unsigned vnodesPerDirection[Dim];
  vnodesPerDirection[0] = vnodes1;
  storeSpacingFields(et, vnodesPerDirection, recurse, vnodes);
}
// 2D
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields(e_dim_tag p1, e_dim_tag p2,
			unsigned vnodes1, unsigned vnodes2,
			bool recurse,int vnodes) {
  e_dim_tag et[2];
  et[0] = p1;
  et[1] = p2;
  unsigned vnodesPerDirection[Dim];
  vnodesPerDirection[0] = vnodes1;
  vnodesPerDirection[1] = vnodes2;
  storeSpacingFields(et, vnodesPerDirection, recurse, vnodes);
}
// 3D
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields(e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
			bool recurse, int vnodes) {
  e_dim_tag et[3];
  et[0] = p1;
  et[1] = p2;
  et[2] = p3;
  unsigned vnodesPerDirection[Dim];
  vnodesPerDirection[0] = vnodes1;
  vnodesPerDirection[1] = vnodes2;
  vnodesPerDirection[2] = vnodes3;
  storeSpacingFields(et, vnodesPerDirection, recurse, vnodes);
}

// TJW: Note: should clean up here eventually, and put redundant code from
// this and the other general storeSpacingFields() implementation into one
// function. Need to check this in quickly for Blanca right now --12/8/98
// The general storeSpacingfields() function; others invoke this internally:
template<typename T, unsigned Dim>
void UniformCartesian<T, Dim>::
storeSpacingFields(e_dim_tag */*p*/,
		   unsigned* /*vnodesPerDirection*/,
		   bool /*recurse*/, int /*vnodes*/)
{
    /*
  // VERTEX-VERTEX SPACINGS (same as CELL-CELL SPACINGS for uniform):
  NDIndex<Dim> cells, verts;
  unsigned int d;
  for (d=0; d<Dim; d++) {
    cells[d] = Index(this->gridSizes_m[d]-1);
    verts[d] = Index(this->gridSizes_m[d]);
  }
  if (!hasSpacingFields_m) {
    // allocate layout and spacing field
    FlCell =
      new FieldLayout<Dim>(cells, p, vnodesPerDirection, recurse, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    // (not really used by Div() etc for UniformCartesian); someday should make
    // this user-settable.
    VertSpacings =
      new BareField<Vector<T,Dim>,Dim>(*FlCell,GuardCellSizes<Dim>(1));
    // Added 12/8/98 --TJW:
    FlVert =
      new FieldLayout<Dim>(verts, p, vnodesPerDirection, recurse, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    CellSpacings =
      new BareField<Vector<T,Dim>,Dim>(*FlVert,GuardCellSizes<Dim>(1));
  }
  BareField<Vector<T,Dim>,Dim>& vertSpacings = *VertSpacings;
  Vector<T,Dim> vertexSpacing;
  for (d=0; d<Dim; d++)
    vertexSpacing[d] = meshSpacing_m[d];
  vertSpacings = vertexSpacing;
  //-------------------------------------------------
  // Now the hard part, filling in the guard cells:
  //-------------------------------------------------
  // The easy part of the hard part is filling so that all the internal
  // guard layers are right:
  vertSpacings.fillGuardCells();
  // The hard part of the hard part is filling the external guard layers,
  // using the mesh BC to figure out how:
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // Temporaries used in loop over faces
  Vector<T,Dim> v0,v1; v0 = 0.0; v1 = 1.0; // Used for Reflective mesh BC
  unsigned int face;
  typedef Vector<T,Dim> T;          // Used multipple places in loop below
  typename BareField<T,Dim>::iterator_if vfill_i; // Iterator used below
  int voffset;             // Pointer offsets used with LField::iterator below
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  for (face=0; face < 2*Dim; face++) {
    // NDIndex's spanning elements and guard elements:
    NDIndex<Dim> vSlab = AddGuardCells(cells,vertSpacings.getGuardCellSizes());
    // Shrink it down to be the guards along the active face:
    d = face/2;
    // The following bitwise AND logical test returns true if face is odd
    // (meaning the "high" or "right" face in the numbering convention) and
    // returns false if face is even (meaning the "low" or "left" face in
    // the numbering convention):
    if ( face & 1u ) {
      vSlab[d] = Index(cells[d].max() + 1,
		       cells[d].max() + vertSpacings.rightGuard(d));
    } else {
      vSlab[d] = Index(cells[d].min() - vertSpacings.leftGuard(d),
		       cells[d].min() - 1);
    }
    // Compute pointer offsets used with LField::iterator below:
    // Treat all as Reflective BC (see Cartesian for comparison); for
    // uniform cartesian mesh, all mesh BC's equivalent for this purpose:
    if ( face & 1u ) {
      voffset = 2*cells[d].max() + 1 - 1;
    } else {
      voffset = 2*cells[d].min() - 1 + 1;
    }

    // +++++++++++++++vertSpacings++++++++++++++
    for (vfill_i=vertSpacings.begin_if();
	 vfill_i!=vertSpacings.end_if(); ++vfill_i)
      {
	// Cache some things we will use often below.
	// Pointer to the data for the current LField (right????):
	LField<T,Dim> &fill = *(*vfill_i).second;
	// NDIndex spanning all elements in the LField, including the guards:
	const NDIndex<Dim> &fill_alloc = fill.getAllocated();
	// If the previously-created boundary guard-layer NDIndex "cSlab"
	// contains any of the elements in this LField (they will be guard
	// elements if it does), assign the values into them here by applying
	// the boundary condition:
	if ( vSlab.touches( fill_alloc ) )
	  {
	    // Find what it touches in this LField.
	    NDIndex<Dim> dest = vSlab.intersect( fill_alloc );

	    // For exrapolation boundary conditions, the boundary guard-layer
	    // elements are typically copied from interior values; the "src"
	    // NDIndex specifies the interior elements to be copied into the
	    // "dest" boundary guard-layer elements (possibly after some
	    // mathematical operations like multipplying by minus 1 later):
	    NDIndex<Dim> src = dest; // Create dest equal to src
	    // Now calculate the interior elements; the voffset variable
	    // computed above makes this right for "low" or "high" face cases:
	    src[d] = voffset - src[d];

	    // TJW: Why is there another loop over LField's here??????????
	    // Loop over the ones that src touches.
	    typename BareField<T,Dim>::iterator_if from_i;
	    for (from_i=vertSpacings.begin_if();
		 from_i!=vertSpacings.end_if(); ++from_i)
	      {
		// Cache a few things.
		LField<T,Dim> &from = *(*from_i).second;
		const NDIndex<Dim> &from_owned = from.getOwned();
		const NDIndex<Dim> &from_alloc = from.getAllocated();
		// If src touches this LField...
		if ( src.touches( from_owned ) )
		  {
		    NDIndex<Dim> from_it = src.intersect( from_alloc );
		    NDIndex<Dim> vfill_it = dest.plugBase( from_it );
		    // Build iterators for the copy...
		    typedef typename LField<T,Dim>::iterator LFI;
		    LFI lhs = fill.begin(vfill_it);
		    LFI rhs = from.begin(from_it);
		    // And do the assignment (reflective BC hardwired):
		    BrickExpression<Dim,LFI,LFI,OpUMeshExtrapolate<T> >
		      (lhs,rhs,OpUMeshExtrapolate<T>(v0,v1)).apply();
		  }
	      }
	  }
      }

  }

  // For uniform cartesian mesh, cell-cell spacings are identical to
  // vert-vert spacings:
  //12/8/98  CellSpacings = VertSpacings;
  // Added 12/8/98 --TJW:
  BareField<Vector<T,Dim>,Dim>& cellSpacings = *CellSpacings;
  cellSpacings = vertexSpacing;

  hasSpacingFields_m = true; // Flag this as having been done to this object.
  */
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
