// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

// UniformCartesian.cpp
// Implementations for UniformCartesian mesh class (uniform spacings)

// include files
#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
#include "Field/BareField.h"
// #include "Field/BrickExpression.h"
#include "Field/LField.h"
#include "Field/Kokkos_Field.h"
// #include "Field/Assign.h"
// #include "Field/AssignDefs.h"

namespace ippl {

//-----------------------------------------------------------------------------
// Setup chores common to all constructors:
//-----------------------------------------------------------------------------
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
setup()
{
  hasSpacingFields = false;
  FlCell = 0;
  FlVert = 0;
  VertSpacings = 0;
  CellSpacings = 0;
  volume = 0.0;
}

//-----------------------------------------------------------------------------
// Constructors from NDIndex object:
//-----------------------------------------------------------------------------
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const NDIndex<Dim>& ndi)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
    meshSpacing[d] = ndi[d].stride();  // Default mesh spacing from stride()
    origin(d) = ndi[d].first();     // Default origin at ndi[d].first
  }
  volume = 1.0;               // Default mesh has unit cell volume.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const NDIndex<Dim>& ndi, MFLOAT* const delX)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
    origin(d) = ndi[d].first();     // Default origin at ndi[d].first
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const NDIndex<Dim>& ndi, MFLOAT* const delX,
                 const Vector<MFLOAT,Dim>& orig)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
//-----------------------------------------------------------------------------
// Constructors from Index objects:
//-----------------------------------------------------------------------------

//===========1D============
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  meshSpacing[0] = I.stride();       // Default mesh spacing from stride()
  origin(0) = I.first();      // Default origin at I.first()

  volume = 1.0;               // Default mesh has unit cell volume.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I, MFLOAT* const delX)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  origin(0) = I.first();      // Default origin at I.first()

  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I, MFLOAT* const delX,
                 const Vector<MFLOAT,Dim>& orig)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}

//===========2D============
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I, const Index& J)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  meshSpacing[0] = I.stride();       // Default mesh spacing from stride()
  meshSpacing[1] = J.stride();
  origin(0) = I.first();      // Default origin at (I.first(),J.first())
  origin(1) = J.first();

  volume = 1.0;               // Default mesh has unit cell volume.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I, const Index& J, MFLOAT* const delX)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  origin(0) = I.first();      // Default origin at (I.first(),J.first())
  origin(1) = J.first();
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I, const Index& J, MFLOAT* const delX,
                 const Vector<MFLOAT,Dim>& orig)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}

//===========3D============
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I, const Index& J, const Index& K)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  meshSpacing[0] = I.stride();       // Default mesh spacing from stride()
  meshSpacing[1] = J.stride();
  meshSpacing[2] = K.stride();
  origin(0) = I.first();   // Default origin at (I.first(),J.first(),K.first())
  origin(1) = J.first();
  origin(2) = K.first();

  volume = 1.0;               // Default mesh has unit cell volume.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I, const Index& J, const Index& K,
                 MFLOAT* const delX)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  origin(0) = I.first();   // Default origin at (I.first(),J.first(),K.first())
  origin(1) = J.first();
  origin(2) = K.first();
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
UniformCartesian<Dim,MFLOAT>::
UniformCartesian(const Index& I, const Index& J, const Index& K,
                 MFLOAT* const delX, const Vector<MFLOAT,Dim>& orig)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}

//-----------------------------------------------------------------------------
// initialize with NDIndex object:
//-----------------------------------------------------------------------------
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const NDIndex<Dim>& ndi)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
    meshSpacing[d] = ndi[d].stride();  // Default mesh spacing from stride()
    origin(d) = ndi[d].first();     // Default origin at ndi[d].first
  }
  volume = 1.0;               // Default mesh has unit cell volume.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const NDIndex<Dim>& ndi, MFLOAT* const delX)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
    origin(d) = ndi[d].first();     // Default origin at ndi[d].first
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const NDIndex<Dim>& ndi, MFLOAT* const delX,
           const Vector<MFLOAT,Dim>& orig)
{
  setup();
  for (unsigned int d=0; d<Dim; d++) {
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
//-----------------------------------------------------------------------------
// initialize from Index objects:
//-----------------------------------------------------------------------------

//===========1D============
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  meshSpacing[0] = I.stride();       // Default mesh spacing from stride()
  origin(0) = I.first();      // Default origin at I.first()

  volume = 1.0;               // Default mesh has unit cell volume.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I, MFLOAT* const delX)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  origin(0) = I.first();      // Default origin at I.first()

  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I, MFLOAT* const delX,
           const Vector<MFLOAT,Dim>& orig)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}

//===========2D============
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  meshSpacing[0] = I.stride();       // Default mesh spacing from stride()
  meshSpacing[1] = J.stride();
  origin(0) = I.first();      // Default origin at (I.first(),J.first())
  origin(1) = J.first();

  volume = 1.0;               // Default mesh has unit cell volume.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, MFLOAT* const delX)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  origin(0) = I.first();      // Default origin at (I.first(),J.first())
  origin(1) = J.first();
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, MFLOAT* const delX,
           const Vector<MFLOAT,Dim>& orig)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}

//===========3D============
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, const Index& K)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  meshSpacing[0] = I.stride();       // Default mesh spacing from stride()
  meshSpacing[1] = J.stride();
  meshSpacing[2] = K.stride();
  origin(0) = I.first();   // Default origin at (I.first(),J.first(),K.first())
  origin(1) = J.first();
  origin(2) = K.first();

  volume = 1.0;               // Default mesh has unit cell volume.
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, const Index& K,
           MFLOAT* const delX)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  origin(0) = I.first();   // Default origin at (I.first(),J.first(),K.first())
  origin(1) = J.first();
  origin(2) = K.first();
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, const Index& K,
           MFLOAT* const delX, const Vector<MFLOAT,Dim>& orig)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
}

//-----------------------------------------------------------------------------
// Set/accessor functions for member data:
//-----------------------------------------------------------------------------
// Set the origin of mesh vertex positions:
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
set_origin(const Vector<MFLOAT,Dim>& o)
{
  origin = o;
  this->notifyOfChange();
}
// Get the origin of mesh vertex positions:
template<unsigned Dim, class MFLOAT>
Vector<MFLOAT,Dim> UniformCartesian<Dim,MFLOAT>::
get_origin() const
{
  return origin;
}

// Set the spacings of mesh vertex positions (recompute Dvc, cell volume):
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
set_meshSpacing(MFLOAT* const del)
{
  unsigned d;
  volume = 1.0;
  for (d=0; d<Dim; ++d) {
    meshSpacing[d] = del[d];
    volume *= del[d];
  }
  set_Dvc();
  // if spacing fields exist, we must recompute values
  if (hasSpacingFields) storeSpacingFields();
  this->notifyOfChange();
}
// Set only the derivative constants, using pre-set spacings:
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
set_Dvc()
{
  unsigned d;
  MFLOAT coef = 1.0;
  for (d=1;d<Dim;++d) coef *= 0.5;

  for (d=0;d<Dim;++d) {
    MFLOAT dvc = coef/meshSpacing[d];
    for (unsigned b=0; b<(1u<<Dim); ++b) {
      int s = ( b&(1<<d) ) ? 1 : -1;
      Dvc[b][d] = s*dvc;
    }
  }
}
// Get the spacings of mesh vertex positions along specified direction:
template<unsigned Dim, class MFLOAT>
MFLOAT UniformCartesian<Dim,MFLOAT>::
get_meshSpacing(unsigned d) const
{
  PAssert_LT(d, Dim);
  MFLOAT ms = meshSpacing[d];
  return ms;
}

// Get the cell volume:
template<unsigned Dim, class MFLOAT>
MFLOAT UniformCartesian<Dim,MFLOAT>::
get_volume() const
{
  return volume;
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
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
storeSpacingFields()
{
  // Set up default FieldLayout parameters:
  e_dim_tag et[Dim];
  for (unsigned int d=0; d<Dim; d++) et[d] = PARALLEL;
  storeSpacingFields(et, -1);
}
// 1D
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag p1, int vnodes)
{
  e_dim_tag et[1];
  et[0] = p1;
  storeSpacingFields(et, vnodes);
}
// 2D
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag p1, e_dim_tag p2, int vnodes)
{
  e_dim_tag et[2];
  et[0] = p1;
  et[1] = p2;
  storeSpacingFields(et, vnodes);
}
// 3D
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, int vnodes)
{
  e_dim_tag et[3];
  et[0] = p1;
  et[1] = p2;
  et[2] = p3;
  storeSpacingFields(et, vnodes);
}
// The general storeSpacingfields() function; others invoke this internally:
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag* et, int vnodes)
{
  // VERTEX-VERTEX SPACINGS (same as CELL-CELL SPACINGS for uniform):
  NDIndex<Dim> cells, verts;
  unsigned int d;
  for (d=0; d<Dim; d++) {
    cells[d] = Index(gridSizes[d]-1);
    verts[d] = Index(gridSizes[d]);
  }
  if (!hasSpacingFields) {
    // allocate layout and spacing field
    FlCell = new FieldLayout<Dim>(cells, et, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    // (not really used by Div() etc for UniformCartesian); someday should make
    // this user-settable.
    VertSpacings =
      new BareField<Vector<MFLOAT,Dim>,Dim>(*FlCell,GuardCellSizes<Dim>(1));
    // Added 12/8/98 --TJW:
    FlVert =
      new FieldLayout<Dim>(verts, et, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    CellSpacings =
      new BareField<Vector<MFLOAT,Dim>,Dim>(*FlVert,GuardCellSizes<Dim>(1));
  }
  BareField<Vector<MFLOAT,Dim>,Dim>& vertSpacings = *VertSpacings;
  Vector<MFLOAT,Dim> vertexSpacing;
  for (d=0; d<Dim; d++)
    vertexSpacing[d] = meshSpacing[d];
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
  Vector<MFLOAT,Dim> v0,v1; v0 = 0.0; v1 = 1.0; // Used for Reflective mesh BC
  unsigned int face;
  typedef Vector<MFLOAT,Dim> T;          // Used multipple places in loop below
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
  BareField<Vector<MFLOAT,Dim>,Dim>& cellSpacings = *CellSpacings;
  cellSpacings = vertexSpacing;

  hasSpacingFields = true; // Flag this as having been done to this object.
}

// These specify both the total number of vnodes and the numbers of vnodes
// along each dimension for the partitioning of the index space. Obviously
// this restricts the number of vnodes to be a product of the numbers along
// each dimension (the constructor implementation checks this): Special
// cases for 1-3 dimensions, ala FieldLayout ctors (see FieldLayout.h for
// more relevant comments, including definition of recurse):

// 1D
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
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
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
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
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
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
template<unsigned Dim, class MFLOAT>
void UniformCartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag *p,
		   unsigned* vnodesPerDirection,
		   bool recurse, int vnodes)
{
  // VERTEX-VERTEX SPACINGS (same as CELL-CELL SPACINGS for uniform):
  NDIndex<Dim> cells, verts;
  unsigned int d;
  for (d=0; d<Dim; d++) {
    cells[d] = Index(gridSizes[d]-1);
    verts[d] = Index(gridSizes[d]);
  }
  if (!hasSpacingFields) {
    // allocate layout and spacing field
    FlCell =
      new FieldLayout<Dim>(cells, p, vnodesPerDirection, recurse, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    // (not really used by Div() etc for UniformCartesian); someday should make
    // this user-settable.
    VertSpacings =
      new BareField<Vector<MFLOAT,Dim>,Dim>(*FlCell,GuardCellSizes<Dim>(1));
    // Added 12/8/98 --TJW:
    FlVert =
      new FieldLayout<Dim>(verts, p, vnodesPerDirection, recurse, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    CellSpacings =
      new BareField<Vector<MFLOAT,Dim>,Dim>(*FlVert,GuardCellSizes<Dim>(1));
  }
  BareField<Vector<MFLOAT,Dim>,Dim>& vertSpacings = *VertSpacings;
  Vector<MFLOAT,Dim> vertexSpacing;
  for (d=0; d<Dim; d++)
    vertexSpacing[d] = meshSpacing[d];
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
  Vector<MFLOAT,Dim> v0,v1; v0 = 0.0; v1 = 1.0; // Used for Reflective mesh BC
  unsigned int face;
  typedef Vector<MFLOAT,Dim> T;          // Used multipple places in loop below
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
  BareField<Vector<MFLOAT,Dim>,Dim>& cellSpacings = *CellSpacings;
  cellSpacings = vertexSpacing;

  hasSpacingFields = true; // Flag this as having been done to this object.
}

//-----------------------------------------------------------------------------
// I/O:
//-----------------------------------------------------------------------------
// Formatted output of UniformCartesian object:
template< unsigned Dim, class MFLOAT >
void
UniformCartesian<Dim,MFLOAT>::
print(std::ostream& out)
{
    Inform info("", out);
    print(info);
}

template< unsigned Dim, class MFLOAT >
void
UniformCartesian<Dim,MFLOAT>::
print(Inform& out)
{
    out << "======UniformCartesian<" << Dim << ",MFLOAT>==begin======\n";
    unsigned int d;
    for (d=0; d < Dim; d++)
        out << "gridSizes[" << d << "] = " << gridSizes[d] << "\n";
    out << "origin = " << origin << "\n";
    for (d=0; d < Dim; d++)
        out << "meshSpacing[" << d << "] = " << meshSpacing[d] << "\n";
    for (d=0; d < (1u<<Dim); d++)
        out << "Dvc[" << d << "] = " << Dvc[d] << "\n";
    out << "cell volume = " << volume << "\n";
    out << "======UniformCartesian<" << Dim << ",MFLOAT>==end========\n";
}

}
