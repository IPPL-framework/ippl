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
#include "Field/BrickExpression.h"
#include "Field/LField.h"
#include "Field/Field.h"
#include "Field/Assign.h"
#include "Field/AssignDefs.h"

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
                 const Vektor<MFLOAT,Dim>& orig)
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
                 const Vektor<MFLOAT,Dim>& orig)
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
                 const Vektor<MFLOAT,Dim>& orig)
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
                 MFLOAT* const delX, const Vektor<MFLOAT,Dim>& orig)
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
           const Vektor<MFLOAT,Dim>& orig)
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
           const Vektor<MFLOAT,Dim>& orig)
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
           const Vektor<MFLOAT,Dim>& orig)
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
           MFLOAT* const delX, const Vektor<MFLOAT,Dim>& orig)
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
set_origin(const Vektor<MFLOAT,Dim>& o)
{
  origin = o;
  this->notifyOfChange();
}
// Get the origin of mesh vertex positions:
template<unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim> UniformCartesian<Dim,MFLOAT>::
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
      new BareField<Vektor<MFLOAT,Dim>,Dim>(*FlCell,GuardCellSizes<Dim>(1));
    // Added 12/8/98 --TJW:
    FlVert =
      new FieldLayout<Dim>(verts, et, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    CellSpacings =
      new BareField<Vektor<MFLOAT,Dim>,Dim>(*FlVert,GuardCellSizes<Dim>(1));
  }
  BareField<Vektor<MFLOAT,Dim>,Dim>& vertSpacings = *VertSpacings;
  Vektor<MFLOAT,Dim> vertexSpacing;
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
  Vektor<MFLOAT,Dim> v0,v1; v0 = 0.0; v1 = 1.0; // Used for Reflective mesh BC
  unsigned int face;
  typedef Vektor<MFLOAT,Dim> T;          // Used multipple places in loop below
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
  BareField<Vektor<MFLOAT,Dim>,Dim>& cellSpacings = *CellSpacings;
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
      new BareField<Vektor<MFLOAT,Dim>,Dim>(*FlCell,GuardCellSizes<Dim>(1));
    // Added 12/8/98 --TJW:
    FlVert =
      new FieldLayout<Dim>(verts, p, vnodesPerDirection, recurse, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    CellSpacings =
      new BareField<Vektor<MFLOAT,Dim>,Dim>(*FlVert,GuardCellSizes<Dim>(1));
  }
  BareField<Vektor<MFLOAT,Dim>,Dim>& vertSpacings = *VertSpacings;
  Vektor<MFLOAT,Dim> vertexSpacing;
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
  Vektor<MFLOAT,Dim> v0,v1; v0 = 0.0; v1 = 1.0; // Used for Reflective mesh BC
  unsigned int face;
  typedef Vektor<MFLOAT,Dim> T;          // Used multipple places in loop below
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
  BareField<Vektor<MFLOAT,Dim>,Dim>& cellSpacings = *CellSpacings;
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

//--------------------------------------------------------------------------
// Various (UniformCartesian) mesh mechanisms:
//--------------------------------------------------------------------------

// Volume of single cell indexed by NDIndex:
template <unsigned Dim, class MFLOAT>
MFLOAT
UniformCartesian<Dim,MFLOAT>::
getCellVolume(const NDIndex<Dim>& ndi) const
{
  unsigned int d;
  for (d=0; d<Dim; d++)
    if (ndi[d].length() != 1)
      ERRORMSG("UniformCartesian::getCellVolume() error: arg is not a NDIndex"
	       << "specifying a single element" << endl);
  return volume;
}
// Field of volumes of all cells:
template <unsigned Dim, class MFLOAT>
Field<MFLOAT,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
UniformCartesian<Dim,MFLOAT>::
getCellVolumeField(Field<MFLOAT,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
		   volumes) const
{
  volumes = volume;
  return volumes;
}
// Volume of range of cells bounded by verticies specified by input NDIndex;
template <unsigned Dim, class MFLOAT>
MFLOAT
UniformCartesian<Dim,MFLOAT>::
getVertRangeVolume(const NDIndex<Dim>& ndi) const
{
  // Get vertex positions of extremal cells:
  Vektor<MFLOAT,Dim> v0, v1;
  NDIndex<Dim> ndi0, ndi1;
  unsigned int d;
  int i0, i1;
  for (d=0; d<Dim; d++) {
    i0 = ndi[d].first();
    i1 = ndi[d].last();
    ndi0[d] = Index(i0,i0,1); // Bounding vertex (from below)
    ndi1[d] = Index(i1,i1,1); // Bounding vertex (from above)
  }
  v0 = getVertexPosition(ndi0);
  v1 = getVertexPosition(ndi1);
  // Compute volume of rectangular solid beweeen these extremal vertices:
  MFLOAT volume = 1.0;
  for (d=0; d<Dim; d++) volume *= abs(v1(d) - v0(d));
  return volume;
}
// Volume of range of cells spanned by input NDIndex (index of cells):
template <unsigned Dim, class MFLOAT>
MFLOAT
UniformCartesian<Dim,MFLOAT>::
getCellRangeVolume(const NDIndex<Dim>& ndi) const
{
  // Get vertex positions bounding extremal cells:
  Vektor<MFLOAT,Dim> v0, v1;
  NDIndex<Dim> ndi0, ndi1;
  unsigned int d;
  int i0, i1;
  for (d=0; d<Dim; d++) {
    i0 = ndi[d].first();
    i1 = ndi[d].last() + 1;
    ndi0[d] = Index(i0,i0,1); // Bounding vertex (from below)
    ndi1[d] = Index(i1,i1,1); // Bounding vertex (from above)
  }
  v0 = getVertexPosition(ndi0);
  v1 = getVertexPosition(ndi1);
  // Compute volume of rectangular solid beweeen these extremal vertices:
  MFLOAT volume = 1.0;
  for (d=0; d<Dim; d++) volume *= abs(v1(d) - v0(d));
  return volume;
}

// Nearest vertex index to (x,y,z):
template <unsigned Dim, class MFLOAT>
NDIndex<Dim>
UniformCartesian<Dim,MFLOAT>::
getNearestVertex(const Vektor<MFLOAT,Dim>& x) const
{
  // Find coordinate vectors of the vertices just above and just below the
  // input point (extremal vertices on cell containing point):
  NDIndex<Dim> ndi;
  int i;
  for (unsigned int d=0; d<Dim; d++) {
    i = (int)((x(d) - origin(d))/meshSpacing[d] + 0.5);
    if (x(d) >= origin(d))
      ndi[d] = Index(i,i);
    else
      ndi[d] = Index(i-1,i-1);
  }
  return ndi;
}
// Nearest vertex index with all vertex coordinates below (x,y,z):
template <unsigned Dim, class MFLOAT>
NDIndex<Dim>
UniformCartesian<Dim,MFLOAT>::
getVertexBelow(const Vektor<MFLOAT,Dim>& x) const
{
  // Find coordinate vectors of the vertices just above and just below the
  // input point (extremal vertices on cell containing point):
  NDIndex<Dim> ndi;
  int i;
  for (unsigned int d=0; d<Dim; d++) {
    i = (int)((x(d) - origin(d))/meshSpacing[d]);
    if (x(d) >= origin(d))
      ndi[d] = Index(i,i);
    else
      ndi[d] = Index(i-1,i-1);
  }
  return ndi;
}
// (x,y,z) coordinates of indexed vertex:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
UniformCartesian<Dim,MFLOAT>::
getVertexPosition(const NDIndex<Dim>& ndi) const
{
  unsigned int d;
  for (d=0; d<Dim; d++) {
    if (ndi[d].length() != 1)
      ERRORMSG("UniformCartesian::getVertexPosition() error: arg is not a"
	       << " NDIndex specifying a single element" << endl);
  }
  // N.B.: following may need modification to be right for periodic Mesh BC:
  Vektor<MFLOAT,Dim> vertexPosition;
  for (d=0; d<Dim; d++)
    vertexPosition(d) = ndi[d].first()*meshSpacing[d] + origin(d);
  return vertexPosition;
}
// Field of (x,y,z) coordinates of all vertices:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Vert>&
UniformCartesian<Dim,MFLOAT>::
getVertexPositionField(Field<Vektor<MFLOAT,Dim>,Dim,
  UniformCartesian<Dim,MFLOAT>,Vert>& vertexPositions) const
{
  unsigned int d;
  int currentLocation[Dim];
  Vektor<MFLOAT,Dim> vertexPosition;
  vertexPositions.Uncompress();  // uncompress field before entering values!
  typename Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Vert>::iterator fi,
    fi_end = vertexPositions.end();
  for (fi = vertexPositions.begin(); fi != fi_end; ++fi) {
    fi.GetCurrentLocation(currentLocation);
    for (d=0; d<Dim; d++)
      vertexPosition(d) = origin(d) + currentLocation[d]*meshSpacing[d];
    *fi = vertexPosition;
  }
  return vertexPositions;
}

// (x,y,z) coordinates of indexed cell:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
UniformCartesian<Dim,MFLOAT>::
getCellPosition(const NDIndex<Dim>& ndi) const
{
  unsigned int d;
  for (d=0; d<Dim; d++) {
    if (ndi[d].length() != 1)
      ERRORMSG("UniformCartesian::getCellPosition() error: arg is not a"
	       << " NDIndex specifying a single element" << endl);
  }
  // N.B.: following may need modification to be right for periodic Mesh BC:
  Vektor<MFLOAT,Dim> cellPosition;
  for (d=0; d<Dim; d++)
    cellPosition(d) = (ndi[d].first()+0.5)*meshSpacing[d] + origin(d);
  return cellPosition;
}
// Field of (x,y,z) coordinates of all cells:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
UniformCartesian<Dim,MFLOAT>::
getCellPositionField(Field<Vektor<MFLOAT,Dim>,Dim,
  UniformCartesian<Dim,MFLOAT>,Cell>& cellPositions) const
{
  unsigned int d;
  int currentLocation[Dim];
  Vektor<MFLOAT,Dim> cellPosition;
  cellPositions.Uncompress();  // uncompress field before entering values!
  typename Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Cell>::iterator fi,
    fi_end = cellPositions.end();
  for (fi = cellPositions.begin(); fi != fi_end; ++fi) {
    fi.GetCurrentLocation(currentLocation);
    for (d=0; d<Dim; d++)
      cellPosition(d) = origin(d) + (currentLocation[d]+0.5)*meshSpacing[d];
    *fi = cellPosition;
  }
  return cellPositions;
}

// Vertex-vertex grid spacing of indexed cell:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
UniformCartesian<Dim,MFLOAT>::
getDeltaVertex(const NDIndex<Dim>& ndi) const
{
  // bfh: updated to compute interval for a whole index range
  Vektor<MFLOAT,Dim> vertexVertexSpacing;
  for (unsigned int d=0; d<Dim; d++)
    vertexVertexSpacing[d] = meshSpacing[d] * ndi[d].length();
  return vertexVertexSpacing;
}

// Field of vertex-vertex grid spacings of all cells:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
UniformCartesian<Dim,MFLOAT>::
getDeltaVertexField(Field<Vektor<MFLOAT,Dim>,Dim,
		    UniformCartesian<Dim,MFLOAT>,Cell>& vertexSpacings) const
{
  Vektor<MFLOAT,Dim> vertexVertexSpacing;
  unsigned int d;
  for (d=0; d<Dim; d++) vertexVertexSpacing(d) = meshSpacing[d];
  vertexSpacings = vertexVertexSpacing;
  return vertexSpacings;
}

// Cell-cell grid spacing of indexed vertex:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
UniformCartesian<Dim,MFLOAT>::
getDeltaCell(const NDIndex<Dim>& ndi) const
{
  // bfh: updated to compute interval for a whole index range
  Vektor<MFLOAT,Dim> cellCellSpacing;
  for (unsigned int d=0; d<Dim; d++)
    cellCellSpacing[d] = meshSpacing[d] * ndi[d].length();
  return cellCellSpacing;
}

// Field of cell-cell grid spacings of all vertices:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Vert>&
UniformCartesian<Dim,MFLOAT>::
getDeltaCellField(Field<Vektor<MFLOAT,Dim>,Dim,
		  UniformCartesian<Dim,MFLOAT>,Vert>& cellSpacings) const
{
  Vektor<MFLOAT,Dim> cellCellSpacing;
  unsigned int d;
  for (d=0; d<Dim; d++) cellCellSpacing(d) = meshSpacing[d];
  cellSpacings = cellCellSpacing;
  return cellSpacings;
}
// Array of surface normals to cells adjoining indexed cell:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>*
UniformCartesian<Dim,MFLOAT>::
getSurfaceNormals(const NDIndex<Dim>& /*ndi*/) const
{
  Vektor<MFLOAT,Dim>* surfaceNormals = new Vektor<MFLOAT,Dim>[2*Dim];
  unsigned int d, i;
  for (d=0; d<Dim; d++) {
    for (i=0; i<Dim; i++) {
      surfaceNormals[2*d](i)   = 0.0;
      surfaceNormals[2*d+1](i) = 0.0;
    }
    surfaceNormals[2*d](d)   = -1.0;
    surfaceNormals[2*d+1](d) =  1.0;
  }
  return surfaceNormals;
}
// Array of (pointers to) Fields of surface normals to all cells:
template <unsigned Dim, class MFLOAT>
void
UniformCartesian<Dim,MFLOAT>::
getSurfaceNormalFields(Field<Vektor<MFLOAT,Dim>, Dim,
		       UniformCartesian<Dim,MFLOAT>,Cell>**
		       surfaceNormalsFields ) const
{
  Vektor<MFLOAT,Dim>* surfaceNormals = new Vektor<MFLOAT,Dim>[2*Dim];
  unsigned int d, i;
  for (d=0; d<Dim; d++) {
    for (i=0; i<Dim; i++) {
      surfaceNormals[2*d](i)   = 0.0;
      surfaceNormals[2*d+1](i) = 0.0;
    }
    surfaceNormals[2*d](d)   = -1.0;
    surfaceNormals[2*d+1](d) =  1.0;
  }
  for (d=0; d<2*Dim; d++) assign((*(surfaceNormalsFields[d])),
				 surfaceNormals[d]);
  //  return surfaceNormalsFields;
}
// Similar functions, but specify the surface normal to a single face, using
// the following numbering convention: 0 means low face of 1st dim, 1 means
// high face of 1st dim, 2 means low face of 2nd dim, 3 means high face of
// 2nd dim, and so on:
// Surface normal to face on indexed cell:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
UniformCartesian<Dim,MFLOAT>::
getSurfaceNormal(const NDIndex<Dim>& /*ndi*/, unsigned face) const
{
  Vektor<MFLOAT,Dim> surfaceNormal;
  unsigned int d;
  // The following bitwise AND logical test returns true if face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if face is even (meaning the "low" or "left" face in the
  // numbering convention):
  if ( face & 1 ) {
    for (d=0; d<Dim; d++) {
      if ((face/2) == d) {
	surfaceNormal(face) = -1.0;
      } else {
	surfaceNormal(face) =  0.0;
      }
    }
  } else {
    for (d=0; d<Dim; d++) {
      if ((face/2) == d) {
	surfaceNormal(face) =  1.0;
      } else {
	surfaceNormal(face) =  0.0;
      }
    }
  }
  return surfaceNormal;
}
// Field of surface normals to face on all cells:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,UniformCartesian<Dim,MFLOAT>,Cell>&
UniformCartesian<Dim,MFLOAT>::
getSurfaceNormalField(Field<Vektor<MFLOAT,Dim>, Dim,
		      UniformCartesian<Dim,MFLOAT>,Cell>& surfaceNormalField,
		      unsigned face) const
{
  Vektor<MFLOAT,Dim> surfaceNormal;
  unsigned int d;
  // The following bitwise AND logical test returns true if face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if face is even (meaning the "low" or "left" face in the
  // numbering convention):
  if ( face & 1 ) {
    for (d=0; d<Dim; d++) {
      if ((face/2) == d) {
	surfaceNormal(face) = -1.0;
      } else {
	surfaceNormal(face) =  0.0;
      }
    }
  } else {
    for (d=0; d<Dim; d++) {
      if ((face/2) == d) {
	surfaceNormal(face) =  1.0;
      } else {
	surfaceNormal(face) =  0.0;
      }
    }
  }
  surfaceNormalField = surfaceNormal;
  return surfaceNormalField;
}


//--------------------------------------------------------------------------
// Global functions
//--------------------------------------------------------------------------


//*****************************************************************************
// Stuff taken from old Cartesian.h, which now applies to UniformCartesian:
//*****************************************************************************

//----------------------------------------------------------------------
// Divergence Vektor/Vert -> Scalar/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  dot(x[I  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1], x.get_mesh().Dvc[1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  dot(x[I  ][J  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1][J  ], x.get_mesh().Dvc[1]) +
	  dot(x[I  ][J+1], x.get_mesh().Dvc[2]) +
	  dot(x[I+1][J+1], x.get_mesh().Dvc[3]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1][J  ][K  ], x.get_mesh().Dvc[1]) +
	  dot(x[I  ][J+1][K  ], x.get_mesh().Dvc[2]) +
	  dot(x[I+1][J+1][K  ], x.get_mesh().Dvc[3]) +
	  dot(x[I  ][J  ][K+1], x.get_mesh().Dvc[4]) +
	  dot(x[I+1][J  ][K+1], x.get_mesh().Dvc[5]) +
	  dot(x[I  ][J+1][K+1], x.get_mesh().Dvc[6]) +
	  dot(x[I+1][J+1][K+1], x.get_mesh().Dvc[7]));
  return r;
}
//----------------------------------------------------------------------
// Divergence Vektor/Cell -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  dot(x[I-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ], x.get_mesh().Dvc[1]));
  return r;
}
//----------------------------------------------------------------------
// (tjw: the one in cv.cpp)
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  dot(x[I-1][J-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ][J-1], x.get_mesh().Dvc[1]) +
	  dot(x[I-1][J  ], x.get_mesh().Dvc[2]) +
	  dot(x[I  ][J  ], x.get_mesh().Dvc[3]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  dot(x[I-1][J-1][K-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ][J-1][K-1], x.get_mesh().Dvc[1]) +
	  dot(x[I-1][J  ][K-1], x.get_mesh().Dvc[2]) +
	  dot(x[I  ][J  ][K-1], x.get_mesh().Dvc[3]) +
	  dot(x[I-1][J-1][K  ], x.get_mesh().Dvc[4]) +
	  dot(x[I  ][J-1][K  ], x.get_mesh().Dvc[5]) +
	  dot(x[I-1][J  ][K  ], x.get_mesh().Dvc[6]) +
	  dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[7]));
  return r;
}
//----------------------------------------------------------------------
// Divergence Vektor/Vert -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  Vektor<T,1U> idx;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);

  assign( r[I] , dot( idx , (x[I+1] - x[I-1]) ) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Vektor<T,2U> idx,idy;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);

  assign( r[I][J] ,
	  dot( idx , (x[I+1][J  ] - x[I-1][J  ])) +
	  dot( idy , (x[I  ][J+1] - x[I  ][J-1])) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  Vektor<T,3U> idx,idy,idz;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 0.5/x.get_mesh().get_meshSpacing(2);

  assign( r[I][J][K] ,
	  dot(idx , (x[I+1][J  ][K  ] - x[I-1][J  ][K  ] )) +
	  dot(idy , (x[I  ][J+1][K  ] - x[I  ][J-1][K  ] )) +
	  dot(idz , (x[I  ][J  ][K+1] - x[I  ][J  ][K-1] )) );
  return r;
}
//----------------------------------------------------------------------
// Divergence Vektor/Edge -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Edge>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  Vektor<T,1U> idx;
  idx[0] = 1.0/x.get_mesh().get_meshSpacing(0);

  assign( r[I] , dot( idx , (x[I] - x[I-1]) ) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Edge>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Vektor<T,2U> idx,idy;
  idx[0] = 1.0/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0/x.get_mesh().get_meshSpacing(1);

  assign( r[I][J] ,
	  dot( idx , (x[I][J] - x[I-1][J  ])) +
	  dot( idy , (x[I][J] - x[I  ][J-1])) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Edge>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  Vektor<T,3U> idx,idy,idz;
  idx[0] = 1.0/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0/x.get_mesh().get_meshSpacing(1);
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 1.0/x.get_mesh().get_meshSpacing(2);

  assign( r[I][J][K] ,
	  dot(idx , (x[I][J][K] - x[I-1][J  ][K  ] )) +
	  dot(idy , (x[I][J][K] - x[I  ][J-1][K  ] )) +
	  dot(idz , (x[I][J][K] - x[I  ][J  ][K-1] )) );
  return r;
}
//----------------------------------------------------------------------
// Divergence Vektor/Cell -> Scalar/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Div(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
    Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  Vektor<T,1U> idx;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);

  assign( r[I] , dot( idx , (x[I+1] - x[I-1]) ) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Div(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
    Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Vektor<T,2U> idx,idy;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);

  assign( r[I][J] ,
	  dot( idx , (x[I+1][J  ] - x[I-1][J  ])) +
	  dot( idy , (x[I  ][J+1] - x[I  ][J-1])) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Div(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
    Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  Vektor<T,3U> idx,idy,idz;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 0.5/x.get_mesh().get_meshSpacing(2);

  assign( r[I][J][K] ,
	  dot(idx , (x[I+1][J  ][K  ] - x[I-1][J  ][K  ] )) +
	  dot(idy , (x[I  ][J+1][K  ] - x[I  ][J-1][K  ] )) +
	  dot(idz , (x[I  ][J  ][K+1] - x[I  ][J  ][K-1] )) );
  return r;
}
//----------------------------------------------------------------------
// Divergence Tenzor/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
    Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  dot(x[I  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1], x.get_mesh().Dvc[1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
    Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  dot(x[I  ][J  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1][J  ], x.get_mesh().Dvc[1]) +
	  dot(x[I  ][J+1], x.get_mesh().Dvc[2]) +
	  dot(x[I+1][J+1], x.get_mesh().Dvc[3]));

  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
inline Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
    Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1][J  ][K  ], x.get_mesh().Dvc[1]) +
	  dot(x[I  ][J+1][K  ], x.get_mesh().Dvc[2]) +
	  dot(x[I+1][J+1][K  ], x.get_mesh().Dvc[3]) +
	  dot(x[I  ][J  ][K+1], x.get_mesh().Dvc[4]) +
	  dot(x[I+1][J  ][K+1], x.get_mesh().Dvc[5]) +
	  dot(x[I  ][J+1][K+1], x.get_mesh().Dvc[6]) +
	  dot(x[I+1][J+1][K+1], x.get_mesh().Dvc[7]));

  return r;
}
//----------------------------------------------------------------------
// Divergence SymTenzor/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
inline Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Div(Field<SymTenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
    Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  dot(x[I  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1], x.get_mesh().Dvc[1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
inline Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Div(Field<SymTenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
    Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  dot(x[I  ][J  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1][J  ], x.get_mesh().Dvc[1]) +
	  dot(x[I  ][J+1], x.get_mesh().Dvc[2]) +
	  dot(x[I+1][J+1], x.get_mesh().Dvc[3]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
inline Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Div(Field<SymTenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
    Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[0]) +
	  dot(x[I+1][J  ][K  ], x.get_mesh().Dvc[1]) +
	  dot(x[I  ][J+1][K  ], x.get_mesh().Dvc[2]) +
	  dot(x[I+1][J+1][K  ], x.get_mesh().Dvc[3]) +
	  dot(x[I  ][J  ][K+1], x.get_mesh().Dvc[4]) +
	  dot(x[I+1][J  ][K+1], x.get_mesh().Dvc[5]) +
	  dot(x[I  ][J+1][K+1], x.get_mesh().Dvc[6]) +
	  dot(x[I+1][J+1][K+1], x.get_mesh().Dvc[7]));
  return r;
}

//----------------------------------------------------------------------
// Divergence Tenzor/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
    Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  dot(x[I-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ], x.get_mesh().Dvc[1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
    Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  dot(x[I-1][J-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ][J-1], x.get_mesh().Dvc[1]) +
	  dot(x[I-1][J  ], x.get_mesh().Dvc[2]) +
	  dot(x[I  ][J  ], x.get_mesh().Dvc[3]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
    Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  dot(x[I-1][J-1][K-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ][J-1][K-1], x.get_mesh().Dvc[1]) +
	  dot(x[I-1][J  ][K-1], x.get_mesh().Dvc[2]) +
	  dot(x[I  ][J  ][K-1], x.get_mesh().Dvc[3]) +
	  dot(x[I-1][J-1][K  ], x.get_mesh().Dvc[4]) +
	  dot(x[I  ][J-1][K  ], x.get_mesh().Dvc[5]) +
	  dot(x[I-1][J  ][K  ], x.get_mesh().Dvc[6]) +
	  dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[7]));
  return r;
}

//----------------------------------------------------------------------
// Divergence SymTenzor/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
    Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  dot(x[I-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ], x.get_mesh().Dvc[1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
    Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  dot(x[I-1][J-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ][J-1], x.get_mesh().Dvc[1]) +
	  dot(x[I-1][J  ], x.get_mesh().Dvc[2]) +
	  dot(x[I  ][J  ], x.get_mesh().Dvc[3]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
    Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  dot(x[I-1][J-1][K-1], x.get_mesh().Dvc[0]) +
	  dot(x[I  ][J-1][K-1], x.get_mesh().Dvc[1]) +
	  dot(x[I-1][J  ][K-1], x.get_mesh().Dvc[2]) +
	  dot(x[I  ][J  ][K-1], x.get_mesh().Dvc[3]) +
	  dot(x[I-1][J-1][K  ], x.get_mesh().Dvc[4]) +
	  dot(x[I  ][J-1][K  ], x.get_mesh().Dvc[5]) +
	  dot(x[I-1][J  ][K  ], x.get_mesh().Dvc[6]) +
	  dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[7]));
  return r;
}

//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  x[I  ] * x.get_mesh().Dvc[0] +
	  x[I+1] * x.get_mesh().Dvc[1]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
     Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  x[I  ][J  ] * x.get_mesh().Dvc[0] +
	  x[I+1][J  ] * x.get_mesh().Dvc[1] +
	  x[I  ][J+1] * x.get_mesh().Dvc[2] +
	  x[I+1][J+1] * x.get_mesh().Dvc[3]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  x[I  ][J  ][K  ] * x.get_mesh().Dvc[0] +
	  x[I+1][J  ][K  ] * x.get_mesh().Dvc[1] +
	  x[I  ][J+1][K  ] * x.get_mesh().Dvc[2] +
	  x[I+1][J+1][K  ] * x.get_mesh().Dvc[3] +
	  x[I  ][J  ][K+1] * x.get_mesh().Dvc[4] +
	  x[I+1][J  ][K+1] * x.get_mesh().Dvc[5] +
	  x[I  ][J+1][K+1] * x.get_mesh().Dvc[6] +
	  x[I+1][J+1][K+1] * x.get_mesh().Dvc[7]);

  return r;
}
//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Edge
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Edge>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Edge>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = Index(domain[0].first(), domain[0].last()-1);

  assign( r[I] ,
	  x[I  ] * x.get_mesh().Dvc[0] +
	  x[I+1] * x.get_mesh().Dvc[1]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Edge>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
     Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Edge>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = Index(domain[0].first(), domain[0].last()-1);
  Index J = Index(domain[1].first(), domain[1].last()-1);

  Vektor<T,2U> idx,idy;
  idx[0] = 1.0/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0/x.get_mesh().get_meshSpacing(1);

  assign( r[I][J] ,
	  (x[I+1][J  ] - x[I  ][J  ]) * idx +
          (x[I  ][J+1] - x[I  ][J  ]) * idy);
  I = Index(domain[0].last(), domain[0].last());
  assign( r[I][J](1),
          (x[I][J+1] - x[I][J]));
  I = Index(domain[0].first(), domain[0].last()-1);
  J = Index(domain[1].last(), domain[1].last());
  assign( r[I][J](0),
          (x[I+1][J] - x[I][J]));
  J = Index(domain[1].first(), domain[1].last()-1);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Edge>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Edge>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = Index(domain[0].first(), domain[0].last()-1);
  Index J = Index(domain[1].first(), domain[1].last()-1);
  Index K = Index(domain[2].first(), domain[2].last()-1);

  Vektor<T,3U> idx,idy,idz;
  idx[0] = 1.0/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0/x.get_mesh().get_meshSpacing(1);
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 1.0/x.get_mesh().get_meshSpacing(2);

  assign( r[I][J][K] ,
	  (x[I+1][J  ][K  ] - x[I  ][J  ][K  ]) * idx +
          (x[I  ][J+1][K  ] - x[I  ][J  ][K  ]) * idy +
          (x[I  ][J  ][K+1] - x[I  ][J  ][K  ]) * idz);
  I = Index(domain[0].last(), domain[0].last());
  assign( r[I][J][K](1),
          (x[I  ][J+1][K  ] - x[I  ][J  ][K  ]));
  assign( r[I][J][K](2),
          (x[I  ][J  ][K+1] - x[I  ][J  ][K  ]));
  I = Index(domain[0].first(), domain[0].last()-1);
  J = Index(domain[1].last(), domain[1].last());
  assign( r[I][J][K](0),
          (x[I+1][J  ][K  ] - x[I  ][J  ][K  ]));
  assign( r[I][J][K](2),
          (x[I  ][J  ][K+1] - x[I  ][J  ][K  ]));
  J = Index(domain[1].first(), domain[1].last()-1);
  K = Index(domain[2].last(), domain[2].last());
  assign( r[I][J][K](0),
          (x[I+1][J  ][K  ] - x[I  ][J  ][K  ]));
  assign( r[I][J][K](1),
          (x[I  ][J+1][K  ] - x[I  ][J  ][K  ]));
  K = Index(domain[2].first(), domain[2].last()-1);

  return r;
}
//----------------------------------------------------------------------
// Grad Scalar/Cell -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  x[I-1] * x.get_mesh().Dvc[0] +
	  x[I  ] * x.get_mesh().Dvc[1]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
     Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  x[I-1][J-1] * x.get_mesh().Dvc[0] +
	  x[I  ][J-1] * x.get_mesh().Dvc[1] +
	  x[I-1][J  ] * x.get_mesh().Dvc[2] +
	  x[I  ][J  ] * x.get_mesh().Dvc[3]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  x[I-1][J-1][K-1] * x.get_mesh().Dvc[0] +
	  x[I  ][J-1][K-1] * x.get_mesh().Dvc[1] +
	  x[I-1][J  ][K-1] * x.get_mesh().Dvc[2] +
	  x[I  ][J  ][K-1] * x.get_mesh().Dvc[3] +
	  x[I-1][J-1][K  ] * x.get_mesh().Dvc[4] +
	  x[I  ][J-1][K  ] * x.get_mesh().Dvc[5] +
	  x[I-1][J  ][K  ] * x.get_mesh().Dvc[6] +
	  x[I  ][J  ][K  ] * x.get_mesh().Dvc[7]);
  return r;
}
//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  Vektor<T,1U> idx;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);

  assign( r[I] ,  idx * (x[I+1] - x[I-1] ) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
     Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  Vektor<T,2U> idx,idy;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);

  assign( r[I][J] ,
	  idx * (x[I+1][J  ] - x[I-1][J  ]) +
	  idy * (x[I  ][J+1] - x[I  ][J-1]) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  Vektor<T,3U> idx,idy,idz;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 0.5/x.get_mesh().get_meshSpacing(2);

  assign(r[I][J][K] ,
	 idx * (x[I+1][J  ][K  ] - x[I-1][J  ][K  ]) +
	 idy * (x[I  ][J+1][K  ] - x[I  ][J-1][K  ]) +
	 idz * (x[I  ][J  ][K+1] - x[I  ][J  ][K-1]));
  return r;
}
//----------------------------------------------------------------------
// Grad Scalar/Cell -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Grad(Field<T,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
     Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  Vektor<T,1U> idx;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);

  assign( r[I] , idx * (x[I+1] - x[I-1] ) );
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Grad(Field<T,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
     Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  Vektor<T,2U> idx,idy;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);

  assign( r[I][J] ,
	  idx * (x[I+1][J  ] - x[I-1][J  ]) +
	  idy * (x[I  ][J+1] - x[I  ][J-1]) );

  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Grad(Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  Vektor<T,3U> idx,idy,idz;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 0.5/x.get_mesh().get_meshSpacing(2);

  assign(r[I][J][K] ,
	 idx * (x[I+1][J  ][K  ] - x[I-1][J  ][K  ]) +
	 idy * (x[I  ][J+1][K  ] - x[I  ][J-1][K  ]) +
	 idz * (x[I  ][J  ][K+1] - x[I  ][J  ][K-1]));

  //approximate at the border up to a term of order O(h^2)
  Vektor<T,3U> xx = Vektor<T,3U>(1.0,0.0,0.0);
  Vektor<T,3U> yy = Vektor<T,3U>(0.0,1.0,0.0);
  Vektor<T,3U> zz = Vektor<T,3U>(0.0,0.0,1.0);

  Index lo(0, 0);

  int maxI = I.length() - 1;
  int maxJ = J.length() - 1;
  int maxK = K.length() - 1;

  Index Iup(maxI, maxI);
  Index Jup(maxJ, maxJ);
  Index Kup(maxK, maxK);

  r[lo][J][K] = (idx * (- 1.0*x[lo+2][J][K]
                        + 4.0*x[lo+1][J][K]
                        - 3.0*x[lo  ][J][K])
                 + yy*r[lo][J][K]
                 + zz*r[lo][J][K]);
  r[Iup][J][K] = (idx * (  1.0*x[Iup-2][J][K]
                         - 4.0*x[Iup-1][J][K]
                         + 3.0*x[Iup  ][J][K])
                  + yy*r[Iup][J][K]
                  + zz*r[Iup][J][K]);

  r[I][lo][K] = (idy * (- 1.0*x[I][lo+2][K]
                        + 4.0*x[I][lo+1][K]
                        - 3.0*x[I][lo  ][K])
                 + xx*r[I][lo][K]
                 + zz*r[I][lo][K]);
  r[I][Jup][K] = (idy * (  1.0*x[I][Jup-2][K]
                         - 4.0*x[I][Jup-1][K]
                         + 3.0*x[I][Jup  ][K])
                  + xx*r[I][Jup][K]
                  + zz*r[I][Jup][K]);

  r[I][J][lo] = (idz * (- 1.0*x[I][J][lo+2]
                        + 4.0*x[I][J][lo+1]
                        - 3.0*x[I][J][lo  ])
                 + xx*r[I][J][lo]
                 + yy*r[I][J][lo]);
  r[I][J][Kup] = (idz * (  1.0*x[I][J][Kup-2]
                         - 4.0*x[I][J][Kup-1]
                         + 3.0*x[I][J][Kup  ])
                  + xx*r[I][J][Kup]
                  + yy*r[I][J][Kup]);

  return r;
}


template < class T, class MFLOAT >
inline Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Grad1Ord(Field<T,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
     Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  Vektor<T,3U> idx,idy,idz;
  idx[0] = 0.5/x.get_mesh().get_meshSpacing(0);
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 0.5/x.get_mesh().get_meshSpacing(1);
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 0.5/x.get_mesh().get_meshSpacing(2);

  assign(r[I][J][K] ,
	 idx * (x[I+1][J  ][K  ] - x[I-1][J  ][K  ]) +
	 idy * (x[I  ][J+1][K  ] - x[I  ][J-1][K  ]) +
	 idz * (x[I  ][J  ][K+1] - x[I  ][J  ][K-1]));
  return r;
}
//----------------------------------------------------------------------
// Grad Vektor/Vert -> Tenzor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  outerProduct( x[I  ] , x.get_mesh().Dvc[0] ) +
	  outerProduct( x[I+1] , x.get_mesh().Dvc[1])) ;
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  outerProduct( x[I  ][J  ] , x.get_mesh().Dvc[0] ) +
	  outerProduct( x[I+1][J  ] , x.get_mesh().Dvc[1] ) +
	  outerProduct( x[I  ][J+1] , x.get_mesh().Dvc[2] ) +
	  outerProduct( x[I+1][J+1] , x.get_mesh().Dvc[3] )) ;
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  outerProduct( x[I  ][J  ][K  ] , x.get_mesh().Dvc[0] ) +
	  outerProduct( x[I+1][J  ][K  ] , x.get_mesh().Dvc[1] ) +
	  outerProduct( x[I  ][J+1][K  ] , x.get_mesh().Dvc[2] ) +
	  outerProduct( x[I+1][J+1][K  ] , x.get_mesh().Dvc[3] ) +
	  outerProduct( x[I  ][J  ][K+1] , x.get_mesh().Dvc[4] ) +
	  outerProduct( x[I+1][J  ][K+1] , x.get_mesh().Dvc[5] ) +
	  outerProduct( x[I  ][J+1][K+1] , x.get_mesh().Dvc[6] ) +
	  outerProduct( x[I+1][J+1][K+1] , x.get_mesh().Dvc[7] ));

  return r;
}
//----------------------------------------------------------------------
// Grad Vektor/Cell -> Tenzor/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,1U>,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  assign( r[I] ,
	  outerProduct( x[I-1] , x.get_mesh().Dvc[0] ) +
	  outerProduct( x[I  ] , x.get_mesh().Dvc[1] ));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,2U>,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  outerProduct( x[I-1][J-1] , x.get_mesh().Dvc[0] ) +
	  outerProduct( x[I  ][J-1] , x.get_mesh().Dvc[1] ) +
	  outerProduct( x[I-1][J  ] , x.get_mesh().Dvc[2] ) +
	  outerProduct( x[I  ][J  ] , x.get_mesh().Dvc[3] ));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,3U>,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  outerProduct( x[I-1][J-1][K-1] , x.get_mesh().Dvc[0] ) +
	  outerProduct( x[I  ][J-1][K-1] , x.get_mesh().Dvc[1] ) +
	  outerProduct( x[I-1][J  ][K-1] , x.get_mesh().Dvc[2] ) +
	  outerProduct( x[I  ][J  ][K-1] , x.get_mesh().Dvc[3] ) +
	  outerProduct( x[I-1][J-1][K  ] , x.get_mesh().Dvc[4] ) +
	  outerProduct( x[I  ][J-1][K  ] , x.get_mesh().Dvc[5] ) +
	  outerProduct( x[I-1][J  ][K  ] , x.get_mesh().Dvc[6] ) +
	  outerProduct( x[I  ][J  ][K  ] , x.get_mesh().Dvc[7] ));
  return r;
}

namespace IPPL {

//----------------------------------------------------------------------
// Weighted average Cell to Vert
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Average(Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
	Field<T2,1U,UniformCartesian<1U,MFLOAT>,Cell>& w,
	Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  assign( r[I] ,
	  ( x[I-1] * w[I-1] + x[I  ] * w[I  ] )/
	  ( w[I-1] + w[I  ] ) );
  return r;
}
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Average(Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
	Field<T2,2U,UniformCartesian<2U,MFLOAT>,Cell>& w,
	Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  ( x[I-1][J-1] * w[I-1][J-1] +
	    x[I  ][J-1] * w[I  ][J-1] +
	    x[I-1][J  ] * w[I-1][J  ] +
	    x[I  ][J  ] * w[I  ][J  ] )/
	  ( w[I-1][J-1] +
	    w[I  ][J-1] +
	    w[I-1][J  ] +
	    w[I  ][J  ] ) );
  return r;
}
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Average(Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
	Field<T2,3U,UniformCartesian<3U,MFLOAT>,Cell>& w,
	Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  ( x[I-1][J-1][K-1] * w[I-1][J-1][K-1] +
	    x[I  ][J-1][K-1] * w[I  ][J-1][K-1] +
	    x[I-1][J  ][K-1] * w[I-1][J  ][K-1] +
	    x[I  ][J  ][K-1] * w[I  ][J  ][K-1] +
	    x[I-1][J-1][K  ] * w[I-1][J-1][K  ] +
	    x[I  ][J-1][K  ] * w[I  ][J-1][K  ] +
	    x[I-1][J  ][K  ] * w[I-1][J  ][K  ] +
	    x[I  ][J  ][K  ] * w[I  ][J  ][K  ] )/
	  ( w[I-1][J-1][K-1] +
	    w[I  ][J-1][K-1] +
	    w[I-1][J  ][K-1] +
	    w[I  ][J  ][K-1] +
	    w[I-1][J-1][K  ] +
	    w[I  ][J-1][K  ] +
	    w[I-1][J  ][K  ] +
	    w[I  ][J  ][K  ] ) );
  return r;
}
//----------------------------------------------------------------------
// Weighted average Vert to Cell
// N.B.: won't work except for unit-stride (& zero-base?) Field's.
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Average(Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,
	Field<T2,1U,UniformCartesian<1U,MFLOAT>,Vert>& w,
	Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  assign( r[I] ,
	  ( x[I+1] * w[I+1] + x[I  ] * w[I  ] )/
	  ( w[I+1] + w[I  ] ) );
  return r;
}
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Average(Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
	Field<T2,2U,UniformCartesian<2U,MFLOAT>,Vert>& w,
	Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  assign( r[I][J] ,
	  ( x[I+1][J+1] * w[I+1][J+1] +
	    x[I  ][J+1] * w[I  ][J+1] +
	    x[I+1][J  ] * w[I+1][J  ] +
	    x[I  ][J  ] * w[I  ][J  ] )/
	  ( w[I+1][J+1] +
	    w[I  ][J+1] +
	    w[I+1][J  ] +
	    w[I  ][J  ] ) );
  return r;
}
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Average(Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
	Field<T2,3U,UniformCartesian<3U,MFLOAT>,Vert>& w,
	Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  assign( r[I][J][K] ,
	  ( x[I+1][J+1][K+1] * w[I+1][J+1][K+1] +
	    x[I  ][J+1][K+1] * w[I  ][J+1][K+1] +
	    x[I+1][J  ][K+1] * w[I+1][J  ][K+1] +
	    x[I  ][J  ][K+1] * w[I  ][J  ][K+1] +
	    x[I+1][J+1][K  ] * w[I+1][J+1][K  ] +
	    x[I  ][J+1][K  ] * w[I  ][J+1][K  ] +
	    x[I+1][J  ][K  ] * w[I+1][J  ][K  ] +
	    x[I  ][J  ][K  ] * w[I  ][J  ][K  ] )/
	  ( w[I+1][J+1][K+1] +
	    w[I  ][J+1][K+1] +
	    w[I+1][J  ][K+1] +
	    w[I  ][J  ][K+1] +
	    w[I+1][J+1][K  ] +
	    w[I  ][J+1][K  ] +
	    w[I+1][J  ][K  ] +
	    w[I  ][J  ][K  ] ) );
  return r;
}

//----------------------------------------------------------------------
// Unweighted average Cell to Vert
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>&
Average(Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>& x,
	Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] = 0.5*(x[I-1] + x[I  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>&
Average(Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>& x,
	Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] = 0.25*(x[I-1][J-1] + x[I  ][J-1] + x[I-1][J  ] + x[I  ][J  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>&
Average(Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>& x,
	Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] = 0.125*(x[I-1][J-1][K-1] + x[I  ][J-1][K-1] + x[I-1][J  ][K-1] +
		      x[I  ][J  ][K-1] + x[I-1][J-1][K  ] + x[I  ][J-1][K  ] +
		      x[I-1][J  ][K  ] + x[I  ][J  ][K  ]);
  return r;
}
//----------------------------------------------------------------------
// Unweighted average Vert to Cell
// N.B.: won't work except for unit-stride (& zero-base?) Field's.
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>&
Average(Field<T1,1U,UniformCartesian<1U,MFLOAT>,Vert>& x,

	Field<T1,1U,UniformCartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] = 0.5*(x[I+1] + x[I  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>&
Average(Field<T1,2U,UniformCartesian<2U,MFLOAT>,Vert>& x,
	Field<T1,2U,UniformCartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] = 0.25*(x[I+1][J+1] + x[I  ][J+1] + x[I+1][J  ] + x[I  ][J  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>&
Average(Field<T1,3U,UniformCartesian<3U,MFLOAT>,Vert>& x,
	Field<T1,3U,UniformCartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] = 0.125*(x[I+1][J+1][K+1] + x[I  ][J+1][K+1] + x[I+1][J  ][K+1] +
		      x[I  ][J  ][K+1] + x[I+1][J+1][K  ] + x[I  ][J+1][K  ] +
		      x[I+1][J  ][K  ] + x[I  ][J  ][K  ]);
  return r;
}

}
