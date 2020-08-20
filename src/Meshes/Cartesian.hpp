// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// Cartesian.cpp
// Implementations for Cartesian mesh class (nonuniform spacings)

// include files
#include "Utility/PAssert.h"
#include "Utility/IpplException.h"
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
Cartesian<Dim,MFLOAT>::
setup()
{
  hasSpacingFields = false;
}

//-----------------------------------------------------------------------------
// Constructors from NDIndex object:
//-----------------------------------------------------------------------------
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const NDIndex<Dim>& ndi)
{
  unsigned int d,i;
  for (d=0; d<Dim; d++)
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  setup();                          // Setup chores, such as array allocations
  for (d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective;     // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective;     // Default mesh: reflective boundary conds
    origin(d) = ndi[d].first();     // Default origin at ndi[d].first()
    // default mesh spacing from stride()
    for (i=0; i < gridSizes[d]-1; i++) {
      (meshSpacing[d])[i] = ndi[d].stride();
      (meshPosition[d])[i] = MFLOAT(i);
    }
    (meshPosition[d])[gridSizes[d]-1] = MFLOAT(gridSizes[d]-1);
  }
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const NDIndex<Dim>& ndi, MFLOAT** const delX)
{
  unsigned int d;
  for (d=0; d<Dim; d++)
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  setup();                          // Setup chores, such as array allocations
  for (d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective;     // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective;     // Default mesh: reflective boundary conds
    origin(d) = ndi[d].first();     // Default origin at ndi[d].first()
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const NDIndex<Dim>& ndi, MFLOAT** const delX,
          const Vektor<MFLOAT,Dim>& orig)
{
  int d;
  for (d=0; d<Dim; d++)
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  setup();                          // Setup chores, such as array allocations
  for (d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective;     // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective;     // Default mesh: reflective boundary conds
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify a MeshBC_E array for mesh boundary conditions:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const NDIndex<Dim>& ndi, MFLOAT** const delX,
          const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc)
{
  int d;
  for (d=0; d<Dim; d++)
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  setup();                          // Setup chores, such as array allocations
  set_origin(orig);           // Set origin.
  set_MeshBC(mbc);            // Set up mesh boundary conditions
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
//-----------------------------------------------------------------------------
// Constructors from Index objects:
//-----------------------------------------------------------------------------

//===========1D============
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();      // Default origin at I.first()
  unsigned int i;
  // Default mesh spacing from stride()
  for (i=0; i < gridSizes[0]-1; i++) {
    (meshSpacing[0])[i] = I.stride();
    (meshPosition[0])[i] = MFLOAT(i);
  }
  (meshPosition[0])[gridSizes[0]-1] = MFLOAT(gridSizes[0]-1);
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, MFLOAT** const delX)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();      // Default origin at I.first()
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, MFLOAT** const delX,
          const Vektor<MFLOAT,Dim>& orig)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify a MeshBC_E array for mesh boundary conditions:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, MFLOAT** const delX,
          const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  set_origin(orig);           // Set origin.
  set_MeshBC(mbc);            // Set up mesh boundary conditions
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}

//===========2D============
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, const Index& J)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();      // Default origin at I.first(),J.first()
  origin(1) = J.first();
  unsigned int i;
  // Default mesh spacing from stride()
  for (i=0; i < gridSizes[0]-1; i++) {
    (meshSpacing[0])[i] = I.stride();
    (meshPosition[0])[i] = MFLOAT(i);
  }
  (meshPosition[0])[gridSizes[0]-1] = MFLOAT(gridSizes[0]-1);
  for (i=0; i < gridSizes[1]-1; i++) {
    (meshSpacing[1])[i] = J.stride();
    (meshPosition[1])[i] = MFLOAT(i);
  }
  (meshPosition[1])[gridSizes[1]-1] = MFLOAT(gridSizes[1]-1);
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, const Index& J, MFLOAT** const delX)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();      // Default origin at I.first(),J.first()
  origin(1) = J.first();
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, const Index& J, MFLOAT** const delX,
          const Vektor<MFLOAT,Dim>& orig)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify a MeshBC_E array for mesh boundary conditions:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, const Index& J, MFLOAT** const delX,
          const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  set_origin(orig);           // Set origin.
  set_MeshBC(mbc);            // Set up mesh boundary conditions
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}

//===========3D============
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, const Index& J, const Index& K)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  // Setup chores, such as array allocations
  setup();
  origin(0) = I.first();    // Default origin at I.first(),J.first(),K.first()
  origin(1) = J.first();
  origin(2) = K.first();
  unsigned int i;
  // Default mesh spacing from stride()
  for (i=0; i < gridSizes[0]-1; i++) {
    (meshSpacing[0])[i] = I.stride();
    (meshPosition[0])[i] = MFLOAT(i);
  }
  (meshPosition[0])[gridSizes[0]-1] = MFLOAT(gridSizes[0]-1);
  for (i=0; i < gridSizes[1]-1; i++) {
    (meshSpacing[1])[i] = J.stride();
    (meshPosition[1])[i] = MFLOAT(i);
  }
  (meshPosition[1])[gridSizes[1]-1] = MFLOAT(gridSizes[1]-1);
  for (i=0; i < gridSizes[2]-1; i++) {
    (meshSpacing[2])[i] = K.stride();
    (meshPosition[2])[i] = MFLOAT(i);
  }
  (meshPosition[2])[gridSizes[2]-1] = MFLOAT(gridSizes[2]-1);

  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, const Index& J, const Index& K,
          MFLOAT** const delX)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();    // Default origin at I.first(),J.first(),K.first()
  origin(1) = J.first();
  origin(2) = K.first();
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, const Index& J, const Index& K,
          MFLOAT** const delX, const Vektor<MFLOAT,Dim>& orig)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify a MeshBC_E array for mesh boundary conditions:
template <unsigned Dim, class MFLOAT>
Cartesian<Dim,MFLOAT>::
Cartesian(const Index& I, const Index& J, const Index& K,
          MFLOAT** const delX, const Vektor<MFLOAT,Dim>& orig,
          MeshBC_E* const mbc)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  set_origin(orig);           // Set origin.
  set_MeshBC(mbc);            // Set up mesh boundary conditions
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}

//-----------------------------------------------------------------------------
// initialize using NDIndex object:
//-----------------------------------------------------------------------------
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const NDIndex<Dim>& ndi)
{
  unsigned int d,i;
  for (d=0; d<Dim; d++)
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  setup();                          // Setup chores, such as array allocations
  for (d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective;     // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective;     // Default mesh: reflective boundary conds
    origin(d) = ndi[d].first();     // Default origin at ndi[d].first()
    // default mesh spacing from stride()
    for (i=0; i < gridSizes[d]-1; i++) {
      (meshSpacing[d])[i] = ndi[d].stride();
      (meshPosition[d])[i] = MFLOAT(i);
    }
    (meshPosition[d])[gridSizes[d]-1] = MFLOAT(gridSizes[d]-1);
  }
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const NDIndex<Dim>& ndi, MFLOAT** const delX)
{
  unsigned int d;
  for (d=0; d<Dim; d++)
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  setup();                          // Setup chores, such as array allocations
  for (d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective;     // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective;     // Default mesh: reflective boundary conds
    origin(d) = ndi[d].first();     // Default origin at ndi[d].first()
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const NDIndex<Dim>& ndi, MFLOAT** const delX,
           const Vektor<MFLOAT,Dim>& orig)
{
  int d;
  for (d=0; d<Dim; d++)
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  setup();                          // Setup chores, such as array allocations
  for (d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective;     // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective;     // Default mesh: reflective boundary conds
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify a MeshBC_E array for mesh boundary conditions:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const NDIndex<Dim>& ndi, MFLOAT** const delX,
           const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc)
{
  int d;
  for (d=0; d<Dim; d++)
    gridSizes[d] = ndi[d].length(); // Number of vertices along this dimension.
  setup();                          // Setup chores, such as array allocations
  set_origin(orig);           // Set origin.
  set_MeshBC(mbc);            // Set up mesh boundary conditions
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
//-----------------------------------------------------------------------------
// initialize using Index objects:
//-----------------------------------------------------------------------------

//===========1D============
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();      // Default origin at I.first()
  unsigned int i;
  // Default mesh spacing from stride()
  for (i=0; i < gridSizes[0]-1; i++) {
    (meshSpacing[0])[i] = I.stride();
    (meshPosition[0])[i] = MFLOAT(i);
  }
  (meshPosition[0])[gridSizes[0]-1] = MFLOAT(gridSizes[0]-1);
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, MFLOAT** const delX)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();      // Default origin at I.first()
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, MFLOAT** const delX,
           const Vektor<MFLOAT,Dim>& orig)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify a MeshBC_E array for mesh boundary conditions:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, MFLOAT** const delX,
           const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc)
{
  PInsist(Dim==1,"Number of Index arguments does not match mesh dimension!!");
  setup();
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  set_origin(orig);           // Set origin.
  set_MeshBC(mbc);            // Set up mesh boundary conditions
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}

//===========2D============
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();      // Default origin at I.first(),J.first()
  origin(1) = J.first();
  unsigned int i;
  // Default mesh spacing from stride()
  for (i=0; i < gridSizes[0]-1; i++) {
    (meshSpacing[0])[i] = I.stride();
    (meshPosition[0])[i] = MFLOAT(i);
  }
  (meshPosition[0])[gridSizes[0]-1] = MFLOAT(gridSizes[0]-1);
  for (i=0; i < gridSizes[1]-1; i++) {
    (meshSpacing[1])[i] = J.stride();
    (meshPosition[1])[i] = MFLOAT(i);
  }
  (meshPosition[1])[gridSizes[1]-1] = MFLOAT(gridSizes[1]-1);
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, MFLOAT** const delX)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();      // Default origin at I.first(),J.first()
  origin(1) = J.first();
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, MFLOAT** const delX,
           const Vektor<MFLOAT,Dim>& orig)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify a MeshBC_E array for mesh boundary conditions:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, MFLOAT** const delX,
           const Vektor<MFLOAT,Dim>& orig, MeshBC_E* const mbc)
{
  PInsist(Dim==2,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  set_origin(orig);           // Set origin.
  set_MeshBC(mbc);            // Set up mesh boundary conditions
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}

//===========3D============
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, const Index& K)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  // Setup chores, such as array allocations
  setup();
  origin(0) = I.first();    // Default origin at I.first(),J.first(),K.first()
  origin(1) = J.first();
  origin(2) = K.first();
  int i;
  // Default mesh spacing from stride()
  for (i=0; i < gridSizes[0]-1; i++) {
    (meshSpacing[0])[i] = I.stride();
    (meshPosition[0])[i] = MFLOAT(i);
  }
  (meshPosition[0])[gridSizes[0]-1] = MFLOAT(gridSizes[0]-1);
  for (i=0; i < gridSizes[1]-1; i++) {
    (meshSpacing[1])[i] = J.stride();
    (meshPosition[1])[i] = MFLOAT(i);
  }
  (meshPosition[1])[gridSizes[1]-1] = MFLOAT(gridSizes[1]-1);
  for (i=0; i < gridSizes[2]-1; i++) {
    (meshSpacing[2])[i] = K.stride();
    (meshPosition[2])[i] = MFLOAT(i);
  }
  (meshPosition[2])[gridSizes[2]-1] = MFLOAT(gridSizes[2]-1);

  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, const Index& K,
           MFLOAT** const delX)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  origin(0) = I.first();    // Default origin at I.first(),J.first(),K.first()
  origin(1) = J.first();
  origin(2) = K.first();
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify mesh spacings and origin:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, const Index& K,
           MFLOAT** const delX, const Vektor<MFLOAT,Dim>& orig)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  for (unsigned int d=0; d<Dim; d++) {
    MeshBC[2*d]   = Reflective; // Default mesh: reflective boundary conds
    MeshBC[2*d+1] = Reflective; // Default mesh: reflective boundary conds
  }
  set_origin(orig);           // Set origin.
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}
// Also specify a MeshBC_E array for mesh boundary conditions:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
initialize(const Index& I, const Index& J, const Index& K,
           MFLOAT** const delX, const Vektor<MFLOAT,Dim>& orig,
           MeshBC_E* const mbc)
{
  PInsist(Dim==3,"Number of Index arguments does not match mesh dimension!!");
  gridSizes[0] = I.length();  // Number of vertices along this dimension.
  gridSizes[1] = J.length();  // Number of vertices along this dimension.
  gridSizes[2] = K.length();  // Number of vertices along this dimension.
  setup();                    // Setup chores, such as array allocations
  set_origin(orig);           // Set origin.
  set_MeshBC(mbc);            // Set up mesh boundary conditions
  set_meshSpacing(delX);      // Set mesh spacings and compute cell volume
  set_Dvc();                  // Set derivative coefficients from spacings.
}

//-----------------------------------------------------------------------------
// Set/accessor functions for member data:
//-----------------------------------------------------------------------------
// Set the origin of mesh vertex positions:
template<unsigned Dim, class MFLOAT>
void Cartesian<Dim,MFLOAT>::
set_origin(const Vektor<MFLOAT,Dim>& o)
{
  origin = o;
  for (unsigned d=0; d<Dim; ++d) {
    (meshPosition[d])[0] = o(d);
    for (unsigned vert=1; vert<gridSizes[d]; ++vert) {
      (meshPosition[d])[vert] = (meshPosition[d])[vert-1] +
                                (meshSpacing[d])[vert-1];
    }
  }
  // Apply the current state of the mesh BC to add guards to meshPosition map:
  for (unsigned face=0; face < 2*Dim; ++face) updateMeshSpacingGuards(face);
  this->notifyOfChange();
}
// Get the origin of mesh vertex positions:
template<unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim> Cartesian<Dim,MFLOAT>::
get_origin() const
{
  return origin;
}

// Set the spacings of mesh vertex positions:
template<unsigned Dim, class MFLOAT>
void Cartesian<Dim,MFLOAT>::
set_meshSpacing(MFLOAT** const del)
{
  unsigned d, cell, face;

  for (d=0;d<Dim;++d) {
    (meshPosition[d])[0] = origin(d);
    for (cell=0; cell < gridSizes[d]-1; cell++) {
      (meshSpacing[d])[cell] = del[d][cell];
      (meshPosition[d])[cell+1] = (meshPosition[d])[cell] + del[d][cell];
    }
  }
  // Apply the current state of the mesh BC to add guards to meshSpacings map:
  for (face=0; face < 2*Dim; ++face) updateMeshSpacingGuards(face);
  // if spacing fields allocated, we must update values
  if (hasSpacingFields) storeSpacingFields();
  this->notifyOfChange();
}

// Set only the derivative constants, using pre-set spacings:
template<unsigned Dim, class MFLOAT>
void Cartesian<Dim,MFLOAT>::
set_Dvc()
{
  unsigned d;
  MFLOAT coef = 1.0;
  for (d=1;d<Dim;++d) coef *= 0.5;

  for (d=0;d<Dim;++d) {
    MFLOAT dvc = coef;
    for (unsigned b=0; b<(1<<Dim); ++b) {
      int s = ( b&(1<<d) ) ? 1 : -1;
      Dvc[b](d) = s*dvc;
    }
  }
}

// Get the spacings of mesh vertex positions along specified direction:
template<unsigned Dim, class MFLOAT>
void Cartesian<Dim,MFLOAT>::
get_meshSpacing(unsigned d, MFLOAT* spacings) const
{
  PAssert_LT(d, Dim);
  for (unsigned int cell=0; cell < gridSizes[d]-1; cell++)
    spacings[cell] = (*(meshSpacing[d].find(cell))).second;
  return;
}
//leak template<unsigned Dim, class MFLOAT>
//leak MFLOAT* Cartesian<Dim,MFLOAT>::
//leak get_meshSpacing(unsigned d) const
//leak {
//leak   PAssert_LT(d, Dim);
//leak   MFLOAT* theMeshSpacing = new MFLOAT[gridSizes[d]-1];
//leak   for (int cell=0; cell < gridSizes[d]-1; cell++)
//leak     theMeshSpacing[cell] = (*(meshSpacing[d].find(cell))).second;
//leak   return theMeshSpacing;
//leak }

///////////////////////////////////////////////////////////////////////////////

// Applicative templates for Mesh BC PETE_apply() functions, used
// by BrickExpression in storeSpacingFields()

// Periodic:
template<class T>
struct OpMeshPeriodic
{
};
template<class T>
inline void PETE_apply(OpMeshPeriodic<T> /*e*/, T& a, T b) { a = b; }

// Reflective/None:
template<class T>
struct OpMeshExtrapolate
{
  OpMeshExtrapolate(T& o, T& s) : Offset(o), Slope(s) {}
  T Offset, Slope;
};
// template<class T>
// inline void apply(OpMeshExtrapolate<T> e, T& a, T b)
template<class T>
inline void PETE_apply(OpMeshExtrapolate<T> e, T& a, T b)
{
  a = b*e.Slope+e.Offset;
}

///////////////////////////////////////////////////////////////////////////////

// Create BareField's of vertex and cell spacings
// Special prototypes taking no args or FieldLayout ctor args:
// No-arg case:
template<unsigned Dim, class MFLOAT>
void Cartesian<Dim,MFLOAT>::
storeSpacingFields()
{
  // Set up default FieldLayout parameters:
  e_dim_tag et[Dim];
  for (unsigned int d=0; d<Dim; d++) et[d] = PARALLEL;
  storeSpacingFields(et, -1);
}
// 1D
template<unsigned Dim, class MFLOAT>
void Cartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag p1, int vnodes)
{
  e_dim_tag et[1];
  et[0] = p1;
  storeSpacingFields(et, vnodes);
}
// 2D
template<unsigned Dim, class MFLOAT>
void Cartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag p1, e_dim_tag p2, int vnodes)
{
  e_dim_tag et[2];
  et[0] = p1;
  et[1] = p2;
  storeSpacingFields(et, vnodes);
}
// 3D
template<unsigned Dim, class MFLOAT>
void Cartesian<Dim,MFLOAT>::
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
void Cartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag* et, int vnodes)
{
  unsigned int d;
  int currentLocation[Dim];
  NDIndex<Dim> cells, verts;
  for (d=0; d<Dim; d++) {
    cells[d] = Index(gridSizes[d]-1);
    verts[d] = Index(gridSizes[d]);
  }
  if (!hasSpacingFields) {
    // allocate layouts and spacing fields
    FlCell = new FieldLayout<Dim>(cells, et, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    VertSpacings =
      new BareField<Vektor<MFLOAT,Dim>,Dim>(*FlCell,GuardCellSizes<Dim>(1));
    FlVert = new FieldLayout<Dim>(verts, et, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    CellSpacings =
      new BareField<Vektor<MFLOAT,Dim>,Dim>(*FlVert,GuardCellSizes<Dim>(1));
  }
  // VERTEX-VERTEX SPACINGS:
  BareField<Vektor<MFLOAT,Dim>,Dim>& vertSpacings = *VertSpacings;
  Vektor<MFLOAT,Dim> vertexSpacing;
  vertSpacings.Uncompress(); // Must do this prior to assign via iterator
  typename BareField<Vektor<MFLOAT,Dim>,Dim>::iterator cfi,
    cfi_end = vertSpacings.end();
  for (cfi = vertSpacings.begin(); cfi != cfi_end; ++cfi) {
    cfi.GetCurrentLocation(currentLocation);
    for (d=0; d<Dim; d++)
      vertexSpacing(d) = (*(meshSpacing[d].find(currentLocation[d]))).second;
    *cfi = vertexSpacing;
  }
  // CELL-CELL SPACINGS:
  BareField<Vektor<MFLOAT,Dim>,Dim>& cellSpacings = *CellSpacings;
  Vektor<MFLOAT,Dim> cellSpacing;
  cellSpacings.Uncompress(); // Must do this prior to assign via iterator
  typename BareField<Vektor<MFLOAT,Dim>,Dim>::iterator vfi,
    vfi_end = cellSpacings.end();
  for (vfi = cellSpacings.begin(); vfi != vfi_end; ++vfi) {
    vfi.GetCurrentLocation(currentLocation);
    for (d=0; d<Dim; d++)
      cellSpacing(d) = 0.5 * ((meshSpacing[d])[currentLocation[d]] +
                              (meshSpacing[d])[currentLocation[d]-1]);
    *vfi = cellSpacing;
  }
  //-------------------------------------------------
  // Now the hard part, filling in the guard cells:
  //-------------------------------------------------
  // The easy part of the hard part is filling so that all the internal
  // guard layers are right:
  cellSpacings.fillGuardCells();
  vertSpacings.fillGuardCells();
  // The hard part of the hard part is filling the external guard layers,
  // using the mesh BC to figure out how:
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // Temporaries used in loop over faces
  Vektor<MFLOAT,Dim> v0,v1; v0 = 0.0; v1 = 1.0; // Used for Reflective mesh BC
  typedef Vektor<MFLOAT,Dim> T;          // Used multipple places in loop below
  typename BareField<T,Dim>::iterator_if cfill_i; // Iterator used below
  typename BareField<T,Dim>::iterator_if vfill_i; // Iterator used below
  int coffset, voffset; // Pointer offsets used with LField::iterator below
  MeshBC_E bct;         // Scalar value of mesh BC used for each face in loop
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  for (unsigned int face=0; face < 2*Dim; face++) {
    // NDIndex's spanning elements and guard elements:
    NDIndex<Dim> cSlab = AddGuardCells(verts,cellSpacings.getGuardCellSizes());
    NDIndex<Dim> vSlab = AddGuardCells(cells,vertSpacings.getGuardCellSizes());
    // Shrink it down to be the guards along the active face:
    d = face/2;
    // The following bitwise AND logical test returns true if face is odd
    // (meaning the "high" or "right" face in the numbering convention) and
    // returns false if face is even (meaning the "low" or "left" face in
    // the numbering convention):
    if ( face & 1 ) {
      cSlab[d] = Index(verts[d].max() + 1,
                       verts[d].max() + cellSpacings.rightGuard(d));
      vSlab[d] = Index(cells[d].max() + 1,
                       cells[d].max() + vertSpacings.rightGuard(d));
    } else {
      cSlab[d] = Index(verts[d].min() - cellSpacings.leftGuard(d),
                       verts[d].min() - 1);
      vSlab[d] = Index(cells[d].min() - vertSpacings.leftGuard(d),
                       cells[d].min() - 1);
    }
    // Compute pointer offsets used with LField::iterator below:
    switch (MeshBC[face]) {
    case Periodic:
      bct = Periodic;
      if ( face & 1 ) {
        coffset = -verts[d].length();
        voffset = -cells[d].length();
      } else {
        coffset = verts[d].length();
        voffset = cells[d].length();
      }
      break;
    case Reflective:
      bct = Reflective;
      if ( face & 1 ) {
        coffset = 2*verts[d].max() + 1;
        voffset = 2*cells[d].max() + 1 - 1;
      } else {
        coffset = 2*verts[d].min() - 1;
        voffset = 2*cells[d].min() - 1 + 1;
      }
      break;
    case NoBC:
      bct = NoBC;
      if ( face & 1 ) {
        coffset = 2*verts[d].max() + 1;
        voffset = 2*cells[d].max() + 1 - 1;
      } else {
        coffset = 2*verts[d].min() - 1;
        voffset = 2*cells[d].min() - 1 + 1;
      }
      break;
    default:
        throw IpplException("Cartesian::storeSpacingFields", "unknown MeshBC type");
    }

    // Loop over all the LField's in the BareField's:
    // +++++++++++++++cellSpacings++++++++++++++
    for (cfill_i=cellSpacings.begin_if();
         cfill_i!=cellSpacings.end_if(); ++cfill_i)
      {
        // Cache some things we will use often below.
        // Pointer to the data for the current LField (right????):
        LField<T,Dim> &fill = *(*cfill_i).second;
        // NDIndex spanning all elements in the LField, including the guards:
        const NDIndex<Dim> &fill_alloc = fill.getAllocated();
        // If the previously-created boundary guard-layer NDIndex "cSlab"
        // contains any of the elements in this LField (they will be guard
        // elements if it does), assign the values into them here by applying
        // the boundary condition:
        if ( cSlab.touches( fill_alloc ) )
          {
            // Find what it touches in this LField.
            NDIndex<Dim> dest = cSlab.intersect( fill_alloc );

            // For exrapolation boundary conditions, the boundary guard-layer
            // elements are typically copied from interior values; the "src"
            // NDIndex specifies the interior elements to be copied into the
            // "dest" boundary guard-layer elements (possibly after some
            // mathematical operations like multipplying by minus 1 later):
            NDIndex<Dim> src = dest; // Create dest equal to src
            // Now calculate the interior elements; the coffset variable
            // computed above makes this right for "low" or "high" face cases:
            src[d] = coffset - src[d];

            // TJW: Why is there another loop over LField's here??????????
            // Loop over the ones that src touches.
            typename BareField<T,Dim>::iterator_if from_i;
            for (from_i=cellSpacings.begin_if();
                 from_i!=cellSpacings.end_if(); ++from_i)
              {
                // Cache a few things.
                LField<T,Dim> &from = *(*from_i).second;
                const NDIndex<Dim> &from_owned = from.getOwned();
                const NDIndex<Dim> &from_alloc = from.getAllocated();
                // If src touches this LField...
                if ( src.touches( from_owned ) )
                  {
                    NDIndex<Dim> from_it = src.intersect( from_alloc );
                    NDIndex<Dim> cfill_it = dest.plugBase( from_it );
                    // Build iterators for the copy...
                    typedef typename LField<T,Dim>::iterator LFI;
                    LFI lhs = fill.begin(cfill_it);
                    LFI rhs = from.begin(from_it);
                    // And do the assignment.
                    if (bct == Periodic) {
                      BrickExpression<Dim,LFI,LFI,OpMeshPeriodic<T> >
                        (lhs,rhs,OpMeshPeriodic<T>()).apply();
                    } else {
                      if (bct == Reflective) {
                        BrickExpression<Dim,LFI,LFI,OpMeshExtrapolate<T> >
                          (lhs,rhs,OpMeshExtrapolate<T>(v0,v1)).apply();
                      } else {
                        if (bct == NoBC) {
                          BrickExpression<Dim,LFI,LFI,OpMeshExtrapolate<T> >
                            (lhs,rhs,OpMeshExtrapolate<T>(v0,v0)).apply();
                        }
                      }
                    }
                  }
              }
          }
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
                    // And do the assignment.
                    if (bct == Periodic) {
                      BrickExpression<Dim,LFI,LFI,OpMeshPeriodic<T> >
                        (lhs,rhs,OpMeshPeriodic<T>()).apply();
                    } else {
                      if (bct == Reflective) {
                        BrickExpression<Dim,LFI,LFI,OpMeshExtrapolate<T> >
                          (lhs,rhs,OpMeshExtrapolate<T>(v0,v1)).apply();
                      } else {
                        if (bct == NoBC) {
                          BrickExpression<Dim,LFI,LFI,OpMeshExtrapolate<T> >
                            (lhs,rhs,OpMeshExtrapolate<T>(v0,v0)).apply();
                        }
                      }
                    }
                  }
              }
          }
      }

  }

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
void Cartesian<Dim,MFLOAT>::
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
void Cartesian<Dim,MFLOAT>::
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
void Cartesian<Dim,MFLOAT>::
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
void Cartesian<Dim,MFLOAT>::
storeSpacingFields(e_dim_tag *p,
                   unsigned* vnodesPerDirection,
                   bool recurse, int vnodes) {
  unsigned int d;
  int currentLocation[Dim];
  NDIndex<Dim> cells, verts;
  for (d=0; d<Dim; d++) {
    cells[d] = Index(gridSizes[d]-1);
    verts[d] = Index(gridSizes[d]);
  }
  if (!hasSpacingFields) {
    // allocate layouts and spacing fields
    FlCell =
      new FieldLayout<Dim>(cells, p, vnodesPerDirection, recurse, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    VertSpacings =
      new BareField<Vektor<MFLOAT,Dim>,Dim>(*FlCell,GuardCellSizes<Dim>(1));
    FlVert =
      new FieldLayout<Dim>(verts, p, vnodesPerDirection, recurse, vnodes);
    // Note: enough guard cells only for existing Div(), etc. implementations:
    CellSpacings =
      new BareField<Vektor<MFLOAT,Dim>,Dim>(*FlVert,GuardCellSizes<Dim>(1));
  }
  // VERTEX-VERTEX SPACINGS:
  BareField<Vektor<MFLOAT,Dim>,Dim>& vertSpacings = *VertSpacings;
  Vektor<MFLOAT,Dim> vertexSpacing;
  vertSpacings.Uncompress(); // Must do this prior to assign via iterator
  typename BareField<Vektor<MFLOAT,Dim>,Dim>::iterator cfi,
    cfi_end = vertSpacings.end();
  for (cfi = vertSpacings.begin(); cfi != cfi_end; ++cfi) {
    cfi.GetCurrentLocation(currentLocation);
    for (d=0; d<Dim; d++)
      vertexSpacing(d) = (*(meshSpacing[d].find(currentLocation[d]))).second;
    *cfi = vertexSpacing;
  }
  // CELL-CELL SPACINGS:
  BareField<Vektor<MFLOAT,Dim>,Dim>& cellSpacings = *CellSpacings;
  Vektor<MFLOAT,Dim> cellSpacing;
  cellSpacings.Uncompress(); // Must do this prior to assign via iterator
  typename BareField<Vektor<MFLOAT,Dim>,Dim>::iterator vfi,
    vfi_end = cellSpacings.end();
  for (vfi = cellSpacings.begin(); vfi != vfi_end; ++vfi) {
    vfi.GetCurrentLocation(currentLocation);
    for (d=0; d<Dim; d++)
      cellSpacing(d) = 0.5 * ((meshSpacing[d])[currentLocation[d]] +
                              (meshSpacing[d])[currentLocation[d]-1]);
    *vfi = cellSpacing;
  }
  //-------------------------------------------------
  // Now the hard part, filling in the guard cells:
  //-------------------------------------------------
  // The easy part of the hard part is filling so that all the internal
  // guard layers are right:
  cellSpacings.fillGuardCells();
  vertSpacings.fillGuardCells();
  // The hard part of the hard part is filling the external guard layers,
  // using the mesh BC to figure out how:
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // Temporaries used in loop over faces
  Vektor<MFLOAT,Dim> v0,v1; v0 = 0.0; v1 = 1.0; // Used for Reflective mesh BC
  unsigned int face;
  typedef Vektor<MFLOAT,Dim> T;          // Used multipple places in loop below
  typename BareField<T,Dim>::iterator_if cfill_i; // Iterator used below
  typename BareField<T,Dim>::iterator_if vfill_i; // Iterator used below
  int coffset, voffset; // Pointer offsets used with LField::iterator below
  MeshBC_E bct;         // Scalar value of mesh BC used for each face in loop
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  for (face=0; face < 2*Dim; face++) {
    // NDIndex's spanning elements and guard elements:
    NDIndex<Dim> cSlab = AddGuardCells(verts,cellSpacings.getGuardCellSizes());
    NDIndex<Dim> vSlab = AddGuardCells(cells,vertSpacings.getGuardCellSizes());
    // Shrink it down to be the guards along the active face:
    d = face/2;
    // The following bitwise AND logical test returns true if face is odd
    // (meaning the "high" or "right" face in the numbering convention) and
    // returns false if face is even (meaning the "low" or "left" face in
    // the numbering convention):
    if ( face & 1 ) {
      cSlab[d] = Index(verts[d].max() + 1,
                       verts[d].max() + cellSpacings.rightGuard(d));
      vSlab[d] = Index(cells[d].max() + 1,
                       cells[d].max() + vertSpacings.rightGuard(d));
    } else {
      cSlab[d] = Index(verts[d].min() - cellSpacings.leftGuard(d),
                       verts[d].min() - 1);
      vSlab[d] = Index(cells[d].min() - vertSpacings.leftGuard(d),
                       cells[d].min() - 1);
    }
    // Compute pointer offsets used with LField::iterator below:
    switch (MeshBC[face]) {
    case Periodic:
      bct = Periodic;
      if ( face & 1 ) {
        coffset = -verts[d].length();
        voffset = -cells[d].length();
      } else {
        coffset = verts[d].length();
        voffset = cells[d].length();
      }
      break;
    case Reflective:
      bct = Reflective;
      if ( face & 1 ) {
        coffset = 2*verts[d].max() + 1;
        voffset = 2*cells[d].max() + 1 - 1;
      } else {
        coffset = 2*verts[d].min() - 1;
        voffset = 2*cells[d].min() - 1 + 1;
      }
      break;
    case NoBC:
      bct = NoBC;
      if ( face & 1 ) {
        coffset = 2*verts[d].max() + 1;
        voffset = 2*cells[d].max() + 1 - 1;
      } else {
        coffset = 2*verts[d].min() - 1;
        voffset = 2*cells[d].min() - 1 + 1;
      }
      break;
    default:
      ERRORMSG("Cartesian::storeSpacingFields(): unknown MeshBC type" << endl);
      break;
    }

    // Loop over all the LField's in the BareField's:
    // +++++++++++++++cellSpacings++++++++++++++
    for (cfill_i=cellSpacings.begin_if();
         cfill_i!=cellSpacings.end_if(); ++cfill_i)
      {
        // Cache some things we will use often below.
        // Pointer to the data for the current LField (right????):
        LField<T,Dim> &fill = *(*cfill_i).second;
        // NDIndex spanning all elements in the LField, including the guards:
        const NDIndex<Dim> &fill_alloc = fill.getAllocated();
        // If the previously-created boundary guard-layer NDIndex "cSlab"
        // contains any of the elements in this LField (they will be guard
        // elements if it does), assign the values into them here by applying
        // the boundary condition:
        if ( cSlab.touches( fill_alloc ) )
          {
            // Find what it touches in this LField.
            NDIndex<Dim> dest = cSlab.intersect( fill_alloc );

            // For exrapolation boundary conditions, the boundary guard-layer
            // elements are typically copied from interior values; the "src"
            // NDIndex specifies the interior elements to be copied into the
            // "dest" boundary guard-layer elements (possibly after some
            // mathematical operations like multipplying by minus 1 later):
            NDIndex<Dim> src = dest; // Create dest equal to src
            // Now calculate the interior elements; the coffset variable
            // computed above makes this right for "low" or "high" face cases:
            src[d] = coffset - src[d];

            // TJW: Why is there another loop over LField's here??????????
            // Loop over the ones that src touches.
            typename BareField<T,Dim>::iterator_if from_i;
            for (from_i=cellSpacings.begin_if();
                 from_i!=cellSpacings.end_if(); ++from_i)
              {
                // Cache a few things.
                LField<T,Dim> &from = *(*from_i).second;
                const NDIndex<Dim> &from_owned = from.getOwned();
                const NDIndex<Dim> &from_alloc = from.getAllocated();
                // If src touches this LField...
                if ( src.touches( from_owned ) )
                  {
                    NDIndex<Dim> from_it = src.intersect( from_alloc );
                    NDIndex<Dim> cfill_it = dest.plugBase( from_it );
                    // Build iterators for the copy...
                    typedef typename LField<T,Dim>::iterator LFI;
                    LFI lhs = fill.begin(cfill_it);
                    LFI rhs = from.begin(from_it);
                    // And do the assignment.
                    if (bct == Periodic) {
                      BrickExpression<Dim,LFI,LFI,OpMeshPeriodic<T> >
                        (lhs,rhs,OpMeshPeriodic<T>()).apply();
                    } else {
                      if (bct == Reflective) {
                        BrickExpression<Dim,LFI,LFI,OpMeshExtrapolate<T> >
                          (lhs,rhs,OpMeshExtrapolate<T>(v0,v1)).apply();
                      } else {
                        if (bct == NoBC) {
                          BrickExpression<Dim,LFI,LFI,OpMeshExtrapolate<T> >
                            (lhs,rhs,OpMeshExtrapolate<T>(v0,v0)).apply();
                        }
                      }
                    }
                  }
              }
          }
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
                    // And do the assignment.
                    if (bct == Periodic) {
                      BrickExpression<Dim,LFI,LFI,OpMeshPeriodic<T> >
                        (lhs,rhs,OpMeshPeriodic<T>()).apply();
                    } else {
                      if (bct == Reflective) {
                        BrickExpression<Dim,LFI,LFI,OpMeshExtrapolate<T> >
                          (lhs,rhs,OpMeshExtrapolate<T>(v0,v1)).apply();
                      } else {
                        if (bct == NoBC) {
                          BrickExpression<Dim,LFI,LFI,OpMeshExtrapolate<T> >
                            (lhs,rhs,OpMeshExtrapolate<T>(v0,v0)).apply();
                        }
                      }
                    }
                  }
              }
          }
      }

  }

  hasSpacingFields = true; // Flag this as having been done to this object.
}


//-----------------------------------------------------------------------------
// I/O:
//-----------------------------------------------------------------------------
// Formatted output of Cartesian object:
template< unsigned Dim, class MFLOAT >
void
Cartesian<Dim,MFLOAT>::
print(std::ostream& out)
{
  unsigned int d;
  out << "======Cartesian<" << Dim << ",MFLOAT>==begin======" << std::endl;
  for (d=0; d < Dim; d++)
    out << "gridSizes[" << d << "] = " << gridSizes[d] << std::endl;
  out << "origin = " << origin << std::endl;
  for (d=0; d < Dim; d++) {
    out << "--------meshSpacing[" << d << "]---------" << std::endl;
    typename std::map<int,MFLOAT>::iterator mi;
    for (mi=meshSpacing[d].begin(); mi != meshSpacing[d].end(); ++mi) {
      out << "meshSpacing[" << d << "][" << (*mi).first << "] = "
          << (*mi).second << std::endl;
    }
  }
  for (unsigned b=0; b < (1<<Dim); b++)
    out << "Dvc[" << b << "] = " << Dvc[b] << std::endl;
  for (d=0; d < Dim; d++)
    out << "MeshBC[" << 2*d << "] = " << Mesh<Dim>::MeshBC_E_Names[MeshBC[2*d]]
        << " ; MeshBC[" << 2*d+1 << "] = " << Mesh<Dim>::MeshBC_E_Names[MeshBC[2*d+1]]
        << std::endl;
  out << "======Cartesian<" << Dim << ",MFLOAT>==end========" << std::endl;
}

//--------------------------------------------------------------------------
// Various (Cartesian) mesh mechanisms:
//--------------------------------------------------------------------------

// Volume of cell indexed by NDIndex:
template <unsigned Dim, class MFLOAT>
MFLOAT
Cartesian<Dim,MFLOAT>::
getCellVolume(const NDIndex<Dim>& ndi) const
{
  MFLOAT volume = 1.0;
  for (unsigned int d=0; d<Dim; d++)
    if (ndi[d].length() != 1) {
      ERRORMSG("Cartesian::getCellVolume() error: arg is not a NDIndex"
               << "specifying a single element" << endl);
    }
    else {
      volume *= (*(meshSpacing[d].find(ndi[d].first()))).second;
    }
  return volume;
}
// Field of volumes of all cells:
template <unsigned Dim, class MFLOAT>
Field<MFLOAT,Dim,Cartesian<Dim,MFLOAT>,Cell>&
Cartesian<Dim,MFLOAT>::
getCellVolumeField(Field<MFLOAT,Dim,Cartesian<Dim,MFLOAT>,Cell>& volumes) const
{
  // N.B.: here, other places taking Field& (in UniformCartesian, too), should
  // have check on domain of input Field& to make sure it's big enough to hold
  // all the values for this mesh object.
  volumes = 1.0;
  int currentLocation[Dim];
  volumes.Uncompress();
  // Iterate through all cells:
  typename Field<MFLOAT,Dim,Cartesian<Dim,MFLOAT>,Cell>::iterator fi,
    fi_end=volumes.end();
  for (fi = volumes.begin(); fi != fi_end; ++fi) {
    fi.GetCurrentLocation(currentLocation);
    for (unsigned int d=0; d<Dim; d++)
      *fi *= (*(meshSpacing[d].find(currentLocation[d]))).second;
  }
  return volumes;
}
// Volume of range of cells bounded by verticies specified by input NDIndex;
template <unsigned Dim, class MFLOAT>
MFLOAT
Cartesian<Dim,MFLOAT>::
getVertRangeVolume(const NDIndex<Dim>& ndi) const
{
  // Get vertex positions of extremal cells:
  Vektor<MFLOAT,Dim> v0, v1;
  unsigned int d;
  int i0, i1;
  for (d=0; d<Dim; d++) {
    i0 = ndi[d].first();
    if ( (i0 < -(int(gridSizes[d])-1)/2) ||
         (i0 > 3*(int(gridSizes[d])-1)/2) )
      ERRORMSG("Cartesian::getVertRangeVolume() error: " << ndi
               << " is an NDIndex ranging outside the mesh and guard layers;"
               << " not allowed." << endl);
    v0(d) = (*(meshPosition[d].find(i0))).second;
    i1 = ndi[d].last();
    if ( (i1 < -(int(gridSizes[d])-1)/2) ||
         (i1 > 3*(int(gridSizes[d])-1)/2) )
      ERRORMSG("Cartesian::getVertRangeVolume() error: " << ndi
               << " is an NDIndex ranging outside the mesh and guard layers;"
               << " not allowed." << endl);
    v1(d) = (*(meshPosition[d].find(i1))).second;
  }
  // Compute volume of rectangular solid beweeen these extremal vertices:
  MFLOAT volume = 1.0;
  for (d=0; d<Dim; d++) volume *= std::abs(v1(d) - v0(d));
  return volume;
}
// Volume of range of cells spanned by input NDIndex (index of cells):
template <unsigned Dim, class MFLOAT>
MFLOAT
Cartesian<Dim,MFLOAT>::
getCellRangeVolume(const NDIndex<Dim>& ndi) const
{
  // Get vertex positions bounding extremal cells:
  Vektor<MFLOAT,Dim> v0, v1;
  int i0, i1;
  for (unsigned int d=0; d<Dim; d++) {
    i0 = ndi[d].first();
    if ( (i0 < -(int(gridSizes[d])-1)/2) ||
         (i0 > 3*(int(gridSizes[d])-1)/2) )
      ERRORMSG("Cartesian::getCellRangeVolume() error: " << ndi
               << " is an NDIndex ranging outside the mesh and guard layers;"
               << " not allowed." << endl);
    v0(d) = (*(meshPosition[d].find(i0))).second;
    i1 = ndi[d].last()+1;
    if ( (i1 < -(int(gridSizes[d])-1)/2) ||
         (i1 > 3*(int(gridSizes[d])-1)/2) )
      ERRORMSG("Cartesian::getCellRangeVolume() error: " << ndi
               << " is an NDIndex ranging outside the mesh and guard layers;"
               << " not allowed." << endl);
    v1(d) = (*(meshPosition[d].find(i1))).second;
  }
  // Compute volume of rectangular solid beweeen these extremal vertices:
  MFLOAT volume = 1.0;
  for (unsigned int d=0; d<Dim; d++) volume *= std::abs(v1(d) - v0(d));
  return volume;
}

// Nearest vertex index to (x,y,z):
template <unsigned Dim, class MFLOAT>
NDIndex<Dim>
Cartesian<Dim,MFLOAT>::
getNearestVertex(const Vektor<MFLOAT,Dim>& x) const
{
  unsigned int d;
  Vektor<MFLOAT,Dim> boxMin, boxMax;
  for (d=0; d<Dim; d++) {
    int gs = (int(gridSizes[d])-1)/2;
    boxMin(d) = (*(meshPosition[d].find(-gs))).second;
    boxMax(d) = (*(meshPosition[d].find(3*gs))).second;
  }
  for (d=0; d<Dim; d++)
    if ( (x(d) < boxMin(d)) || (x(d) > boxMax(d)) )
      ERRORMSG("Cartesian::getNearestVertex() - input point is outside"
               << " mesh boundary and guard layers; not allowed." << endl);

  // Find coordinate vectors of the vertices just above and just below the
  // input point (extremal vertices on cell containing point);
  MFLOAT xVertexBelow, xVertexAbove, xVertex;
  int vertBelow, vertAbove, vertNearest[Dim];
  for (d=0; d<Dim; d++) {
    vertBelow = -(int(gridSizes[d])-1)/2;
    vertAbove = 3*(int(gridSizes[d])-1)/2;
    xVertexBelow = (*(meshPosition[d].find(vertBelow))).second;
    xVertexAbove = (*(meshPosition[d].find(vertAbove))).second;
    // check for out of bounds
    if (x(d) < xVertexBelow) {
      vertNearest[d] = vertBelow;
      continue;
    }
    if (x(d) > xVertexAbove) {
      vertNearest[d] = vertAbove;
      continue;
    }
    while (vertAbove > vertBelow+1) {
      vertNearest[d] = (vertAbove+vertBelow)/2;
      xVertex = (*(meshPosition[d].find(vertNearest[d]))).second;
      if (x(d) > xVertex) {
        vertBelow = vertNearest[d];
        xVertexBelow = xVertex;
      }
      else if (x(d) < xVertex) {
        vertAbove = vertNearest[d];
        xVertexAbove = xVertex;
      }
      else {  // found exact match!
        vertAbove = vertBelow;
      }
    }
    if (vertAbove != vertBelow) {
      if ((x(d)-xVertexBelow)<(xVertexAbove-x(d))) {
        vertNearest[d] = vertBelow;
      }
      else {
        vertNearest[d] = vertAbove;
      }
    }
  }

  // Construct the NDIndex for nearest vert get its position vector:
  NDIndex<Dim> ndi;
  for (d=0; d<Dim; d++) ndi[d] = Index(vertNearest[d],vertNearest[d],1);

  return ndi;
}
// Nearest vertex index with all vertex coordinates below (x,y,z):
template <unsigned Dim, class MFLOAT>
NDIndex<Dim>
Cartesian<Dim,MFLOAT>::
getVertexBelow(const Vektor<MFLOAT,Dim>& x) const
{
  unsigned int d;
  Vektor<MFLOAT,Dim> boxMin, boxMax;
  for (d=0; d<Dim; d++) {
    int gs = (int(gridSizes[d]) - 1)/2;
    boxMin(d) = (*(meshPosition[d].find(-gs))).second;
    boxMax(d) = (*(meshPosition[d].find(3*gs))).second;
  }
  for (d=0; d<Dim; d++)
    if ( (x(d) < boxMin(d)) || (x(d) > boxMax(d)) )
      ERRORMSG("Cartesian::getVertexBelow() - input point is outside"
               << " mesh boundary and guard layers; not allowed." << endl);

  // Find coordinate vectors of the vertices just below the input point;
  MFLOAT xVertexBelow, xVertexAbove, xVertex;
  int vertBelow, vertAbove, vertNearest[Dim];
  for (d=0; d<Dim; d++) {
    vertBelow = -(int(gridSizes[d])-1)/2;
    vertAbove = 3*(int(gridSizes[d])-1)/2;
    xVertexBelow = (*(meshPosition[d].find(vertBelow))).second;
    xVertexAbove = (*(meshPosition[d].find(vertAbove))).second;
    // check for out of bounds
    if (x(d) < xVertexBelow) {
      vertNearest[d] = vertBelow;
      continue;
    }
    if (x(d) > xVertexAbove) {
      vertNearest[d] = vertAbove;
      continue;
    }
    while (vertAbove > vertBelow+1) {
      vertNearest[d] = (vertAbove+vertBelow)/2;
      xVertex = (*(meshPosition[d].find(vertNearest[d]))).second;
      if (x(d) > xVertex) {
        vertBelow = vertNearest[d];
        xVertexBelow = xVertex;
      }
      else if (x(d) < xVertex) {
        vertAbove = vertNearest[d];
        xVertexAbove = xVertex;
      }
      else {  // found exact match!
        vertAbove = vertBelow;
      }
    }
    if (vertAbove != vertBelow) {
      vertNearest[d] = vertBelow;
    }
  }

  // Construct the NDIndex for nearest vert get its position vector:
  NDIndex<Dim> ndi;
  for (d=0; d<Dim; d++) ndi[d] = Index(vertNearest[d],vertNearest[d],1);

  return ndi;
}
// (x,y,z) coordinates of indexed vertex:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
Cartesian<Dim,MFLOAT>::
getVertexPosition(const NDIndex<Dim>& ndi) const
{
  unsigned int d;
  int i;
  Vektor<MFLOAT,Dim> vertexPosition;
  for (d=0; d<Dim; d++) {
    if (ndi[d].length() != 1)
      ERRORMSG("Cartesian::getVertexPosition() error: " << ndi
               << " is not an NDIndex specifying a single element" << endl);
    i = ndi[d].first();
    if ( (i < -(int(gridSizes[d])-1)/2) ||
         (i > 3*(int(gridSizes[d])-1)/2) )
      ERRORMSG("Cartesian::getVertexPosition() error: " << ndi
               << " is an NDIndex outside the mesh and guard layers;"
               << " not allowed." << endl);
    vertexPosition(d) = (*(meshPosition[d].find(i))).second;
  }
  return vertexPosition;
}
// Field of (x,y,z) coordinates of all vertices:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Vert>&
Cartesian<Dim,MFLOAT>::
getVertexPositionField(Field<Vektor<MFLOAT,Dim>,Dim,
                       Cartesian<Dim,MFLOAT>,Vert>& vertexPositions) const
{
  int currentLocation[Dim];
  Vektor<MFLOAT,Dim> vertexPosition;
  vertexPositions.Uncompress();
  typename Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Vert>::iterator fi,
    fi_end = vertexPositions.end();
  for (fi = vertexPositions.begin(); fi != fi_end; ++fi) {
    // Construct a NDIndex for each field element:
    fi.GetCurrentLocation(currentLocation);
    for (unsigned int d=0; d<Dim; d++) {
      vertexPosition(d) = (*(meshPosition[d].find(currentLocation[d]))).second;
    }
    *fi = vertexPosition;
  }
  return vertexPositions;
}

// (x,y,z) coordinates of indexed cell:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
Cartesian<Dim,MFLOAT>::
getCellPosition(const NDIndex<Dim>& ndi) const
{
  unsigned int d;
  int i;
  Vektor<MFLOAT,Dim> cellPosition;
  for (d=0; d<Dim; d++) {
    if (ndi[d].length() != 1)
      ERRORMSG("Cartesian::getCellPosition() error: " << ndi
               << " is not an NDIndex specifying a single element" << endl);
    i = ndi[d].first();
    if ( (i < -(int(gridSizes[d])-1)/2) ||
         (i >= 3*(int(gridSizes[d])-1)/2) )
      ERRORMSG("Cartesian::getCellPosition() error: " << ndi
               << " is an NDIndex outside the mesh and guard layers;"
               << " not allowed." << endl);
    cellPosition(d) = 0.5 * ( (*(meshPosition[d].find(i))).second +
                              (*(meshPosition[d].find(i+1))).second );
  }
  return cellPosition;
}
// Field of (x,y,z) coordinates of all cells:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Cell>&
Cartesian<Dim,MFLOAT>::
getCellPositionField(Field<Vektor<MFLOAT,Dim>,Dim,
                     Cartesian<Dim,MFLOAT>,Cell>& cellPositions) const
{
  int currentLocation[Dim];
  Vektor<MFLOAT,Dim> cellPosition;
  cellPositions.Uncompress();
  typename Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Cell>::iterator fi,
    fi_end = cellPositions.end();
  for (fi = cellPositions.begin(); fi != fi_end; ++fi) {
    // Construct a NDIndex for each field element:
    fi.GetCurrentLocation(currentLocation);
    for (unsigned int d=0; d<Dim; d++) {
      cellPosition(d) =
        0.5 * ( (*(meshPosition[d].find(currentLocation[d]))).second +
                (*(meshPosition[d].find(currentLocation[d]+1))).second );
    }
    *fi = cellPosition;
  }
  return cellPositions;
}

// Vertex-vertex grid spacing of indexed cell:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
Cartesian<Dim,MFLOAT>::
getDeltaVertex(const NDIndex<Dim>& ndi) const
{
  // return value
  Vektor<MFLOAT,Dim> vertexVertexSpacing(0);

  for (unsigned int d=0; d<Dim; d++) {
    // endpoints of the index range ... make sure they are in ascending order
    int a = ndi[d].first();
    int b = ndi[d].last();
    if (b < a) {
      int tmpa = a; a = b; b = tmpa;
    }

    // make sure we have valid endpoints
    if (a < -((int(gridSizes[d])-1)/2) || b >= 3*(int(gridSizes[d])-1)/2) {
      ERRORMSG("Cartesian::getDeltaVertex() error: " << ndi
               << " is an NDIndex ranging outside"
               << " the mesh and guard layers region; not allowed."
               << endl);
    }

    // add up all the values between the endpoints
    // N.B.: following may need modification to be right for periodic Mesh BC:
    while (a <= b)
      vertexVertexSpacing[d] += (*(meshSpacing[d].find(a++))).second;
  }

  return vertexVertexSpacing;
}

// Field of vertex-vertex grid spacings of all cells:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Cell>&
Cartesian<Dim,MFLOAT>::
getDeltaVertexField(Field<Vektor<MFLOAT,Dim>,Dim,
                    Cartesian<Dim,MFLOAT>,Cell>& vertexSpacings) const
{
  int currentLocation[Dim];
  Vektor<MFLOAT,Dim> vertexVertexSpacing;
  vertexSpacings.Uncompress();
  typename Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Cell>::iterator fi,
    fi_end = vertexSpacings.end();
  for (fi = vertexSpacings.begin(); fi != fi_end; ++fi) {
    fi.GetCurrentLocation(currentLocation);
    for (unsigned int d=0; d<Dim; d++)
      vertexVertexSpacing[d]=(*(meshSpacing[d].find(currentLocation[d]))).second;
    *fi = vertexVertexSpacing;
  }
  return vertexSpacings;
}

// Cell-cell grid spacing of indexed cell:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>
Cartesian<Dim,MFLOAT>::
getDeltaCell(const NDIndex<Dim>& ndi) const
{
  // return value
  Vektor<MFLOAT,Dim> cellCellSpacing(0);

  for (unsigned int d=0; d<Dim; d++) {
    // endpoints of the index range ... make sure they are in ascending order
    int a = ndi[d].first();
    int b = ndi[d].last();
    if (b < a) {
      int tmpa = a; a = b; b = tmpa;
    }

    // make sure the endpoints are valid
    if (a <= -(int(gridSizes[d])-1)/2 || b >= 3*(int(gridSizes[d])-1)/2) {
      ERRORMSG("Cartesian::getDeltaCell() error: " << ndi
               << " is an NDIndex ranging outside"
               << " the mesh and guard layers region; not allowed."
               << endl);
    }

    // add up the contributions along the interval ...
    while (a <= b) {
      cellCellSpacing[d] += ((*(meshSpacing[d].find(a))).second +
                             (*(meshSpacing[d].find(a-1))).second) * 0.5;
      a++;
    }
  }
  return cellCellSpacing;
}

// Field of cell-cell grid spacings of all cells:
template <unsigned Dim, class MFLOAT>
Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Vert>&
Cartesian<Dim,MFLOAT>::
getDeltaCellField(Field<Vektor<MFLOAT,Dim>,Dim,
                  Cartesian<Dim,MFLOAT>,Vert>& cellSpacings) const
{
  int currentLocation[Dim];
  Vektor<MFLOAT,Dim> cellCellSpacing;
  cellSpacings.Uncompress();
  typename Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Vert>::iterator fi,
    fi_end = cellSpacings.end();
  for (fi = cellSpacings.begin(); fi != fi_end; ++fi) {
    fi.GetCurrentLocation(currentLocation);
    for (unsigned int d=0; d<Dim; d++)
      cellCellSpacing[d]+=((*(meshSpacing[d].find(currentLocation[d]))).second +
                   (*(meshSpacing[d].find(currentLocation[d]-1))).second) * 0.5;
    *fi = cellCellSpacing;
  }
  return cellSpacings;
}
// Array of surface normals to cells adjoining indexed cell:
template <unsigned Dim, class MFLOAT>
Vektor<MFLOAT,Dim>*
Cartesian<Dim,MFLOAT>::
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
Cartesian<Dim,MFLOAT>::
getSurfaceNormalFields(Field<Vektor<MFLOAT,Dim>, Dim,
                       Cartesian<Dim,MFLOAT>,Cell>**
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
Cartesian<Dim,MFLOAT>::
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
Field<Vektor<MFLOAT,Dim>,Dim,Cartesian<Dim,MFLOAT>,Cell>&
Cartesian<Dim,MFLOAT>::
getSurfaceNormalField(Field<Vektor<MFLOAT,Dim>, Dim,
                      Cartesian<Dim,MFLOAT>,Cell>& surfaceNormalField,
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

// Set up mesh boundary conditions:
// Face specifies the mesh face, following usual numbering convention.
// MeshBC_E "type" specifies the kind of BC reflective/periodic/none.
// One face at a time:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
set_MeshBC(unsigned face, MeshBC_E meshBCType)
{
  MeshBC[face] = meshBCType;
  updateMeshSpacingGuards(face);
  // if spacing fields allocated, we must update values
  if (hasSpacingFields) storeSpacingFields();
}
// All faces at once:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
set_MeshBC(MeshBC_E* meshBCTypes)
{
  for (unsigned int face=0; face < 2*Dim; face++) {
    MeshBC[face] = meshBCTypes[face];
    updateMeshSpacingGuards(face);
  }
  // if spacing fields allocated, we must update values
  if (hasSpacingFields) storeSpacingFields();
}
// Helper function to update guard layer values of mesh spacings:
template <unsigned Dim, class MFLOAT>
void
Cartesian<Dim,MFLOAT>::
updateMeshSpacingGuards(int face)
{
  // Apply the current state of the mesh BC to add guards to meshSpacings map
  // Assume worst case of needing ngridpts/2 guard layers (for periodic, most
  // likely):
  int d = face/2;
  unsigned int cell, guardLayer;
  // The following bitwise AND logical test returns true if face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if face is even (meaning the "low" or "left" face in
  // the numbering convention):
  if ( face & 1 ) {
    // "High" guard cells:
    switch (MeshBC[d*2]) {
    case Periodic:
      for (guardLayer = 0; guardLayer <= (gridSizes[d]-1)/2; guardLayer++) {
        cell = gridSizes[d] - 1 + guardLayer;
        (meshSpacing[d])[cell] = (meshSpacing[d])[guardLayer];
        (meshPosition[d])[cell+1] = (meshPosition[d])[cell] +
                                    (meshSpacing[d])[cell];
      }
      break;
    case Reflective:
      for (guardLayer = 0; guardLayer <= (gridSizes[d]-1)/2; guardLayer++) {
        cell = gridSizes[d] - 1 + guardLayer;
        (meshSpacing[d])[cell] = (meshSpacing[d])[cell - guardLayer - 1];
        (meshPosition[d])[cell+1] = (meshPosition[d])[cell] +
                                    (meshSpacing[d])[cell];
      }
      break;
    case NoBC:
      for (guardLayer = 0; guardLayer <= (gridSizes[d]-1)/2; guardLayer++) {
        cell = gridSizes[d] - 1 + guardLayer;
        (meshSpacing[d])[cell] = 0;
        (meshPosition[d])[cell+1] = (meshPosition[d])[cell] +
                                    (meshSpacing[d])[cell];
      }
      break;
    default:
      ERRORMSG("Cartesian::updateMeshSpacingGuards(): unknown MeshBC type"
               << endl);
      break;
    }
  }
  else {
    // "Low" guard cells:
    switch (MeshBC[d]) {
    case Periodic:
      for (guardLayer = 0; guardLayer <= (gridSizes[d]-1)/2; guardLayer++) {
        cell = -1 - guardLayer;
        (meshSpacing[d])[cell] = (meshSpacing[d])[gridSizes[d] + cell];
        (meshPosition[d])[cell] = (meshPosition[d])[cell+1] -
                                  (meshSpacing[d])[cell];
      }
      break;
    case Reflective:
      for (guardLayer = 0; guardLayer <= (gridSizes[d]-1)/2; guardLayer++) {
        cell = -1 - guardLayer;
        (meshSpacing[d])[cell] = (meshSpacing[d])[-cell - 1];
        (meshPosition[d])[cell] = (meshPosition[d])[cell+1] -
                                  (meshSpacing[d])[cell];
      }
      break;
    case NoBC:
      for (guardLayer = 0; guardLayer <= (gridSizes[d]-1)/2; guardLayer++) {
        cell = -1 - guardLayer;
        (meshSpacing[d])[cell] = 0;
        (meshPosition[d])[cell] = (meshPosition[d])[cell+1] -
                                  (meshSpacing[d])[cell];
      }
      break;
    default:
      ERRORMSG("Cartesian::updateMeshSpacingGuards(): unknown MeshBC type"
               << endl);
      break;
    }
  }
}

// Get mesh boundary conditions:
// One face at a time
template <unsigned Dim, class MFLOAT>
MeshBC_E
Cartesian<Dim,MFLOAT>::
get_MeshBC(unsigned face) const
{
  MeshBC_E mb;
  mb = MeshBC[face];
  return mb;
}
// All faces at once
template <unsigned Dim, class MFLOAT>
MeshBC_E*
Cartesian<Dim,MFLOAT>::
get_MeshBC() const
{
  MeshBC_E* mb = new MeshBC_E[2*Dim];
  for (unsigned int b=0; b < 2*Dim; b++) mb[b] = MeshBC[b];
  return mb;
}



//--------------------------------------------------------------------------
// Global functions
//--------------------------------------------------------------------------


//*****************************************************************************
// Stuff taken from old Cartesian.h, modified for new nonuniform Cartesian:
//*****************************************************************************

//----------------------------------------------------------------------
// Divergence Vektor/Vert -> Scalar/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,Cartesian<1U,MFLOAT>,Cell>&
Div(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x,
    Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] =
    dot(x[I  ], x.get_mesh().Dvc[0]/vertSpacings[I]) +
    dot(x[I+1], x.get_mesh().Dvc[1]/vertSpacings[I]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,Cartesian<2U,MFLOAT>,Cell>&
Div(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x,
    Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] =
    dot(x[I  ][J  ], x.get_mesh().Dvc[0]/vertSpacings[I][J]) +
    dot(x[I+1][J  ], x.get_mesh().Dvc[1]/vertSpacings[I][J]) +
    dot(x[I  ][J+1], x.get_mesh().Dvc[2]/vertSpacings[I][J]) +
    dot(x[I+1][J+1], x.get_mesh().Dvc[3]/vertSpacings[I][J]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,Cartesian<3U,MFLOAT>,Cell>&
Div(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x,
    Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] =
    dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[0]/vertSpacings[I][J][K]) +
    dot(x[I+1][J  ][K  ], x.get_mesh().Dvc[1]/vertSpacings[I][J][K]) +
    dot(x[I  ][J+1][K  ], x.get_mesh().Dvc[2]/vertSpacings[I][J][K]) +
    dot(x[I+1][J+1][K  ], x.get_mesh().Dvc[3]/vertSpacings[I][J][K]) +
    dot(x[I  ][J  ][K+1], x.get_mesh().Dvc[4]/vertSpacings[I][J][K]) +
    dot(x[I+1][J  ][K+1], x.get_mesh().Dvc[5]/vertSpacings[I][J][K]) +
    dot(x[I  ][J+1][K+1], x.get_mesh().Dvc[6]/vertSpacings[I][J][K]) +
    dot(x[I+1][J+1][K+1], x.get_mesh().Dvc[7]/vertSpacings[I][J][K]);
  return r;
}
//----------------------------------------------------------------------
// Divergence Vektor/Cell -> Scalar/Vert
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,Cartesian<1U,MFLOAT>,Vert>&
Div(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x,
    Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] =
    dot(x[I-1], x.get_mesh().Dvc[0]/cellSpacings[I]) +
    dot(x[I  ], x.get_mesh().Dvc[1]/cellSpacings[I]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,Cartesian<2U,MFLOAT>,Vert>&
Div(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x,
    Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] =
    dot(x[I-1][J-1], x.get_mesh().Dvc[0]/cellSpacings[I][J]) +
    dot(x[I  ][J-1], x.get_mesh().Dvc[1]/cellSpacings[I][J]) +
    dot(x[I-1][J  ], x.get_mesh().Dvc[2]/cellSpacings[I][J]) +
    dot(x[I  ][J  ], x.get_mesh().Dvc[3]/cellSpacings[I][J]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,Cartesian<3U,MFLOAT>,Vert>&
Div(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x,
    Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] =
    dot(x[I-1][J-1][K-1], x.get_mesh().Dvc[0]/cellSpacings[I][J][K]) +
    dot(x[I  ][J-1][K-1], x.get_mesh().Dvc[1]/cellSpacings[I][J][K]) +
    dot(x[I-1][J  ][K-1], x.get_mesh().Dvc[2]/cellSpacings[I][J][K]) +
    dot(x[I  ][J  ][K-1], x.get_mesh().Dvc[3]/cellSpacings[I][J][K]) +
    dot(x[I-1][J-1][K  ], x.get_mesh().Dvc[4]/cellSpacings[I][J][K]) +
    dot(x[I  ][J-1][K  ], x.get_mesh().Dvc[5]/cellSpacings[I][J][K]) +
    dot(x[I-1][J  ][K  ], x.get_mesh().Dvc[6]/cellSpacings[I][J][K]) +
    dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[7]/cellSpacings[I][J][K]);
  return r;
}


// TJW: I've attempted to update these differential operators from uniform
// cartesian implementations to workfor (nonuniform) Cartesian class, but they
// may really need to get changed in other ways, such as something besides
// simple centered differncing for cell-cell and vert-vert cases. Flag these
// operator implementations since they may be a source of trouble until tested
// and debugged further. The Grad() operators, especially, may be wrong. All
// that being said, I have tested quite a few of these, including the following
// two needed by Tecolote: 1) Div SymTenzor/Cell->Vektor/Vert 2) Grad
// Vektor/Vert->Tenzor/Cell

// BEGIN FLAGGED DIFFOPS REGION I

//----------------------------------------------------------------------
// Divergence Vektor/Vert -> Scalar/Vert
// (Re-coded 1/20/1998 tjw. Hope it's right....???)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,Cartesian<1U,MFLOAT>,Vert>&
Div(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x,
    Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& vS = *(x.get_mesh().VertSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  Vektor<MFLOAT,1U> idx;
  idx[0] = 1.0;
  r[I] = dot(idx, (x[I+1] - x[I-1])/(vS[I  ] + vS[I-1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,Cartesian<2U,MFLOAT>,Vert>&
Div(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x,
    Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& vS = *(x.get_mesh().VertSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Vektor<MFLOAT,2U> idx,idy;
  idx[0] = 1.0;
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0;
  r[I][J] =
    dot(idx, (x[I+1][J  ] - x[I-1][J  ])/(vS[I  ][J  ] + vS[I-1][J  ])) +
    dot(idy, (x[I  ][J+1] - x[I  ][J-1])/(vS[I  ][J  ] + vS[I  ][J-1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,Cartesian<3U,MFLOAT>,Vert>&
Div(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x,
    Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& vS = *(x.get_mesh().VertSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  Vektor<MFLOAT,3U> idx,idy,idz;
  idx[0] = 1.0;
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0;
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 1.0;
  r[I][J][K] =
    dot(idx, ((x[I+1][J  ][K  ] - x[I-1][J  ][K  ])/
              (vS[I  ][J  ][K  ] + vS[I-1][J  ][K  ]))) +
    dot(idy, ((x[I  ][J+1][K  ] - x[I  ][J-1][K  ])/
              (vS[I  ][J  ][K  ] + vS[I  ][J-1][K  ]))) +
    dot(idz, ((x[I  ][J  ][K+1] - x[I  ][J  ][K-1])/
              (vS[I  ][J  ][K  ] + vS[I  ][J  ][K-1])));
  return r;
}
//----------------------------------------------------------------------
// Divergence Vektor/Cell -> Scalar/Cell (???right? tjw 3/10/97)
// (Re-coded 1/20/1998 tjw. Hope it's right....???)
// TJW 5/14/1999: I think there's a bug here, in the denominators. For example,
//                the 1D denom should be (cs[I+1] + cs[I]) as I see it.
//                This one wasn't in test/simple/TestCartesian.cpp.
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,1U,Cartesian<1U,MFLOAT>,Cell>&
Div(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x,
    Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& cS = *(x.get_mesh().CellSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  Vektor<MFLOAT,1U> idx;
  idx[0] = 1.0;
  r[I] = dot(idx, (x[I+1] - x[I-1])/(cS[I  ] + cS[I-1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,2U,Cartesian<2U,MFLOAT>,Cell>&
Div(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x,
    Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& cS = *(x.get_mesh().CellSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Vektor<MFLOAT,2U> idx,idy;
  idx[0] = 1.0;
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0;
  r[I][J] =
    dot(idx, (x[I+1][J  ] - x[I-1][J  ])/(cS[I  ][J  ] + cS[I-1][J  ])) +
    dot(idy, (x[I  ][J+1] - x[I  ][J-1])/(cS[I  ][J  ] + cS[I  ][J-1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<T,3U,Cartesian<3U,MFLOAT>,Cell>&
Div(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x,
    Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& cS = *(x.get_mesh().CellSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  Vektor<MFLOAT,3U> idx,idy,idz;
  idx[0] = 1.0;
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0;
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 1.0;
  r[I][J][K] =
    dot(idx, ((x[I+1][J  ][K  ] - x[I-1][J  ][K  ])/
              (cS[I  ][J  ][K  ] + cS[I-1][J  ][K  ]))) +
    dot(idy, ((x[I  ][J+1][K  ] - x[I  ][J-1][K  ])/
              (cS[I  ][J  ][K  ] + cS[I  ][J-1][K  ]))) +
    dot(idz, ((x[I  ][J  ][K+1] - x[I  ][J  ][K-1])/
              (cS[I  ][J  ][K  ] + cS[I  ][J  ][K-1])));
  return r;
}
//----------------------------------------------------------------------
// Divergence Tenzor/Vert -> Vektor/Cell (???dot right thing? tjw 1/20/1998)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x,
    Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] =
    dot(x[I  ], x.get_mesh().Dvc[0]/vertSpacings[I]) +
    dot(x[I+1], x.get_mesh().Dvc[1]/vertSpacings[I]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x,
    Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] =
    dot(x[I  ][J  ], x.get_mesh().Dvc[0]/vertSpacings[I][J]) +
    dot(x[I+1][J  ], x.get_mesh().Dvc[1]/vertSpacings[I][J]) +
    dot(x[I  ][J+1], x.get_mesh().Dvc[2]/vertSpacings[I][J]) +
    dot(x[I+1][J+1], x.get_mesh().Dvc[3]/vertSpacings[I][J]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>&
Div(Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x,
    Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] =
    dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[0]/vertSpacings[I][J][K]) +
    dot(x[I+1][J  ][K  ], x.get_mesh().Dvc[1]/vertSpacings[I][J][K]) +
    dot(x[I  ][J+1][K  ], x.get_mesh().Dvc[2]/vertSpacings[I][J][K]) +
    dot(x[I+1][J+1][K  ], x.get_mesh().Dvc[3]/vertSpacings[I][J][K]) +
    dot(x[I  ][J  ][K+1], x.get_mesh().Dvc[4]/vertSpacings[I][J][K]) +
    dot(x[I+1][J  ][K+1], x.get_mesh().Dvc[5]/vertSpacings[I][J][K]) +
    dot(x[I  ][J+1][K+1], x.get_mesh().Dvc[6]/vertSpacings[I][J][K]) +
    dot(x[I+1][J+1][K+1], x.get_mesh().Dvc[7]/vertSpacings[I][J][K]);
  return r;
}
//----------------------------------------------------------------------
// Divergence SymTenzor/Vert -> Vektor/Cell (???dot right thing? tjw 1/20/1998)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>&
Div(Field<SymTenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x,
    Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] =
    dot(x[I  ], x.get_mesh().Dvc[0]/vertSpacings[I]) +
    dot(x[I+1], x.get_mesh().Dvc[1]/vertSpacings[I]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>&
Div(Field<SymTenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x,
    Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] =
    dot(x[I  ][J  ], x.get_mesh().Dvc[0]/vertSpacings[I][J]) +
    dot(x[I+1][J  ], x.get_mesh().Dvc[1]/vertSpacings[I][J]) +
    dot(x[I  ][J+1], x.get_mesh().Dvc[2]/vertSpacings[I][J]) +
    dot(x[I+1][J+1], x.get_mesh().Dvc[3]/vertSpacings[I][J]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>&
Div(Field<SymTenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x,
    Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] =
    dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[0]/vertSpacings[I][J][K]) +
    dot(x[I+1][J  ][K  ], x.get_mesh().Dvc[1]/vertSpacings[I][J][K]) +
    dot(x[I  ][J+1][K  ], x.get_mesh().Dvc[2]/vertSpacings[I][J][K]) +
    dot(x[I+1][J+1][K  ], x.get_mesh().Dvc[3]/vertSpacings[I][J][K]) +
    dot(x[I  ][J  ][K+1], x.get_mesh().Dvc[4]/vertSpacings[I][J][K]) +
    dot(x[I+1][J  ][K+1], x.get_mesh().Dvc[5]/vertSpacings[I][J][K]) +
    dot(x[I  ][J+1][K+1], x.get_mesh().Dvc[6]/vertSpacings[I][J][K]) +
    dot(x[I+1][J+1][K+1], x.get_mesh().Dvc[7]/vertSpacings[I][J][K]);
  return r;
}

//----------------------------------------------------------------------
// Divergence Tenzor/Cell -> Vektor/Vert (???dot right thing? tjw 1/20/1998)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x,
    Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] =
    dot(x[I-1], x.get_mesh().Dvc[0]/cellSpacings[I]) +
    dot(x[I  ], x.get_mesh().Dvc[1]/cellSpacings[I]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x,
    Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] =
    dot(x[I-1][J-1], x.get_mesh().Dvc[0]/cellSpacings[I][J]) +
    dot(x[I  ][J-1], x.get_mesh().Dvc[1]/cellSpacings[I][J]) +
    dot(x[I-1][J  ], x.get_mesh().Dvc[2]/cellSpacings[I][J]) +
    dot(x[I  ][J  ], x.get_mesh().Dvc[3]/cellSpacings[I][J]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>&
Div(Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x,
    Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] =
    dot(x[I-1][J-1][K-1], x.get_mesh().Dvc[0]/cellSpacings[I][J][K]) +
    dot(x[I  ][J-1][K-1], x.get_mesh().Dvc[1]/cellSpacings[I][J][K]) +
    dot(x[I-1][J  ][K-1], x.get_mesh().Dvc[2]/cellSpacings[I][J][K]) +
    dot(x[I  ][J  ][K-1], x.get_mesh().Dvc[3]/cellSpacings[I][J][K]) +
    dot(x[I-1][J-1][K  ], x.get_mesh().Dvc[4]/cellSpacings[I][J][K]) +
    dot(x[I  ][J-1][K  ], x.get_mesh().Dvc[5]/cellSpacings[I][J][K]) +
    dot(x[I-1][J  ][K  ], x.get_mesh().Dvc[6]/cellSpacings[I][J][K]) +
    dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[7]/cellSpacings[I][J][K]);
  return r;
}

//----------------------------------------------------------------------
// Divergence SymTenzor/Cell -> Vektor/Vert (???dot right thing? tjw 1/20/1998)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x,
    Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] =
    dot(x[I-1], x.get_mesh().Dvc[0]/cellSpacings[I]) +
    dot(x[I  ], x.get_mesh().Dvc[1]/cellSpacings[I]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x,
    Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] =
    dot(x[I-1][J-1], x.get_mesh().Dvc[0]/cellSpacings[I][J]) +
    dot(x[I  ][J-1], x.get_mesh().Dvc[1]/cellSpacings[I][J]) +
    dot(x[I-1][J  ], x.get_mesh().Dvc[2]/cellSpacings[I][J]) +
    dot(x[I  ][J  ], x.get_mesh().Dvc[3]/cellSpacings[I][J]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>&
Div(Field<SymTenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x,
    Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] =
    dot(x[I-1][J-1][K-1], x.get_mesh().Dvc[0]/cellSpacings[I][J][K]) +
    dot(x[I  ][J-1][K-1], x.get_mesh().Dvc[1]/cellSpacings[I][J][K]) +
    dot(x[I-1][J  ][K-1], x.get_mesh().Dvc[2]/cellSpacings[I][J][K]) +
    dot(x[I  ][J  ][K-1], x.get_mesh().Dvc[3]/cellSpacings[I][J][K]) +
    dot(x[I-1][J-1][K  ], x.get_mesh().Dvc[4]/cellSpacings[I][J][K]) +
    dot(x[I  ][J-1][K  ], x.get_mesh().Dvc[5]/cellSpacings[I][J][K]) +
    dot(x[I-1][J  ][K  ], x.get_mesh().Dvc[6]/cellSpacings[I][J][K]) +
    dot(x[I  ][J  ][K  ], x.get_mesh().Dvc[7]/cellSpacings[I][J][K]);
  return r;
}

// END FLAGGED DIFFOPS REGION I

//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>&
Grad(Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& x,
     Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] = (x[I  ]*x.get_mesh().Dvc[0] +
          x[I+1]*x.get_mesh().Dvc[1])/vertSpacings[I];
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2u>,2U,Cartesian<2U,MFLOAT>,Cell>&
Grad(Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& x,
     Field<Vektor<T,2u>,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] = (x[I  ][J  ]*x.get_mesh().Dvc[0] +
             x[I+1][J  ]*x.get_mesh().Dvc[1] +
             x[I  ][J+1]*x.get_mesh().Dvc[2] +
             x[I+1][J+1]*x.get_mesh().Dvc[3])/vertSpacings[I][J];
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>&
Grad(Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& vertSpacings = *(x.get_mesh().VertSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] = (x[I  ][J  ][K  ]*x.get_mesh().Dvc[0] +
                x[I+1][J  ][K  ]*x.get_mesh().Dvc[1] +
                x[I  ][J+1][K  ]*x.get_mesh().Dvc[2] +
                x[I+1][J+1][K  ]*x.get_mesh().Dvc[3] +
                x[I  ][J  ][K+1]*x.get_mesh().Dvc[4] +
                x[I+1][J  ][K+1]*x.get_mesh().Dvc[5] +
                x[I  ][J+1][K+1]*x.get_mesh().Dvc[6] +
                x[I+1][J+1][K+1]*x.get_mesh().Dvc[7])/vertSpacings[I][J][K];
  return r;
}
//----------------------------------------------------------------------
// Grad Scalar/Cell -> Vektor/Vert
// (cellSpacings[I,J,K]->[I-1,....] 1/20/1998 tjw)
// (reverted to cellSpacings[I-1,J-1,K-1]->[I,....] 2/2/1998 tjw)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>&
Grad(Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& x,
     Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] = (x[I-1]*x.get_mesh().Dvc[0] +
          x[I  ]*x.get_mesh().Dvc[1])/cellSpacings[I];
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>&
Grad(Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& x,
     Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] = (x[I-1][J-1]*x.get_mesh().Dvc[0] +
             x[I  ][J-1]*x.get_mesh().Dvc[1] +
             x[I-1][J  ]*x.get_mesh().Dvc[2] +
             x[I  ][J  ]*x.get_mesh().Dvc[3])/cellSpacings[I][J];
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>&
Grad(Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& x,
     Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  Vektor<MFLOAT,3U> dvc[1<<3U];
  for (unsigned int d=0; d < 1<<3U; d++) dvc[d] = x.get_mesh().Dvc[d];
  r[I][J][K] = (x[I-1][J-1][K-1]*dvc[0] +
                x[I  ][J-1][K-1]*dvc[1] +
                x[I-1][J  ][K-1]*dvc[2] +
                x[I  ][J  ][K-1]*dvc[3] +
                x[I-1][J-1][K  ]*dvc[4] +
                x[I  ][J-1][K  ]*dvc[5] +
                x[I-1][J  ][K  ]*dvc[6] +
                x[I  ][J  ][K  ]*dvc[7])/
    cellSpacings[I][J][K];
  return r;
}

// TJW: I've attempted to update these differential operators from uniform
// cartesian implementations to workfor (nonuniform) Cartesian class, but they
// may really need to get changed in other ways, such as something besides
// simple centered differncing for cell-cell and vert-vert cases. Flag these
// operator implementations since they may be a source of trouble until tested
// and debugged further. The Grad() operators, especially, may be wrong. All
// that being said, I have tested quite a few of these, including the following
// two needed by Tecolote: 1) Div SymTenzor/Cell->Vektor/Vert 2) Grad
// Vektor/Vert->Tenzor/Cell

// BEGIN FLAGGED DIFFOPS REGION II

//----------------------------------------------------------------------
// Grad Scalar/Vert -> Vektor/Vert (???right? tjw 1/16/98)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>&
Grad(Field<T,1U,Cartesian<1U,MFLOAT>,Vert>& x,
     Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& vertSpacings =
    *(x.get_mesh().VertSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  Vektor<MFLOAT,1U> idx;
  idx[0] = 1.0;

  r[I] =  idx*((x[I+1] - x[I-1])/(vertSpacings[I] + vertSpacings[I-1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>&
Grad(Field<T,2U,Cartesian<2U,MFLOAT>,Vert>& x,
     Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& vertSpacings =
    *(x.get_mesh().VertSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  Vektor<MFLOAT,2U> idx,idy;
  idx[0] = 1.0;
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0;

  r[I][J] =
    idx*((x[I+1][J  ] - x[I-1][J  ])/
         (vertSpacings[I][J] + vertSpacings[I-1][J])) +
    idy*((x[I  ][J+1] - x[I  ][J-1])/
         (vertSpacings[I][J] + vertSpacings[I][J-1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>&
Grad(Field<T,3U,Cartesian<3U,MFLOAT>,Vert>& x,
     Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& vertSpacings =
    *(x.get_mesh().VertSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  Vektor<MFLOAT,3U> idx,idy,idz;
  idx[0] = 1.0;
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0;
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 1.0;

  r[I][J][K] =
    idx*((x[I+1][J  ][K  ] - x[I-1][J  ][K  ])/
         (vertSpacings[I][J][K] + vertSpacings[I-1][J][K])) +
    idy*((x[I  ][J+1][K  ] - x[I  ][J-1][K  ])/
         (vertSpacings[I][J][K] + vertSpacings[I][J-1][K])) +
    idz*((x[I  ][J  ][K+1] - x[I  ][J  ][K-1])/
         (vertSpacings[I][J][K] + vertSpacings[I][J][K-1]));
  return r;
}
//----------------------------------------------------------------------
// Grad Scalar/Cell -> Vektor/Cell
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>&
Grad(Field<T,1U,Cartesian<1U,MFLOAT>,Cell>& x,
     Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& cellSpacings =
    *(x.get_mesh().CellSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];

  Vektor<MFLOAT,1U> idx;
  idx[0] = 1.0;

  r[I] =  idx*((x[I+1] - x[I-1])/(cellSpacings[I] + cellSpacings[I+1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,2u>,2U,Cartesian<2U,MFLOAT>,Cell>&
Grad(Field<T,2U,Cartesian<2U,MFLOAT>,Cell>& x,
     Field<Vektor<T,2u>,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& cellSpacings =
    *(x.get_mesh().CellSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  Vektor<MFLOAT,2U> idx,idy;
  idx[0] = 1.0;
  idx[1] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0;

  r[I][J] =
    idx*((x[I+1][J  ] - x[I-1][J  ])/
         (cellSpacings[I][J] + cellSpacings[I+1][J])) +
    idy*((x[I  ][J+1] - x[I  ][J-1])/
         (cellSpacings[I][J] + cellSpacings[I][J+1]));
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>&
Grad(Field<T,3U,Cartesian<3U,MFLOAT>,Cell>& x,
     Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& cellSpacings =
    *(x.get_mesh().CellSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  Vektor<MFLOAT,3U> idx,idy,idz;
  idx[0] = 1.0;
  idx[1] = 0.0;
  idx[2] = 0.0;
  idy[0] = 0.0;
  idy[1] = 1.0;
  idy[2] = 0.0;
  idz[0] = 0.0;
  idz[1] = 0.0;
  idz[2] = 1.0;

  r[I][J][K] =
    idx*((x[I+1][J  ][K  ] - x[I-1][J  ][K  ])/
         (cellSpacings[I][J][K] + cellSpacings[I+1][J][K])) +
    idy*((x[I  ][J+1][K  ] - x[I  ][J-1][K  ])/
         (cellSpacings[I][J][K] + cellSpacings[I][J+1][K])) +
    idz*((x[I  ][J  ][K+1] - x[I  ][J  ][K-1])/
         (cellSpacings[I][J][K] + cellSpacings[I][J][K+1]));
  return r;
}
//----------------------------------------------------------------------
// Grad Vektor/Vert -> Tenzor/Cell (???o.p. right thing? tjw 1/20/1998)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& vS = *(x.get_mesh().VertSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] =
    outerProduct(x[I  ], x.get_mesh().Dvc[0]/vS[I]) +
    outerProduct(x[I+1], x.get_mesh().Dvc[1]/vS[I]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& vS = *(x.get_mesh().VertSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] =
    outerProduct(x[I  ][J  ], x.get_mesh().Dvc[0]/vS[I][J]) +
    outerProduct(x[I+1][J  ], x.get_mesh().Dvc[1]/vS[I][J]) +
    outerProduct(x[I  ][J+1], x.get_mesh().Dvc[2]/vS[I][J]) +
    outerProduct(x[I+1][J+1], x.get_mesh().Dvc[3]/vS[I][J]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>&
Grad(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& x,
     Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& vS = *(x.get_mesh().VertSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] =
    outerProduct(x[I  ][J  ][K  ], x.get_mesh().Dvc[0]/vS[I][J][K]) +
    outerProduct(x[I+1][J  ][K  ], x.get_mesh().Dvc[1]/vS[I][J][K]) +
    outerProduct(x[I  ][J+1][K  ], x.get_mesh().Dvc[2]/vS[I][J][K]) +
    outerProduct(x[I+1][J+1][K  ], x.get_mesh().Dvc[3]/vS[I][J][K]) +
    outerProduct(x[I  ][J  ][K+1], x.get_mesh().Dvc[4]/vS[I][J][K]) +
    outerProduct(x[I+1][J  ][K+1], x.get_mesh().Dvc[5]/vS[I][J][K]) +
    outerProduct(x[I  ][J+1][K+1], x.get_mesh().Dvc[6]/vS[I][J][K]) +
    outerProduct(x[I+1][J+1][K+1], x.get_mesh().Dvc[7]/vS[I][J][K]);

  return r;
}
//----------------------------------------------------------------------
// Grad Vektor/Cell -> Tenzor/Vert (???o.p. right thing? tjw 1/20/1998)
// (cellSpacings[I,J,K]->[I-1,....] 1/20/1998 tjw)
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,1U>,1U,Cartesian<1U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,1U>,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,1U>,1U>& cellSpacings = *(x.get_mesh().CellSpacings);
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] =
    outerProduct(x[I-1], x.get_mesh().Dvc[0]/cellSpacings[I-1]) +
    outerProduct(x[I  ], x.get_mesh().Dvc[1]/cellSpacings[I-1]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,2U>,2U,Cartesian<2U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,2U>,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,2U>,2U>& cS = *(x.get_mesh().CellSpacings);
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] =
    outerProduct(x[I-1][J-1], x.get_mesh().Dvc[0]/cS[I-1][J-1]) +
    outerProduct(x[I  ][J-1], x.get_mesh().Dvc[1]/cS[I-1][J-1]) +
    outerProduct(x[I-1][J  ], x.get_mesh().Dvc[2]/cS[I-1][J-1]) +
    outerProduct(x[I  ][J  ], x.get_mesh().Dvc[3]/cS[I-1][J-1]);
  return r;
}
//----------------------------------------------------------------------
template < class T, class MFLOAT >
Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>&
Grad(Field<Vektor<T,3U>,3U,Cartesian<3U,MFLOAT>,Cell>& x,
     Field<Tenzor<T,3U>,3U,Cartesian<3U,MFLOAT>,Vert>& r)
{
  PAssert_EQ(x.get_mesh().hasSpacingFields, true);
  BareField<Vektor<MFLOAT,3U>,3U>& cS = *(x.get_mesh().CellSpacings);
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];
  r[I][J][K] =
    outerProduct(x[I-1][J-1][K-1], x.get_mesh().Dvc[0]/cS[I-1][J-1][K-1]) +
    outerProduct(x[I  ][J-1][K-1], x.get_mesh().Dvc[1]/cS[I-1][J-1][K-1]) +
    outerProduct(x[I-1][J  ][K-1], x.get_mesh().Dvc[2]/cS[I-1][J-1][K-1]) +
    outerProduct(x[I  ][J  ][K-1], x.get_mesh().Dvc[3]/cS[I-1][J-1][K-1]) +
    outerProduct(x[I-1][J-1][K  ], x.get_mesh().Dvc[4]/cS[I-1][J-1][K-1]) +
    outerProduct(x[I  ][J-1][K  ], x.get_mesh().Dvc[5]/cS[I-1][J-1][K-1]) +
    outerProduct(x[I-1][J  ][K  ], x.get_mesh().Dvc[6]/cS[I-1][J-1][K-1]) +
    outerProduct(x[I  ][J  ][K  ], x.get_mesh().Dvc[7]/cS[I-1][J-1][K-1]);
  return r;
}

// END FLAGGED DIFFOPS REGION II

namespace IPPL {

//----------------------------------------------------------------------
// Weighted average Cell to Vert
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>&
Average(Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& x,
        Field<T2,1U,Cartesian<1U,MFLOAT>,Cell>& w,
        Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] = (x[I-1]*w[I-1] + x[I  ]*w[I  ])/(w[I-1] + w[I  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>&
Average(Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& x,
        Field<T2,2U,Cartesian<2U,MFLOAT>,Cell>& w,
        Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  r[I][J] = (x[I-1][J-1] * w[I-1][J-1] +
             x[I  ][J-1] * w[I  ][J-1] +
             x[I-1][J  ] * w[I-1][J  ] +
             x[I  ][J  ] * w[I  ][J  ])/
    (w[I-1][J-1] +
     w[I  ][J-1] +
     w[I-1][J  ] +
     w[I  ][J  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>&
Average(Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& x,
        Field<T2,3U,Cartesian<3U,MFLOAT>,Cell>& w,
        Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  r[I][J][K] = (x[I-1][J-1][K-1] * w[I-1][J-1][K-1] +
                x[I  ][J-1][K-1] * w[I  ][J-1][K-1] +
                x[I-1][J  ][K-1] * w[I-1][J  ][K-1] +
                x[I  ][J  ][K-1] * w[I  ][J  ][K-1] +
                x[I-1][J-1][K  ] * w[I-1][J-1][K  ] +
                x[I  ][J-1][K  ] * w[I  ][J-1][K  ] +
                x[I-1][J  ][K  ] * w[I-1][J  ][K  ] +
                x[I  ][J  ][K  ] * w[I  ][J  ][K  ] )/
    (w[I-1][J-1][K-1] +
     w[I  ][J-1][K-1] +
     w[I-1][J  ][K-1] +
     w[I  ][J  ][K-1] +
     w[I-1][J-1][K  ] +
     w[I  ][J-1][K  ] +
     w[I-1][J  ][K  ] +
     w[I  ][J  ][K  ]);
  return r;
}
//----------------------------------------------------------------------
// Weighted average Vert to Cell
// N.B.: won't work except for unit-stride (& zero-base?) Field's.
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>&
Average(Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& x,
        Field<T2,1U,Cartesian<1U,MFLOAT>,Vert>& w,
        Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] = (x[I+1]*w[I+1] + x[I  ]*w[I  ])/(w[I+1] + w[I  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>&
Average(Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& x,
        Field<T2,2U,Cartesian<2U,MFLOAT>,Vert>& w,
        Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];

  r[I][J] = (x[I+1][J+1] * w[I+1][J+1] +
             x[I  ][J+1] * w[I  ][J+1] +
             x[I+1][J  ] * w[I+1][J  ] +
             x[I  ][J  ] * w[I  ][J  ])/
    (w[I+1][J+1] +
     w[I  ][J+1] +
     w[I+1][J  ] +
     w[I  ][J  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class T2, class MFLOAT >
Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>&
Average(Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& x,
        Field<T2,3U,Cartesian<3U,MFLOAT>,Vert>& w,
        Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& r)
{
  const NDIndex<3U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  Index K = domain[2];

  r[I][J][K] = (x[I+1][J+1][K+1] * w[I+1][J+1][K+1] +
                x[I  ][J+1][K+1] * w[I  ][J+1][K+1] +
                x[I+1][J  ][K+1] * w[I+1][J  ][K+1] +
                x[I  ][J  ][K+1] * w[I  ][J  ][K+1] +
                x[I+1][J+1][K  ] * w[I+1][J+1][K  ] +
                x[I  ][J+1][K  ] * w[I  ][J+1][K  ] +
                x[I+1][J  ][K  ] * w[I+1][J  ][K  ] +
                x[I  ][J  ][K  ] * w[I  ][J  ][K  ])/
    (w[I+1][J+1][K+1] +
     w[I  ][J+1][K+1] +
     w[I+1][J  ][K+1] +
     w[I  ][J  ][K+1] +
     w[I+1][J+1][K  ] +
     w[I  ][J+1][K  ] +
     w[I+1][J  ][K  ] +
     w[I  ][J  ][K  ]);
  return r;
}

//----------------------------------------------------------------------
// Unweighted average Cell to Vert
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>&
Average(Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& x,
        Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] = 0.5*(x[I-1] + x[I  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>&
Average(Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& x,
        Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] = 0.25*(x[I-1][J-1] + x[I  ][J-1] + x[I-1][J  ] + x[I  ][J  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>&
Average(Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& x,
        Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& r)
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
Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>&
Average(Field<T1,1U,Cartesian<1U,MFLOAT>,Vert>& x,
        Field<T1,1U,Cartesian<1U,MFLOAT>,Cell>& r)
{
  const NDIndex<1U>& domain = r.getDomain();
  Index I = domain[0];
  r[I] = 0.5*(x[I+1] + x[I  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>&
Average(Field<T1,2U,Cartesian<2U,MFLOAT>,Vert>& x,
        Field<T1,2U,Cartesian<2U,MFLOAT>,Cell>& r)
{
  const NDIndex<2U>& domain = r.getDomain();
  Index I = domain[0];
  Index J = domain[1];
  r[I][J] = 0.25*(x[I+1][J+1] + x[I  ][J+1] + x[I+1][J  ] + x[I  ][J  ]);
  return r;
}
//----------------------------------------------------------------------
template < class T1, class MFLOAT >
Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>&
Average(Field<T1,3U,Cartesian<3U,MFLOAT>,Vert>& x,
        Field<T1,3U,Cartesian<3U,MFLOAT>,Cell>& r)
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

/***************************************************************************
 * $RCSfile: Cartesian.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: Cartesian.cpp,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $
 ***************************************************************************/