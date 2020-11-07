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


// include files
#include "FieldLayout/CenteredFieldLayout.h"
#include "Meshes/Centering.h"
#include "Meshes/CartesianCentering.h"
#include "Utility/PAssert.h"


//=============================================================================
// Helper global functions:
// The constructors call these specialized global functions as a workaround for
// lack of partial specialization:
//=============================================================================

//===========================Arbitrary mesh type=============================

//-----------------------------------------------------------------------------
// These specify only a total number of vnodes, allowing the constructor
// complete control on how to do the vnode partitioning of the index space:
// Constructor for arbitrary dimension with parallel/serial specifier array:

//------------------Cell centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Cell> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++)
    ndi[d] = ippl::Index(mesh.getGridsize(d) - 1);
  cfl.initialize(ndi, edt);
}

//------------------Vert centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Vert> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++)
    ndi[d] = ippl::Index(mesh.getGridsize(d));
  cfl.initialize(ndi, edt);
}

//------------------Edge centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Edge> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++)
    ndi[d] = ippl::Index(mesh.getGridsize(d));
  cfl.initialize(ndi, edt);
}

//------------------CartesianCentering centering-------------------------------
template<CenteringEnum* CE, unsigned Dim, class Mesh,
         unsigned NComponents>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,
		   CartesianCentering<CE,Dim,NComponents> > & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt)
{
  NDIndex<Dim> ndi;
  // For componentwise layout of Field of multicomponent object, like
  // Field<Vektor...>, allow for maximal number of objects needed per
  // dimension (the number for the object component requiring the maximal
  // number):
  unsigned npts[Dim], nGridPts;
  unsigned int d, c;
  for (d=0; d<Dim; d++) {
    nGridPts = mesh.getGridsize(d);
    npts[d] = 0;
    for (c=0; c<NComponents; c++) {
      if (CE[c + d*NComponents] == CELL) {
        npts[d] = std::max(npts[d], (nGridPts - 1));
      } else {
        npts[d] = std::max(npts[d], nGridPts);
      }
    }
  }
  for (d=0; d<Dim; d++) ndi[d] = ippl::Index(npts[d]);
  cfl.initialize(ndi, edt);
}

//------------------Cell centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Cell> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt,
		   unsigned* vnodesAlongDirection)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++) ndi[d] = ippl::Index(mesh.getGridsize(d) - 1);
  cfl.initialize(ndi, edt, vnodesAlongDirection);
}

//------------------Cell centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Cell> & cfl,
		   const Mesh& mesh,
		   const NDIndex<Dim> *dombegin,
		   const NDIndex<Dim> *domend,
		   const int *nbegin,
		   const int *nend)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++) ndi[d] = ippl::Index(mesh.getGridsize(d) - 1);
  cfl.initialize(ndi, dombegin, domend, nbegin, nend);
}

//------------------Vert centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Vert> & cfl,
		   const Mesh& mesh,
		   const NDIndex<Dim> *dombegin,
		   const NDIndex<Dim> *domend,
		   const int *nbegin,
		   const int *nend)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++) ndi[d] = ippl::Index(mesh.getGridsize(d));
  cfl.initialize(ndi, dombegin, domend, nbegin, nend);
}

//------------------Edge centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Edge> & cfl,
		   const Mesh& mesh,
		   const NDIndex<Dim> *dombegin,
		   const NDIndex<Dim> *domend,
		   const int *nbegin,
		   const int *nend)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++) ndi[d] = ippl::Index(mesh.getGridsize(d));
  cfl.initialize(ndi, dombegin, domend, nbegin, nend);
}

//------------------CartesianCentering centering-------------------------------
template<CenteringEnum* CE, unsigned Dim, class Mesh,
         unsigned NComponents>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,
		   CartesianCentering<CE,Dim,NComponents> > & cfl,
		   const Mesh& mesh,
		   const NDIndex<Dim> *dombegin,
		   const NDIndex<Dim> *domend,
		   const int *nbegin,
		   const int *nend)
{
  // For componentwise layout of Field of multicomponent object, like
  // Field<Vektor...>, allow for maximal number of objects needed per
  // dimension (the number for the object component requiring the maximal
  // number):
  unsigned npts[Dim], nGridPts;
  unsigned int d, c;
  for (d=0; d<Dim; d++) {
    nGridPts = mesh.getGridsize(d);
    npts[d] = 0;
    for (c=0; c<NComponents; c++) {
      if (CE[c + d*NComponents] == CELL) {
	npts[d] = max(npts[d], (nGridPts - 1));
      } else {
	npts[d] = max(npts[d], nGridPts);
      }
    }
  }

  NDIndex<Dim> ndi;
  for (d=0; d<Dim; d++) ndi[d] = ippl::Index(npts[d]);
  cfl.initialize(ndi, dombegin, domend, nbegin, nend);
}


//=============================================================================
// General ctor calls specializations of global function (workaround for lack
// of partial specialization:
//=============================================================================

// Constructor for arbitrary dimension with parallel/serial specifier array:
// This one also works if nothing except mesh is specified:
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag *p)
{

  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  centeredInitialize(*this, mesh, p);
}


// Constructors for 1 ... 6 dimensions with parallel/serial specifiers:
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1)
{

  PInsist(Dim==1,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  centeredInitialize(*this, mesh, &p1);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2)
{

  PInsist(Dim==2,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[2];
  edt[0] = p1; edt[1] = p2;
  centeredInitialize(*this, mesh, edt);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3)
{

  PInsist(Dim==3,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[3];
  edt[0] = p1; edt[1] = p2; edt[2] = p3;
  centeredInitialize(*this, mesh, edt);
}

//-----------------------------------------------------------------------------

// Constructor for arbitrary dimension with parallel/serial specifier array:
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag *p,
		    unsigned* vnodesAlongDirection)
{

  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  centeredInitialize(*this, mesh, p, vnodesAlongDirection);
}

//-----------------------------------------------------------------------------
// A constructor for a completely user-specified partitioning of the
// mesh space.

template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    const NDIndex<Dim> *dombegin,
		    const NDIndex<Dim> *domend,
		    const int *nbegin,
		    const int *nend)
{

  centeredInitialize(*this, mesh, dombegin, domend, nbegin, nend);
}