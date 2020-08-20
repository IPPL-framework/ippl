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
		   e_dim_tag* edt,
		   int vnodes)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++)
    ndi[d] = Index(mesh.gridSizes[d] - 1);
  cfl.initialize(ndi, edt, vnodes);
}

//------------------Vert centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Vert> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt,
		   int vnodes)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++)
    ndi[d] = Index(mesh.gridSizes[d]);
  cfl.initialize(ndi, edt, vnodes);
}

//------------------Edge centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Edge> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt,
		   int vnodes)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++)
    ndi[d] = Index(mesh.gridSizes[d]);
  cfl.initialize(ndi, edt, vnodes);
}

//------------------CartesianCentering centering-------------------------------
template<CenteringEnum* CE, unsigned Dim, class Mesh,
         unsigned NComponents>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,
		   CartesianCentering<CE,Dim,NComponents> > & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt,
		   int vnodes)
{
  NDIndex<Dim> ndi;
  // For componentwise layout of Field of multicomponent object, like
  // Field<Vektor...>, allow for maximal number of objects needed per
  // dimension (the number for the object component requiring the maximal
  // number):
  unsigned npts[Dim], nGridPts;
  unsigned int d, c;
  for (d=0; d<Dim; d++) {
    nGridPts = mesh.gridSizes[d];
    npts[d] = 0;
    for (c=0; c<NComponents; c++) {
      if (CE[c + d*NComponents] == CELL) {
        npts[d] = std::max(npts[d], (nGridPts - 1));
      } else {
        npts[d] = std::max(npts[d], nGridPts);
      }
    }
  }
  for (d=0; d<Dim; d++) ndi[d] = Index(npts[d]);
  cfl.initialize(ndi, edt, vnodes);
}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// These specify both the total number of vnodes and the numbers of vnodes
// along each dimension for the partitioning of the index space. Obviously this
// restricts the number of vnodes to be a product of the numbers along each
// dimension (the constructor implementation checks this):

//------------------Cell centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Cell> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt,
		   unsigned* vnodesAlongDirection,
		   bool recurse,
		   int vnodes)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++) ndi[d] = Index(mesh.gridSizes[d] - 1);
  cfl.initialize(ndi, edt, vnodesAlongDirection, recurse, vnodes);
}

//------------------Vert centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Vert> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt,
		   unsigned* vnodesAlongDirection,
		   bool recurse,
		   int vnodes)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++) ndi[d] = Index(mesh.gridSizes[d]);
  cfl.initialize(ndi, edt, vnodesAlongDirection, recurse, vnodes);
}

//------------------Edge centering---------------------------------------------
template<unsigned Dim, class Mesh>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,Edge> & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt,
		   unsigned* vnodesAlongDirection,
		   bool recurse,
		   int vnodes)
{
  NDIndex<Dim> ndi;
  for (unsigned int d=0; d<Dim; d++) ndi[d] = Index(mesh.gridSizes[d]);
  cfl.initialize(ndi, edt, vnodesAlongDirection, recurse, vnodes);
}

//------------------CartesianCentering centering-------------------------------
template<CenteringEnum* CE, unsigned Dim, class Mesh,
         unsigned NComponents>
inline void
centeredInitialize(CenteredFieldLayout<Dim,Mesh,
		   CartesianCentering<CE,Dim,NComponents> > & cfl,
		   const Mesh& mesh,
		   e_dim_tag* edt,
		   unsigned* vnodesAlongDirection,
		   bool recurse,
		   int vnodes)
{
  NDIndex<Dim> ndi;
  // For componentwise layout of Field of multicomponent object, like
  // Field<Vektor...>, allow for maximal number of objects needed per
  // dimension (the number for the object component requiring the maximal
  // number):
  unsigned npts[Dim], nGridPts;
  unsigned int d, c;
  for (d=0; d<Dim; d++) {
    nGridPts = mesh.gridSizes[d];
    npts[d] = 0;
    for (c=0; c<NComponents; c++) {
      if (CE[c + d*NComponents] == CELL) {
	npts[d] = max(npts[d], (nGridPts - 1));
      } else {
	npts[d] = max(npts[d], nGridPts);
      }
    }
  }
  for (d=0; d<Dim; d++) ndi[d] = Index(npts[d]);
  cfl.initialize(ndi, edt, vnodesAlongDirection, recurse, vnodes);
}

//-----------------------------------------------------------------------------
// This initializer just specifies a completely user-provided partitioning.

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
  for (unsigned int d=0; d<Dim; d++) ndi[d] = Index(mesh.gridSizes[d] - 1);
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
  for (unsigned int d=0; d<Dim; d++) ndi[d] = Index(mesh.gridSizes[d]);
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
  for (unsigned int d=0; d<Dim; d++) ndi[d] = Index(mesh.gridSizes[d]);
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
    nGridPts = mesh.gridSizes[d];
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
  for (d=0; d<Dim; d++) ndi[d] = Index(npts[d]);
  cfl.initialize(ndi, dombegin, domend, nbegin, nend);
}


//=============================================================================
// General ctor calls specializations of global function (workaround for lack
// of partial specialization:
//=============================================================================

//-----------------------------------------------------------------------------
// These specify only a total number of vnodes, allowing the constructor
// complete control on how to do the vnode partitioning of the index space:
// Constructor for arbitrary dimension with parallel/serial specifier array:

// Constructor for arbitrary dimension with parallel/serial specifier array:
// This one also works if nothing except mesh is specified:
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag *p,
		    int vnodes)
{

  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  centeredInitialize(*this, mesh, p, vnodes);
}


// Constructors for 1 ... 6 dimensions with parallel/serial specifiers:
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1,
		    int vnodes)
{

  PInsist(Dim==1,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  centeredInitialize(*this, mesh, &p1, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2,
		    int vnodes)
{

  PInsist(Dim==2,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[2];
  edt[0] = p1; edt[1] = p2;
  centeredInitialize(*this, mesh, edt, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
		    int vnodes)
{

  PInsist(Dim==3,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[3];
  edt[0] = p1; edt[1] = p2; edt[2] = p3;
  centeredInitialize(*this, mesh, edt, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		    int vnodes)
{

  PInsist(Dim==4,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[4];
  edt[0] = p1; edt[1] = p2; edt[2] = p3; edt[3] = p4;
  centeredInitialize(*this, mesh, edt, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		    e_dim_tag p5,
		    int vnodes)
{

  PInsist(Dim==5,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[5];
  edt[0] = p1; edt[1] = p2; edt[2] = p3; edt[3] = p4; edt[4] = p5;
  centeredInitialize(*this, mesh, edt, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		    e_dim_tag p5, e_dim_tag p6,
		    int vnodes)
{

  PInsist(Dim==6,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[6];
  edt[0] = p1; edt[1] = p2; edt[2] = p3; edt[3] = p4; edt[4] = p5; edt[5] = p6;
  centeredInitialize(*this, mesh, edt, vnodes);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// These specify both the total number of vnodes and the numbers of vnodes
// along each dimension for the partitioning of the index space. Obviously this
// restricts the number of vnodes to be a product of the numbers along each
// dimension (the constructor implementation checks this):

// Constructor for arbitrary dimension with parallel/serial specifier array:
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag *p,
		    unsigned* vnodesAlongDirection,
		    bool recurse,
		    int vnodes)
{

  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  centeredInitialize(*this, mesh, p, vnodesAlongDirection, recurse, vnodes);
}
// Constructors for 1 ... 6 dimensions with parallel/serial specifiers:
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1,
		    unsigned vnodes1,
		    bool recurse,
		    int vnodes)
{

  PInsist(Dim==1,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  centeredInitialize(*this, mesh, &p1, &vnodes1, recurse, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2,
		    unsigned vnodes1, unsigned vnodes2,
		    bool recurse,
		    int vnodes)
{

  PInsist(Dim==2,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[2];
  edt[0] = p1; edt[1] = p2;
  unsigned vad[2];
  vad[0] = vnodes1; vad[1] = vnodes2;
  centeredInitialize(*this, mesh, edt, vad, recurse, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
		    unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		    bool recurse,
		    int vnodes)
{

  PInsist(Dim==3,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[3];
  edt[0] = p1; edt[1] = p2; edt[2] = p3;
  unsigned vad[3];
  vad[0] = vnodes1; vad[1] = vnodes2; vad[2] = vnodes3;
  centeredInitialize(*this, mesh, edt, vad, recurse, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		    unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		    unsigned vnodes4,
		    bool recurse,
		    int vnodes)
{

  PInsist(Dim==4,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[4];
  edt[0] = p1; edt[1] = p2; edt[2] = p3; edt[3] = p4;
  unsigned vad[4];
  vad[0] = vnodes1; vad[1] = vnodes2; vad[2] = vnodes3;
  vad[3] = vnodes4;
  centeredInitialize(*this, mesh, edt, vad, recurse, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		    e_dim_tag p5,
		    unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		    unsigned vnodes4, unsigned vnodes5,
		    bool recurse,
		    int vnodes)
{

  PInsist(Dim==5,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[5];
  edt[0] = p1; edt[1] = p2; edt[2] = p3; edt[3] = p4; edt[4] = p5;
  unsigned vad[5];
  vad[0] = vnodes1; vad[1] = vnodes2; vad[2] = vnodes3;
  vad[3] = vnodes4; vad[4] = vnodes5;
  centeredInitialize(*this, mesh, edt, vad, recurse, vnodes);
}
template<unsigned Dim, class Mesh, class Centering>
CenteredFieldLayout<Dim,Mesh,Centering>::
CenteredFieldLayout(Mesh& mesh,
		    e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		    e_dim_tag p5, e_dim_tag p6,
		    unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		    unsigned vnodes4, unsigned vnodes5, unsigned vnodes6,
		    bool recurse,
		    int vnodes)
{

  PInsist(Dim==6,
    "Number of arguments does not match dimension of CenteredFieldLayout!!");
  PInsist(Dim<=Mesh::Dimension,
    "CenteredFieldLayout dimension cannot be greater than Mesh dimension!!");
  e_dim_tag edt[6];
  edt[0] = p1; edt[1] = p2; edt[2] = p3; edt[3] = p4; edt[4] = p5; edt[5] = p6;
  unsigned vad[6];
  vad[0] = vnodes1; vad[1] = vnodes2; vad[2] = vnodes3;
  vad[3] = vnodes4; vad[4] = vnodes5; vad[5] = vnodes6;
  centeredInitialize(*this, mesh, edt, vad, recurse, vnodes);
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


/***************************************************************************
 * $RCSfile: CenteredFieldLayout.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: CenteredFieldLayout.cpp,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $
 ***************************************************************************/