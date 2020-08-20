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
#include "Particle/ParticleBalancer.h"
#include "Particle/IpplParticleBase.h"
#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleUniformLayout.h"
#include "Particle/ParticleAttrib.h"
#include "Particle/IntNGP.h"
#include "Region/RegionLayout.h"
#include "Index/NDIndex.h"
#include "FieldLayout/FieldLayout.h"
#include "FieldLayout/BinaryBalancer.h"
#include "Utility/IpplInfo.h"



/////////////////////////////////////////////////////////////////////////////
// calculate a new RegionLayout for a given ParticleBase, and distribute the
// new RegionLayout to all the nodes.  This uses a Field BinaryBalancer.
template < class T, unsigned Dim, class Mesh, class CachingPolicy>
bool
BinaryRepartition(IpplParticleBase<ParticleSpatialLayout<T,Dim,Mesh,CachingPolicy> >& PB, double offset) {



  static IntNGP interp; // to scatter particle density

  //Inform dbgmsg("Particle BinaryRepartition", INFORM_ALL_NODES);
  //dbgmsg << "Performing particle load balancing, for ";
  //dbgmsg << PB.getTotalNum() << " particles ..." << endl;

  // get the internal FieldLayout from the Particle object's internal
  // RegionLayout.  From this, we make a new Field (we do not need a
  RegionLayout<T,Dim,Mesh>& RL = PB.getLayout().getLayout();
  if ( ! RL.initialized()) {
    ERRORMSG("Cannot repartition particles: uninitialized layout." << endl);
    return false;
  }
  FieldLayout<Dim>& FL = RL.getFieldLayout();
  Mesh& mesh = RL.getMesh();

  // NDIndex which describes the entire domain ... if a particle is
  // outside this region, we are in trouble!
  const NDIndex<Dim>& TotalDomain = FL.getDomain();

  // for all the particles, do the following:
  //   1. get the position, and invert to the 'FieldLayout' index space
  //   2. increment the field at the position near this index position
  NDIndex<Dim> indx;

  // By default, we do the number density computation and repartition of
  // index space on a cell-centered Field.  If FieldLayout is vertex-centered,
  // we'll need to make some adjustments here.
  bool CenterOffset[Dim];
  int CenteringTotal = 0;
  unsigned int d;
  for (d=0; d<Dim; ++d) {
    CenterOffset[d] = (TotalDomain[d].length() < mesh.gridSizes[d]);
    CenteringTotal += CenterOffset[d];
  }


  if (CenteringTotal == Dim) { // allCell centering
    Field<double,Dim,Mesh,Cell> BF(mesh,FL,GuardCellSizes<Dim>(1));

    // Now do a number density scatter on this Field
    // Afterwards, the Field will be deleted, and will checkout of the
    // FieldLayout.  This is desired so that when we repartition the
    // FieldLayout, we do not waste time redistributing the Field's data.
    BF = offset;
    scatter(BF,PB.R,interp);

    // calculate a new repartitioning of the field, and use this to repartition
    // the FieldLayout used inside the Particle object
    try
	{
		indx = CalcBinaryRepartition(FL, BF);
	}
	catch(BinaryRepartitionFailed bf)
	{
		return false;
	}
  }
  else if (CenteringTotal == 0) { // allVert centering
    Field<double,Dim,Mesh,Vert> BF(mesh,FL,GuardCellSizes<Dim>(1));

    // Now do a number density scatter on this Field
    // Afterwards, the Field will be deleted, and will checkout of the
    // FieldLayout.  This is desired so that when we repartition the
    // FieldLayout, we do not waste time redistributing the Field's data.
    BF = offset;
    scatter(BF,PB.R,interp);

    // calculate a new repartitioning of the field, and use this to repartition
    // the FieldLayout used inside the Particle object
	try
	{
		indx = CalcBinaryRepartition(FL, BF);
	}
	catch(BinaryRepartitionFailed bf)
	{
		return false;
	}
  }
  else {
    ERRORMSG("Not implemented for face- and edge-centered Fields!!" << endl);
    Ippl::abort();
  }

  // now, we can repartition the FieldLayout within the RegionLayout
  RL.RepartitionLayout(indx);
  PB.update();
  return true;
}


// the same, but taking a uniform layout (this will not actually do anything)
template<class T, unsigned Dim>
bool
BinaryRepartition(IpplParticleBase<ParticleUniformLayout<T,Dim> >& /*PB*/, double /*offset*/) {
  // for a uniform layout, this repartition method does nothing, so just
  // exit
  return true;
}

