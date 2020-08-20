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
 ***************************************************************************/

#include "Ippl.h"
#include "Particle/ParticleBalancer.h"


/***************************************************************************
 * A test program for particle load balancing
 ***************************************************************************/


// dimension of our positions
const unsigned Dim = 2;

// type of our positions
typedef double Pos_t;

// type of our Particle Layout
typedef ParticleSpatialLayout<Pos_t,Dim> playout_t;

// type of our Particle object
typedef GenParticle<playout_t, int> Particle_t;

int main(int argc, char *argv[])
{
  Ippl ippl(argc, argv);
  Inform testmsg(argv[0], INFORM_ALL_NODES);

  testmsg << "Particle test: BinaryRepartition: Begin." << endl;

  // define our geometry
  // FieldLayout Index objects are 1 smaller than Mesh Index objects,
  // implying cell-centered Fields.
  Vektor<Pos_t,Dim> origin(-50.0, 10.0);
  Pos_t spacing[2];
  spacing[0] = 7.5;
  spacing[1] = 0.5;
  Index I1(21), J1(21);
  UniformCartesian<Dim,Pos_t> grid(I1, J1, spacing, origin);
  CenteredFieldLayout<Dim,UniformCartesian<Dim,Pos_t>,Cell> flayout(grid);

  // a regionlayout to store where to put things
  RegionLayout<Pos_t,Dim> r1(flayout, grid);

  // a new particle object, using a spatial layout
  playout_t* PL = new playout_t(r1);
  Particle_t P(PL);

  // keep adding new particles and doing load balancing
  for (unsigned loopi=0; loopi < 4; ++loopi) {
    P.create(10);
    P.data = Ippl::myNode();
    //    P.getLayout().setInteractionRadius(3.0);
    double frac = (loopi+1)/4.0;
    for (int i=0; i < Dim; ++i)
      assign(P.R(i), r1.getDomain()[i].min() +
                     IpplRandom*frac*r1.getDomain()[i].length());
    P.update();

    // do repartitioning ... should be mostly even distribution
    testmsg << "---------------------------------------------" << endl;
    testmsg << "Performing repartition " << loopi << " ..." << endl;
    BinaryRepartition(P);
    testmsg << "GenParticle after repartition: " << P << endl;
  }

  // print out neighbors of particles on node 0
  /*
  testmsg << "\n---------------------------------------------" << endl;
  testmsg << "Finding nearest neighbors of particles on node 0 ..." << endl;
  for (int i=0; i < P.getLocalNum(); ++i) {
    Particle_t::pair_iterator plisti;
    Particle_t::pair_iterator plistf;
    P.getLayout().getPairlist(i, plisti, plistf);

    testmsg << "For ID = " << P.ID[i] << ", num neighbors = ";
    testmsg << (plistf - plisti) << ", R = " << P.R[i] << endl;

    // perform calculation for each neighbor of this particle.
    for ( ; plisti != plistf ; ++plisti ) {

      // get local index and sep^2 of neighbor
      Particle_t::pair_t nbrData = *plisti;

      if (Ippl::myNode() == 0) {
	// print out their values
	testmsg << "  Neighbor:";
	testmsg <<  " ID="             << P.ID[nbrData.first];
	testmsg << ", sep^2="          << nbrData.second;
	testmsg << ", pos="            << P.R[nbrData.first] << endl;
      }
    }
  }
  */
  testmsg << "Particle test: BinaryRepartition: End." << endl;

  return 0;
}

/***************************************************************************
 * $RCSfile: loadbalance.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:38 $
 * IPPL_VERSION_ID: $Id: loadbalance.cpp,v 1.1.1.1 2003/01/23 07:40:38 adelmann Exp $ 
 ***************************************************************************/
