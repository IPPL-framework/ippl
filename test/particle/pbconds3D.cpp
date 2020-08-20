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

/***************************************************************************
 * Particle Boundary Conditions test ... initializes a set of 2D particles,
 * changes their position to move beyond the boundary, and prints out the
 * resulting positions.
 ***************************************************************************/

#include "Ippl.h"

// dimension of our positions
const unsigned Dim = 3;

// data type for our positions
typedef double Real;

// typedef for our particle layout type
typedef ParticleSpatialLayout<Real,Dim> playout_t;



int main(int argc, char *argv[])
{

  unsigned int i, j;
  Ippl ippl(argc, argv);
  Inform testmsg(argv[0]);

  testmsg << "Particle test: Boundary conditions: Begin." << endl;

  // create layout objects
  FieldLayout<Dim> FL(Index(100), Index(100), Index(100), SERIAL, PARALLEL, PARALLEL, 8);

  // create an empty particle object, with 2D position and 2
  // integer attributes, using a uniform layout
  playout_t* PL = new playout_t(FL);
  GenArrayParticle<playout_t,int,3> P(PL);

  // initialize the particle object:
  // The basic method is this: on just one node, call create(N)
  // and then set the values for the N new particles.  Then, on
  // every node, call update to distribute the particles where
  // they are supposed to go.  This can be done several times
  // if the number of particles is very large and cannot all fit
  // on one processor.
  int breakvar = (argc > 1 ? atoi(argv[1]) : 0);
  while (breakvar != 0);

  testmsg << "Start of initialization ..." << endl;
  if (P.singleInitNode()) {
    P.create(20); // makes new particles at end of current array
    // put ten particles near right edge, and ten near left edge
    for (j=0; j < 10; j++) { // initialize new values
      P.R[j] = Vektor<Real,Dim>(5, j * 10,j * 10);
      P.data[0][j] = Ippl::myNode();
      P.data[1][j] = 5;
    }
    for (j=10; j < 20; j++) { // initialize new values
      P.R[j] = Vektor<Real,Dim>(95, (j-10) * 10 + 2,(j-10) * 10 + 2);
      P.data[0][j] = Ippl::myNode();
      P.data[1][j] = 95;
    }
  }
  testmsg << "Performing initial update ..." << endl;
  P.update();         // do update to move particles to proper loc
  testmsg << "End of initialization" << i << ": localnum=";
  testmsg << P.getLocalNum() << ", totalnum=";
  testmsg << P.getTotalNum() << endl;

  // Just print out the contents of node 0's data
  testmsg << "------------------" << endl;
  for (i=0; i < P.getLocalNum(); i++) {
    testmsg << "LID=" << i << ", GID=" << P.ID[i] << ", data[0]=";
    testmsg << P.data[0][i] << ", data[1]=" << P.data[1][i] << ", R=";
    testmsg << P.R[i] << endl;
  }

  //
  // test 1:
  // add offset to R positions, moving particles beyond the border
  // bconds: all open
  //

  testmsg << "Testing open boundary conditions ... moving by (5,-1,10):" << endl;
  for (i=0; i < 2*Dim; ++i)
    P.getBConds()[i] = ParticleNoBCond;

  Vektor<Real,Dim> offset1(5,-1,10);
  assign(P.R, P.R + offset1);
  P.update();
  testmsg << "------------------" << endl;
  for (i=0; i < P.getLocalNum(); i++) {
    testmsg << "LID=" << i << ", GID=" << P.ID[i] << ", data[0]=";
    testmsg << P.data[0][i] << ", data[1]=" << P.data[1][i] << ", R=";
    testmsg << P.R[i] << endl;
  }

  //
  // test 2:
  // add offset to X Y & Z positions
  // BC: open X, Y
  //     periodc Z

  testmsg << "Testing periodic boundary conditions ... moving by (5,-1,10)";
  testmsg << endl;
  P.getBConds()[4] = ParticlePeriodicBCond;
  P.getBConds()[5] = ParticlePeriodicBCond;

  Vektor<Real,Dim> offset2(5,-1,10);
  assign(P.R, P.R + offset2);
  P.update();
  testmsg << "------------------" << endl;
  for (i=0; i < P.getLocalNum(); i++) {
    testmsg << "LID=" << i << ", GID=" << P.ID[i] << ", data[0]=";
    testmsg << P.data[0][i] << ", data[1]=" << P.data[1][i] << ", R=";
    testmsg << P.R[i] << endl;
  }
  testmsg << "Particle test: Spatial layout: End." << endl;
  return 0;
}