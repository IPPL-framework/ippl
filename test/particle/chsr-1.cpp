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
 * Visit http://www.acl.lanl.gov/POOMS for more details
 *
 ***************************************************************************/

/***************************************************************************

 Hi Court and Rui I follow up on discussions at the AAC 2002
 workshop with Court, on modelling CHSR. More precicely I am interested
 in using Lagrangian type of methods in general for Vlasov type of
 simulations. Court
 you mentioned that Rui has some ideas on Lagrangian methods for 
 CHSR. On my side,
 I am currently experimenting (building) a 6D parallel framework
 i.e: 6D scalar field and the needed operators for (semi) Lagrangian
 methods (by the way the dimension are presision are template parameters).
 
 This test program sets up a 6-d field with random values and tests
 some of the operators used for Lagrangian like solvers.

 usage: chsr-1 <gridSize>
 
 note: gridSize is used for all 6 dimensions in this testprogram
 
***************************************************************************/

#include "Ippl.h"

// dimension of f
const unsigned Dim = 6;

// presision
typedef double T;

// typedef our particle layout type
typedef ParticleSpatialLayout<T,Dim> playout_t;

int n;

const T fMax = 1.0;       // maximum value for f


int main(int argc, char *argv[]){
  Ippl ippl(argc, argv);
  Inform msg(argv[0]);
  
  n = atoi(argv[1]);

  msg << "Dim=" << Dim << endl;

  // create layout objects for max DIM==6
  const Index I(n), J(n), K(n);          // spacial
  const Index L(n), N(n), M(n);          // momenta

  // spacial half space
  const Index Ih(n/2,n,1), Jh(n/2,n,1), Kh(n/2,n,1);          

  // Initialize domain objects
  
  NDIndex<Dim> domain, domain1;

  domain[0] = n;
  domain[1] = n;
  domain[2] = n;
  domain[3] = n;
  
  domain1[0] = n+1;
  domain1[1] = n+1;
  domain1[2] = n+1;
  domain1[3] = n+1;

  if (Dim==6) {
    domain[4] = n;
    domain[5] = n;
    domain1[4] = n+1;
    domain1[5] = n+1;
  }

  UniformCartesian<Dim> mymesh(domain1);
  FieldLayout<Dim> FL(domain);

  /* 
     initialize f
     Note: we may do not need a mesh, however in order to
           convert to physical units it maybee helps
  */

  Field<T,Dim> f(mymesh,FL);

  T s = 0.0;


  assign(f[I][J][K][L][M][N],IpplRandom*fMax);
  assign(f[I][J][K][L][M][N],fMax);
  s = sum(f[I][J][K][L][M][N]);

  msg << "s= " << s << " full space " << endl;


  s = sum(f[Ih][Jh][Kh][L][M][N]);

  msg << "s= " << s << " half space " << endl;

  Ippl::Comm->barrier();
  msg << "done ...." << endl;
  return 0;
}