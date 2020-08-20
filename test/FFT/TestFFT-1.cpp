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
 *
 ***************************************************************************/

// Tests the use of the (parallel) FFT class.

/*

  Usage: $MYMPIRUN -machinefile nodefile -np 4 ./TestFFT -grid 32 32 32 -test1 -Loop 100 --commlib mpi

*/


#include "Ippl.h"

#include <complex>
using namespace std;

enum TestCases {test1};

bool Configure(int argc, char *argv[], 
	       unsigned int *nx, unsigned int *ny, unsigned int *nz,
	       TestCases *test2Perform,
	       int *serialDim, unsigned int *processes,  unsigned int *nLoop) 
{

  Inform msg("Configure ");
  Inform errmsg("Error ");

  for (int i=1; i < argc; ++i) {
    string s(argv[i]);
    if (s == "-grid") {
      *nx = atoi(argv[++i]);
      *ny = atoi(argv[++i]);
      *nz = atoi(argv[++i]);
    } else if (s == "-test1") {
      *test2Perform = test1;
    } else if (s == "-Loop") {
      *nLoop = atoi(argv[++i]);
    } else if (s == "-Decomp") {
      *serialDim = atoi(argv[++i]);
    } 
    else {
      errmsg << "Illegal format for or unknown option '" << s.c_str() << "'.";
      errmsg << endl;
    }
  }
  if (*serialDim == 0)
    msg << "Serial dimension is x" << endl;
  else if (*serialDim == 1)
    msg << "Serial dimension is y" << endl;
  else if (*serialDim == 2)
    msg << "Serial dimension is z" << endl;
  else {
    msg << "All parallel" << endl;
    *serialDim = -1;
  }

  *processes = Ippl::getNodes();

  return true;
}

int main(int argc, char *argv[])
{
  Ippl ippl(argc,argv);
  Inform testmsg(NULL,0);

  unsigned int processes;
  int serialDim = -1;
  TestCases test2do;
  unsigned int nx,ny,nz;
  unsigned int nLoop;
 
  /*bool res = */ Configure(argc, argv,  &nx, &ny, &nz, 
                            &test2do, &serialDim, &processes, &nLoop); 


  // The preceding cpp definition causes compile-time setting of D:
  const unsigned D=3U;
  testmsg << "Dimensionality: D= " << D << " P= " << processes;
  testmsg << " nx= " << nx << " ny= " << ny << " nz= " << nz << endl;
  
  unsigned ngrid[D];   // grid sizes

  // Used in evaluating correctness of results:
  double realDiff;
  
  double pi = acos(-1.0);
  double twopi = 2.0*pi;
  
  Timer timer;
  
  // Layout information:
 
  e_dim_tag allParallel[D];    // Specifies SERIAL, PARALLEL dims
  for (unsigned int d=0; d<D; d++) 
    allParallel[d] = PARALLEL;

  // Compression of temporaries:
  bool compressTemps = false ;
 
  ngrid[0]=nx;
  ngrid[1]=ny;
  ngrid[2]=nz;

  //------------------------complex<-->complex------------------------------- 
  // Complex test Fields
  // create standard domain
  NDIndex<D> ndiStandard;
  for (unsigned int d=0; d<D; d++)
    ndiStandard[d] = Index(ngrid[d]);
   
   // all parallel layout, standard domain, normal axis order
   FieldLayout<D> layoutPPStan(ndiStandard,allParallel,processes);
        
   // create test Fields for complex-to-complex FFT
   BareField<std::complex<double>,D> CFieldPPStan(layoutPPStan);
   BareField<std::complex<double>,D> CFieldPPStan_save(layoutPPStan);
   BareField<double,D> diffFieldPPStan(layoutPPStan);
       
   // Rather more complete test functions (sine or cosine mode):
   std::complex<double> sfact(1.0,0.0);      // (1,0) for sine mode; (0,0) for cosine mode
   std::complex<double> cfact(0.0,0.0);      // (0,0) for sine mode; (1,0) for cosine mode

   double xfact, kx, yfact, ky, zfact, kz;
   xfact = pi/(ngrid[0] + 1.0);
   yfact = 2.0*twopi/(ngrid[1]);
   zfact = 2.0*twopi/(ngrid[2]);
   kx = 1.0; ky = 2.0; kz = 3.0; // wavenumbers
 
   CFieldPPStan[ndiStandard[0]][ndiStandard[1]][ndiStandard[2]] = 
      sfact * ( sin( (ndiStandard[0]+1) * kx * xfact +
                      ndiStandard[1]    * ky * yfact +
                      ndiStandard[2]    * kz * zfact ) +
		sin( (ndiStandard[0]+1) * kx * xfact -
                      ndiStandard[1]    * ky * yfact -
                      ndiStandard[2]    * kz * zfact ) ) + 
      cfact * (-cos( (ndiStandard[0]+1) * kx * xfact +
                      ndiStandard[1]    * ky * yfact +
                      ndiStandard[2]    * kz * zfact ) + 
		cos( (ndiStandard[0]+1) * kx * xfact -
                      ndiStandard[1]    * ky * yfact -
                      ndiStandard[2]    * kz * zfact ) );

   CFieldPPStan_save = CFieldPPStan; // Save values for checking later

   // Instantiate complex<-->complex FFT object Transform in all directions
   FFT<CCTransform,D,double> ccfft(ndiStandard, compressTemps);

   testmsg << "Test 3D complex<-->complex transform (simple test: forward then inverse transform" << endl;
   timer.start();
    
   // ================BEGIN INTERACTION LOOP====================================
   for (unsigned int i=0; i<nLoop; i++)  {
      ccfft.transform( -1 , CFieldPPStan);
      ccfft.transform( +1 , CFieldPPStan);
      
      diffFieldPPStan = Abs(CFieldPPStan - CFieldPPStan_save);
      realDiff = max(diffFieldPPStan);
      testmsg << "CC <-> CC: fabs(realDiff) = " << fabs(realDiff) << endl;
      //-------------------------------------------------------------------------
      CFieldPPStan = CFieldPPStan_save;
    }
    
    timer.stop();
    testmsg << " CPU time used = " << timer.cpu_time() << " secs. NP= " << Ippl::getNodes() << endl;
    return 0;
}

/***************************************************************************
 * $RCSfile: TestFFT.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 ***************************************************************************/

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $ 
 ***************************************************************************/

