// -*- C++ -*-
/* *************************************************************************
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

 Tests the use of the (parallel) FFT class.

  
  Usage all parallel, i.e. 3D domain decomposition:
 
   $MYMPIRUN -machinefile nodefile -np 4 ./TestFFT-1 -grid 32 32 32 -test1 -Loop 100  --commlib mpi

  Usage 2D domain decomposition (z is serial): 

   $MYMPIRUN -machinefile nodefile -np 4 ./TestFFT-1 -grid 32 32 32 -test1 -Loop 100 -Decomp 2 --commlib mpi

*/


#include "Ippl.h"

#include <complex>
#include <string>

using namespace std;

enum TestCases {test1,test2};

bool Configure(int argc, char *argv[], 
	       unsigned int *nx, unsigned int *ny, unsigned int *nz,
	       TestCases *test2Perform,
	       int *serialDim, unsigned int *nLoop) 
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
    } else if (s == "-test2") {
      *test2Perform = test2;
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

  return true;
}

int main(int argc, char *argv[])
{
  Ippl ippl(argc,argv);
  Inform testmsg(NULL,0);

  int serialDim = -1;
  TestCases test2do;
  unsigned int nx,ny,nz;
  unsigned int nLoop;

  static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
  static IpplTimings::TimerRef fftfTimer = IpplTimings::getTimer("fft-forward");
  static IpplTimings::TimerRef fftbTimer = IpplTimings::getTimer("fft-backward");
  static IpplTimings::TimerRef fieldassTimer = IpplTimings::getTimer("field-assigment");
  static IpplTimings::TimerRef fieldcompTimer = IpplTimings::getTimer("field-computation");

  /*bool res = */ Configure(argc, argv,  &nx, &ny, &nz, 
                            &test2do, &serialDim, &nLoop); 


  // The preceding cpp definition causes compile-time setting of D:
  const unsigned D=3U;
  testmsg << "Dimensionality: D= " << D << " P= " << Ippl::getNodes();
  testmsg << " nx= " << nx << " ny= " << ny << " nz= " << nz << endl;
  
  unsigned ngrid[D];   // grid sizes

  // Used in evaluating correctness of results:
  double realDiff;
  
  double pi = acos(-1.0);
  double twopi = 2.0*pi;

  // Layout information: 
  e_dim_tag allParallel[D];    // Specifies SERIAL, PARALLEL dims
  for (unsigned int d=0; d<D; d++) 
   allParallel[d] = PARALLEL;
  
  if(serialDim == 0)
   allParallel[0] = SERIAL;
  else if (serialDim == 1)
   allParallel[1] = SERIAL;
  else if (serialDim == 2)
   allParallel[2] = SERIAL;

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
   FieldLayout<D> layoutPPStan(ndiStandard,allParallel,Ippl::getNodes());
        
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

   IpplTimings::startTimer(mainTimer); 

   // ================BEGIN INTERACTION LOOP====================================
   for (unsigned int i=0; i<nLoop; i++)  {

      IpplTimings::startTimer(fftfTimer); 
      ccfft.transform( -1 , CFieldPPStan);
      IpplTimings::stopTimer(fftfTimer); 
      IpplTimings::startTimer(fftbTimer); 
      ccfft.transform( +1 , CFieldPPStan);
      IpplTimings::stopTimer(fftbTimer); 
      IpplTimings::startTimer(fieldcompTimer); 
      diffFieldPPStan = Abs(CFieldPPStan - CFieldPPStan_save);
      realDiff = max(diffFieldPPStan);
      IpplTimings::stopTimer(fieldcompTimer); 

      testmsg << "CC <-> CC: fabs(realDiff) = " << fabs(realDiff) << endl;

      IpplTimings::startTimer(fieldassTimer); 
      CFieldPPStan = CFieldPPStan_save;
      IpplTimings::stopTimer(fieldassTimer); 
    }
    IpplTimings::stopTimer(mainTimer); 
  
    if(serialDim == -1)
     testmsg << "3D domain decomp  Nodes = " << Ippl::getNodes() << " Fields " << nx << "^3" << endl;
    else
     testmsg << "2D domain decomp  Nodes = " << Ippl::getNodes() << " Fields " << nx << "^3" << endl;

    IpplTimings::print();

    return 0;
}

/***************************************************************************
 * $RCSfile: TestFFT-1.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 ***************************************************************************/

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $ 
 ***************************************************************************/

