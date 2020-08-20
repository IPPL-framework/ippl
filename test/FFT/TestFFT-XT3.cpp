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

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// TestFFT.cpp , Tim Williams 1/27/1997
// Updated by Julian Cummings, 3/31/98

// Tests the use of the (parallel) FFT class.

#include "Ippl.h"

#include <complex>
#include <string>

using namespace std;

#define THREED

enum FsT {FFTSolver,Pt2PtSolver,TreeSolver,None};
enum InterPolT {NGP,CIC};
enum BCT {OOO,OOP,PPP,DDD,DDO,DDP};   // OOO == all dim, open BC
enum TestCases {test1};

bool Configure(int argc, char *argv[], InterPolT *interPol, 
	       unsigned int *nx, unsigned int *ny, unsigned int *nz,
	       TestCases *test2Perform,
	       int *serialDim, unsigned int *processes,  unsigned int *nLoop) 
{

  Inform msg("Configure ");
  Inform errmsg("Error ");

  string bc_str;
  string interPol_str;
  string dist_str;

  for (int i=1; i < argc; ++i) {
    string s(argv[i]);
    if (s == "-grid") {
      *nx = atoi(argv[++i]);
      *ny = atoi(argv[++i]);
      *nz = atoi(argv[++i]);
    } else if (s == "-test1") {
      *test2Perform = test1;
    } else if (s == "-NGP") {
      *interPol = NGP;
      msg << "Interploation NGP" << endl;
    } else if (s == "-CIC") {
      *interPol = CIC;
      msg << "Interploation CIC" << endl;
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
  Inform msg(NULL,0);

  unsigned int processes;
  int serialDim;
  TestCases test2do;
  unsigned int nx,ny,nz;
  unsigned int nLoop;
  InterPolT interPol;

  /*bool res = */ Configure(argc, argv, &interPol, &nx, &ny, &nz, 
                            &test2do, &serialDim, &processes, &nLoop); 


  // The preceding cpp definition causes compile-time setting of D:
  const unsigned D=3U;
  msg << "Dimensionality: D= " << D << " P= " << processes;
  msg << " nx= " << nx << " ny= " << ny << " nz= " << nz << endl;
  
  unsigned ngrid[D];   // grid sizes

  // Used in evaluating correctness of results:
  double realDiff;
  
  // Various counters, constants, etc:
  unsigned int d;
  
  /*int tag = */ Ippl::Comm->next_tag(IPPL_APP_TAG0);
  double pi = acos(-1.0);
  double twopi = 2.0*pi;
  
  // Layout information:
  e_dim_tag domainDec[D];    // Specifies SERIAL, PARALLEL dims
  for (d=0; d<D; d++) {
      if (d != 2)
	  domainDec[d] = PARALLEL;
      else
	  domainDec[d] = SERIAL;
  }
  // Compression of temporaries:
  bool compressTemps;
  compressTemps = false;
  ngrid[0]=nx;
  ngrid[1]=ny;
  ngrid[2]=nz;

  msg << "In-place CC transform using PPS layout ..." << endl;

  //------------------------complex<-->complex-------------------------------
  // Complex test Fields
  // create standard domain
  NDIndex<D> ndiStandard;
  for (d=0; d<D; d++) 
      ndiStandard[d] = Index(ngrid[d]);

  // create half-size domain for RC transform along zeroth axis
  NDIndex<D> ndiStandard0h = ndiStandard;
  ndiStandard0h[0] = Index(ngrid[0]/2+1);
  
  // layout, standard domain, normal axis order
  FieldLayout<D> layoutPPStan(ndiStandard,domainDec,processes);

  // layout, standard domain, normal axis order
  FieldLayout<D> layoutPPStan0h(ndiStandard0h,domainDec,processes);

    
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
  
  FFT<CCTransform,D,double> ccfft(ndiStandard, compressTemps);

  msg << "start complex<-->complex " << endl;
  
  for (unsigned int i=0; i<nLoop; i++)  {
      // Test complex<-->complex transform 
      // (simple test: forward then inverse transform, see if get back original values.
      ccfft.transform( 1 , CFieldPPStan);
      ccfft.transform(-1 , CFieldPPStan);
      diffFieldPPStan = Abs(CFieldPPStan - CFieldPPStan_save);
      realDiff = max(diffFieldPPStan);
      msg << "CC <-> CC: fabs(realDiff) = " << fabs(realDiff) << endl;
      CFieldPPStan = CFieldPPStan_save;
  }
     
  BareField<double,D>   RFieldPPStan(layoutPPStan);
  BareField<double,D>   RFieldPPStan_save(layoutPPStan);

  BareField<std::complex<double>,D> CFieldPPStan0h(layoutPPStan0h);
  FFT<RCTransform,D,double> rcfft(ndiStandard,  ndiStandard0h, compressTemps);      

  RFieldPPStan_save = RFieldPPStan;
 
  msg << "start real<-->complex " << endl;
  for (unsigned int i=0; i<nLoop; i++)  {
      rcfft.transform( 1, RFieldPPStan,   CFieldPPStan0h);
      rcfft.transform(-1, CFieldPPStan0h, RFieldPPStan);
      diffFieldPPStan = Abs(RFieldPPStan - RFieldPPStan_save);
      realDiff = max(diffFieldPPStan);
      msg << "RR <-> CC: fabs(realDiff) = " << realDiff << endl;
  }

  return 0;
}

/***************************************************************************
 * $RCSfile: TestFFT.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 ***************************************************************************/
