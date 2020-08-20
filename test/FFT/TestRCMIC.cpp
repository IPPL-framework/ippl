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

#include <complex>
#include <string>

using namespace std;

bool Configure(int argc, char *argv[],
               unsigned int *nx, unsigned int *ny, unsigned int *nz,
               int *serialDim, unsigned int *processes,  unsigned int *nLoop)
{

  Inform msg("Configure ");
  Inform errmsg("Error ");

  string bc_str;
  string dist_str;

  for (int i=1; i < argc; ++i) {
    string s(argv[i]);
    if (s == "-grid") {
      *nx = atoi(argv[++i]);
      *ny = atoi(argv[++i]);
      *nz = atoi(argv[++i]);
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
    *serialDim = 0;
    msg << "Serial dimension is x" << endl;
  }
  *processes = Ippl::getNodes();

  return true;
}


std::complex<double> printComplex(std::complex<double> in) {
  double r = 0.0;
  double i = 0.0;
  if (in.real() > 0.00001 || in.real() < -0.00001)
    r = in.real();
  if (in.imag() > 0.00001 || in.imag() < -0.00001)
    i = in.imag();

  return std::complex<double>(r, i);
}



int main(int argc, char *argv[])
{

  Ippl ippl(argc,argv);
  Inform testmsg(NULL,0);

  static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
  static IpplTimings::TimerRef fftTimer = IpplTimings::getTimer("fftTimer");

  IpplTimings::startTimer(mainTimer);

  const unsigned D=3U;
  bool compressTemps = false;
  bool constInput    = true;  // preserve input field in two-field transform

  testmsg << "%%%%%%% Dimensionality: D = " << D << " %%%%%%%" << endl;


  unsigned int processes;
  int serialDim;
  unsigned int nx,ny,nz;
  unsigned int nLoop;

  Configure(argc, argv, &nx, &ny, &nz, &serialDim, &processes, &nLoop);

  int vnodes = processes;
  unsigned ngrid[D];   // grid sizes
  ngrid[0] = nx;
  if (D > 1) ngrid[1] = ny;
  if (D > 2) ngrid[2] = nz;

  // Used in evaluating correctness of results:
  double realDiff;

  // Various counters, constants, etc:

  double pi = acos(-1.0);
  double twopi = 2.0*pi;

  e_dim_tag allParallel[D];    // Specifies SERIAL, PARALLEL dims
  for (unsigned int d=0; d<D; d++)
    allParallel[d] = PARALLEL;

  e_dim_tag serialParallel[D]; // Specifies SERIAL, PARALLEL dims
  for (unsigned int d=0; d<D; d++)
    serialParallel[d] = PARALLEL;
  serialParallel[serialDim] = SERIAL;

  // create standard domain
  NDIndex<D> ndiStandard;
  for (unsigned int d=0; d<D; d++)
    ndiStandard[d] = Index(ngrid[d]);
  // create new domain with axes permuted to match FFT output

  // create half-size domain for RC transform along zeroth axis
  NDIndex<D> ndiStandard0h = ndiStandard;
  ndiStandard0h[0] = Index(ngrid[0]/2+1);

  /*
    unsigned ndiVNodes[D];
    for (unsigned int d=0; d<D; d++)
    ndiVNodes[d] = 1;
    ndiVNodes[D-1] = vnodes;
  */

  // all parallel layout, standard domain, normal axis order
  FieldLayout<D> layoutPPStan(ndiStandard,allParallel,vnodes);
  //FieldLayout<D> layoutPPStan(ndiStandard, allParallel, ndiVNodes);

  // zeroth axis serial, standard domain, normal axis order
  FieldLayout<D> layoutSPStan(ndiStandard,serialParallel,vnodes);
  //FieldLayout<D> layoutSPStan(ndiStandard,serialParallel,ndiVNodes);

  // all parallel layout, zeroth axis half-size domain, normal axis order
  FieldLayout<D> layoutPPStan0h(ndiStandard0h,allParallel,vnodes);
  //FieldLayout<D> layoutPPStan0h(ndiStandard0h,allParallel,ndiVNodes);

  // zeroth axis serial, zeroth axis half-size domain, normal axis order
  FieldLayout<D> layoutSPStan0h(ndiStandard0h,serialParallel,vnodes);
  //FieldLayout<D> layoutSPStan0h(ndiStandard0h,serialParallel,ndiVNodes);

  // create test Fields for complex-to-complex FFT
  BareField<std::complex<double>,D> CFieldPPStan(layoutPPStan);

  BareField<std::complex<double>,D> CFieldSPStan(layoutSPStan);

  BareField<double,D> diffFieldSPStan(layoutSPStan);

  // create test Fields for real-to-complex FFT
  BareField<double,D>   RFieldSPStan(layoutSPStan);
  BareField<double,D>   RFieldSPStan_save(layoutSPStan);
  BareField<std::complex<double>,D> CFieldSPStan0h(layoutSPStan0h);

  INFOMSG("RFieldSPStan   layout= " << layoutSPStan << endl;);
  INFOMSG("CFieldSPStan0h layout= " << layoutSPStan0h << endl;);

  // For calling FieldDebug functions from debugger, set up output format:
  setFormat(4,3);

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


  //Initialize Real Field separately instead of RFieldSPStan = real(CFieldPPStan) due to conversion problems with intel compiler
  RFieldSPStan[ndiStandard[0]][ndiStandard[1]][ndiStandard[2]] =
    1.0 * ( sin( (ndiStandard[0]+1) * kx * xfact +
                   ndiStandard[1]    * ky * yfact +
                   ndiStandard[2]    * kz * zfact ) +
              sin( (ndiStandard[0]+1) * kx * xfact -
                   ndiStandard[1]    * ky * yfact -
                   ndiStandard[2]    * kz * zfact ) ) +
    0.0 * (-cos( (ndiStandard[0]+1) * kx * xfact +
                   ndiStandard[1]    * ky * yfact +
                   ndiStandard[2]    * kz * zfact ) +
             cos( (ndiStandard[0]+1) * kx * xfact -
                  ndiStandard[1]    * ky * yfact -
                  ndiStandard[2]    * kz * zfact ) );

  //RFieldSPStan = real(CFieldPPStan);
  CFieldSPStan0h = std::complex<double>(0.0,0.0);

  // create RC FFT object
  FFT<RCTransform,D,double> rcfft(ndiStandard, ndiStandard0h, compressTemps);
  // set direction names
  rcfft.setDirectionName(+1, "forward");
  rcfft.setDirectionName(-1, "inverse");

  testmsg << "RC transform using layout with zeroth dim serial ..." << endl;

  //do one fft before timing begins
  rcfft.transform("forward", RFieldSPStan,  CFieldSPStan0h, constInput);
  rcfft.transform("forward", CFieldSPStan0h, RFieldSPStan, constInput);

  testmsg << "Initial FFTS using rcfft.transform DONE " << endl;

  testmsg << " SETUP FFT DONE" << endl;

  for (unsigned i=0; i<nLoop; i++) {
          testmsg << "start new loop iteration now" << endl;
    RFieldSPStan_save = RFieldSPStan;

    IpplTimings::startTimer(fftTimer);

    rcfft.transform("forward", RFieldSPStan,  CFieldSPStan0h, constInput);
    rcfft.transform("forward", CFieldSPStan0h, RFieldSPStan, constInput);
    IpplTimings::stopTimer(fftTimer);


    diffFieldSPStan = Abs(RFieldSPStan - RFieldSPStan_save);
    realDiff = max(diffFieldSPStan);
    //if (realDiff > 1e-5)
      testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
    //testmsg << "FFT " << (i+1) << " done" << endl;

  }

  IpplTimings::stopTimer(mainTimer);
  IpplTimings::print();
  IpplTimings::print(std::string("TestRC.timing"));

  return 0;
}
/***************************************************************************
 * $RCSfile: TestRC.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 ***************************************************************************/

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $
 ***************************************************************************/
