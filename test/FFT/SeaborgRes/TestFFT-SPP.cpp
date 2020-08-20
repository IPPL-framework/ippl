// TestFFT.cpp , Tim Williams 1/27/1997
// Updated by Julian Cummings, 3/31/98

// Tests the use of the (parallel) FFT class.

#include "Ippl.h"

#include <complex>
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
  Inform testmsg(NULL,0);

  unsigned int processes;
  int serialDim = -1;
  TestCases test2do;
  unsigned int nx,ny,nz;
  unsigned int nLoop;
  InterPolT interPol;

  /*bool res = */
  Configure(argc, argv, &interPol, &nx, &ny, &nz,
            &test2do, &serialDim, &processes, &nLoop);

  // The preceding cpp definition causes compile-time setting of D:
  const unsigned D=3U;
  testmsg << "Dimensionality: D= " << D << " P= " << processes;
  testmsg << " nx= " << nx << " ny= " << ny << " nz= " << nz << endl;

  unsigned ngrid[D];   // grid sizes

  // Used in evaluating correctness of results:
  double realDiff = -1.0;

  // Various counters, constants, etc:
  unsigned int d;

  /*int tag = */ Ippl::Comm->next_tag(IPPL_APP_TAG0);
  double pi = acos(-1.0);
  double twopi = 2.0*pi;
  // Timer:
  Timer timer;

  // Layout information:
  unsigned vnodes;             // number of vnodes; input at cmd line
  e_dim_tag allParallel[D];    // Specifies SERIAL, PARALLEL dims
  for (d=0; d<D; d++)
    allParallel[d] = PARALLEL;

  allParallel[0] = SERIAL;

  // Compression of temporaries:
  bool compressTemps;
  vnodes = processes;
  compressTemps = false;
  ngrid[0]=nx;
  ngrid[1]=ny;
  ngrid[2]=nz;

  testmsg << "In-place CC transform using all-parallel layout ..." << endl;

  // ================BEGIN INTERACTION LOOP====================================

    //------------------------complex<-->complex-------------------------------
    // Complex test Fields
    // create standard domain
    NDIndex<D> ndiStandard;
    for (d=0; d<D; d++)
      ndiStandard[d] = Index(ngrid[d]);

    // create new domain with axes permuted to match FFT output
    NDIndex<D> ndiPermuted;
    ndiPermuted[0] = ndiStandard[D-1];
    for (d=1; d<D; d++)
      ndiPermuted[d] = ndiStandard[d-1];

    // create half-size domain for RC transform along zeroth axis
    NDIndex<D> ndiStandard0h = ndiStandard;
    ndiStandard0h[0] = Index(ngrid[0]/2+1);
    // create new domain with axes permuted to match FFT output
    NDIndex<D> ndiPermuted0h;
    ndiPermuted0h[0] = ndiStandard0h[D-1];
    for (d=1; d<D; d++)
      ndiPermuted0h[d] = ndiStandard0h[d-1];

    // create half-size domain for sine transform along zeroth axis
    // and RC transform along first axis
    NDIndex<D> ndiStandard1h = ndiStandard;
    ndiStandard1h[1] = Index(ngrid[1]/2+1);
    // create new domain with axes permuted to match FFT output
    NDIndex<D> ndiPermuted1h;
    ndiPermuted1h[0] = ndiStandard1h[D-1];
    for (d=1; d<D; d++)
      ndiPermuted1h[d] = ndiStandard1h[d-1];

    // all parallel layout, standard domain, normal axis order
    FieldLayout<D> layoutPPStan(ndiStandard,allParallel,vnodes);

    FieldLayout<D> layoutPPStan0h(ndiStandard0h,allParallel,vnodes);

    FieldLayout<D> layoutPPStan1h(ndiStandard1h,allParallel,vnodes);

    // create test Fields for complex-to-complex FFT
    BareField<std::complex<double>,D> CFieldPPStan(layoutPPStan);
    BareField<std::complex<double>,D> CFieldPPStan_save(layoutPPStan);
    BareField<double,D> diffFieldPPStan(layoutPPStan);

    // For calling FieldDebug functions from debugger, set up output format:
    setFormat(4,3);

    // Rather more complete test functions (sine or cosine mode):
    std::complex<double> sfact(1.0,0.0);      // (1,0) for sine mode; (0,0) for cosine mode
    std::complex<double> cfact(0.0,0.0);      // (0,0) for sine mode; (1,0) for cosine mode

    // Conditionally-compiled loading functions (couldn't make these
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

    // CC FFT tests
    // Instantiate complex<-->complex FFT object
    // Transform in all directions
    FFT<CCTransform,D,double> ccfft(ndiStandard, compressTemps);

    // set direction names
    ccfft.setDirectionName(+1, "forward");
    ccfft.setDirectionName(-1, "inverse");
    timer.start();
    for (unsigned int i=0; i<nLoop; i++)  {
      // Test complex<-->complex transform (simple test: forward then inverse transform, see if get back original values.
      ccfft.transform("forward", CFieldPPStan);
      ccfft.transform("inverse", CFieldPPStan);
      diffFieldPPStan = Abs(CFieldPPStan - CFieldPPStan_save);
      realDiff = max(diffFieldPPStan);
      testmsg << "CC <-> CC: fabs(realDiff) = " << fabs(realDiff) << endl;
      //-------------------------------------------------------------------------
      CFieldPPStan = CFieldPPStan_save;
    }
    timer.stop();
    testmsg << "Results " << nLoop << " times CC <-> CC: CPU time = " << timer.cpu_time() << " secs.";
    testmsg <<  " Dimensionality: D= " << D << " P= " << processes;
    testmsg << " nx= " << nx << " ny= " << ny << " nz= " << nz;
    testmsg << " ||d||= " << fabs(realDiff) << endl;
    return 0;
}