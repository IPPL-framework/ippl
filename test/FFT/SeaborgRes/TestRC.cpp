// TestFFT.cpp , Tim Williams 1/27/1997
// Updated by Julian Cummings, 3/31/98

// Tests the use of the (parallel) FFT class.

#include "Ippl.h"

#include <complex>
using namespace std;

int main(int argc, char *argv[])
{

  Ippl ippl(argc,argv);
  Inform testmsg(NULL,0);

  const unsigned D=3U;
  testmsg << "%%%%%%% Dimensionality: D = " << D << " %%%%%%%" << endl;

  unsigned ngrid[D];   // grid sizes

  // Used in evaluating correctness of results:
  double realDiff;
  const double errorTol = 1.0e-10;
  bool correct = true;

  // Various counters, constants, etc:
  unsigned int d;
  int Parent = 0;
  int tag = Ippl::Comm->next_tag(IPPL_APP_TAG0);
  double pi = acos(-1.0);
  double twopi = 2.0*pi;
  // Timer:
  Timer timer;
  bool constInput;  // preserve input field in two-field transform

  // Layout information:
  unsigned vnodes = 1;             // number of vnodes; input at cmd line
  e_dim_tag allParallel[D], serialParallel[D];       // Specifies SERIAL, PARALLEL dims
  for (d=0; d<D; d++) {
    allParallel[d]    = PARALLEL;
    serialParallel[d] = SERIAL;
  }

  // Compression of temporaries:
  bool compressTemps;

  // ================BEGIN INTERACTION LOOP====================================
  while (true) {
    // read in vnodes etc. off of node 0
    if( Ippl::Comm->myNode() == Parent ) {
      bool vnodesOK = false;
      while (!vnodesOK) {
        compressTemps = false;
        constInput = false;
        for (d=0; d<D; d++)
          ngrid[d] = 32;

        // now broadcast data to other nodes
        Message *mess = new Message();
        putMessage( *mess, vnodes );
        if (vnodes > 0) {
          putMessage(*mess, compressTemps);
          putMessage(*mess, constInput);
          for (d=0; d<D; d++)
            putMessage( *mess, ngrid[d] );
        }
        Ippl::Comm->broadcast_all(mess, tag);
      }
    }
    // now each node recieves the data
    unsigned pe = Ippl::Comm->myNode();
    Message *mess = Ippl::Comm->receive_block(Parent, tag);
    PAssert(mess);
    getMessage( *mess, vnodes );
    if (vnodes <= 0) break;
    getMessage(*mess, compressTemps);
    getMessage(*mess, constInput);
    for (d=0; d<D; d++)
      getMessage( *mess, ngrid[d] );
    delete mess;

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
    // all parallel layout, zeroth axis half-size domain, normal axis order
    FieldLayout<D> layoutPPStan0h(ndiStandard0h,allParallel,vnodes);

#ifndef ONED
    // zeroth axis serial, standard domain, normal axis order
    FieldLayout<D> layoutSPStan(ndiStandard,serialParallel,vnodes);
    // zeroth axis serial, standard domain, permuted axis order
    FieldLayout<D> layoutSPPerm(ndiPermuted,serialParallel,vnodes);
    // zeroth axis serial, zeroth axis half-size domain, normal axis order
    FieldLayout<D> layoutSPStan0h(ndiStandard0h,serialParallel,vnodes);
    // zeroth axis serial, zeroth axis half-size domain, permuted axis order
    FieldLayout<D> layoutSPPerm0h(ndiPermuted0h,serialParallel,vnodes);
#endif
    // all parallel layout, first axis half-size domain, normal axis order
    FieldLayout<D> layoutPPStan1h(ndiStandard1h,allParallel,vnodes);
#ifndef ONED
    // zeroth axis serial, first axis half-size domain, normal axis order
    FieldLayout<D> layoutSPStan1h(ndiStandard1h,serialParallel,vnodes);
    // zeroth axis serial, first axis half-size domain, permuted axis order
    FieldLayout<D> layoutSPPerm1h(ndiPermuted1h,serialParallel,vnodes);
#endif

    // create test Fields for complex-to-complex FFT
    BareField<std::complex<double>,D> CFieldPPStan(layoutPPStan);
    BareField<std::complex<double>,D> CFieldPPStan_save(layoutPPStan);
    BareField<double,D> diffFieldPPStan(layoutPPStan);
#ifndef ONED
    BareField<std::complex<double>,D> CFieldSPStan(layoutSPStan);
    BareField<std::complex<double>,D> CFieldSPStan_save(layoutSPStan);
    BareField<double,D> diffFieldSPStan(layoutSPStan);
    BareField<std::complex<double>,D> CFieldSPPerm(layoutSPPerm);
#endif

    // create test Fields for real-to-complex FFT
    BareField<double,D> RFieldPPStan(layoutPPStan);
    BareField<double,D> RFieldPPStan_save(layoutPPStan);
    BareField<std::complex<double>,D> CFieldPPStan0h(layoutPPStan0h);
#ifndef ONED
    BareField<double,D> RFieldSPStan(layoutSPStan);
    BareField<double,D> RFieldSPStan_save(layoutSPStan);
    BareField<std::complex<double>,D> CFieldSPStan0h(layoutSPStan0h);
    BareField<std::complex<double>,D> CFieldSPPerm0h(layoutSPPerm0h);
#endif

    // create test Fields for sine transform and real-to-complex FFT
    BareField<std::complex<double>,D> CFieldPPStan1h(layoutPPStan1h);
#ifndef ONED
    BareField<std::complex<double>,D> CFieldSPStan1h(layoutSPStan1h);
    BareField<std::complex<double>,D> CFieldSPPerm1h(layoutSPPerm1h);
#endif


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
    CFieldPPStan_save = CFieldPPStan; // Save values for checking later

    // RC FFT tests
    RFieldPPStan = real(CFieldPPStan_save);
    CFieldPPStan0h = std::complex<double>(0.0,0.0);

    RFieldPPStan_save = RFieldPPStan;  // save input data for checking results

    // create RC FFT object
    FFT<RCTransform,D,double> rcfft(ndiStandard, ndiStandard0h, compressTemps);

    // set direction names
    rcfft.setDirectionName(+1, "forward");
    rcfft.setDirectionName(-1, "inverse");

    testmsg << "RC transform using all-parallel layout ..." << endl;
    timer.start();
    // Test real<-->complex transform (simple test: forward then inverse
    // transform, see if we get back original values.

    cout << "PE " << pe << " about to invoke forward FFT::transform()" << endl;
    testmsg << "Forward transform ..." << endl;
    rcfft.transform("forward", RFieldPPStan, CFieldPPStan0h, constInput);

    testmsg << "Inverse transform ..." << endl;
    rcfft.transform("inverse", CFieldPPStan0h, RFieldPPStan, constInput);

    timer.stop();

    diffFieldPPStan = Abs(RFieldPPStan - RFieldPPStan_save);
    realDiff = max(diffFieldPPStan);
    if (fabs(realDiff) > errorTol) {
      correct = false;
      testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
    }
    testmsg << "CPU time used = " << timer.cpu_time() << " secs." << endl;
    timer.clear();
    //-------------------------------------------------------------------------

    testmsg << "RC transform using layout with zeroth dim serial ..." << endl;
    timer.start();
    // Test real<-->complex transform (simple test: forward then inverse
    // transform, see if we get back original values.

    testmsg << "Forward transform ..." << endl;
    rcfft.transform("forward", RFieldSPStan, CFieldSPStan0h, constInput);


    testmsg << "Inverse transform ..." << endl;
    rcfft.transform("inverse", CFieldSPStan0h, RFieldSPStan, constInput);

    timer.stop();

    diffFieldSPStan = Abs(RFieldSPStan - RFieldSPStan_save);
    realDiff = max(diffFieldSPStan);
    if (fabs(realDiff) > errorTol) {
      correct = false;
      testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
    }
    testmsg << "CPU time used = " << timer.cpu_time() << " secs." << endl;
    timer.clear();
    //-------------------------------------------------------------------------

    RFieldSPStan = RFieldSPStan_save;  // restore field, just in case ...

    testmsg << "RC transform using layout with axes permuted ..." << endl;
    timer.start();
    // Test real<-->complex transform (simple test: forward then inverse
    // transform, see if we get back original values.

    testmsg << "Forward transform ..." << endl;
    rcfft.transform("forward", RFieldSPStan, CFieldSPPerm0h, constInput);

    testmsg << "Inverse transform ..." << endl;
    rcfft.transform("inverse", CFieldSPPerm0h, RFieldSPStan, constInput);

    timer.stop();

    diffFieldSPStan = Abs(RFieldSPStan - RFieldSPStan_save);
    realDiff = max(diffFieldSPStan);
    if (fabs(realDiff) > errorTol) {
      correct = false;
      testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
    }
    testmsg << "CPU time used = " << timer.cpu_time() << " secs." << endl;
    timer.clear();
  }
  testmsg << "test is correct: " << correct << endl;
  return 0;
}