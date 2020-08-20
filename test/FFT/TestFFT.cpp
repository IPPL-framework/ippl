// -*- C++ -*-
/***************************************************************************
 *
 ***************************************************************************/

#include "Ippl.h"
#include "Utilities/Timer.h"

#include <fstream>
#include <complex>
using namespace std;

bool Configure(int argc, char *argv[],
	       unsigned int *nx, unsigned int *ny, unsigned int *nz,
	       int *domainDec, unsigned *processes,  unsigned int *nLoop) 
{

  Inform msg("Configure ");
  Inform errmsg("Error ");

  for (int i=1; i < argc; ++i) {
    string s(argv[i]);
    if (s == "-grid") {
      *nx = atoi(argv[++i]);
      *ny = atoi(argv[++i]);
      *nz = atoi(argv[++i]);
    }   else if (s == "-Loop") {
      *nLoop = atoi(argv[++i]);
    } else if (s == "-Decomp") {
      *domainDec = atoi(argv[++i]);
    } 
    else {
      errmsg << "Illegal format for or unknown option '" << s.c_str() << "'.";
      errmsg << endl;
    }
  }

  *processes = Ippl::getNodes();
  return true;
}


void writeMemoryHeader(std::ofstream &outputFile)
{

    std::string dateStr("no time");
    std::string timeStr("no time");
    std::string indent("        ");

    IpplMemoryUsage::IpplMemory_p memory = IpplMemoryUsage::getInstance();

    outputFile << "SDDS1" << std::endl;
    outputFile << "&description\n"
               << indent << "text=\"Memory statistics '"
               << "TestFFT" << "' "
               << dateStr << "" << timeStr << "\",\n"
               << indent << "contents=\"stat parameters\"\n"
               << "&end\n";
    outputFile << "&parameter\n"
               << indent << "name=processors,\n"
               << indent << "type=long,\n"
               << indent << "description=\"Number of Cores used\"\n"
               << "&end\n";
    outputFile << "&parameter\n"
               << indent << "name=revision,\n"
               << indent << "type=string,\n"
               << indent << "description=\"git revision of TestFFT\"\n"
               << "&end\n";
    outputFile << "&parameter\n"
               << indent << "name=flavor,\n"
               << indent << "type=string,\n"
               << indent << "description=\"n.a\"\n"
               << "&end\n";
    outputFile << "&column\n"
               << indent << "name=t,\n"
               << indent << "type=double,\n"
               << indent << "units=ns,\n"
               << indent << "description=\"1 Time\"\n"
               << "&end\n";
    outputFile << "&column\n"
               << indent << "name=memory,\n"
               << indent << "type=double,\n"
               << indent << "units=" + memory->getUnit() + ",\n"
               << indent << "description=\"2 Total Memory\"\n"
               << "&end\n";

    unsigned int columnStart = 3;

    for (int p = 0; p < Ippl::getNodes(); ++p) {
        outputFile << "&column\n"
                   << indent << "name=processor-" << p << ",\n"
                   << indent << "type=double,\n"
                   << indent << "units=" + memory->getUnit() + ",\n"
                   << indent << "description=\"" << columnStart
                   << " Memory per processor " << p << "\"\n"
                   << "&end\n";
        ++columnStart;
    }

    outputFile << "&data\n"
               << indent << "mode=ascii,\n"
               << indent << "no_row_counts=1\n"
               << "&end\n";

    outputFile << Ippl::getNodes() << std::endl;
    outputFile << "IPPL test" << " " << "no version" << " git rev. #" << "no version " << std::endl;
    outputFile << "IPPL test" << std::endl;
}

void open_m(std::ofstream& os, const std::string& fileName, const std::ios_base::openmode mode) {
    os.open(fileName.c_str(), mode);
    os.precision(15);
    os.setf(std::ios::scientific, std::ios::floatfield);
}

void writeMemoryData(std::ofstream &os_memData, unsigned int pwi, unsigned int step)
{
    os_memData << step << "\t";     // 1

    IpplMemoryUsage::IpplMemory_p memory = IpplMemoryUsage::getInstance();

    int nProcs = Ippl::getNodes();
    double total = 0.0;
    for (int p = 0; p < nProcs; ++p) {
        total += memory->getMemoryUsage(p);
    }

    os_memData << total << std::setw(pwi) << "\t";

    for (int p = 0; p < nProcs; p++) {
        os_memData << memory->getMemoryUsage(p)  << std::setw(pwi);

        if ( p < nProcs - 1 )
            os_memData << "\t";

    }
    os_memData << std::endl;
}

int main(int argc, char *argv[])
{
  Ippl ippl(argc,argv);
  Inform testmsg(NULL,0);
  const unsigned D=3U;

  unsigned int nx,ny,nz;
  int domDec;
  unsigned processes;
  unsigned int nLoop; 

  Configure(argc, argv, &nx, &ny, &nz, &domDec, &processes, &nLoop);
  
  // Compression of temporaries:
  bool compressTemps = true;
  bool constInput = true;  // preserve input field in two-field transform

  unsigned ngrid[D];   // grid sizes
  ngrid[0] = nx;
  ngrid[1] = ny;
  ngrid[2] = ny;

  // Used in evaluating correctness of results:
  double realDiff;
  const double errorTol = 1.0e-10;
  //  bool correct = true;

  // Various counters, constants, etc:
  //  double pi = acos(-1.0);
  //double twopi = 2.0*pi;


  // probes etc.

  std::string baseFn = std::string(argv[0])  + 
    std::string("-mx=") + std::to_string(nx) +
    std::string("-my=") + std::to_string(ny) +
    std::string("-mz=") + std::to_string(nz) +
    std::string("-p=") + std::to_string(processes);

  //  unsigned int pwi = 10;
  //  std::ios_base::openmode mode_m = std::ios::out;

  //  std::ofstream os_memData;
  //open_m(os_memData, baseFn+std::string(".mem"), mode_m);

  static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");           
  IpplTimings::startTimer(mainTimer);                                                    

  static IpplTimings::TimerRef fftccppTimer = IpplTimings::getTimer("FFTCCPP");           
  static IpplTimings::TimerRef ifftccppTimer = IpplTimings::getTimer("IFFTCCPP");
  static IpplTimings::TimerRef fftccpsTimer = IpplTimings::getTimer("FFTCCPS");           
  static IpplTimings::TimerRef ifftccpsTimer = IpplTimings::getTimer("IFFTCCPS");

  static IpplTimings::TimerRef fftrcppTimer = IpplTimings::getTimer("FFTRCPP");           
  static IpplTimings::TimerRef ifftrcppTimer = IpplTimings::getTimer("IFFTRCPP");
  static IpplTimings::TimerRef fftrcpsTimer = IpplTimings::getTimer("FFTRCPS");           
  static IpplTimings::TimerRef ifftrcpsTimer = IpplTimings::getTimer("IFFTRCPS");
           
  static IpplTimings::TimerRef fEvalccppTimer = IpplTimings::getTimer("fEvalCCPP");
  static IpplTimings::TimerRef fEvalccpsTimer = IpplTimings::getTimer("fEvalCCPS");

  static IpplTimings::TimerRef fEvalrcppTimer = IpplTimings::getTimer("fEvalRCPP");
  static IpplTimings::TimerRef fEvalrcpsTimer = IpplTimings::getTimer("fEvalRCPS");
           
  static IpplTimings::TimerRef fInitTimer = IpplTimings::getTimer("init fields");           
  static IpplTimings::TimerRef fsetupTimer = IpplTimings::getTimer("setup fields");           

  //  writeMemoryHeader(os_memData);
  
  e_dim_tag allParallel[D];    // Specifies SERIAL, PARALLEL dims
  for (unsigned int d=0; d<D; d++)
    allParallel[d] = PARALLEL;

  e_dim_tag serialParallel[D]; // Specifies SERIAL, PARALLEL dims
  serialParallel[0] = SERIAL;
  for (unsigned int d=1; d<D; d++) 
    serialParallel[d] = PARALLEL;

  testmsg << "Make fields" << endl;
  IpplTimings::startTimer(fsetupTimer);

  //------------------------complex<-->complex-------------------------------
  // Complex test Fields

  // create standard domain
  NDIndex<D> ndiStandard;
  for (unsigned int d=0; d<D; d++) 
    ndiStandard[d] = Index(ngrid[d]);
    // create new domain with axes permuted to match FFT output
  NDIndex<D> ndiPermuted;
  ndiPermuted[0] = ndiStandard[D-1];
  for (unsigned int d=1; d<D; d++) 
    ndiPermuted[d] = ndiStandard[d-1];

  // create half-size domain for RC transform along zeroth axis
  NDIndex<D> ndiStandard0h = ndiStandard;
  ndiStandard0h[0] = Index(ngrid[0]/2+1);

  // create new domain with axes permuted to match FFT output
  NDIndex<D> ndiPermuted0h;
  ndiPermuted0h[0] = ndiStandard0h[D-1];
  for (unsigned int d=1; d<D; d++) 
    ndiPermuted0h[d] = ndiStandard0h[d-1];

  // create half-size domain for sine transform along zeroth axis
  // and RC transform along first axis
  NDIndex<D> ndiStandard1h = ndiStandard;
  ndiStandard1h[1] = Index(ngrid[1]/2+1);

  // create new domain with axes permuted to match FFT output
  NDIndex<D> ndiPermuted1h;
  ndiPermuted1h[0] = ndiStandard1h[D-1];
  for (unsigned int d=1; d<D; d++) 
    ndiPermuted1h[d] = ndiStandard1h[d-1];

  // all parallel layout, standard domain, normal axis order
  FieldLayout<D> layoutPPStan(ndiStandard,allParallel,processes);

  // zeroth axis serial, standard domain, normal axis order
  FieldLayout<D> layoutSPStan(ndiStandard,serialParallel,processes);
  // zeroth axis serial, standard domain, permuted axis order
  FieldLayout<D> layoutSPPerm(ndiPermuted,serialParallel,processes);

  // all parallel layout, zeroth axis half-size domain, normal axis order
  FieldLayout<D> layoutPPStan0h(ndiStandard0h,allParallel,processes);

  // zeroth axis serial, zeroth axis half-size domain, normal axis order
  FieldLayout<D> layoutSPStan0h(ndiStandard0h,serialParallel,processes);
  // zeroth axis serial, zeroth axis half-size domain, permuted axis order
  FieldLayout<D> layoutSPPerm0h(ndiPermuted0h,serialParallel,processes);


  // all parallel layout, first axis half-size domain, normal axis order
  FieldLayout<D> layoutPPStan1h(ndiStandard1h,allParallel,processes);

  // zeroth axis serial, first axis half-size domain, normal axis order
  FieldLayout<D> layoutSPStan1h(ndiStandard1h,serialParallel,processes);
  // zeroth axis serial, first axis half-size domain, permuted axis order
  FieldLayout<D> layoutSPPerm1h(ndiPermuted1h,serialParallel,processes);


  // create test Fields for complex-to-complex FFT
  BareField<std::complex<double>,D> CFieldPPStan(layoutPPStan);
  BareField<std::complex<double>,D> CFieldPPStan_save(layoutPPStan);
  BareField<double,D> diffFieldPPStan(layoutPPStan);

  BareField<std::complex<double>,D> CFieldSPStan(layoutSPStan);
  BareField<std::complex<double>,D> CFieldSPStan_save(layoutSPStan);
  BareField<double,D> diffFieldSPStan(layoutSPStan);
  BareField<std::complex<double>,D> CFieldSPPerm(layoutSPPerm);

  // create test Fields for real-to-complex FFT
  BareField<double,D> RFieldPPStan(layoutPPStan);
  BareField<double,D> RFieldPPStan_save(layoutPPStan);
  BareField<std::complex<double>,D> CFieldPPStan0h(layoutPPStan0h);

  BareField<double,D> RFieldSPStan(layoutSPStan);
  BareField<double,D> RFieldSPStan_save(layoutSPStan);
  BareField<std::complex<double>,D> CFieldSPStan0h(layoutSPStan0h);
  BareField<std::complex<double>,D> CFieldSPPerm0h(layoutSPPerm0h);

  // create test Fields for sine transform and real-to-complex FFT
  BareField<std::complex<double>,D> CFieldPPStan1h(layoutPPStan1h);

  BareField<std::complex<double>,D> CFieldSPStan1h(layoutSPStan1h);
  BareField<std::complex<double>,D> CFieldSPPerm1h(layoutSPPerm1h);

  testmsg << "Initialize fields ..." << endl;
  IpplTimings::stopTimer(fsetupTimer);

  IpplTimings::startTimer(fInitTimer);
  // Rather more complete test functions (sine or cosine mode):
  // std::complex<double> sfact(1.0,0.0);      // (1,0) for sine mode; (0,0) for cosine mode
  // std::complex<double> cfact(0.0,0.0);      // (0,0) for sine mode; (1,0) for cosine mode

 /*
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
  */
  CFieldPPStan = std::complex<double>(0.0,0.0);// 
  CFieldSPStan = std::complex<double>(0.0,0.0); //CFieldPPStan;
  CFieldSPPerm = std::complex<double>(0.0,0.0);

  CFieldPPStan_save = CFieldPPStan; // Save values for checking later
  CFieldSPStan_save = CFieldSPStan;
  IpplTimings::stopTimer(fInitTimer);

  // CC FFT tests
  // Instantiate complex<-->complex FFT object
  // Transform in all directions
  FFT<CCTransform,D,double> ccfft(ndiStandard, compressTemps);
  
  // set direction names
  ccfft.setDirectionName(+1, "forward");
  ccfft.setDirectionName(-1, "inverse");

  testmsg << nLoop << " x In-place CC transform using all-parallel layout ..." << endl;
  for (uint i=0; i<nLoop; i++) {
    IpplTimings::startTimer(fftccppTimer);
    ccfft.transform("forward", CFieldPPStan);
    IpplTimings::stopTimer(fftccppTimer);
    IpplTimings::startTimer(ifftccppTimer);
    ccfft.transform("inverse", CFieldPPStan);
    IpplTimings::stopTimer(ifftccppTimer);
    IpplTimings::startTimer(fEvalccppTimer);
    diffFieldPPStan = Abs(CFieldPPStan - CFieldPPStan_save);
    realDiff = max(diffFieldPPStan);
    if (fabs(realDiff) > errorTol) {
      //      correct = false;
      testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
    }
    IpplTimings::stopTimer(fEvalccppTimer);
  }

  testmsg << "In-place CC transform using layout with zeroth dim serial ..." << endl;
  for (uint i=0; i<nLoop; i++) {
    IpplTimings::startTimer(fftccpsTimer);
    ccfft.transform("forward", CFieldSPStan);
    IpplTimings::stopTimer(fftccpsTimer);
    IpplTimings::startTimer(ifftccpsTimer);
    ccfft.transform("inverse", CFieldSPStan);
    IpplTimings::stopTimer(ifftccpsTimer);
    IpplTimings::startTimer(fEvalccpsTimer);
    diffFieldSPStan = Abs(CFieldSPStan - CFieldSPStan_save);
    realDiff = max(diffFieldSPStan);
    if (fabs(realDiff) > errorTol) {
      //      correct = false;
      testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
    }
    IpplTimings::stopTimer(fEvalccpsTimer);
  }

  /*

  //-------------------------------------------------------------------------
    CFieldSPStan = CFieldSPStan_save;  // restore field, just in case ...

    timer.start();
    // Test complex<-->complex transform (simple test: forward then inverse
    // transform, see if get back original values.
    testmsg << "Forward transform ..." << endl;
    ccfft.transform("forward", CFieldSPStan, CFieldSPPerm, constInput);
    testmsg << "Inverse transform ..." << endl;
    ccfft.transform("inverse", CFieldSPPerm, CFieldSPStan, constInput);
    timer.stop();

    diffFieldSPStan = Abs(CFieldSPStan - CFieldSPStan_save);
    realDiff = max(diffFieldSPStan);
    if (fabs(realDiff) > errorTol) {
      correct = false;
      testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
    }
    testmsg << "CPU time used = " << timer.cpu_time() << " secs." << endl;
    timer.clear();
    //-------------------------------------------------------------------------

    */

    // RC FFT tests
    RFieldPPStan = real(CFieldPPStan_save);
    CFieldPPStan0h = std::complex<double>(0.0,0.0);
    RFieldSPStan = real(CFieldSPStan_save);
    CFieldSPStan0h = std::complex<double>(0.0,0.0);
    CFieldSPPerm0h = std::complex<double>(0.0,0.0);

    RFieldPPStan_save = RFieldPPStan;  // save input data for checking results
    RFieldSPStan_save = RFieldSPStan;

    // create RC FFT object
    FFT<RCTransform,D,double> rcfft(ndiStandard, ndiStandard0h, compressTemps);

    rcfft.setDirectionName(+1, "forward");
    rcfft.setDirectionName(-1, "inverse");

    testmsg << "RC transform using all-parallel layout ..." << endl;
    for (uint i=0; i<nLoop; i++) {
      IpplTimings::startTimer(fftrcppTimer);
      rcfft.transform("forward", RFieldPPStan, CFieldPPStan0h, constInput);
      IpplTimings::stopTimer(fftrcppTimer);
      IpplTimings::startTimer(ifftrcppTimer);
      rcfft.transform("inverse", CFieldPPStan0h, RFieldPPStan, constInput);
      IpplTimings::stopTimer(ifftrcppTimer);
      IpplTimings::startTimer(fEvalrcppTimer);
      diffFieldPPStan = Abs(RFieldPPStan - RFieldPPStan_save);
      realDiff = max(diffFieldPPStan);
      if (fabs(realDiff) > errorTol) {
	//	correct = false;
	testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
      }
      IpplTimings::stopTimer(fEvalrcppTimer);
    }

    testmsg << "RC transform using layout with zeroth dim serial ..." << endl;

    for (uint i=0; i<nLoop; i++) {
      IpplTimings::startTimer(fftrcpsTimer);
      rcfft.transform("forward", RFieldSPStan, CFieldSPStan0h, constInput);
      IpplTimings::stopTimer(fftrcpsTimer);
      IpplTimings::startTimer(ifftrcpsTimer);
      rcfft.transform("inverse", CFieldSPStan0h, RFieldSPStan, constInput);
      IpplTimings::stopTimer(ifftrcpsTimer);
      IpplTimings::startTimer(fEvalrcpsTimer);
      diffFieldSPStan = Abs(RFieldSPStan - RFieldSPStan_save);
      realDiff = max(diffFieldSPStan);
      if (fabs(realDiff) > errorTol) {
	//	correct = false;
	testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
      }
      IpplTimings::stopTimer(fEvalrcpsTimer);
    }

    /*

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

  */

    /*
    // define zeroth axis to be sine transform
    bool sineTransformDims[D];
    sineTransformDims[0] = true;
    for (unsigned int d=1; d<D; ++d) sineTransformDims[d] = false;

    // Sine and RC transform tests

    RFieldPPStan = RFieldPPStan_save;
    CFieldPPStan1h = std::complex<double>(0.0,0.0);
    RFieldSPStan = RFieldSPStan_save;
    CFieldSPStan1h = std::complex<double>(0.0,0.0);
    CFieldSPPerm1h = std::complex<double>(0.0,0.0);

    // create Sine FFT object
    FFT<SineTransform,D,double> sinefft(ndiStandard, ndiStandard1h,
                                        sineTransformDims, compressTemps);
    // set direction names
    sinefft.setDirectionName(+1,"forward");
    sinefft.setDirectionName(-1,"inverse");

    testmsg << "Sine and RC transform using all-parallel layout ..." << endl;
    timer.start();
    // Test sine transform (simple test: forward then inverse
    // transform, see if we get back original values.
    testmsg << "Forward transform ..." << endl;
    sinefft.transform("forward", RFieldPPStan, CFieldPPStan1h, constInput);
    testmsg << "Inverse transform ..." << endl;
    sinefft.transform("inverse", CFieldPPStan1h, RFieldPPStan, constInput);
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

    testmsg << "Sine and RC transform using layout with zeroth dim serial ..."
            << endl;
    timer.start();
    // Test sine transform (simple test: forward then inverse
    // transform, see if we get back original values.
    testmsg << "Forward transform ..." << endl;
    sinefft.transform("forward", RFieldSPStan, CFieldSPStan1h, constInput);
    testmsg << "Inverse transform ..." << endl;
    sinefft.transform("inverse", CFieldSPStan1h, RFieldSPStan, constInput);
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

    testmsg << "Sine and RC transform using layout with axes permuted ..."
            << endl;
    timer.start();
    // Test sine transform (simple test: forward then inverse
    // transform, see if we get back original values.
    testmsg << "Forward transform ..." << endl;
    sinefft.transform("forward", RFieldSPStan, CFieldSPPerm1h, constInput);
    testmsg << "Inverse transform ..." << endl;
    sinefft.transform("inverse", CFieldSPPerm1h, RFieldSPStan, constInput);
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

    // Sine transform tests
    RFieldPPStan = RFieldPPStan_save;
    RFieldSPStan = RFieldSPStan_save;

    // create Sine FFT object
    FFT<SineTransform,D,double> sinefft2(ndiStandard, sineTransformDims,
                                         compressTemps);
    // set direction names
    sinefft2.setDirectionName(+1,"forward");
    sinefft2.setDirectionName(-1,"inverse");

    testmsg << "In-place sine transform using all-parallel layout ..." << endl;
    timer.start();
    // Test sine transform (simple test: forward then inverse
    // transform, see if we get back original values.
    testmsg << "Forward transform ..." << endl;
    sinefft2.transform("forward", RFieldPPStan);
    testmsg << "Inverse transform ..." << endl;
    sinefft2.transform("inverse", RFieldPPStan);
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

    testmsg << "In-place sine transform using layout with zeroth dim serial ..." << endl;
    timer.start();
    // Test sine transform (simple test: forward then inverse
    // transform, see if we get back original values.
    testmsg << "Forward transform ..." << endl;
    sinefft2.transform("forward", RFieldSPStan);
    testmsg << "Inverse transform ..." << endl;
    sinefft2.transform("inverse", RFieldSPStan);
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

    RFieldSPStan = RFieldSPStan_save;  // restore input field, just in case

    testmsg << "Two-field sine transform using layout with zeroth dim serial ..." << endl;
    timer.start();
    // Test sine transform (simple test: forward then inverse
    // transform, see if we get back original values.
    testmsg << "Forward transform ..." << endl;
    sinefft2.transform("forward", RFieldSPStan, RFieldSPStan, constInput);
    testmsg << "Inverse transform ..." << endl;
    sinefft2.transform("inverse", RFieldSPStan, RFieldSPStan, constInput);
    timer.stop();

    diffFieldSPStan = Abs(RFieldSPStan - RFieldSPStan_save);
    realDiff = max(diffFieldSPStan);
    if (fabs(realDiff) > errorTol) {
      correct = false;
      testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
    }
    testmsg << "CPU time used = " << timer.cpu_time() << " secs." << endl;
    timer.clear();
    */
    IpplTimings::stopTimer(mainTimer);                                                    
    IpplTimings::print();
    IpplTimings::print(baseFn+std::string(".timing"));
    return 0;
}
/***************************************************************************
 * $RCSfile: TestFFT.cpp.org,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 ***************************************************************************/
