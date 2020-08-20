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
#include "Utilities/Timer.h"

#include <fstream>

#include <complex>
using namespace std;

bool Configure(int argc, char *argv[],
	       unsigned int *nx, unsigned int *ny, unsigned int *nz,
	       int *domainDec, unsigned int *processes,  unsigned int *nLoop) 
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
  Inform msg(NULL,0);

  const unsigned D=3U;
  bool compressTemps = true;
  bool constInput    = true;  // preserve input field in two-field transform

  unsigned int processes;
  int domDec;
  unsigned int nx,ny,nz;
  unsigned int nLoop;

  Configure(argc, argv, &nx, &ny, &nz, &domDec, &processes, &nLoop); 

  /*
    domDec == 3: 3D domain dec all parallel          
    domDec == 1: 1D domain dec x,y serial    z: parallel 
    domDec == 2: 2D domain dec x,y parallel  z: serial
   */

  if (domDec == 3)
    msg << "3D domain decomposition " << endl;
  else if (domDec == 1)
    msg << "1D domain decomposition: serial dimension is x,y" << endl;
  else if (domDec == 2)
    msg << "2D domain decomposition: serial dimension is z" << endl;
  else {
    msg << "Domain decompostion not known going for 3D, all parallel" << endl;
    domDec = 3;
  }

  std::string baseFn = std::string(argv[0])  + 
    std::string("-mx=") + std::to_string(nx) +
    std::string("-my=") + std::to_string(ny) +
    std::string("-mz=") + std::to_string(nz) +
    std::string("-p=") + std::to_string(processes) +
    std::string("-ddec=") + std::to_string(domDec) ;

  unsigned int pwi = 10;
  std::ios_base::openmode mode_m = std::ios::out;

  std::ofstream os_memData;
  open_m(os_memData, baseFn+std::string(".mem"), mode_m);

  static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");           
  IpplTimings::startTimer(mainTimer);                                                    

  static IpplTimings::TimerRef fftTimer = IpplTimings::getTimer("FFT");           
  static IpplTimings::TimerRef ifftTimer = IpplTimings::getTimer("IFFT");           
  static IpplTimings::TimerRef fEvalTimer = IpplTimings::getTimer("fEval");           
  static IpplTimings::TimerRef fInitTimer = IpplTimings::getTimer("init");           

  writeMemoryHeader(os_memData);

  int vnodes = processes;
  unsigned ngrid[D];   // grid sizes
  ngrid[0] = nx;
  ngrid[1] = ny;
  ngrid[2] = nz;

  // Used in evaluating correctness of results:
  double realDiff;
   
  // Various counters, constants, etc:
  
  const double pi = acos(-1.0);
  const double twopi = 2.0*pi;

  e_dim_tag allParallel[D];    // Specifies SERIAL, PARALLEL dims
  e_dim_tag serialParallel[D]; // Specifies SERIAL, PARALLEL dims

  if (domDec == 3) {
    for (unsigned int d=0; d<D; d++) 
      allParallel[d] = PARALLEL;
  }
  if (domDec == 2) {
    for (unsigned int d=0; d<D; d++) 
      serialParallel[d] = PARALLEL;
    serialParallel[2] = SERIAL;
  } 
  else {
    for (unsigned int d=0; d<D; d++) 
      serialParallel[d] = SERIAL;
    serialParallel[2] = PARALLEL;
  }

  IpplTimings::startTimer(fInitTimer);

  // create standard domain
  NDIndex<D> ndiStandard;
  for (unsigned int d=0; d<D; d++) 
    ndiStandard[d] = Index(ngrid[d]);

  // create new domain with axes permuted to match FFT output
  
  // create half-size domain for RC transform along zeroth axis
  NDIndex<D> ndiStandard0h = ndiStandard;
  ndiStandard0h[0] = Index(ngrid[0]/2+1);
  
  // all parallel layout, standard domain, normal axis order
  FieldLayout<D> layoutPPStan(ndiStandard,allParallel,vnodes);
  // zeroth axis serial, standard domain, normal axis order
  FieldLayout<D> layoutSPStan(ndiStandard,serialParallel,vnodes);
  
  // all parallel layout, zeroth axis half-size domain, normal axis order
  FieldLayout<D> layoutPPStan0h(ndiStandard0h,allParallel,vnodes);
  // zeroth axis serial, zeroth axis half-size domain, normal axis order
  FieldLayout<D> layoutSPStan0h(ndiStandard0h,serialParallel,vnodes);
    

  // create test Fields for complex-to-complex FFT
  BareField<std::complex<double>,D> CFieldPPStan(layoutPPStan);
  
  BareField<std::complex<double>,D> CFieldSPStan(layoutSPStan);
  
  BareField<double,D> diffFieldSPStan(layoutSPStan);
  
  // create test Fields for real-to-complex FFT
  BareField<double,D> RFieldSPStan(layoutSPStan);
  BareField<double,D> RFieldSPStan_save(layoutSPStan);
  BareField<std::complex<double>,D> CFieldSPStan0h(layoutSPStan0h);
  
  // Rather more complete test functions (sine or cosine mode):
  std::complex<double> sfact(1.0,0.0);      // (1,0) for sine mode; (0,0) for cosine mode
  std::complex<double> cfact(0.0,0.0);      // (0,0) for sine mode; (1,0) for cosine mode
  
  double xfact, kx, yfact, ky, zfact, kz;
  xfact = pi/(ngrid[0] + 1.0);
  yfact = 2.0*twopi/(ngrid[1]);
  zfact = 2.0*twopi/(ngrid[2]);
  kx = 1.0; ky = 2.0; kz = 32.0; // wavenumbers

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
  // RC FFT tests  

  RFieldSPStan = real(CFieldPPStan);

  CFieldSPStan0h = std::complex<double>(0.0,0.0);
  IpplTimings::stopTimer(fInitTimer);

  // create RC FFT object
  FFT<RCTransform,D,double> rcfft(ndiStandard, ndiStandard0h, compressTemps);

  // set direction names
  rcfft.setDirectionName(+1, "forward");
  rcfft.setDirectionName(-1, "inverse");

  msg << "RC transform using compress tmp and  constInput" << endl;

  IpplTimings::startTimer(fftTimer);                                                    
  rcfft.transform("forward", RFieldSPStan,  CFieldSPStan0h, constInput);
  IpplTimings::stopTimer(fftTimer);                                       
             
  IpplTimings::startTimer(ifftTimer);                                                    
  rcfft.transform("inverse", CFieldSPStan0h, RFieldSPStan, constInput);
  IpplTimings::stopTimer(ifftTimer);                                       

  IpplMemoryUsage::IpplMemory_p memory = IpplMemoryUsage::getInstance();
  memory->sample();
  writeMemoryData(os_memData, pwi, 0);

  for (unsigned i=0; i<nLoop; i++) {
    RFieldSPStan_save = RFieldSPStan;

    IpplTimings::startTimer(fftTimer);                                                    
    rcfft.transform("forward", RFieldSPStan,  CFieldSPStan0h, constInput);
    IpplTimings::stopTimer(fftTimer);                                                    

    IpplTimings::startTimer(ifftTimer);                                                    
    rcfft.transform("inverse", CFieldSPStan0h, RFieldSPStan, constInput);
    IpplTimings::stopTimer(ifftTimer);                                                    

    IpplTimings::startTimer(fEvalTimer);                                         
    diffFieldSPStan = Abs(RFieldSPStan - RFieldSPStan_save);
    realDiff = max(diffFieldSPStan);
    IpplTimings::stopTimer(fEvalTimer);                                         
    msg << "fabs(realDiff) = " << fabs(realDiff) << endl;

    memory->sample();
    writeMemoryData(os_memData, pwi, i+1);

  }

  IpplTimings::stopTimer(mainTimer);                                                    
  IpplTimings::print();
  IpplTimings::print(baseFn+std::string(".timing"));
  return 0;
}




  /* cout << "TYPEINFO:" << endl;
  cout << typeid(RFieldSPStan[0][0][0]).name() << endl;
  cout << typeid(CFieldPPStan[0][0][0]).name() << endl;
  */
  
  /*
  for(int x = ndiStandard[0].first(); x <= ndiStandard[0].last(); x++) {
    for(int y = ndiStandard[1].first(); y <= ndiStandard[1].last(); y++) {
      for(int z = ndiStandard[2].first(); z <= ndiStandard[2].last(); z++) {
	RFieldSPStan[x][y][z] = real(CFieldPPStan[x][y][z].get());
      }
    }
  }
  */


  //  Inform fo2(NULL,"FFTrealField.dat",Inform::OVERWRITE);
  


    //double total_time = 0;
    //total_time+= timer.cpu_time();
    /*
      Inform fo2(NULL,"FFTrealResult.dat",Inform::OVERWRITE);
      for(int x = ndiStandard[0].first(); x <= ndiStandard[0].last(); x++) {
      for(int y = ndiStandard[1].first(); y <= ndiStandard[1].last(); y++) {
      for(int z = ndiStandard[2].first(); z <= ndiStandard[2].last(); z++) {
      fo2 << x << " " << y << " " << z << " " <<  RFieldSPStan[x][y][z].get() << endl;
      }
      }
      }
    */


/***************************************************************************
 * $RCSfile: TestRC.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 ***************************************************************************/

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $ 
 ***************************************************************************/

