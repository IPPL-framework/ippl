/***************************************************************************
 *
 * The IPPL Framework
 *
 * Filename: TestFFTCos.cpp
 *
 * Usage: TestFFTCos #GridPoints
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/
#include "Ippl.h"
#include <complex>
using namespace std;

int main(int argc, char *argv[])
{
  Ippl ippl(argc,argv);
  static IpplTimings::TimerRef mainprgTimer = IpplTimings::getTimer("mainprg");
  static IpplTimings::TimerRef cosFFTTimer = IpplTimings::getTimer("cosFFT");
  IpplTimings::startTimer(mainprgTimer);

  Inform msg("TestFFTCos ",0);
  Inform msg2all("TestFFTCos ",INFORM_ALL_NODES);



  //msg << "HEllo " << endl;

  //msg2all << "HEllo " << endl;

  const unsigned D=3U;

  unsigned ngrid[D];   // grid sizes

  double realDiff;

  // Various counters, constants, etc:  
  double pi = acos(-1.0);
  double twopi = 2.0*pi;
  
  // Layout information:
  e_dim_tag allParallel[D];    // Specifies SERIAL, PARALLEL dims
  for (unsigned int d=0; d<D; d++) 
      allParallel[d] = PARALLEL;

  // Compression of temporaries:
  bool compressTemps = false;

  for (unsigned int d=0; d<D; d++) 
      ngrid[d] = atoi(argv[1]);
  
  NDIndex<D> ndiStandard;
  for (unsigned int d=0; d<D; d++) 
      ndiStandard[d] = Index(ngrid[d]);

  msg << ndiStandard << endl;

  // all parallel layout, standard domain, normal axis order
  FieldLayout<D> layoutPPStan(ndiStandard,allParallel);

  msg << layoutPPStan << endl;

  BareField<double,D> RField(layoutPPStan);
  BareField<double,D> RField_save(layoutPPStan);
  BareField<double,D> diffField(layoutPPStan);    

  double xfact, kx, yfact, ky, zfact, kz;
  xfact = pi/(ngrid[0] + 1.0);
  yfact = 2.0*twopi/(ngrid[1]);
  zfact = 2.0*twopi/(ngrid[2]);
  kx = 1.0; ky = 2.0; kz = 3.0; // wavenumbers


  RField[ndiStandard[0]][ndiStandard[1]][ndiStandard[2]] = 

    1.0  * ( sin( (ndiStandard[0]+1) * kx * xfact +
		   ndiStandard[1]    * ky * yfact +
		   ndiStandard[2]    * kz * zfact ) +
	       sin( (ndiStandard[0]+1) * kx * xfact -
		   ndiStandard[1]    * ky * yfact -
		   ndiStandard[2]    * kz * zfact ) ) + 
    0.0  * (-cos( (ndiStandard[0]+1) * kx * xfact +
		   ndiStandard[1]    * ky * yfact +
		   ndiStandard[2]    * kz * zfact ) + 
	       cos( (ndiStandard[0]+1) * kx * xfact -
		   ndiStandard[1]    * ky * yfact -
		   ndiStandard[2]    * kz * zfact ) );
  
  RField_save = RField;

  bool cosTransformDims[D];
  for (unsigned int d=0; d<D; ++d) 
      cosTransformDims[d] = true;
  
  // ToDo
  // FFT<CosTransform,D,double> cosfft2(ndiStandard, cosTransformDims, compressTemps);
  
  FFT<SineTransform,D,double> sinefft2(ndiStandard, cosTransformDims, compressTemps);

  msg << &sinefft2 << endl;
  msg << "In-place cosine transform using all-parallel layout ..." << endl;

  IpplTimings::startTimer(cosFFTTimer);
  sinefft2.transform(-1, RField);
  sinefft2.transform( 1, RField);
  IpplTimings::stopTimer(cosFFTTimer);

  // ToDo
  // cosfft2.transform(-1, RField);
  // cosfft2.transform( 1, RField);
  
  diffField = Abs(RField - RField_save);
  realDiff = max(diffField);
  msg << "fabs(realDiff) = " << fabs(realDiff) << endl;

  IpplTimings::stopTimer(mainprgTimer);
  IpplTimings::print();
  IpplTimings::print(string(argv[0])+string(".timing"));
  return 0;
}
/***************************************************************************
 * $RCSfile: TestFFT.cpp.org,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 ***************************************************************************/
