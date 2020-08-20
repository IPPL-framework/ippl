/***************************************************************************
 *
 * The IPPL Framework
 *
 * Filename: TestFFTSin.cpp
 *
 * Usage: TestFFTSin #GridPoints
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
  Inform testmsg(NULL,0);

  const unsigned D=3U;
  unsigned ngrid[D];   // grid sizes

  double realDiff;

  // Various counters, constants, etc:
  unsigned int d;
  double pi = acos(-1.0);
  double twopi = 2.0*pi;
  
  Timer timer;

  // Layout information:
  e_dim_tag allParallel[D];    // Specifies SERIAL, PARALLEL dims
  for (d=0; d<D; d++) 
    allParallel[d] = PARALLEL;

  // Compression of temporaries:
  bool compressTemps = false;

  for (d=0; d<D; d++) 
      ngrid[d] = atoi(argv[1]);
  
  NDIndex<D> ndiStandard;
  for (d=0; d<D; d++) 
      ndiStandard[d] = Index(ngrid[d]);

  // all parallel layout, standard domain, normal axis order
  FieldLayout<D> layoutPPStan(ndiStandard,allParallel);

  BareField<double,D> RFieldPPStan(layoutPPStan);
  BareField<double,D> RFieldPPStan_save(layoutPPStan);
  BareField<double,D> diffFieldPPStan(layoutPPStan);    

  double xfact, kx, yfact, ky, zfact, kz;
  xfact = pi/(ngrid[0] + 1.0);
  yfact = 2.0*twopi/(ngrid[1]);
  zfact = 2.0*twopi/(ngrid[2]);
  kx = 1.0; ky = 2.0; kz = 3.0; // wavenumbers
  RFieldPPStan[ndiStandard[0]][ndiStandard[1]][ndiStandard[2]] = 
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
  
  bool sineTransformDims[D];
  for (d=0; d<D; ++d) 
    sineTransformDims[d] = true;
  
  RFieldPPStan_save = RFieldPPStan[ndiStandard[0]][ndiStandard[1]][ndiStandard[2]];

  RFieldPPStan = RFieldPPStan_save;
    
  FFT<SineTransform,D,double> sinefft2(ndiStandard, sineTransformDims, compressTemps);

  testmsg << "In-place sine transform using all-parallel layout ..." << endl;
  timer.start();
  sinefft2.transform(-1, RFieldPPStan);
  sinefft2.transform( 1, RFieldPPStan);
  timer.stop();
  
  diffFieldPPStan = Abs(RFieldPPStan - RFieldPPStan_save);
  realDiff = max(diffFieldPPStan);
  testmsg << "fabs(realDiff) = " << fabs(realDiff) << endl;
  testmsg << "CPU time used = " << timer.cpu_time() << " secs." << endl;
  timer.clear();
  return 0;
}
/***************************************************************************
 * $RCSfile: TestFFT.cpp.org,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 ***************************************************************************/
