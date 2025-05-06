/*
   Initializer:
   Cosmology.h

      Defines CosmoStuff class. This will be used 
      for calculating many quantities -- transfer function, 
      mass variance on certain scale, growth factor. 
      See Cosmology.cpp for more explanations and 
      actual implementation. 

                        Zarija Lukic, February 2009
                              zarija@lanl.gov
*/

#ifndef Cosmology_Header_Included
#define Cosmology_Header_Included

#include "TypesAndDefs.h"

#ifdef USENAMESPACE
namespace initializer {
#endif
class GlobalStuff;

class CosmoClass{
      
   public :
      CosmoClass() { };
      void SetParameters(GlobalStuff& DataBase, const char *tfName);
      real Sigma_r(real R, real norm);
      void GrowthFactor(real z, real *gf, real *g_dot);
      real TransferFunction(real k);
      void FDVelocity(real &x, real &y, real &z);
      ~CosmoClass();
      
   private :
      real cobe_temp, tt;
      real Omega_m, Omega_bar, Omega_nu, Omega_r, h, n_s, w_de, z_in;
      real sound_horizon, alpha_nu, beta_c, f_nu, N_nu;
      int TFFlag;
      real *table_kk, *table_tf;
      unsigned long table_size;

      real kh_tmp, R_M, Pk_norm, gf_tmp_x, last_k;
      
      //neutrino stuff
      real *neut_p, *neut_c;
      real neut_vv0;
      static const unsigned int neut_nmax = 1000;
      //FIXME: check upper limit for integral
      //static const unsigned int neut_pmax = 3.0e5;
      
      real (CosmoClass::*tan_f)(real k);
      real integrate(real (CosmoClass::*func)(real), real a, real b);
      real midpoint(real (CosmoClass::*func)(real), real a, real b, int n);
      real interpolate(real xx[], real yy[], unsigned long n, real x);
      void locate(real xx[], unsigned long n, real x, unsigned long *j);
      void hunt(real xx[], unsigned long n, real x, unsigned long *jlo);
      real eisenstein_hu(real k);
      real klypin_holtzmann(real k);
      real hu_sugiyama(real k);
      real peacock_dodds(real k);
      real bbks(real k);
      real cmbfast(real k);
      real sigma2(real k);
   
  void growths(real a, real y[], real dydx[]);
  void odesolve(real ystart[], int nvar, real x1, real x2, real eps, real h1,
		void (CosmoClass::*derivs)(real, real [], real []), bool print_stat);
  void rkqs(real y[], real dydx[], int n, real *x, real htry, real eps, 
	    real yscal[], real *hdid, real *hnext, int *feval, 
	    void (CosmoClass::*derivs)(real, real [], real []));
  void rkck(real y[], real dydx[], int n, real x, real h, real yout[],
	    real yerr[], void (CosmoClass::*derivs)(real, real [], real []));

  //neutrino stuff
  void genFDDist();
  real FD(real vv);
  real Maxwell(real vv);
};

#ifdef USENAMESPACE
}
#endif

#endif
