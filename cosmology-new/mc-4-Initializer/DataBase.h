/*
   Initializer:
   DataBase.h
 
      Defines GlobalStuff class. (main makes a globaly visible 
      instance of that class -- DataBase.) 
      This class will hold simulation 
      and cosmology parameters needed by most of routines. 
 
                     Zarija Lukic, February 2009
                          zarija@lanl.gov
 */

#ifndef DataBase_Header_Included
#define DataBase_Header_Included

#include "TypesAndDefs.h"
#include "InputParser.h"
#include <string>

#ifdef USENAMESPACE
namespace initializer {
#endif

  //class ParallelTools;

class GlobalStuff {
public:
  static GlobalStuff& instance();
  GlobalStuff(const GlobalStuff&) = delete;
  GlobalStuff& operator=(const GlobalStuff&) = delete;
  GlobalStuff(GlobalStuff&&) = delete;
  GlobalStuff& operator=(GlobalStuff&&) = delete;
private:
  GlobalStuff() = default;

public :
  void GetParameters(InputParser& par);

  // Simulation parameters:
  integer ngrid;       // Number of mesh points in any direction
  real box_size;       // Lenght of the box size in Mpc/h
  integer dim;         // dimensionality of domain decomposition
  unsigned long seed;  // Master seed for random numbers
  real z_in;           // Starting redshift
	
  // Cosmological parameters:
  real Omega_m;    // Total matter fraction (today)
  real Omega_bar;  // Baryon fraction (today)
  real Omega_nu;   // neutrino fraction
  real Hubble;     // Hubble constant
  real Sigma_8;    // Mass RMS in 8Mpc/h spheres
  real n_s;        // Spectral index
  real w_de;       // Dark energy EOS parameter
  real f_NL;       // f_NL for non-gaussianity
  integer TFFlag;  /* Transfer function used: 0=CMBFast, 1=KH, 
		      2=HS, 3=PD, 4=BBKS */
	
  // Neutrino parameters:
  int N_nu, nu_pairs;
  
  // Code stuff:
  integer PrintFormat;  /* 0 - serial ASCII, 1 - serial binary, 
			   2 - parallel binary, 3 - HDF5, 
			   4 - Gadget */
};

#ifdef USENAMESPACE
}
#endif

#endif
