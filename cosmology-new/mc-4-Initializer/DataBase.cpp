/*
   Initializer:
   DataBase.cpp

      Has only 
         GetParameters(Basedata& bdata, ParallelTools& Parallel)
      routine which reads in simulation and cosmology 
      parameters from the main code's Basedata and stors them into 
      DataBase class (defined in DataBase.h).

                     Zarija Lukic, February 2009
                           zarija@lanl.gov
*/

#include <iostream>
#include <fstream>
#include "TypesAndDefs.h"
#include "Ippl.h"
#include "DataBase.h"

initializer::GlobalStuff& initializer::GlobalStuff::instance() {
    static initializer::GlobalStuff instance;
    return instance;
}

#include "InputParser.h"

#ifdef USENAMESPACE
namespace initializer {
#endif

  void GlobalStuff::GetParameters(InputParser& par){

    Inform msg ("DataBase ");

    int np;
    if(!par.getByName("np", np))  // Grid size is equal to np^3
      msg << "Error: np not found!" << endl;
    ngrid = np;
   
    if(!par.getByName("box_size", box_size))
       msg << "Error: box_size not found!" << endl;
   
    int decomposition_type=3;
    
    dim = decomposition_type;

    int sseed;
    if(!par.getByName("seed", sseed))
      msg << "Error: seed not found!" << endl;
    seed = sseed;

   if(!par.getByName("z_in", z_in))
       msg << "Error: z_in not found!" << endl;
      
   // Get cosmology parameters:
   if(!par.getByName("hubble", Hubble))
       msg << "Error: hubble not found!" << endl;

   if(!par.getByName("Omega_m", Omega_m))
       msg << "Error: Omega_m not found!" << endl;

   if(!par.getByName("Omega_bar", Omega_bar))
       msg << "Error: Omega_bar not found!" << endl;

   if(!par.getByName("Omega_nu", Omega_nu))
       msg << "Error: Omega_nu not found!" << endl;

   if(!par.getByName("Omega_r", Omega_r))
     msg << "Error: Omega_r not found set Omega_r to zero!" << endl;
   else
     Omega_r = 0.0;
   
   if(!par.getByName("Sigma_8", Sigma_8))
       msg << "Error: Sigma_8 not found!" << endl;
   
   if(!par.getByName("n_s", n_s))
       msg << "Error: n_s not found!" << endl;
   
   if(!par.getByName("w_de", w_de))
       msg << "Error: w_de not found!" << endl;
   
   if(!par.getByName("TFFlag", TFFlag))
       msg << "Error: TFFlag not found!" << endl;

   // Get neutrino parameters:
   if(!par.getByName("N_nu", N_nu))
       msg << "Error: N_nu not found!" << endl;
   
   if(!par.getByName("nu_pairs", nu_pairs))
       msg << "Error: nu_pairs not found!" << endl;
   
   // Get non-gaussianity parameters:
   if(!par.getByName("f_NL", f_NL)){
     msg << "Warning: f_NL not found, assuming it is zero!" << endl;
     f_NL = 0.0;
   }
   
   // Get code parameters:
   if(!par.getByName("PrintFormat", PrintFormat))
      msg << "Error: PrintFormat not found!" << endl;

   
   // Make some sanity checks:
   // Errors:
   if (Omega_m > 1.0)
     throw IpplException("DataBase", "Cannot initialize closed universe!");
   
   if (Omega_bar > Omega_m)
     throw IpplException("DataBase", "More baryons than total matter content!");
   if (dim < 1 || dim > 3)
     throw IpplException("DataBase", "1, 2 or 3D decomposition is possible!");
   if (Omega_nu > 0 && nu_pairs < 1)
     throw IpplException("DataBase", "Omega_nu > 0, yet nu_pairs < 1!");
      
   // Warnings:
   if (z_in > 1000.0)
     msg << "Starting before CMB epoch. Make sure you want that." << endl;
   if (Sigma_8 > 1.0)
     msg << "Sigma(8) > 1. Make sure you want that." << endl;
   if (Hubble > 1.0)
     msg << "H > 100 km/s/Mpc. Make sure you want that." << endl;

   msg << "Done loading parameters." << endl;
   return;
}

#ifdef USENAMESPACE
}
#endif
