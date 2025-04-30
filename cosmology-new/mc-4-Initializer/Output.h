/*
 *  Output.h
 *  Initializer
 *
 *  Created by Zarija on 1/27/10.
 *
 */

#ifndef Output_Header_Included
#define Output_Header_Included

#include <string>
#include "InputParser.h"

#ifdef USENAMESPACE
namespace initializer {
#endif
using namespace std;
   
class Output {

public:   
   void grid2phys(real *pos_x, real *pos_y, real *pos_z, 	 
                  real *vel_x, real *vel_y, real *vel_z, 	 
                  int Npart, int np, float rL);
   void write_hcosmo_ascii(real *pos_x, real *pos_y, real *pos_z,
                           real *vel_x, real *vel_y, real *vel_z,
                           int Npart, string outBase);
   void write_hcosmo_serial(real *pos_x, real *pos_y, real *pos_z, 	 
                            real *vel_x, real *vel_y, real *vel_z, 	 
                            integer *id, int Npart, string outBase);
   void write_hcosmo_parallel(real *pos_x, real *pos_y, real *pos_z, 	 
                              real *vel_x, real *vel_y, real *vel_z, 	 
                              integer *id, int Npart, string outBase);
   void write_gadget(InputParser& par, real *pos_x, real *pos_y, real *pos_z,
                     real *vel_x, real *vel_y, real *vel_z, integer *id, 
                     int Npart, string outBase);

private:
   
   struct gadget_header
   {
      int      npart[6];
      double   mass[6];
      double   time;
      double   redshift;
      int      flag_sfr;
      int      flag_feedback;
      unsigned int  npartTotal[6];
      int      flag_cooling;
      int      num_files;
      double   BoxSize;
      double   Omega0;
      double   OmegaLambda;
      double   HubbleParam;
      int      flag_stellarage;
      int      flag_metals;
      unsigned int  npartTotalHighWord[6];
      int      flag_entropy_instead_u;
      char     fill[60];  /* fills to 256 Bytes */
   };
};

#ifdef USENAMESPACE
}   
#endif
   
#endif
