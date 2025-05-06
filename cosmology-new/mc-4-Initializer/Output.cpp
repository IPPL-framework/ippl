/*
 *  Output.cpp
 *  Initializer
 *
 *  Created by Zarija on 1/27/10.
 *
 */

#include <mpi.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "TypesAndDefs.h"
#include "Output.h"
#include "InputParser.h"


#ifdef USENAMESPACE
namespace initializer {
#endif
using namespace std;

void Output::grid2phys(real *pos_x, real *pos_y, real *pos_z, 	 
               real *vel_x, real *vel_y, real *vel_z, 	 
               int Npart, int np, float rL) { 	 
   long i; 	 
   
   float grid2phys_pos = 1.0*rL/np; 	 
   float grid2phys_vel = 100.0*rL/np; 	 
   
   for(i=0; i<Npart; i++) { 	 
      pos_x[i] *= grid2phys_pos; 	 
      pos_y[i] *= grid2phys_pos; 	 
      pos_z[i] *= grid2phys_pos; 	 
      vel_x[i] *= grid2phys_vel; 	 
      vel_y[i] *= grid2phys_vel; 	 
      vel_z[i] *= grid2phys_vel; 	 
   } 	 
   
   return; 	 
} 	 

   
void Output::write_hcosmo_serial(real *pos_x, real *pos_y, real *pos_z, 	 
                  real *vel_x, real *vel_y, real *vel_z, 	 
                  int *id, int Npart, string outBase) { 	 
   FILE *outFile;
   ostringstream outName;
   long i;
   int MyPE, NumPEs, MasterPE, proc;
   MPI_Status status;
   
   MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
   MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
   MasterPE = 0;
   
   outName << outBase << ".bin";
   if (MyPE == MasterPE){
      outFile = fopen(outName.str().c_str(), "wb");
      for(i=0; i<Npart; i++) { 
         fwrite(&pos_x[i], sizeof(real), 1, outFile); 	 
         fwrite(&vel_x[i], sizeof(real), 1, outFile); 	 
         fwrite(&pos_y[i], sizeof(real), 1, outFile); 	 
         fwrite(&vel_y[i], sizeof(real), 1, outFile); 	 
         fwrite(&pos_z[i], sizeof(real), 1, outFile); 	 
         fwrite(&vel_z[i], sizeof(real), 1, outFile); 	 
         fwrite(&id[i], sizeof(int), 1, outFile); 	 
      }
      for (proc=0; proc<NumPEs; ++proc){  // Get data from other processors:
         if (proc == MasterPE) continue;
         MPI_Recv(pos_x, Npart, MY_MPI_REAL, proc, 101, MPI_COMM_WORLD, &status);
         MPI_Recv(pos_y, Npart, MY_MPI_REAL, proc, 102, MPI_COMM_WORLD, &status);
         MPI_Recv(pos_z, Npart, MY_MPI_REAL, proc, 103, MPI_COMM_WORLD, &status);
         MPI_Recv(vel_x, Npart, MY_MPI_REAL, proc, 104, MPI_COMM_WORLD, &status);
         MPI_Recv(vel_y, Npart, MY_MPI_REAL, proc, 105, MPI_COMM_WORLD, &status);
         MPI_Recv(vel_z, Npart, MY_MPI_REAL, proc, 106, MPI_COMM_WORLD, &status);
         MPI_Recv(id, Npart, MY_MPI_INT, proc, 107, MPI_COMM_WORLD, &status);
         for (i=0; i<Npart; ++i) {
            fwrite(&pos_x[i], sizeof(real), 1, outFile); 	 
            fwrite(&vel_x[i], sizeof(real), 1, outFile); 	 
            fwrite(&pos_y[i], sizeof(real), 1, outFile); 	 
            fwrite(&vel_y[i], sizeof(real), 1, outFile); 	 
            fwrite(&pos_z[i], sizeof(real), 1, outFile); 	 
            fwrite(&vel_z[i], sizeof(real), 1, outFile); 	 
            fwrite(&id[i], sizeof(int), 1, outFile);
         }
      }
      fclose(outFile);
   }
   else {  // Send data to master processor:
      MPI_Send(pos_x, Npart, MY_MPI_REAL, MasterPE, 101, MPI_COMM_WORLD);
      MPI_Send(pos_y, Npart, MY_MPI_REAL, MasterPE, 102, MPI_COMM_WORLD);
      MPI_Send(pos_z, Npart, MY_MPI_REAL, MasterPE, 103, MPI_COMM_WORLD);
      MPI_Send(vel_x, Npart, MY_MPI_REAL, MasterPE, 104, MPI_COMM_WORLD);
      MPI_Send(vel_y, Npart, MY_MPI_REAL, MasterPE, 105, MPI_COMM_WORLD);
      MPI_Send(vel_z, Npart, MY_MPI_REAL, MasterPE, 106, MPI_COMM_WORLD);
      MPI_Send(id, Npart, MY_MPI_INT, MasterPE, 107, MPI_COMM_WORLD);
   }
   
   return; 	 
}
   
   
void Output::write_hcosmo_parallel(real *pos_x, real *pos_y, real *pos_z, 	 
                                   real *vel_x, real *vel_y, real *vel_z, 	 
                                   int *id, int Npart, string outBase) { 	 
   FILE *outFile;
   ostringstream outName;
   long i;
   int MyPE;
   
   MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
   
   outName << outBase << ".bin." << MyPE;
   outFile = fopen(outName.str().c_str(), "wb"); 	 
   for(i=0; i<Npart; i++) { 
      fwrite(&pos_x[i], sizeof(real), 1, outFile); 	 
      fwrite(&vel_x[i], sizeof(real), 1, outFile); 	 
      fwrite(&pos_y[i], sizeof(real), 1, outFile); 	 
      fwrite(&vel_y[i], sizeof(real), 1, outFile); 	 
      fwrite(&pos_z[i], sizeof(real), 1, outFile); 	 
      fwrite(&vel_z[i], sizeof(real), 1, outFile); 	 
      fwrite(&id[i], sizeof(int), 1, outFile); 	 
   }
   fclose(outFile); 	 
   
   return; 	 
}


void Output::write_hcosmo_ascii(real *pos_x, real *pos_y, real *pos_z,
                       real *vel_x, real *vel_y, real *vel_z,
                       int Npart, string outBase) {
   long i;
   int MyPE, NumPEs, MasterPE, proc;
   MPI_Status status;
   ostringstream outName;
   ofstream of;
   
   MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
   MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
   MasterPE = 0;
   
   outName << outBase << ".txt";   
   
   if (MyPE == MasterPE){
      of.open(outName.str().c_str(), ios::out);
      for (i=0; i<Npart; ++i) {
         of  << pos_x[i] << ' ';
         of <<  vel_x[i] << ' ';
         of <<  pos_y[i] << ' ';
         of <<  vel_y[i] << ' ';
         of <<  pos_z[i] << ' ';
         of <<  vel_z[i] << endl;
      }
      for (proc=0; proc<NumPEs; ++proc){  // Get data from other processors:
         if (proc == MasterPE) continue;
         MPI_Recv(pos_x, Npart, MY_MPI_REAL, proc, 101, MPI_COMM_WORLD, &status);
         MPI_Recv(pos_y, Npart, MY_MPI_REAL, proc, 102, MPI_COMM_WORLD, &status);
         MPI_Recv(pos_z, Npart, MY_MPI_REAL, proc, 103, MPI_COMM_WORLD, &status);
         MPI_Recv(vel_x, Npart, MY_MPI_REAL, proc, 104, MPI_COMM_WORLD, &status);
         MPI_Recv(vel_y, Npart, MY_MPI_REAL, proc, 105, MPI_COMM_WORLD, &status);
         MPI_Recv(vel_z, Npart, MY_MPI_REAL, proc, 106, MPI_COMM_WORLD, &status);
         for (i=0; i<Npart; ++i) {
            of  << pos_x[i] << ' ';
            of <<  vel_x[i] << ' ';
            of <<  pos_y[i] << ' ';
            of <<  vel_y[i] << ' ';
            of <<  pos_z[i] << ' ';
            of <<  vel_z[i] << endl;
         }
      }
      of.close();
   }
   else {  // Send data to master processor:
      MPI_Send(pos_x, Npart, MY_MPI_REAL, MasterPE, 101, MPI_COMM_WORLD);
      MPI_Send(pos_y, Npart, MY_MPI_REAL, MasterPE, 102, MPI_COMM_WORLD);
      MPI_Send(pos_z, Npart, MY_MPI_REAL, MasterPE, 103, MPI_COMM_WORLD);
      MPI_Send(vel_x, Npart, MY_MPI_REAL, MasterPE, 104, MPI_COMM_WORLD);
      MPI_Send(vel_y, Npart, MY_MPI_REAL, MasterPE, 105, MPI_COMM_WORLD);
      MPI_Send(vel_z, Npart, MY_MPI_REAL, MasterPE, 106, MPI_COMM_WORLD);
   }
   
   return;
}


void Output::write_gadget(InputParser& par, real *pos_x, real *pos_y, real *pos_z,
                           real *vel_x, real *vel_y, real *vel_z, int *id, 
                           int Npart, string outBase) {
   FILE *outFile;
   ostringstream outName;
   long i;
   int np;
   real rL, Om, h, redshift;
   gadget_header ghead;
   int MyPE, NumPEs, MasterPE, proc;
   MPI_Status status;
   float x, y, z, vx, vy, vz;
   unsigned int gID;
   
   MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
   MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
   MasterPE = 0;
   outName << outBase << ".gadget";
   
   // Gadget header -- dark matter only:
   ghead.flag_sfr = 0;
   ghead.flag_feedback = 0;
   ghead.flag_cooling = 0;
   ghead.num_files = 1;
   ghead.flag_stellarage = 0;
   ghead.flag_metals = 0;
   for (i=0; i<6; ++i) {ghead.npartTotalHighWord[i]=0;}
   ghead.flag_entropy_instead_u = 0;
   
   par.getByName("box_size", rL);
   par.getByName("Omega_m", Om);
   par.getByName("hubble", h);
   par.getByName("z_in", redshift);
   ghead.BoxSize = (double)rL; // In Mpc/h
   ghead.Omega0  = (double)Om;
   ghead.OmegaLambda = (double)(1.0-Om);  // Flat universe
   ghead.HubbleParam = (double)h;
   ghead.redshift = (double)redshift;
   ghead.time = (double)(1.0/(1.0+redshift));
   
   for (i=0; i<6; ++i) {
      ghead.npart[i] = 0;
      ghead.npartTotal[i] = 0;
      ghead.mass[i] = 0.0;
   }
   par.getByName("np", np);
   ghead.npart[1] = np*np*np;
   ghead.npartTotal[1] = np*np*np;
   ghead.mass[1] = (double)27.75197*Om*pow(rL/np, 3); // In 10^10 M_sun/h
   
   // Write it:
   int blksize;
#define SKIP {fwrite(&blksize, sizeof(int), 1, outFile);}
   blksize = sizeof(ghead);
   if (MyPE == MasterPE){
      outFile = fopen(outName.str().c_str(), "wb");
      SKIP
      fwrite(&ghead, sizeof(ghead), 1, outFile);
      SKIP
      blksize = np*np*np*3*sizeof(float);
      SKIP
      for(i=0; i<Npart; i++) { 
         x = (float)pos_x[i];  // In Mpc/h
         y = (float)pos_y[i];
         z = (float)pos_z[i];
         fwrite(&x, sizeof(float), 1, outFile); 	 
         fwrite(&y, sizeof(float), 1, outFile); 	 
         fwrite(&z, sizeof(float), 1, outFile);
      }
      for (proc=0; proc<NumPEs; ++proc){  // Get data from other processors:
         if (proc == MasterPE) continue;
         MPI_Recv(pos_x, Npart, MY_MPI_REAL, proc, 101, MPI_COMM_WORLD, &status);
         MPI_Recv(pos_y, Npart, MY_MPI_REAL, proc, 102, MPI_COMM_WORLD, &status);
         MPI_Recv(pos_z, Npart, MY_MPI_REAL, proc, 103, MPI_COMM_WORLD, &status);
         for (i=0; i<Npart; ++i) {
            x = (float)pos_x[i];  // In Mpc/h
            y = (float)pos_y[i];
            z = (float)pos_z[i];
            fwrite(&x, sizeof(float), 1, outFile); 	 
            fwrite(&y, sizeof(float), 1, outFile); 	 
            fwrite(&z, sizeof(float), 1, outFile);
         }
      }
      SKIP
   }
   else {  // Send data to master processor:
      MPI_Send(pos_x, Npart, MY_MPI_REAL, MasterPE, 101, MPI_COMM_WORLD);
      MPI_Send(pos_y, Npart, MY_MPI_REAL, MasterPE, 102, MPI_COMM_WORLD);
      MPI_Send(pos_z, Npart, MY_MPI_REAL, MasterPE, 103, MPI_COMM_WORLD);
   }
   if (MyPE == MasterPE){
      blksize = np*np*np*3*sizeof(float);
      SKIP
      for(i=0; i<Npart; i++) {
         vx = (float)vel_x[i];    // In km/s
         vy = (float)vel_y[i];
         vz = (float)vel_z[i];
         fwrite(&vx, sizeof(float), 1, outFile); 	 
         fwrite(&vy, sizeof(float), 1, outFile); 	 
         fwrite(&vz, sizeof(float), 1, outFile);
      }
      for (proc=0; proc<NumPEs; ++proc){  // Get data from other processors:
         if (proc == MasterPE) continue;
         MPI_Recv(vel_x, Npart, MY_MPI_REAL, proc, 104, MPI_COMM_WORLD, &status);
         MPI_Recv(vel_y, Npart, MY_MPI_REAL, proc, 105, MPI_COMM_WORLD, &status);
         MPI_Recv(vel_z, Npart, MY_MPI_REAL, proc, 106, MPI_COMM_WORLD, &status);
         for (i=0; i<Npart; ++i) {
            vx = (float)vel_x[i];    // In km/s
            vy = (float)vel_y[i];
            vz = (float)vel_z[i];
            fwrite(&vx, sizeof(float), 1, outFile); 	 
            fwrite(&vy, sizeof(float), 1, outFile); 	 
            fwrite(&vz, sizeof(float), 1, outFile);
         }
      }
      SKIP
   }
   else {  // Send data to master processor:
      MPI_Send(vel_x, Npart, MY_MPI_REAL, MasterPE, 104, MPI_COMM_WORLD);
      MPI_Send(vel_y, Npart, MY_MPI_REAL, MasterPE, 105, MPI_COMM_WORLD);
      MPI_Send(vel_z, Npart, MY_MPI_REAL, MasterPE, 106, MPI_COMM_WORLD);
   }
   if (MyPE == MasterPE){
      blksize = np*np*np*sizeof(int);
      SKIP
      for(i=0; i<Npart; i++) {
         gID = (int)id[i];
         fwrite(&gID, sizeof(int), 1, outFile);
      }
      for (proc=0; proc<NumPEs; ++proc){  // Get data from other processors:
         if (proc == MasterPE) continue;
         MPI_Recv(id, Npart, MY_MPI_INT, proc, 107, MPI_COMM_WORLD, &status);
         for (i=0; i<Npart; ++i) {
            gID = (int)id[i];
            fwrite(&gID, sizeof(int), 1, outFile);
         }
      }
      SKIP
      fclose(outFile);
   }
   else {  // Send data to master processor:
      MPI_Send(id, Npart, MY_MPI_INT, MasterPE, 107, MPI_COMM_WORLD);
   }
   
   return;
}
   
#ifdef USENAMESPACE
}
#endif
