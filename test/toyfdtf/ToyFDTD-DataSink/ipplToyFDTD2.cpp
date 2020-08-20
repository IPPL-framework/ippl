/** 
    ipplToyFDTD2, version 1.0, 599 lines for a full parallel 3D Maxwell solver -- that rocks !
  
    This is a very simple Yee algorithm 3D FDTD code in C++ implementing the free space form of Maxwell's equations on a Cartesian grid.
    The code is based on IP^2L framework which takes case of the parallelization. 
    There are no internal materials or geometry. The code as delivered simulates an idealized rectangular waveguide by treating the 
    interior of the mesh as free space/air and enforcing PEC (Perfect Electric Conductor) conditions on the faces of the mesh.
    The problem is taken from Field and Wave Electromagnetics, 2nd ed., by David K. Cheng, pages 554-555.  It is a WG-16 waveguide
    useful for X-band applications, interior width = 2.29cm, interior height = 1.02cm.  The frequency (10 GHz) is chosen to be
    in the middle of the frequency range for TE10 operation.
 
    3D output: The electric and magnetic fields are written in vtk format to the directory Data which must exist.
    The output frequency to the file is controled by PLOT_MODULUS (and for the last timestep).

    This code implements a Cartesian mesh with space differentials of dx, dy, dz.
    This means that a point in the mesh has neighboring points dx meters away in the direction of the positive and negative x-axis,
    and neighboring points dy meters away in the directions of the +- y-axis, and neighboring points dz meters away
    in the directions of the +- z-axis.
    The mesh has nx cells in the x direction, ny in the y direction, and nz in the z direction.

    EFD and HDF refer to the parallel fields holding the electric and magnetic field respectively

    dt is the time differential -- the length of each timestep in seconds.

    vtk is a common unsed file format and used for E and B-field output.

    Acknowledment: 
    The idea (base) of this code comes from the if-I-can-do-it-you-can-do-it FDTD!
    Copyright (C) 1998,1999 Laurie E. Miller, Paul Hayes, Matthew O'Keefe
    Copyright (C) 1999 John Schneider
*/

#include "Ippl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string>
#include <fstream>

#include "DataSink.h"

// program control constants
#define MAXIMUM_ITERATION 100
// total number of timesteps to be computed

#define PLOT_MODULUS 5
// The program will output 3D data every PLOT_MODULUS timesteps,
//     except for the last iteration computed, which is always
//     output.  So if MAXIMUM_ITERATION is not an integer
//     multiple of PLOT_MODULUS, the last timestep output will
//     come after a shorter interval than that separating
//     previous outputs.

#define FREQUENCY 10.0e9
// frequency of the stimulus in Hertz

#define GUIDE_WIDTH 0.0916
// meters
#define GUIDE_HEIGHT 0.0408
// meters
#define LENGTH_IN_WAVELENGTHS 5.0
// length (meters) of the waveguide in wavelengths of the stimulus wave

#define CELLS_PER_WAVELENGTH 25.0
// minimum number of grid cells per wavelength in the x, y, and z directions

// physical constants
#define LIGHT_SPEED 299792458.0
// speed of light in a vacuum in meters/second
#define LIGHT_SPEED_SQUARED 89875517873681764.0
// m^2/s^2
#define MU_0 1.2566370614359172953850573533118011536788677597500423283899778369231265625144835994512139301368468271e-6
// permeability of free space in henry/meter
#define EPSILON_0 8.8541878176203898505365630317107502606083701665994498081024171524053950954599821142852891607182008932e-12
// permittivity of free space in farad/meter

#define MODE_k 2.0;
#define MODE_l 1.0;
#define MODE_m 0.0;

enum EVALTYPE_T {HX,HY,HZ,EX,EY,EZ};
typedef enum EVALTYPE_T EVALTYPE;

void initialize(Field<Vektor<double, 3>, 3> &EFD, Field<Vektor<double, 3>, 3> &HFD, NDIndex<3> lDom,
                const int &nx, const int &ny, const int &nz, 
                const double &kx, const double &ky, const double &kz, 
                const double &phase_x, const double &phase_y, const double &phase_z,
                const double &dt, const double &omega) {


  NDIndex<3> elem;
  double B_0 = omega * sin(-omega*0.5*dt) / (LIGHT_SPEED*LIGHT_SPEED * M_PI);
  for (int i = lDom[0].first() - 2; i <= lDom[0].last() + 2; ++ i) {
    elem[0] = Index(i,i);
    for (int j = lDom[1].first() - 2; j <= lDom[1].last() + 2; ++ j) {
      elem[1] = Index(j,j);
      for (int k = lDom[2].first() - 2; k <= lDom[2].last() + 2; ++ k) {
	elem[2] = Index(k,k);

	// E-field edge centered 
	EFD.localElement(elem) = Vektor<double, 3>((ky*ky + kz*kz) * cos((0.5 + i) * phase_x) * sin(j * phase_y) * sin(k * phase_z),
						   -kx * ky * sin(i * phase_x) * cos((0.5 + j) * phase_y) * sin(k * phase_z),
						   -kx * kz * sin(i * phase_x) * sin(j * phase_y) * cos((0.5 + k) * phase_z));
	// H-field face centered 
	HFD.localElement(elem) = Vektor<double, 3>(0.0,
						   B_0 * kz * cos((0.5 + i) * phase_x) * sin(j * phase_y) * cos((0.5 + k) * phase_z),
						   B_0 * ky * cos((0.5 + i) * phase_x) * cos((0.5 + j) * phase_y) * sin(k * phase_z));

	// E-field face centered		
	//                 EFD.localElement(elem) = Vektor<double, 3>((ky*ky + kz*kz) * cos(i * phase_x) * sin((0.5 + j) * phase_y) * sin((0.5 + k) * phase_z),
	//                                                            -kx * ky * sin((0.5 + i) * phase_x) * cos(j * phase_y) * sin((0.5 + k) * phase_z),
	//                                                           -kx * kz * sin((0.5 + i) * phase_x) * sin((0.5 + j) * phase_y) * cos(k * phase_z));
	//              H-field edge centered 
	//                 HFD.localElement(elem) = Vektor<double, 3>(0.0,
	//                                                            B_0 * kz * cos(i * phase_x) * sin((0.5 + j) * phase_y) * cos(k * phase_z),
	//                                                            B_0 * ky * cos(i * phase_x) * cos(j * phase_y) * sin((0.5 + k) * phase_z));
      }
    }
  }
  if (lDom[0].first() == 0) {
    elem[0] = Index(-2,-1);
    elem[1] = Index(lDom[1].first() - 2, lDom[1].last() + 2);
    elem[2] = Index(lDom[2].first() - 2, lDom[2].last() + 2);
    EFD.localElement(elem) = 0.0;
    HFD.localElement(elem) = 0.0;
  }

  if (lDom[0].last() == nx) {
    elem[0] = Index(nx, nx + 2);
    elem[1] = Index(lDom[1].first() - 2, lDom[1].last() + 2);
    elem[2] = Index(lDom[2].first() - 2, lDom[2].last() + 2);
    EFD.localElement(elem) = 0.0;
    HFD.localElement(elem) = 0.0;
  }

  if (lDom[1].first() == 0) {
    elem[0] = Index(lDom[0].first() - 2, lDom[0].last() + 2);
    elem[1] = Index(-2, -1);
    elem[2] = Index(lDom[2].first() - 2, lDom[2].last() + 2);
    EFD.localElement(elem) = 0.0;
    HFD.localElement(elem) = 0.0;
  }

  if (lDom[1].last() == ny) {
    elem[0] = Index(lDom[0].first() - 2, lDom[0].last() + 2);
    elem[1] = Index(ny, ny + 2);
    elem[2] = Index(lDom[2].first() - 2, lDom[2].last() + 2);
    EFD.localElement(elem) = 0.0;
    HFD.localElement(elem) = 0.0;
  }

  if (lDom[2].first() == 0) {
    elem[0] = Index(lDom[0].first() - 2, lDom[0].last() + 2);
    elem[1] = Index(lDom[1].first() - 2, lDom[1].last() + 2);
    elem[2] = Index(-2, -1);
    EFD.localElement(elem) = 0.0;
    HFD.localElement(elem) = 0.0;
  }

  if (lDom[2].last() == nz) {
    elem[0] = Index(lDom[0].first() - 2, lDom[0].last() + 2);
    elem[1] = Index(lDom[1].first() - 2, lDom[1].last() + 2);
    elem[2] = Index(nz, nz + 2);
    EFD.localElement(elem) = 0.0;
    HFD.localElement(elem) = 0.0;
  }
}

void updateIJK(Index &II, Index &JJ, Index &KK, NDIndex<3> lDom, const int nx, const int ny, const int nz, EVALTYPE whatToDo ) 
{
  /**
     This function calculates the local indexes depending on the
     field (E or H) and the dimension (x,y or z).

  */

  II = lDom[0];
  JJ = lDom[1];
  KK = lDom[2];

  switch (whatToDo) {
  case HX:
    if (lDom[0].min() == 0)
      II = Index(1, lDom[0].max());
    else
      II = Index(lDom[0].min(), lDom[0].max());

    if (lDom[0].max() == nx)
      II = Index(II.min(), nx-1);
        
    if (lDom[1].max() == ny)
      JJ = Index(JJ.min(), ny-1);
	
    if (lDom[2].max() == nz)
      KK = Index(KK.min(), nz-1);
    break;
  case HY:
    if (lDom[1].min() == 0)
      JJ = Index(1,lDom[1].max());
    else
      JJ = Index(lDom[1].min(),lDom[1].max());
    if (lDom[1].max() == ny)
      JJ = Index(JJ.min(), ny-1);

    II = lDom[0];
    if (lDom[0].max() == nx)
      II = Index(II.min(), nx-1);

    KK = lDom[2];
    if (lDom[2].max() == nz)
      KK = Index(KK.min(), nz-1);
    break;
  case HZ:
    if (lDom[2].max() == nz)
      KK = Index(lDom[2].min(),nz-1);
    else
      KK = lDom[2];
    if (lDom[2].max() == nz)
      KK = Index(KK.min(), nz-1);

    II = lDom[0];
    if (lDom[0].max() == nx)
      II = Index(II.min(), nx-1);

    JJ = lDom[1];
    if (lDom[1].max() == ny)
      JJ = Index(JJ.min(), ny-1);
    break;
  case EX:
    if (lDom[0].max() == nx)
      II = Index(II.min(), nx-1);

    if (lDom[1].min() == 0)
      JJ = Index(1, lDom[1].max());
    else
      JJ = lDom[1];
    if (lDom[1].max() == ny)
      JJ = Index(JJ.min(), ny-1);

    if (lDom[2].min() == 0)
      KK = Index(1, lDom[2].max());
    else
      KK = lDom[2];
    if (lDom[2].max() == nz)
      KK = Index(KK.min(), nz-1);
    break;
  case EY:
    JJ = lDom[1];
    if (lDom[1].max() == ny)
      JJ = Index(JJ.min(), ny-1);

    if (lDom[0].min() == 0)
      II = Index(1, lDom[0].max());
    else
      II = lDom[0];
    if (lDom[0].max() == nx)
      II = Index(II.min(), nx-1);

    if (lDom[2].min() == 0)
      KK = Index(1, lDom[2].max());
    else
      KK = lDom[2];
    if (lDom[2].max() == nz)
      KK = Index(KK.min(), nz-1);
    break;
  case EZ:
    KK = lDom[2];
    if (lDom[2].max() == nz)
      KK = Index(KK.min(), nz-1);

    if (lDom[0].min() == 0)
      II = Index(1, lDom[0].max());
    else
      II = lDom[0];
    if (lDom[0].max() == nx)
      II = Index(II.min(), nx-1);

    if (lDom[1].min() == 0)
      JJ = Index(1, lDom[1].max());
    else
      JJ = lDom[1];
    if (lDom[1].max() == ny)
      JJ = Index(JJ.min(), ny-1);
    break;
  }
}

int main(int argc, char *argv[]){
  Ippl ippl(argc, argv);
  Inform msg(argv[0]);
  Inform msg2all(argv[0],INFORM_ALL_NODES);

  // indices of the 3D array of cells
  int nx, ny, nz;
  // total number of cells along the x, y, and z axes, respectively

  int iteration = 0;
  // counter to track how many timesteps have been computed
  double currentSimulatedTime = 0.0;
  // time in seconds that will be simulated by the program
  double omega;
  // angular frequency in radians/second
  double lambda;
  // wavelength of the stimulus in meters
  double dx, dy, dz;
  // space differentials (or dimensions of a single cell) in meters
  double dt;
  // time differential (how much time between timesteps) in seconds
  double dtmudx, dtepsdx;
  // physical constants used in the field update equations
  double dtmudy, dtepsdy;
  // physical constants used in the field update equations
  double dtmudz, dtepsdz;
  // physical constants used in the field update equations

  double kx, ky, kz;
  double phase_x;
  double phase_y;
  double phase_z;

  bool dump_binary = true;

  static IpplTimings::TimerRef ipplToyFDTDTimer = IpplTimings::getTimer("ipplToyFDTDTimer");
  static IpplTimings::TimerRef fieldUpdateTimer = IpplTimings::getTimer("fieldUpdate");
  static IpplTimings::TimerRef outputTimer = IpplTimings::getTimer("output");

  IpplTimings::startTimer(ipplToyFDTDTimer);

  /**
     setting up the problem to be modeled
    
     David K. Cheng, Field And Wave Electromagnetics, 2nd Ed.,
     Pages 554-555.
     Rectangular waveguide, interior width = 2.29cm, interior height = 1.02cm.
     This is a WG-16 waveguide useful for X-band applications.
    
     Choosing nx, ny, and nz:
     There should be at least 20 cells per wavelength in each direction,
     but we'll go with 25 so the animation will look prettier.
     (CELLS_PER_WAVELENGTH was set to 25.0 in the global
     constants at the beginning of the code.)
     The number of cells along the width of the guide and the width of
     those cells should fit the guide width exactly, so that ny*dy
     = GUIDE_WIDTH meters.
     The same should be true for nz*dz = GUIDE_HEIGHT meters.
     dx is chosen to be dy or dz -- whichever is smaller
    
     nx is chosen to make the guide LENGTH_IN_WAVELENGTHS wavelengths long.
    
     dt is chosen for Courant stability; the timestep must be kept small
     enough so that the plane wave only travels one cell length
     (one dx) in a single timestep.  Otherwise FDTD cannot keep up
     with the signal propagation, since FDTD computes a cell only from
     it's immediate neighbors.
  */
	
  // wavelength in meters:
  lambda = LIGHT_SPEED/FREQUENCY;
    
  // angular frequency in radians/second:
  omega = 2.0*M_PI*FREQUENCY;

  // set ny and dy start with a small ny:
  ny = 3;
  // calculate dy from the guide width and ny:
  dy = GUIDE_WIDTH/ny;
  // until dy is less than a twenty-fifth of a wavelength, increment ny and recalculate dy:
  while(dy >= lambda/CELLS_PER_WAVELENGTH) {
    ny++;
    dy = GUIDE_WIDTH/ny;
  }
  ky = 1.0 / GUIDE_WIDTH * MODE_k;
  phase_y = ky * M_PI * dy;
  // set nz and dz: start with a small nz:
  nz = 3;
  // calculate dz from the guide height and nz:
  dz = GUIDE_HEIGHT/nz;
  // until dz is less than a twenty-fifth of a wavelength, increment nz and recalculate dz:
  while(dz >= lambda/CELLS_PER_WAVELENGTH) {
    nz++;
    dz = GUIDE_HEIGHT/nz;
  }
  kz = 1.0 / GUIDE_HEIGHT * MODE_l;
  phase_z = kz * M_PI * dz;
    
  // set dx equal to dy or dz, whichever is smaller:
  dx = (dy < dz) ? dy : dz;
    
  // choose nx to make the guide LENGTH_IN_WAVELENGTHS wavelengths long:
  nx = (int)(LENGTH_IN_WAVELENGTHS*lambda/dx);
  kx = 1.0 / (dx * nx) * MODE_m;
  phase_x = kx * M_PI * dx;

  // chose dt for Courant stability:
  dt = 0.80/(LIGHT_SPEED*sqrt(1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)));

  // constants used in the field update equations:
  dtmudx = dt/(MU_0*dx);
  dtepsdx = dt/(EPSILON_0*dx);
  dtmudy = dt/(MU_0*dy);
  dtepsdy = dt/(EPSILON_0*dy);
  dtmudz = dt/(MU_0*dz);
  dtepsdz = dt/(EPSILON_0*dz);

  // here we can choose the parallel decomposition scheme
  e_dim_tag decomp[3];
  decomp[0] = SERIAL;
  decomp[1] = PARALLEL;
  decomp[2] = PARALLEL;

  Index I(0,nx+1), J(0,ny+1), K(0,nz+1);
  Index II,JJ,KK; 
  NDIndex<3> elem;
  UniformCartesian<3> mymesh(I,J,K);
  CenteredFieldLayout<3,UniformCartesian<3>,Cell> FL(mymesh,decomp);
  NDIndex<3> lDom = FL.getLocalNDIndex();


  msg << " ----------------------------------------------------" << endl;
  msg << "    L O C A L  D O M A I N E S ... " << endl;
  msg << " ----------------------------------------------------" << endl;
  msg2all << lDom << endl;
  Ippl::Comm->barrier();
  msg << " ----------------------------------------------------" << endl;
  msg << endl;
  msg << " ----------------------------------------------------" << endl;
  /**
     The mesh is set up so that tangential E vectors form the outer faces of
     the simulation volume.  There are nx*ny*nz cells in the mesh, but
     there are nx*(ny+1)*(nz+1) ex component vectors in the mesh.
     There are (nx+1)*ny*(nz+1) ey component vectors in the mesh.
     There are (nx+1)*(ny+1)*nz ez component vectors in the mesh.

     If you draw out a 2-dimensional slice of the mesh, you'll see why
     this is.  For example if you have a 3x3x3 cell mesh, and you
     draw the E field components on the z=0 face, you'll see that
     the face has 12 ex component vectors, 3 in the x-direction
     and 4 in the y-direction.  That face also has 12 ey components,
     4 in the x-direction and 3 in the y-direction.
  */
  Field<Vektor<double,3>,3> EFD(mymesh,FL,GuardCellSizes<3>(2));
  EFD = 0.0;

  /**
     Allocate the HFD field: Since the HDF arrays is staggered half a step off
     from EFD in every direction, the H arrays are one cell smaller in the x, y, and z
     directions than the corresponding E arrays.
     By this arrangement, the outer faces of the mesh consist of E components only, and the H
     components lie only in the interior of the mesh.
  */

  Field<Vektor<double,3>,3> HFD(mymesh,FL,GuardCellSizes<3>(2));
  HFD = 0.0;

  // good named to not need much explanation
  DataSink VTKDump(mymesh,FL,2,nx,ny,nz,dx,dy,dz,dump_binary,lDom);

  msg << "ipplToyFDTD2 version 1.0 " << endl;	
    
  msg << "exsize= " << nx*(ny+1)*(nz+1) 
      << " eysize= " << (nx+1)*ny*(nz+1)
      << " ezsize= " << (nx+1)*(ny+1)*nz << endl;

  msg << endl;
   

  // print out some simulation parameters:
  msg << "PLOT_MODULUS = " <<  PLOT_MODULUS << endl;
  msg << "Meshing parameters: Cells " <<  nx << "  " <<  ny << "  " <<  nz;
  msg << " d_xyz " <<  dx << "  " <<  dy << "  " <<  dz << endl;
  msg << "meter^3 simulation region " << GUIDE_WIDTH << " " <<  GUIDE_HEIGHT << "  " <<  LENGTH_IN_WAVELENGTHS*lambda << endl;
  initialize(EFD, HFD, lDom, nx, ny, nz, kx, ky, kz, phase_x, phase_y, phase_z, dt, omega);

  for(iteration = 0; iteration < MAXIMUM_ITERATION; iteration++) {

    // time in simulated seconds that the simulation has progressed:
    currentSimulatedTime = dt*(double)iteration;

    // 3D data output every PLOT_MODULUS timesteps:
    if ( (iteration % PLOT_MODULUS) == 0) {
      IpplTimings::startTimer(outputTimer);
      VTKDump.setIteration(iteration/PLOT_MODULUS);
      VTKDump.dump(EFD,HFD);
      IpplTimings::stopTimer(outputTimer);
    }

    msg	<< "Iteration " << iteration << " current Simulation time " << currentSimulatedTime << endl;
    IpplTimings::startTimer(fieldUpdateTimer);

    /**
       Update the interior of the mesh all vector components except those on the faces of the mesh.
       Update all the H field vector components within the mesh. Since all H vectors are internal, 
       all H values are updated here.
       Note that the normal H vectors on the faces of the mesh are not
       computed here, and in fact were never allocated -- the normal
       H components on the faces of the mesh are never used to update
       any other value.
       
       Update in turn the hx, hy and hz values:
    */

    updateIJK(II, JJ, KK, lDom, nx, ny, nz, HX);
    HFD[II][JJ][KK](0) += dtmudz*(EFD[II][JJ  ][KK+1](1) - EFD[II][JJ][KK](1)) -
      dtmudy*(EFD[II][JJ+1][KK  ](2) - EFD[II][JJ][KK](2));

    updateIJK(II, JJ, KK, lDom, nx, ny, nz, HY);
    HFD[II][JJ][KK](1) += dtmudx*(EFD[II+1][JJ][KK](2) - EFD[II][JJ][KK](2)) -
      dtmudz*(EFD[II][JJ][KK+1](0) - EFD[II][JJ][KK](0));

    updateIJK(II, JJ, KK, lDom, nx, ny, nz, HZ);
    HFD[II][JJ][KK](2) += dtmudy*(EFD[II][JJ+1][KK](0) - EFD[II][JJ][KK](0)) -
      dtmudx*(EFD[II+1][JJ][KK](1) - EFD[II][JJ][KK](1));

    /** 
	Update the E field vector components. The values on the faces of the mesh 
	are not updated here are handled by the boundary condition computation

	Update in turn the ex, ey and ez values:
    */
		
    updateIJK(II, JJ, KK, lDom, nx, ny, nz, EX);
    EFD[II][JJ][KK](0) += dtepsdy*(HFD[II][JJ][KK](2) - HFD[II][JJ-1][KK  ](2)) -
      dtepsdz*(HFD[II][JJ][KK](1) - HFD[II][JJ  ][KK-1](1));

    updateIJK(II, JJ, KK, lDom, nx, ny, nz, EY);
    EFD[II][JJ][KK](1) += dtepsdz*(HFD[II][JJ][KK](0) - HFD[II  ][JJ][KK-1](0)) -
      dtepsdx*(HFD[II][JJ][KK](2) - HFD[II-1][JJ][KK  ](2));

    updateIJK(II, JJ, KK, lDom, nx, ny, nz, EZ);
    EFD[II][JJ][KK](2) += dtepsdx*(HFD[II][JJ][KK](1) - HFD[II-1][JJ  ][KK](1)) -
      dtepsdy*(HFD[II][JJ][KK](0) - HFD[II  ][JJ-1][KK](0));

    IpplTimings::stopTimer(fieldUpdateTimer);	
		
    /**
       Compute the boundary conditions this is usually the place where the coffee
       bill starts to kick in ....
       OK, so I'm yanking your chain on this one.  The PEC condition is
       enforced by setting the tangential E field components on all the
       faces of the mesh to zero every timestep  But the lazy/efficient way out is 
       to initialize those vectors to zero and never compute them again, which is exactly
       what happens in this code.

    */
  }

  // time in simulated seconds that the simulation has progressed:
  currentSimulatedTime = dt*(double)iteration;
  msg << "iteration " <<  iteration << " currentSimulatedTime " << currentSimulatedTime << endl;
  IpplTimings::startTimer(outputTimer);
  VTKDump.setIteration(iteration/PLOT_MODULUS);
  VTKDump.dump(EFD,HFD);
  IpplTimings::stopTimer(outputTimer);
  msg << "Parallel 3D simulation continuous plane wave rectangular waveguide done ..." << endl;
  IpplTimings::stopTimer(ipplToyFDTDTimer);
  IpplTimings::print();
}
