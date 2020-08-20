// ToyFDTD2, version 1.0
//      The if-I-can-do-it-you-can-do-it FDTD!
// Copyright (C) 1998,1999 Laurie E. Miller, Paul Hayes, Matthew O'Keefe
// Copyright (C) 1999 John Schneider

// This program is free software; you can redistribute it and/or
//     modify it under the terms of the GNU General Public License
//     as published by the Free Software Foundation; either version 2
//     of the License, or any later version, with the following conditions
//     attached in addition to any and all conditions of the GNU
//     General Public License:
//     When reporting or displaying any results or animations created
//     using this code or modification of this code, make the appropriate
//     citation referencing ToyFDTD2 by name and including the version
//     number.
//
// This program is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty
//     of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//     See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
//     along with this program; if not, write to the Free Software
//     Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
//     02111-1307  USA

// Contacting the perpetratorss:
//
// Laurie E. Miller, Paul Hayes, Matthew O'Keefe
// Department of Electrical and Computer Engineering
//      200 Union Street S. E.
//      Minneapolis, MN 55455
//
// lemiller@borg.umn.edu
//
// http://www.borg.umn.edu/toyfdtd/
// http://www.borg.umn.edu/toyfdtd/ToyFDTD2.html
// http://www.toyfdtd.org/

// This code is here for everyone, but not everyone will need something
//      so simple, and not everyone will need to read all the comments.
// This file is over 700 lines long, but less than 400 of that is actually
//      code.

///////////////////////////////////////////////////////////////////////////////
// ToyFDTD2 builds into ToyFDTD1 a way of allocating memory so that each
//      field verctor array is forced to be be contiguous, which the
//      ToyFDTD1 scheme does not.  In many cases this will be faster
//      than the ToyFDTD1 scheme, but in some cases it will be slower;
//      which will be better is highly platform-dependent.  Some optimizing
//      compilers simply force the ToyFDTD1 arrays to be contiguous.  It's
//      worth testing more than one option before undertaking a large run.
//      Both schemes are designed to be easy to read.  In ToyFDTD2, macros
//      are defined for accessing the arrays to make the notation easy to
//      read: i.e. Ez(i,j,k).  The modifications were written by
//      John Schneider.
// This ToyFDTD2 is a stripped-down, minimalist 3D FDTD code.  It
//      illustrates the minimum factors that must be considered to
//      create a simple FDTD simulation.
///////////////////////////////////////////////////////////////////////////////

// This is a very simple Yee algorithm 3D FDTD code in C implementing
//      the free space form of Maxwell's equations on a Cartesian grid.
// There are no internal materials or geometry.
// The code as delivered simulates an idealized rectangular waveguide
//      by treating the interior of the mesh as free space/air and enforcing
//      PEC (Perfect Electric Conductor) conditions on the faces of the mesh.
// The problem is taken from Field and Wave Electromagnetics, 2nd ed., by
//      David K. Cheng, pages 554-555.  It is a WG-16 waveguide
//      useful for X-band applications, interior width = 2.29cm,
//      interior height = 1.02cm.  The frequency (10 GHz) is chosen to be
//      in the middle of the frequency range for TE10 operation.
// Boundaries: PEC (Perfect Electric Conductor).
// Stimulus: A simplified sinusoidal plane wave emanates from x = 0 face.
// 3D output: The electric field intensity vector components in the direction
//      of the height of the guide (ez) are output to file every
//      PLOT_MODULUS timesteps (and for the last timestep), scaled to
//      the range of integers from zero through 254.  (The colormap
//      included in the tar file assigns rgb values to the range zero through
//      255.)   Scaling is performed on each timestep individually, since
//      it is not known in advance what the maximum and minimum
//      values will be for the entire simulation.  The integer value 127 is
//      held to be equal to the data zero for every timestep.  This method
//      of autoscaling every timestep can be very helpful in a simulation
//      where the intensities are sometimes strong and sometimes faint,
//      since it will highlight the presence and structure of faint signals
//      when stronger signals have left the mesh.
//      Each timestep has it's own output file.  This data output file format
//      can be used in several visualization tools, such as animabob.
// Some terminology used here:
//
// This code implements a Cartesian mesh with space differentials
//     of dx, dy, dz.
// This means that a point in the mesh has neighboring points dx meters
//     away in the direction of the positive and negative x-axis,
//     and neighboring points dy meters away in the directions
//     of the +- y-axis, and neighboring points dz meters away
//     in the directions of the +- z-axis,
// The mesh has nx cells in the x direction, ny in the y direction,
//     and nz in the z direction.
// ex, ey, and ez refer to the arrays of electric field intensity vectors
//     -- for example, ex is a 3-dimensional array consisting of the
//     x component of the E field intensity vector for every point in the
//     mesh.  ex[i][j][k] refers to the x component of the E field intensity
//     vector at point [i][j][k].
// hx, hy, and hz refer to the arrays of magnetic field intensity vectors.
//
// dt is the time differential -- the length of each timestep in seconds.
//
// vtk is a common unsed file format and used for E and B-field output.
//
#include "Ippl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string>
#include <fstream>

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
#define GUIDE_WIDTH 0.0229
// meters
#define GUIDE_HEIGHT 0.0102
// meters
#define LENGTH_IN_WAVELENGTHS 5.0
// length of the waveguide in wavelengths of the stimulus wave
#define CELLS_PER_WAVELENGTH 25.0
// minimum number of grid cells per wavelength in the x, y, and
//     z directions

// physical constants
#define LIGHT_SPEED 299792458.0
// speed of light in a vacuum in meters/second
#define LIGHT_SPEED_SQUARED 89875517873681764.0
// m^2/s^2
#define MU_0 1.2566370614359172953850573533118011536788677597500423283899778369231265625144835994512139301368468271e-6
// permeability of free space in henry/meter
#define EPSILON_0 8.8541878176203898505365630317107502606083701665994498081024171524053950954599821142852891607182008932e-12
// permittivity of free space in farad/meter


void dumpVTK(Field<Vektor<double,3>,3> &EFD, Field<Vektor<double,3>,3> &HFD, NDIndex<3> lDom, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(ios::scientific, ios::floatfield);

    std::stringstream fname;
    fname << "data/c_";
    fname << setw(4) << setfill('0') << iteration;
    fname << ".vtk";

    //SERIAL at the moment
    //if (Ippl::myNode() == 0) {

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), ios::out);
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "toyfdtd" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx << " " << ny << " " << nz << endl;
    vtkout << "ORIGIN 0 0 0" << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "POINT_DATA " << nx*ny*nz << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for(int z=lDom[2].first(); z<lDom[2].last(); z++) {
        for(int y=lDom[1].first(); y<lDom[1].last(); y++) {
            for(int x=lDom[0].first(); x<lDom[0].last(); x++) {
                Vektor<double, 3> tmp = EFD[x][y][z].get();
                vtkout << tmp(0) << "\t"
                       << tmp(1) << "\t"
                       << tmp(2) << endl;
            }
        }
    }

    vtkout << "VECTORS B-Field float" << endl;
    for(int z=lDom[2].first(); z<lDom[2].last(); z++) {
        for(int y=lDom[1].first(); y<lDom[1].last(); y++) {
            for(int x=lDom[0].first(); x<lDom[0].last(); x++) {
                Vektor<double, 3> tmp = HFD[x][y][z].get();
                vtkout << tmp(0) << "\t"
                       << tmp(1) << "\t"
                       << tmp(2) << endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();

}

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    // variable declarations
    int i,j,k;
    // indices of the 3D array of cells
    int nx, ny, nz;
    // total number of cells along the x, y, and z axes, respectively

    int iteration = 0;
    // counter to track how many timesteps have been computed
    double stimulus = 0.0;
    // value of the stimulus at a given timestep
    double currentSimulatedTime = 0.0;
    // time in simulated seconds that the simulation has progressed
    double totalSimulatedTime = 0.0;
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

    static IpplTimings::TimerRef ipplToyFDTDTimer = IpplTimings::getTimer("ipplToyFDTDTimer");
    static IpplTimings::TimerRef fieldUpdateTimer = IpplTimings::getTimer("fieldUpdate");
    static IpplTimings::TimerRef outputTimer = IpplTimings::getTimer("output");

    IpplTimings::startTimer(ipplToyFDTDTimer);

    // setting up the problem to be modeled
    //
    // David K. Cheng, Field and Wave Electromagnetics, 2nd ed.,
    //     pages 554-555.
    // Rectangular waveguide, interior width = 2.29cm, interior height = 1.02cm.
    // This is a WG-16 waveguide useful for X-band applications.
    //
    // Choosing nx, ny, and nz:
    // There should be at least 20 cells per wavelength in each direction,
    //     but we'll go with 25 so the animation will look prettier.
    //     (CELLS_PER_WAVELENGTH was set to 25.0 in the global
    //     constants at the beginning of the code.)
    // The number of cells along the width of the guide and the width of
    //     those cells should fit the guide width exactly, so that ny*dy
    //     = GUIDE_WIDTH meters.
    //     The same should be true for nz*dz = GUIDE_HEIGHT meters.
    // dx is chosen to be dy or dz -- whichever is smaller
    // nx is chosen to make the guide LENGTH_IN_WAVELENGTHS
    //     wavelengths long.
    //
    // dt is chosen for Courant stability; the timestep must be kept small
    //     enough so that the plane wave only travels one cell length
    //     (one dx) in a single timestep.  Otherwise FDTD cannot keep up
    //     with the signal propagation, since FDTD computes a cell only from
    //     it's immediate neighbors.

    // wavelength in meters:
    lambda = LIGHT_SPEED/FREQUENCY;
    // angular frequency in radians/second:
    omega = 2.0*M_PI*FREQUENCY;

    // set ny and dy:
    // start with a small ny:
    ny = 3;
    // calculate dy from the guide width and ny:
    dy = GUIDE_WIDTH/ny;
    // until dy is less than a twenty-fifth of a wavelength,
    //     increment ny and recalculate dy:
    while(dy >= lambda/CELLS_PER_WAVELENGTH) {
        ny++;
        dy = GUIDE_WIDTH/ny;
    }

    // set nz and dz:
    // start with a small nz:
    nz = 3;
    // calculate dz from the guide height and nz:
    dz = GUIDE_HEIGHT/nz;
    // until dz is less than a twenty-fifth of a wavelength,
    //     increment nz and recalculate dz:
    while(dz >= lambda/CELLS_PER_WAVELENGTH) {
        nz++;
        dz = GUIDE_HEIGHT/nz;
    }

    // set dx, nx, and dt:
    // set dx equal to dy or dz, whichever is smaller:
    dx = (dy < dz) ? dy : dz;
    // choose nx to make the guide LENGTH_IN_WAVELENGTHS
    //     wavelengths long:
    nx = (int)(LENGTH_IN_WAVELENGTHS*lambda/dx);
    // chose dt for Courant stability:
    dt = 1.0/(LIGHT_SPEED*sqrt(1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)));
    // time in seconds that will be simulated by the program:
    totalSimulatedTime = MAXIMUM_ITERATION*dt;

    // constants used in the field update equations:
    dtmudx = dt/(MU_0*dx);
    dtepsdx = dt/(EPSILON_0*dx);
    dtmudy = dt/(MU_0*dy);
    dtepsdy = dt/(EPSILON_0*dy);
    dtmudz = dt/(MU_0*dz);
    dtepsdz = dt/(EPSILON_0*dz);

    e_dim_tag decomp[3];
    decomp[0] = SERIAL;
    decomp[1] = PARALLEL;
    decomp[2] = PARALLEL;

    Index I(0,nx+1), J(0,ny+1), K(0,nz+1);
    Index II,JJ,KK; // to hold the local index
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

    // memory allocation for the FDTD mesh:
    // The mesh is set up so that tangential E vectors form the outer faces of
    //     the simulation volume.  There are nx*ny*nz cells in the mesh, but
    //     there are nx*(ny+1)*(nz+1) ex component vectors in the mesh.
    //     There are (nx+1)*ny*(nz+1) ey component vectors in the mesh.
    //     There are (nx+1)*(ny+1)*nz ez component vectors in the mesh.
    // If you draw out a 2-dimensional slice of the mesh, you'll see why
    //     this is.  For example if you have a 3x3x3 cell mesh, and you
    //     draw the E field components on the z=0 face, you'll see that
    //     the face has 12 ex component vectors, 3 in the x-direction
    //     and 4 in the y-direction.  That face also has 12 ey components,
    //     4 in the x-direction and 3 in the y-direction.
    Field<Vektor<double,3>,3> EFD(mymesh,FL,GuardCellSizes<3>(2));
    EFD = 0.0;

    // Allocate the H field arrays:
    // Since the H arrays are staggered half a step off
    //     from the E arrays in every direction, the H
    //     arrays are one cell smaller in the x, y, and z
    //     directions than the corresponding E arrays.
    // By this arrangement, the outer faces of the mesh
    //     consist of E components only, and the H
    //     components lie only in the interior of the mesh.
    Field<Vektor<double,3>,3> HFD(mymesh,FL,GuardCellSizes<3>(2));
    HFD = 0.0;

    msg << "exsize= " << nx*(ny+1)*(nz+1)
        << " eysize= " << (nx+1)*ny*(nz+1)
        << " ezsize= " << (nx+1)*(ny+1)*nz << endl;

    // print out some identifying information

    msg << endl;
    msg << "ipplToyFDTD2 version 1.0 " << endl;

    // print out some simulation parameters:
    msg << "PLOT_MODULUS = " <<  PLOT_MODULUS << endl;
    msg << "rectangular waveguide " << endl;
    msg << "Stimulus = Hertz " <<  FREQUENCY << endl;
    msg << "continuous plane wave " << endl;
    msg << " " << endl;
    msg << "Meshing parameters: " << endl;

    msg << "cells " <<  nx << "  " <<  ny << "  " <<  nz << endl;
    msg << "dxyz " <<  dx << "  " <<  dy << "  " <<  dz << endl;
    msg << "meter^3 simulation region " << GUIDE_WIDTH << " " <<  GUIDE_HEIGHT << "  " <<  LENGTH_IN_WAVELENGTHS*lambda << endl;
    msg << " " << endl;
    msg << "3D output scaling parameters: " << endl;
    msg << "Autoscaling every timestep " << endl;
    msg << " " << endl;
    msg << " " << endl;
    // print out some info on each timestep:
    msg << "Following is the iteration number and current " << endl;
    msg << "simulated time for each timestep/iteration of " << endl;
    msg << "the simulation.  For each timestep that 3D data is " << endl;
    msg << "output to file, the maximum and minimum data " << endl;
    msg << "values are printed here with the maximum and " << endl;
    msg << "minimum scaled values in parentheses. " << endl;
    msg << " " << endl;

    // main loop:

    for(iteration = 0; iteration < MAXIMUM_ITERATION; iteration++) {
        //Ez Output section:

        // time in simulated seconds that the simulation has progressed:
        currentSimulatedTime = dt*(double)iteration;

        // 3D data output every PLOT_MODULUS timesteps:
        //     The first time through the main loop all the data written to
        //     file will be zeros.  If anything is nonzero, there's a bug.  :>

        if ( (iteration % PLOT_MODULUS) == 0) {
            IpplTimings::startTimer(outputTimer);
            dumpVTK(EFD,HFD,lDom,nx,ny,nz,iteration,dx,dy,dz);
            IpplTimings::stopTimer(outputTimer);
        }

        // Compute the stimulus: a plane wave emanates from the x=0 face:
        //     The length of the guide lies in the x-direction, the width of the
        //     guide lies in the y-direction, and the height of the guide lies
        //     in the z-direction.  So the guide is sourced by all the ez
        //     components on the stimulus face.

        stimulus = sin(omega*currentSimulatedTime);
        // print to standard output the iteration number
        //     and current simulated time:
        msg << "Iteration " << iteration << " current Simulation time "
            <<  currentSimulatedTime << " stimulus= " << stimulus << endl;
        IpplTimings::startTimer(fieldUpdateTimer);
        JJ = lDom[1]; KK = lDom[2];
        assign(EFD[Index(0,0)][JJ][KK], Vektor<double,3>(0.0,0.0,stimulus));


        // Update the interior of the mesh:
        //    all vector components except those on the faces of the mesh.
        //
        // Update all the H field vector components within the mesh:
        //     Since all H vectors are internal, all H values are updated here.
        //     Note that the normal H vectors on the faces of the mesh are not
        //     computed here, and in fact were never allocated -- the normal
        //     H components on the faces of the mesh are never used to update
        //     any other value, so they are left out of the memory allocation
        //     entirely.

        // Update the hx values:
        if (lDom[0].max() == (nx+1))
            II = Index(lDom[0].min(),nx-1);
        else
            II = lDom[0];
        JJ = lDom[1];
        KK = lDom[2];

        HFD[II][JJ][KK](0) += dtmudz*(EFD[II+1][JJ  ][KK+1](1) - EFD[II+1][JJ][KK](1)) -
            dtmudy*(EFD[II+1][JJ+1][KK  ](2) - EFD[II+1][JJ][KK](2));

        // Update the hy values:
        if (lDom[1].max() == (ny+1))
            JJ = Index(lDom[1].min(),ny-1);
        else
            JJ = lDom[1];
        II = lDom[0];
        KK = lDom[2];

        HFD[II][JJ][KK](1) += dtmudx*(EFD[II+1][JJ+1][KK  ](2) - EFD[II][JJ+1][KK](2)) -
            dtmudz*(EFD[II  ][JJ+1][KK+1](0) - EFD[II][JJ+1][KK](0));

        // Update the hz values:
        if (lDom[2].max() == (nz+1))
            KK = Index(lDom[2].min(),nz-1);
        else
            KK = lDom[2];
        II = lDom[0];
        JJ = lDom[1];

        HFD[II][JJ][KK](2) += dtmudy*(EFD[II  ][JJ+1][KK+1](0) - EFD[II][JJ][KK+1](0)) -
            dtmudx*(EFD[II+1][JJ  ][KK+1](1) - EFD[II][JJ][KK+1](1));

        // Update the E field vector components.
        // The values on the faces of the mesh are not updated here; they
        //      are handled by the boundary condition computation
        //      (and stimulus computation).


        // Update the ex values:
        // II = Index(0,nx-1); JJ = Index(1,ny-1); KK = Index(1,nz-1);

        II = lDom[0];

        if (lDom[1].min() == 0)
            JJ = Index(1,lDom[1].max());
        else
            JJ = lDom[1];

        if (lDom[2].min() == 0)
            KK = Index(1,lDom[2].max());
        else
            KK = lDom[2];

        EFD[II][JJ][KK](0) += dtepsdy*(HFD[II][JJ  ][KK-1](2) - HFD[II][JJ-1][KK-1](2)) -
            dtepsdz*(HFD[II][JJ-1][KK  ](1) - HFD[II][JJ-1][KK-1](1));

        // Update the ey values:
        // II = Index(1,nx-1); JJ = Index(0,ny-1); KK = Index(1,nz-1);

        JJ = lDom[1];

        if (lDom[0].min() == 0)
            II = Index(1,lDom[0].max());
        else
            II = lDom[0];

        if (lDom[2].min() == 0)
            KK = Index(1,lDom[2].max());
        else
            KK = lDom[2];

        EFD[II][JJ][KK](1) += dtepsdz*(HFD[II-1][JJ][KK  ](0) - HFD[II-1][JJ][KK-1](0)) -
            dtepsdx*(HFD[II  ][JJ][KK-1](2) - HFD[II-1][JJ][KK-1](2));

        // Update the ez values:
        // II = Index(1,nx-1); JJ = Index(1,ny-1); KK = Index(0,nz-1);

        KK = lDom[2];

        if (lDom[0].min() == 0)
            II = Index(1,lDom[0].max());
        else
            II = lDom[0];

        if (lDom[1].min() == 0)
            JJ = Index(1,lDom[1].max());
        else
            JJ = lDom[1];

        EFD[II][JJ][KK](2) += dtepsdx*(HFD[II  ][JJ-1][KK](1) - HFD[II-1][JJ-1][KK](1)) -
            dtepsdy*(HFD[II-1][JJ  ][KK](0) - HFD[II-1][JJ-1][KK](0));

        IpplTimings::stopTimer(fieldUpdateTimer);

        msg << " " << endl;


        // Compute the boundary conditions:

        // OK, so I'm yanking your chain on this one.  The PEC condition is
        // enforced by setting the tangential E field components on all the
        // faces of the mesh to zero every timestep (except the stimulus
        // face).  But the lazy/efficient way out is to initialize those
        // vectors to zero and never compute them again, which is exactly
        // what happens in this code.

    }// end mainloop


    // Output section:
    // The output routine is repeated one last time to write out
    // the last data computed.

    // time in simulated seconds that the simulation has progressed:
    currentSimulatedTime = dt*(double)iteration;
    // print to standard output the iteration number
    //     and current simulated time:
    msg << "iteration " <<  iteration << " currentSimulatedTime " << currentSimulatedTime << endl;
    IpplTimings::startTimer(outputTimer);
    dumpVTK(EFD,HFD,lDom,nx,ny,nz,iteration,dx,dy,dz);
    IpplTimings::stopTimer(outputTimer);

    // write some progress notes to standard output:

    msg << " " << endl;
    msg << " " << endl;
    msg << "rectangular waveguide " << endl;
    msg << "Stimulus = %lg Hertz " << FREQUENCY << endl ;
    msg << "continuous plane wave " << endl;
    msg << " " << endl;
    msg << "Meshing parameters: " << endl;
    msg << "cells " <<  nx <<  ny << nz << endl;
    msg << " " << endl;
    msg << "3D output scaling parameters: " << endl;
    msg << "Autoscaling every timestep " << endl;
    msg << " " << endl;

    IpplTimings::stopTimer(ipplToyFDTDTimer);
    IpplTimings::print();

}
