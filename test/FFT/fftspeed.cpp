// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 ***************************************************************************/

// test program to time FFT operations under various conditions
#include "Ippl.h"

const int DDIM = 3;

typedef double                                      DTYPE;

typedef Vert                                        Center_t;
typedef UniformCartesian<DDIM, DTYPE>               Mesh_t;
typedef CenteredFieldLayout<DDIM, Mesh_t, Center_t> FieldLayout_t;
typedef Field<DTYPE, DDIM, Mesh_t, Center_t>        Field_t;
typedef Field<std::complex<double>, DDIM, Mesh_t, Center_t>     CxField_t;
typedef FFT<RCTransform, DDIM, DTYPE>               FFT_t;

int main(int argc, char *argv[]) {
  int i, d;

  // initialize IPPL
  Ippl ippl(argc,argv);
  Inform msg(argv[0]);

  // get the FFT information from the command line.  Usage:
  //   fftspeed size iterations
  int size = 0;
  int iters = 0;
  int numserial = 1;
  if (argc < 3 || argc > 4 ||
      (size = atoi(argv[1])) < 1 ||
      (iters = atoi(argv[2])) < 1) {
    msg << "Usage: " << argv[0] << " <size> <iters>, where" << endl;
    msg << "  <size> is the number of grid points in each dimension." << endl;
    msg << "  <iters> is the number of iterations to run." << endl;
    msg << endl;
    Ippl::abort(0);
  }

  if (argc > 3) {
    numserial = atoi(argv[3]);
    if (numserial < 1 || numserial > 2)
      numserial = 1;
  }

  e_dim_tag decomp[DDIM];
  e_dim_tag decomp2[DDIM];
  for (d=0; d < DDIM; ++d) {
    decomp[d] = (d == 0) ? SERIAL : PARALLEL;
    decomp2[d] = (d < numserial ? SERIAL : PARALLEL);
  }

  // create the domain and mesh information for the real field
  NDIndex<DDIM> NDI;
  for (d=0; d < DDIM; ++d)
    NDI[d] = Index(size);
  Mesh_t mesh(NDI);
  FieldLayout_t layout(mesh, decomp);

  // create the original real field
  msg << "Creating real field with domain = " << NDI << " ..." << endl;
  Field_t rho(layout);
  rho = IpplRandom() + 1.0;

  // create the domain for the complex field, by permuting the real domain
  // and setting the 'numserial'th axis to n/2 + 1.  For one serial axis,
  // we permute by one.  For two serial axes, we permute the first two
  // axes again, since the final field will be left in that state.
  NDIndex<DDIM> NDI2;
  for (d=0; d < DDIM; ++d)
    NDI2[(d+numserial) % DDIM] = NDI[d];
  NDI2[numserial] = Index(NDI2[numserial].length() / 2 + 1);

  // permute the first numserial axes one down
  if (numserial > 1) {
    Index NDI2tmp = NDI2[numserial - 1];
    for (d=0; d < (numserial - 1); ++d)
      NDI2[numserial - d - 1] = NDI2[numserial - d - 2];
    NDI2[0] = NDI2tmp;
  }

  // finally, create the mesh and layout for the complex field
  Mesh_t mesh2(NDI2);
  FieldLayout_t layout2(mesh2, decomp2);

  // create the complex field to store the FFT results
  msg << "Creating complex field with domain = " << NDI2 << " ..." << endl;
  CxField_t crho(layout2);

  // from the complex domain, construct a permuted domain to get n/2+1 as the
  // first index.
  NDIndex<DDIM> NDItemp;
  NDIndex<DDIM> NDI3 = layout2.getDomain();
  // have to permute the first axes back again if we're using more than
  // one serial axis
  if (numserial > 1) {
    Index NDI3tmp = NDI3[0];
    for (d=0; d < (numserial - 1); ++d)
      NDI3[d] = NDI3[d + 1];
    NDI3[numserial - 1] = NDI3tmp;
  }
  // now copy NDI3 into NDItmp, permuting everything
  for (d=0; d < DDIM; ++d)
    NDItemp[d] = NDI3[(d+numserial) % DDIM];
  

  // create the FFT object.
  // NOTE: We must create the FFT object with a complex domain (the second
  // argument) that has (x/2+1, y, z) size, NOT with the domain of the final
  // field.  This is due to a quasi-bug in the FFT code.  We can still FFT
  // into a field with domain (z, x/2+1, y) and layout (S, P, P) after doing
  // this construction this way, in order to save a transpose.  If we are
  // using 2 serial axes, the final field should have layout (z, y, x/2 + 1)
  // on a S, S, P layout, but the domain given to the constuctor should be the
  // same as for 1 serial axis (x/2+1, y, z).
  msg << "Creating the FFT object with constructor complex domain = ";
  msg << NDItemp << " ..." << endl;
  FFT_t fft(layout.getDomain(), NDItemp, false, numserial);

  // FFT the field several times, timing the results
  msg << "Performing " << iters << " FFT operations ..." << endl;

  //  static Timings::TimerRef ftimer = Timings::getTimer("FFT");
  for (i=0; i < iters; ++i) {
    msg << "----------------------------" << endl;

    msg << "Doing transform at iteration " << i << " ..." << endl;
    IpplCounter ca("RC Forward Transform");
    ca.startCounter();
    fft.transform(-1, rho, crho);
    ca.stopCounter();
    ca.printIt();

    msg << "After transform, sum(crho) = " << sum(crho) << endl;

    msg << "Doing inverse transform at iteration " << i << " ..." << endl;
    IpplCounter cb("RC Reverse Transform");
    cb.startCounter();
    fft.transform(+1, crho, rho);
    cb.stopCounter();
    cb.printIt();

    msg << "After transform, sum(rho) = " << sum(rho) << endl;  

    msg << "Waiting at barrier at end of iteration " << i << " ..." << endl;
    Ippl::Comm->barrier();
  }

  // print out results
  msg << endl;
  IpplTimings::print();

  return 0;
}

/***************************************************************************
 * $RCSfile: fftspeed.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:36 $
 * IPPL_VERSION_ID: $Id: fftspeed.cpp,v 1.1.1.1 2003/01/23 07:40:36 adelmann Exp $ 
 ***************************************************************************/
