// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 * This program was prepared by PSI. 
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
//-----------------------------------------------------------------------------
// The IPPL Framework - Visit http://people.web.psi.ch/adelmann/ for more details
//
// This program was prepared by the Regents of the University of California at
// ParticleDebugFunctions.h , Tim Williams 8/6/1998
// Helper functions to print out (formatted ASCII) ParticleAttrib elements.
// Intended mainly for use from within a debugger, called interactively, but
// also callable as template functions from source code. To call from many
// debuggers, the user has to provide nontemplate wrapper functions, as
// described in ParticleDebugFunctions.cpp.

// include files

#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
#include "Utility/ParticleDebugFunctions.h"

// forward class declarations
//class Inform;

//----------------------------------------------------------------------
// Set up the I/O "stream" (IPPL Inform object) format where the output goes.
// This "sticks around" once set until re-set.
//----------------------------------------------------------------------
Inform* PtclDbgInform;           // Pointer to active Inform object
bool PtclDbgInformIsSet = false; // Flags whether pointer is set
void setPtclDbgInform(Inform& inform) {
  
  PtclDbgInform = &inform;
  PtclDbgInformIsSet = true;
}

//-----------------------------------------------------------------------------
// Set up some pre-instantiated common template particle debug functions, for
// common ParticleAttrib types.
// Uncallable from the debuggers I've tried, so comment out and use named
// nontemplate function workaraound below. --TJW 8/7/98
//-----------------------------------------------------------------------------
// Forward template function declarations
// template<class T>
// void pap(ParticleAttrib<T>& pattr, bool docomm);
// template<class T>
// void epap(ParticleAttrib<T>& pattr, int i, bool docomm);
// template<class T>
// void spap(ParticleAttrib<T>& pattr, 
// 	  int ibase, int ibound, int istride, bool docomm);
// #include "Utility/ParticleDebug.h"

// Need these additional forward class declarations as well
// template<class T> class ParticleAttrib;
// template<class T, unsigned D> class Vektor;

// template void epap(ParticleAttrib<double>& pattr, int i, bool docomm);
// template void spap(ParticleAttrib<double>& pattr, int base, int bnd, 
// 		   int stride, bool docomm);
// template void epap(ParticleAttrib<int>& pattr, int i, bool docomm);
// template void spap(ParticleAttrib<int>& pattr, int base, int bnd, 
// 		   int stride, bool docomm);

//-----------------------------------------------------------------------------
// Named nontemplate function workaraound: --TJW 8/7/98
// Users should put functions like these examples somewhere in their own source
// modules, to get access from brain-dead debuggers like the SGI debugger cvd,
// which can't invoke template functions, and gets confused by multipple
// prototypes (making it necessary to embed all this information into the names
// of separately-defined functions).
//-----------------------------------------------------------------------------
// void  dpap(ParticleAttrib<double>& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void depap(ParticleAttrib<double>& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void dspap(ParticleAttrib<double>& pattr, int base, int bnd, int stride, 
// 	   bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  fpap(ParticleAttrib<float>& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void fepap(ParticleAttrib<float>& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void fspap(ParticleAttrib<float>& pattr, int base, int bnd, int stride, 
// 	   bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  ipap(ParticleAttrib<int>& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void iepap(ParticleAttrib<int>& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void ispap(ParticleAttrib<int>& pattr, int base, int bnd, int stride, 
// 	   bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  bpap(ParticleAttrib<bool>& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void bepap(ParticleAttrib<bool>& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void bspap(ParticleAttrib<bool>& pattr, int base, int bnd, int stride, 
// 	   bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  dv1pap(ParticleAttrib<Vektor<double,1> >& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void dv1epap(ParticleAttrib<Vektor<double,1> >& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void dv1spap(ParticleAttrib<Vektor<double,1> >& pattr, int base, int bnd, 
// 	     int stride, bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  dv2pap(ParticleAttrib<Vektor<double,2> >& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void dv2epap(ParticleAttrib<Vektor<double,2> >& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void dv2spap(ParticleAttrib<Vektor<double,2> >& pattr, int base, int bnd, 
// 	     int stride, bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  dv3pap(ParticleAttrib<Vektor<double,3> >& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void dv3epap(ParticleAttrib<Vektor<double,3> >& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void dv3spap(ParticleAttrib<Vektor<double,3> >& pattr, int base, int bnd, 
// 	     int stride, bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  fv1pap(ParticleAttrib<Vektor<float,1> >& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void fv1epap(ParticleAttrib<Vektor<float,1> >& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void fv1spap(ParticleAttrib<Vektor<float,1> >& pattr, int base, int bnd, 
// 	     int stride, bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  fv2pap(ParticleAttrib<Vektor<float,2> >& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void fv2epap(ParticleAttrib<Vektor<float,2> >& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void fv2spap(ParticleAttrib<Vektor<float,2> >& pattr, int base, int bnd, 
// 	     int stride, bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }
// void  fv3pap(ParticleAttrib<Vektor<float,3> >& pattr, bool docomm) {
//   pap(pattr, docomm);
// }
// void fv3epap(ParticleAttrib<Vektor<float,3> >& pattr, int i, bool docomm) {
//   epap(pattr, i, docomm);
// }
// void fv3spap(ParticleAttrib<Vektor<float,3> >& pattr, int base, int bnd, 
// 	     int stride, bool docomm) {
//   spap(pattr, base, bnd, stride, docomm);
// }

// $Id: ParticleDebugFunctions.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $ 
 ***************************************************************************/

