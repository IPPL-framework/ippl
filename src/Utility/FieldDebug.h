// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// FieldDebug.h , Tim Williams 10/23/1996
// Helper functions to print out (formatted ASCII) Field elements.
// Intended mainly for use from within a debugger, called interactively.
// See comments in FieldDebug.cpp for more details.

#ifndef FIELD_DEBUG_H
#define FIELD_DEBUG_H

#include "Utility/FieldDebugFunctions.h"

// forward declarations
class Inform;
template<class T, unsigned Dim> class BareField;

// extern declarations of global variables in FieldDebugFunctions.cpp
extern Inform* FldDbgInform;
extern bool FldDbgInformIsSet;
extern int elementsPerLine;
extern int digitsPastDecimal;
extern int widthOfElements;

//=============================================================================
// Helper functions to print out (formatted ASCII) Field elements.
// Intended mainly for use from within a debugger, called interactively.
// fp[1,2,3](Field&) print all elements of [1D,2D,3D] Field
//      decimal point
// efp[1,2,3](Field&,....) prints single element of [1D,2D,3D] Field; requires 
//                         specification of integer index for each 
// sfp[1,2,3](Field&,....) prints strided slice of [1D,2D,3D] Field; requires 
//                         specification of (base,bound,stride) for each 
//                         dimension
// Implementation notes: The specially-named functions for [1D,2D,3D] are
// needed because the SGI debuggers don't handle single-name functions with
// multipple prototypes for different argument types/numbers properly.
//=============================================================================

//------------------------------------------------------
// Function prototypes; see FieldDebug.cpp for comments:
//------------------------------------------------------
// For printing all elements of a Field:
template<class T>
void fp1(BareField<T, 1U>& field, bool docomm = true);
template<class T>
void fp2(BareField<T, 2U>& field, bool docomm = true);
template<class T>
void fp3(BareField<T, 3U>& field, bool docomm = true);

// For printing all elements of a Field, including global guard layers:
template<class T>
void ggfp1(BareField<T, 1U>& field, bool docomm = true);
template<class T>
void ggfp2(BareField<T, 2U>& field, bool docomm = true);
template<class T>
void ggfp3(BareField<T, 3U>& field, bool docomm = true);

// For printing all elements of a Field, including all guard layers (global and
// internal). These will print the data in a different format: all data for a
// vnode, one vnode at a time:
template<class T>
void agfp1(BareField<T, 1U>& field);
template<class T>
void agfp2(BareField<T, 2U>& field);
template<class T>
void agfp3(BareField<T, 3U>& field);

// For printing one element of a Field:
template<class T>
void efp1(BareField<T, 1U>& field, int i, bool docomm = true);
template<class T>
void efp2(BareField<T, 2U>& field, int i, int j, bool docomm = true);
template<class T>
void efp3(BareField<T, 3U>& field, int i, int j, int k, bool docomm = true);

// For printing strided subrange of  elements of a Field:
template<class T>
void sfp1(BareField<T, 1U>& field, 
	  int ibase, int ibound, int istride, bool docomm = true);
template<class T>
void sfp2(BareField<T, 2U>& field, 
	  int ibase, int ibound, int istride, 
	  int jbase, int jbound, int jstride, bool docomm = true);
template<class T>
void sfp3(BareField<T, 3U>& field, 
	  int ibase, int ibound, int istride,
	  int jbase, int jbound, int jstride, 
	  int kbase, int kbound, int kstride, bool docomm = true);

//-----------------------------------------------------------------------------
// Specializations, so that these types are available from debugger:
//
// USAGE:
// For now, all of these are commented out in this FieldDebug.h header file,
// so that libippl.a doesn't take forever to build and isn't unnecessarily
// large. (In fact, the KCC compiler dies if you have too many of these in
// here.) It is up to the user to include a line like these in his source 
// (most logically in the main()), in order to have access to Field-printing
// for his special types of Field's from the debugger.
//
// To call the functions from the *code* (not from the debugger), the user need
// not include any lines like these; he can just invoke the generic 
// parameterized functions from the list above: [e,s]fp[1,2,3](). 
//-----------------------------------------------------------------------------

// User must put lines like the commented-out ones in his own code to get 
// access to specialized Field-type functions like these from the debugger.

// Scalar Field's of double's:-------------------------------------------------
//void  dfp1(BareField<double, 1U>& f) {fp1(f);}
//void defp1(BareField<double, 1U>& f, int i) {efp1(f,i);}
//void dsfp1(BareField<double,1U>& f,
//	   int base1, int bound1, int stride1) {sfp1(f,base1,bound1,stride1);}
//void dfp2(BareField<double, 2U>& f) {fp2(f);}
//void defp2(BareField<double, 2U>& f, int i, int j) {efp2(f,i,j);}
//void dsfp2(BareField<double,2U>& f,
//	   int base1, int bound1, int stride1,
//	   int base2, int bound2, int stride2) {
//  sfp2(f,base1,bound1,stride1,base2,bound2,stride2);}
//void  dfp3(BareField<double, 3U>& f) {fp3(f);}
//void defp3(BareField<double, 3U>& f, int i, int j, int k) {efp3(f,i,j,k);}
//void dsfp3(BareField<double,3U>& f,
//	   int base1, int bound1, int stride1,
//	   int base2, int bound2, int stride2,
//	   int base3, int bound3, int stride3) {
//  sfp3(f,base1,bound1,stride1,base2,bound2,stride2,base3,bound3,stride3);}

// Scalar Field's of float's:--------------------------------------------------
//void  ffp1(BareField<float, 1U>& f) {fp1(f);}
//void fefp1(BareField<float, 1U>& f, int i) {efp1(f,i);}
//void fsfp1(BareField<float,1U>& f,
//	   int base1, int bound1, int stride1) {sfp1(f,base1,bound1,stride1);}
//void  ffp2(BareField<float, 2U>& f) {fp2(f);}
//void fefp2(BareField<float, 2U>& f, int i, int j) {efp2(f,i,j);}
//void fsfp2(BareField<float,2U>& f,
//	   int base1, int bound1, int stride1,
//	   int base2, int bound2, int stride2) {
//  sfp2(f,base1,bound1,stride1,base2,bound2,stride2);}
//void  ffp3(BareField<float, 3U>& f) {fp3(f);}
//void fefp3(BareField<float, 3U>& f, int i, int j, int k) {efp3(f,i,j,k);}
//void fsfp3(BareField<float,3U>& f,
//	   int base1, int bound1, int stride1,
//	   int base2, int bound2, int stride2,
//	   int base3, int bound3, int stride3) {
//  sfp3(f,base1,bound1,stride1,base2,bound2,stride2,base3,bound3,stride3);}

// Vector Field's of double's:-------------------------------------------------
//void  vdfp2(BareField<Vektor<double,2U>, 2U>& f) {fp2(f);}
//void vdefp2(BareField<Vektor<double,2U>, 2U>& f, int i, int j) {efp2(f,i,j);}
//void vdsfp2(BareField<Vektor<double,2U>,2U>& f,
//	    int base1, int bound1, int stride1,
//	    int base2, int bound2, int stride2) {
//  sfp2(f,base1,bound1,stride1,base2,bound2,stride2);}
//void  vdfp3(BareField<Vektor<double,3U>, 3U>& f) {fp3(f);}
//void vdefp3(BareField<Vektor<double,3U>, 3U>& f, int i, int j, int k) {
//  efp3(f,i,j,k);}
//void vdsfp3(BareField<Vektor<double,3U>,3U>& f,
//	   int base1, int bound1, int stride1,
//	   int base2, int bound2, int stride2,
//	   int base3, int bound3, int stride3) {
//  sfp3(f,base1,bound1,stride1,base2,bound2,stride2,base3,bound3,stride3);}

// Vector Field's of float's:--------------------------------------------------
//void  vffp2(BareField<Vektor<float,2U>, 2U>& f) {fp2(f);}
//void vfefp2(BareField<Vektor<float,2U>, 2U>& f, int i, int j) {efp2(f,i,j);}
//void vfsfp2(BareField<Vektor<float,2U>,2U>& f,
//	    int base1, int bound1, int stride1,
//	    int base2, int bound2, int stride2) {
//  sfp2(f,base1,bound1,stride1,base2,bound2,stride2);}
//void  vffp3(BareField<Vektor<float,3U>, 3U>& f) {fp3(f);}
//void vfefp3(BareField<Vektor<float,3U>, 3U>& f, int i, int j, int k) {
//  efp3(f,i,j,k);}
//void vfsfp3(BareField<Vektor<float,3U>,3U>& f,
//	   int base1, int bound1, int stride1,
//	   int base2, int bound2, int stride2,
//	   int base3, int bound3, int stride3) {
//  sfp3(f,base1,bound1,stride1,base2,bound2,stride2,base3,bound3,stride3);}

#include "Utility/FieldDebug.hpp"

#endif // FIELD_DEBUG_H

/***************************************************************************
 * $RCSfile: FieldDebug.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: FieldDebug.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
