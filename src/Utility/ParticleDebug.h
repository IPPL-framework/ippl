// -*- C++ -*-
//-----------------------------------------------------------------------------
// The IPPL Framework - Visit http://people.web.psi.ch/adelmann/ for more details
//
// This program was prepared by the Regents of the University of California at
// ParticleDebug.h , Tim Williams 8/6/1998
// Helper functions to print out (formatted ASCII) ParticleAttrib elements.
// Intended mainly for use from within a debugger, called interactively, but
// also callable as template functions from source code. To call from many
// debuggers, the user has to provide nontemplate wrapper functions, as
// described in ParticleDebugFunctions.cpp.

#ifndef PARTICLE_DEBUG_H
#define PARTICLE_DEBUG_H

#include "Utility/ParticleDebugFunctions.h"

// forward declarations
class Inform;
template<class T> class ParticleAttrib;

// extern declarations of global variables in ParticleDebugFunctions.cpp
extern Inform* PtclDbgInform;
extern bool PtclDbgInformIsSet;
// extern declarations of global variables in FieldDebugFunctions.cpp
extern Inform* FldDbgInform;
extern bool FldDbgInformIsSet;
extern int elementsPerLine;
extern int digitsPastDecimal;
extern int widthOfElements;


//=============================================================================
// Helper functions to print out (formatted ASCII) ParticleAttrib elements.
// Intended mainly for use from within a debugger, called interactively.
// pap(ParticleAttrib&) print all elements of ParticleAttrib
// epap(ParticleAttrib&,....) prints single element of ParticleAttrib 
//                         specification of integer particle index 
// spap(ParticleAttrib&,....) prints strided range of elements of 
//                            ParticleAttrib; requires specification
//                            of (base,bound,stride) for particle index values 
//=============================================================================

//------------------------------------------------------
// Function prototypes; see ParticleDebug.cpp for comments:
//------------------------------------------------------
// For printing all elements of a ParticleAttrib:
template<class T>
void pap(ParticleAttrib<T>& pattr, bool docomm = true);

// For printing one element of a ParticleAttrib:
template<class T>
void epap(ParticleAttrib<T>& pattr, int i, bool docomm = true);

// For printing strided subrange of elements of a ParticleAttrib:
template<class T>
void spap(ParticleAttrib<T>& pattr, 
	  int ibase, int ibound, int istride, bool docomm = true);

#include "Utility/ParticleDebug.hpp"

#endif // PARTICLE_DEBUG_H

// $Id: ParticleDebug.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $ 
 ***************************************************************************/

