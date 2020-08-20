// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef RANDOM_NUMBER_GEN_H
#define RANDOM_NUMBER_GEN_H

/***********************************************************************
 *
 * class RandomNumberGen
 *
 * RandomNumberGen is actually just a typedef for one of the many
 * RNGSequence classes available in IPPL.  It is selected by examining the
 * values of #define's.  Other RNGSequence classes may be used in a program,
 * but using RandomNumberGen give you the 'default' RNGSequence type as
 * selected at compile time.
 *
 * When using Random or Distributed number sequences, include this file.
 * 
 ***********************************************************************/

#include "Utility/RNGXDiv.h"

typedef RNGXDivSequence RandomNumberGen;

// a default RandomNumberGen object for use in the Framework.  When
// running in parallel, the Ippl object will advance this by the
// node number so as to have different RNG sequences on each node.  If the
// same RNG sequence is needed on each node, the user must instantiate their
// own RNG sequence object and use that.
extern RandomNumberGen IpplRandom;

#endif // RANDOM_NUMBER_GEN_H
