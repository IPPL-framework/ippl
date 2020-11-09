// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 ***************************************************************************/

// Ippl.h
// Comprehensive include file for IPPL applications
// Includes all of the IPPL header files typically needed

#ifndef IPPL_H
#define IPPL_H

// #include <complex>

// Kokkos
#include "Field/BareField.h"
#include "Field/Field.h"

// IPPL Communicate classes
#include "Message/Communicate.h"
#include "Message/GlobalComm.h"

// IPPL Utilities
// #include "Utility/Timer.h"
// #include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
// #include "Utility/IpplCounter.h"
// #include "Utility/IpplStats.h"
// #include "Utility/IpplTimings.h"
// #include "Utility/IpplMemoryUsage.h"

// // IPPL Field Indexing
// #include "Index/Index.h"
// #include "Index/NDIndex.h"

// // IPPL Field Layout
#include "FieldLayout/FieldLayout.h"

// // IPPL Meshes
// #include "Meshes/UniformCartesian.h"
// #include "Meshes/Cartesian.h"

// // IPPL Field classes
// #include "Field/FieldSpec.h"
// #include "Field/Field.h"
// #include "Field/Assign.h"
// #include "Field/AssignDefs.h"
// #include "Field/IndexedBareField.h"
// #include "Field/IndexedField.h"
// #include "Field/GuardCellSizes.h"

// IPPL Particles classes
#include "Particle/ParticleBase.h"
// #include "Particle/ParticleSpatialLayout.h"
// #include "Particle/ParticleBalancer.h"


// // IPPL Field <--> Particle interpolators
// #include "Particle/IntNGP.h"

// IPPL Math Types
#include "Types/Vector.h"
// #include "AppTypes/Tenzor.h"
// #include "AppTypes/SymTenzor.h"
// #include "AppTypes/AntiSymTenzor.h"

// // IPPL FFTs
// #include "FFT/FFT.h"

// // IPPL Load balancing
// #include "FieldLayout/BinaryBalancer.h"

#endif
