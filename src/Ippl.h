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

#include <complex>

// IPPL Communicate classes
#include "Message/Communicate.h"
#include "Message/GlobalComm.h"

// IPPL Utilities
#include "Utility/Timer.h"
#include "Utility/RandomNumberGen.h"
#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
#include "Utility/IpplCounter.h"
#include "Utility/IpplStats.h"
#include "Utility/IpplTimings.h"
#include "Utility/IpplMemoryUsage.h"

// IPPL Field Indexing
#include "Index/Index.h"
#include "Index/NDIndex.h"
#include "Index/SIndex.h"
#include "Index/IndexedSIndex.h"
#include "Index/SIndexAssign.h"

// IPPL Field Layout
#include "FieldLayout/FieldLayout.h"
#include "FieldLayout/CenteredFieldLayout.h"

// IPPL Meshes
#include "Meshes/UniformCartesian.h"
#include "Meshes/Cartesian.h"

// IPPL Expression Templates
#include "PETE/IpplExpressions.h"

// IPPL Field classes
#include "Field/FieldSpec.h"
#include "Field/Field.h"
#include "Field/Assign.h"
#include "Field/AssignDefs.h"
#include "Field/IndexedBareField.h"
#include "Field/IndexedField.h"
#include "Field/GuardCellSizes.h"
#include "Utility/FieldPrint.h"
#include "Utility/FieldDebugPrint.h"
#include "Utility/FieldDebug.h"
#include "Utility/DiscField.h"

// IPPL Particles classes
#include "Particle/IpplParticleBase.h"
#include "Particle/PAssign.h"
#include "Particle/PAssignDefs.h"
#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleUniformLayout.h"
#include "Particle/ParticleInteractLayout.h"
#include "Particle/ParticleCashedLayout.h"
#include "Particle/ParticleBalancer.h"
#include "Particle/GenArrayParticle.h"
#include "Particle/GenParticle.h"
#include "Particle/NoParticleCachingPolicy.h"
#include "Particle/BoxParticleCachingPolicy.h"
#include "Particle/CellParticleCachingPolicy.h"
#include "Particle/PairBuilder/HashPairBuilder.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodic.h"
#include "Particle/PairBuilder/PairConditions.h"


#include "Utility/DiscParticle.h"
#include "Utility/ParticleDebug.h"

// IPPL Field <--> Particle interpolators
#include "Particle/IntNGP.h"
#include "Particle/IntCIC.h"
#include "Particle/IntTSC.h"
#include "Particle/IntSUDS.h"

// IPPL sparse index expression operations
#include "SubField/SubFieldAssign.h"
#include "SubParticle/SubParticleAssign.h"

// IPPL Math Types
#include "AppTypes/Vektor.h"
#include "AppTypes/Tenzor.h"
#include "AppTypes/SymTenzor.h"
#include "AppTypes/AntiSymTenzor.h"

// IPPL Data Connection classes
#include "DataSource/DataConnectCreator.h"
#include "DataSource/FileDataConnect.h"

// IPPL FFTs
#include "FFT/FFT.h"

// IPPL Load balancing
#include "FieldLayout/BinaryBalancer.h"

#endif
