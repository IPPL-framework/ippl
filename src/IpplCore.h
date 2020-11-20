//
// IpplCore
//   Core header files.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_CORE_H
#define IPPL_CORE_H


#include "Field/BareField.h"
#include "Field/Field.h"
#include "Field/BConds.h"

// IPPL Utilities
// #include "Utility/Timer.h"
// #include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
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
#include "Particle/ParticleSpatialLayout.h"
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
