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

// #include "Index/Index.h"
// #include "Index/NDIndex.h"

#include "FieldLayout/FieldLayout.h"

#ifdef ENABLE_FFT
    #include "FFT/FFT.h"
#endif

// // IPPL Meshes
// #include "Meshes/UniformCartesian.h"

#include "Particle/ParticleBase.h"
#include "Particle/ParticleSpatialLayout.h"


#include "Types/Vector.h"

// // IPPL Load balancing
// #include "Particle/ParticleBalancer.h"
// #include "FieldLayout/BinaryBalancer.h"

#endif
