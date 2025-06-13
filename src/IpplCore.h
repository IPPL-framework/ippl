//
// IpplCore
//   Core header files.
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

#ifdef IPPL_ENABLE_FFT
#include "FFT/FFT.h"
#endif

// // IPPL Meshes
// #include "Meshes/UniformCartesian.h"

#include "Types/Vector.h"

#include "Particle/ParticleBase.h"
#include "Particle/ParticleSpatialLayout.h"

// // IPPL Load balancing
#include "Decomposition/OrthogonalRecursiveBisection.h"

// FEM
// // FEM Elements
#include "FEM/Elements/EdgeElement.h"
#include "FEM/Elements/HexahedralElement.h"
#include "FEM/Elements/QuadrilateralElement.h"

// // FEM Quadrature
#include "FEM/Quadrature/GaussJacobiQuadrature.h"
#include "FEM/Quadrature/MidpointQuadrature.h"

// // FEM Spaces
#include "FEM/LagrangeSpace.h"

#endif
