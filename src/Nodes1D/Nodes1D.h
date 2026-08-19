/**
 * @file Nodes1D.h
 * @brief Facade header for the Nodes1D module.
 *
 * Provides Gauss–Legendre, Gauss–Chebyshev, Gauss–Jacobi, and Gauss–Lobatto–Legendre (GLL)
 * nodes and weights on the native interval [-1, 1], plus affine maps to other intervals.
 *
 * Host-only: all Node computation runs serially on the host (meant to be computed once 
 * and saved not rerun each iteration step). Kokkos View overloads
 * compute on the host and deep-copy into device-accessible memory; they do not launch
 * parallel_for (or other Kokkos kernels) for node/weight setup. A few small helpers use
 * KOKKOS_INLINE_FUNCTION for IPPL-style header inlining only; nothing in this module is
 * invoked from a device lambda currently.
 */
//
// Nodes1D — freestanding 1D node/weight finders for quadrature and interpolation.
// Native interval for classical GL / GLL / Jacobi: [-1, 1].
//
#ifndef IPPL_NODES1D_H
#define IPPL_NODES1D_H

#include "Nodes1D/AffineMap.h"
#include "Nodes1D/GaussChebyshev1D.h"
#include "Nodes1D/GaussJacobi1D.h"
#include "Nodes1D/GaussLegendre1D.h"
#include "Nodes1D/GaussLobatto1D.h"

/**
 * @namespace ippl::nodes1d
 * @brief 1D quadrature and interpolation nodes and weights.
 *
 * Classical rules are defined on [-1, 1]. Use affineMapPoint / affineMapWeight (or
 * affineMapNodesWeights) to map onto another interval. Entry points: computeGaussLegendre,
 * computeGaussChebyshev, computeGaussJacobi, computeGaussLobatto.
 */

#endif
