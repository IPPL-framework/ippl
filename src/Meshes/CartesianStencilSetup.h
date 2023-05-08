// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef CARTESIAN_STENCIL_SETUP_H
#define CARTESIAN_STENCIL_SETUP_H

// CartesianStencilSetup.h
// Common stencil setup header for Cartesian and UniformCartesian classes

//----------------------------------------------------------------------
//
// Define the operator class types for the stencils.
//
// These get plugged into UnaryElem as the operator to indicate
// what kind of operation is to be done.
//
//----------------------------------------------------------------------
template <unsigned Dim>
struct Divergence {};
template <unsigned Dim>
struct Gradient {};

//----------------------------------------------------------------------
//
// Now derive from these general tags specific ones that include
// the mesh and centerings.
//
// These define operator_type internally instead of using
// the OperatorClass external polymorphism because that would require
// partial specialization.
//
//----------------------------------------------------------------------

template <unsigned Dim, class Mesh, class From, class To>
struct CenteredDivergence {
    typedef Divergence<Dim> operator_type;
};

template <unsigned Dim, class Mesh, class From, class To>
class CenteredGradient {
    typedef Gradient<Dim> operator_type;
};

#endif  // CARTESIAN_STENCIL_SETUP_H

/***************************************************************************
 * $RCSfile: CartesianStencilSetup.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: CartesianStencilSetup.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $
 ***************************************************************************/
