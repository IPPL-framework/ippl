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
template<unsigned Dim> struct Divergence {};
template<unsigned Dim> struct Gradient   {};

//----------------------------------------------------------------------
//
// Define the return types.
//
// These are the element by element return types.  They just encode 
// the fact that divergence of a Vektor is a scalar, and gradient of
// a scalar is a Vektor.  As such they are independent of centering
// and use the tag base classes.
//
// These need to be filled out with specializations for Tenzors...
// 
//----------------------------------------------------------------------
template<> struct PETEUnaryReturn< Divergence<1>, Vektor<double,1> > 
{ typedef double type; };
template<> struct PETEUnaryReturn< Divergence<2>, Vektor<double,2> > 
{ typedef double type; };
template<> struct PETEUnaryReturn< Divergence<3>, Vektor<double,3> > 
{ typedef double type; };

template<> struct PETEUnaryReturn< Divergence<1>, Vektor<float,1> > 
{ typedef float type; };
template<> struct PETEUnaryReturn< Divergence<2>, Vektor<float,2> > 
{ typedef float type; };
template<> struct PETEUnaryReturn< Divergence<3>, Vektor<float,3> > 
{ typedef float type; };

template<> struct PETEUnaryReturn< Gradient<1>, double > 
{ typedef Vektor<double,1> type; };
template<> struct PETEUnaryReturn< Gradient<2>, double > 
{ typedef Vektor<double,2> type; };
template<> struct PETEUnaryReturn< Gradient<3>, double > 
{ typedef Vektor<double,3> type; };

template<> struct PETEUnaryReturn< Gradient<1>, float > 
{ typedef Vektor<float,1> type; };
template<> struct PETEUnaryReturn< Gradient<2>, float > 
{ typedef Vektor<float,2> type; };
template<> struct PETEUnaryReturn< Gradient<3>, float > 
{ typedef Vektor<float,3> type; };

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

template<unsigned Dim, class Mesh, class From, class To>
struct CenteredDivergence
{
  typedef Divergence<Dim> operator_type;
};

template<unsigned Dim, class Mesh, class From, class To>
class CenteredGradient
{
  typedef Gradient<Dim> operator_type;
};

#endif // CARTESIAN_STENCIL_SETUP_H

/***************************************************************************
 * $RCSfile: CartesianStencilSetup.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: CartesianStencilSetup.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/

