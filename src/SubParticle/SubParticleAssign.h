// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SUB_PARTICLE_ASSIGN_H
#define SUB_PARTICLE_ASSIGN_H

// include files
#include "Field/Assign.h"
#include "SubParticle/SubParticleAttrib.h"


/***************************************************************************
 Versions of the assign function used to assign values to a SubParticleAttrib
 in an expression.  SubParticleAssign uses information in SubParticleAssignDefs
 for specific definitions of for_each operations used in assign.

 This assignment is used to gather values from an expression at selected
 points into a ParticleAttrib.  The points used are stored in an SIndex
 object, and only the local points are gathered.  So, at the end, you will
 have a ParticleAttrib on each processor with a length equal to the number
 of points in the SIndex object.  The syntax for this looks like

   SIndex<Dim> S = (some expression)
   ParticleAttrib<T> P;
   BareField<T,Dim> A, B, C;
   P[S] = A[S] * B[S] + C[S];

 Or, you can use +=, -=, etc, which requires that the attribute already
 be of the right length.

 ***************************************************************************/


//////////////////////////////////////////////////////////////////////
// SubParticleAttrib op Expression 
template<class PA, class T, unsigned Dim, class RHS, class OP>
void
assign(SubParticleAttrib<PA,T,Dim> a, RHS b, OP op, ExprTag<true>);


//////////////////////////////////////////////////////////////////////
// SubBareField op constant
template<class PA, class T, unsigned Dim, class OP>
inline void
assign(SubParticleAttrib<PA,T,Dim> a, const T& b, OP op, ExprTag<true>) {
  assign(a, PETE_Scalar<T>(b), op, ExprTag<true>());
}


//////////////////////////////////////////////////////////////////////
// SubParticleAttrib op constant
#define SUBPARTICLE_ASSIGNMENT_OPERATORS(FUNC,OP)		\
								\
template<class PA, class T, unsigned Dim>			\
inline void							\
FUNC(const SubParticleAttrib<PA,T,Dim>& lhs, const T& rhs)	\
{								\
  assign(lhs, PETE_Scalar<T>(rhs), OP(), ExprTag<true>());	\
}


SUBPARTICLE_ASSIGNMENT_OPERATORS(assign,OpAssign)
SUBPARTICLE_ASSIGNMENT_OPERATORS(operator<<,OpAssign)
SUBPARTICLE_ASSIGNMENT_OPERATORS(operator+=,OpAddAssign)
SUBPARTICLE_ASSIGNMENT_OPERATORS(operator-=,OpSubtractAssign)
SUBPARTICLE_ASSIGNMENT_OPERATORS(operator*=,OpMultipplyAssign)
SUBPARTICLE_ASSIGNMENT_OPERATORS(operator/=,OpDivideAssign)
SUBPARTICLE_ASSIGNMENT_OPERATORS(mineq,OpMinAssign)
SUBPARTICLE_ASSIGNMENT_OPERATORS(maxeq,OpMaxAssign)

#include "SubParticle/SubParticleAssign.hpp"

#endif

/***************************************************************************
 * $RCSfile: SubParticleAssign.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubParticleAssign.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
