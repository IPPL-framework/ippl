/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef SUB_FIELD_ASSIGN_H
#define SUB_FIELD_ASSIGN_H

// include files
#include "Field/Assign.h"

// forward declarations
template<class T, unsigned int Dim, class S> class SubBareField;


/***************************************************************************
 Versions of the assign function used to assign values to a SubBareField
 in an expression.  SubFieldAssign uses information in SubFieldAssignDefs
 for specific definitions of for_each operations used in assign.

 So far, only the assign function for the simple case where everything is
 perfectly aligned is implemented ... to complete this, assign must handle
 assigments between non-conforming layouts which requires general
 communication.  Also, support will have to be added for:
   - Assigning single-element subfield's to scalars
   - Slice operations between fields of unequal dimension
 ***************************************************************************/


//////////////////////////////////////////////////////////////////////
// SubBareField = Expression 
template<class T1, unsigned Dim, class S, class RHS, class OP>
void
assign(SubBareField<T1,Dim,S> a, RHS b, OP op, ExprTag<true>);


//////////////////////////////////////////////////////////////////////
// SubBareField = constant
template<class T, unsigned D, class S, class OP>
inline void
assign(SubBareField<T,D,S> a, const T& b, OP op, ExprTag<true>) {
  assign(a, PETE_Scalar<T>(b), op, ExprTag<true>());
}


//////////////////////////////////////////////////////////////////////

#define SUB_ASSIGNMENT_OPERATORS(FUNC,OP)				 \
									 \
template<class T, unsigned D, class S>					 \
inline void								 \
FUNC(const SubBareField<T,D,S>& lhs, const T& rhs)			 \
{									 \
  assign(lhs,PETE_Scalar<T>(rhs), OP(),ExprTag<true>());		 \
}


SUB_ASSIGNMENT_OPERATORS(assign,OpAssign)
SUB_ASSIGNMENT_OPERATORS(operator<<,OpAssign)
SUB_ASSIGNMENT_OPERATORS(operator+=,OpAddAssign)
SUB_ASSIGNMENT_OPERATORS(operator-=,OpSubtractAssign)
SUB_ASSIGNMENT_OPERATORS(operator*=,OpMultipplyAssign)
SUB_ASSIGNMENT_OPERATORS(operator/=,OpDivideAssign)
SUB_ASSIGNMENT_OPERATORS(mineq,OpMinAssign)
SUB_ASSIGNMENT_OPERATORS(maxeq,OpMaxAssign)

#include "SubField/SubFieldAssign.hpp"

#endif // SUB_FIELD_ASSIGN_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
