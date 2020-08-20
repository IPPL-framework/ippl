// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PASSIGN_H
#define PASSIGN_H

// include files
#include "Particle/PAssignDefs.h"
#include "PETE/IpplExpressions.h"
#include "AppTypes/AppTypeTraits.h"


//////////////////////////////////////////////////////////////////////

// ParticleAttrib = Expression 
template<class T, class RHS, class OP>
void
assign(const ParticleAttrib<T>& a, RHS b, OP op);

// ParticleAttribElem = Expression 
template<class T, unsigned Dim, class RHS, class OP>
void
assign(const ParticleAttribElem<T,Dim>& a, RHS b, OP op);


//////////////////////////////////////////////////////////////////////


// ParticleAttrib = constant; 
template<class T, class OP>
inline void
assign(const ParticleAttrib<T>& a, const T& b, OP op)
{
  assign(a, PETE_Scalar<T>(b), op);
}

// ParticleAttribElem = constant; 
template<class T, unsigned D, class OP>
inline void
assign(const ParticleAttribElem<T,D>& a,
       const typename AppTypeTraits<T>::Element_t& b, OP op)
{
  assign(a, PETE_Scalar<typename AppTypeTraits<T>::Element_t>(b), op);
}


//////////////////////////////////////////////////////////////////////

#define ASSIGNMENT_OPERATORS_PTCL(FUNC,OP)				    \
									    \
template<class T, unsigned D, class RHS>				    \
inline void								    \
FUNC(const ParticleAttribElem<T,D>& lhs, const PETE_Expr<RHS>& rhs)	    \
{									    \
  assign(lhs,rhs.PETE_unwrap().MakeExpression(),OP());		            \
}									    \
									    \
template<class T, unsigned D>						    \
inline void								    \
FUNC(const ParticleAttribElem<T,D>& lhs,                                    \
     typename AppTypeTraits<T>::Element_t rhs)                              \
{									    \
  assign(lhs,PETE_Scalar<typename AppTypeTraits<T>::Element_t>(rhs),OP());  \
}									    \
									    \
template<class T, class RHS>				                    \
inline void								    \
FUNC(const ParticleAttrib<T>& lhs, const PETE_Expr<RHS>& rhs)	            \
{									    \
  assign(lhs,rhs.PETE_unwrap().MakeExpression(),OP());		            \
}									    \
									    \
template<class T> 						            \
inline void								    \
FUNC(const ParticleAttrib<T>& lhs, T rhs)			            \
{									    \
  assign(lhs,PETE_Scalar<T>(rhs),OP());		 	                    \
}

ASSIGNMENT_OPERATORS_PTCL(assign,OpAssign)
ASSIGNMENT_OPERATORS_PTCL(operator<<,OpAssign)
ASSIGNMENT_OPERATORS_PTCL(operator+=,OpAddAssign)
ASSIGNMENT_OPERATORS_PTCL(operator-=,OpSubtractAssign)			      
ASSIGNMENT_OPERATORS_PTCL(operator*=,OpMultipplyAssign)
ASSIGNMENT_OPERATORS_PTCL(operator/=,OpDivideAssign)
ASSIGNMENT_OPERATORS_PTCL(mineq,OpMinAssign)
ASSIGNMENT_OPERATORS_PTCL(maxeq,OpMaxAssign)

#include "Particle/PAssign.hpp"

#endif // PASSIGN_H

/***************************************************************************
 * $RCSfile: PAssign.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: PAssign.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
