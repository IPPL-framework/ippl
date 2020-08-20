// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SEQUENCE_GEN_H
#define SEQUENCE_GEN_H

/***********************************************************************
 * 
 * class SequenceGen
 *
 * SequenceGen generates a sequence of doubles from some algorithm, such
 * a uniform or gaussian random number generator.  It is templated on
 * a class which actually generates the number sequence.  SequenceGen
 * provides an expression-template-aware wrapper around this generator.
 *
 * A SequenceGen is created by giving it a reference to an instance
 * of the specific generator it should use.  It keeps a copy of this
 * generator, so the classes used as template parameters for SequenceGen
 * must have a copy constructor.  It should also have a default constructor.
 * Also, it uses the () operator to get
 * the next number in the sequence.  The particular generator itself
 * may be accessed through the 'getGenerator' method.
 *
 * The generator class used as the template parameter must also provide
 * a typedef indicating the return type of the generator; this typedef
 * must be called Return_t.
 *
 ***********************************************************************/

// include files
#include "PETE/IpplExpressions.h"

// some macro definitions to set up basic binary math operations

#define RNG_OPERATOR_WITH_SCALAR(GEN,SCA,OP,APP)		        \
									\
inline PETEBinaryReturn<GEN,SCA,APP>::type		                \
OP(const GEN& lhs, SCA sca)					        \
{									\
  return PETE_apply( APP(), lhs(), sca );                               \
}									\
									\
inline PETEBinaryReturn<SCA,GEN,APP>::type		                \
OP(SCA sca, const GEN& rhs)					        \
{									\
  return PETE_apply( APP(), sca, rhs() );                               \
}

#define RNG_OPERATOR(GEN,OP,APP)				        \
									\
RNG_OPERATOR_WITH_SCALAR(GEN,short,OP,APP)			        \
RNG_OPERATOR_WITH_SCALAR(GEN,int,OP,APP)			        \
RNG_OPERATOR_WITH_SCALAR(GEN,long,OP,APP)			        \
RNG_OPERATOR_WITH_SCALAR(GEN,float,OP,APP)			        \
RNG_OPERATOR_WITH_SCALAR(GEN,double,OP,APP)			        \
RNG_OPERATOR_WITH_SCALAR(GEN,std::complex<double>,OP,APP)

#define RNG_BASIC_MATH(GEN)                                             \
                                                                        \
RNG_OPERATOR(GEN,operator+,OpAdd)                                       \
RNG_OPERATOR(GEN,operator-,OpSubtract)                                  \
RNG_OPERATOR(GEN,operator*,OpMultipply)                                  \
RNG_OPERATOR(GEN,operator/,OpDivide)


// the guts of the sequence generator
template <class GT>
class SequenceGen : public PETE_Expr< SequenceGen<GT> >
{

public:
  SequenceGen() { }
  SequenceGen(const GT& gen) : Gen(gen) { }
  // Interface for PETE.
  enum { IsExpr = 1 };  // Treat SequenceGen as a PETE_Expr in expressions.
  typedef SequenceGen<GT> PETE_Expr_t;
  typedef typename GT::Return_t PETE_Return_t;
  const PETE_Expr_t& MakeExpression() const { return *this; }
  PETE_Expr_t&       MakeExpression()       { return *this; }

  // typedefs for functions below
  typedef typename GT::Return_t Return_t;

  // return access to the generator
  GT& getGenerator() { return Gen; }
  const GT& getGenerator() const { return Gen; }

  // get the next value in the sequence
  inline Return_t operator()(void) const { return Gen(); }

private:
  // the number generator
  GT Gen;
};

// define for_each operations for SequenceGen objects
#include "Utility/RNGAssignDefs.h"

#endif // SEQUENCE_GEN_H

/***************************************************************************
 * $RCSfile: SequenceGen.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SequenceGen.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
