// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef TSV_META_H
#define TSV_META_H

// include files
#include "AppTypes/TSVMetaAssign.h"
#include "AppTypes/TSVMetaUnary.h"
#include "AppTypes/TSVMetaBinary.h"
#include "AppTypes/TSVMetaDot.h"
#include "AppTypes/TSVMetaCross.h"
#include "AppTypes/TSVMetaDotDot.h"
#include "AppTypes/TSVMetaCompare.h"

// forward declarations
template<class T, unsigned D> class Vektor;
template<class T, unsigned D> class Tenzor;
template<class T, unsigned D> class SymTenzor;
template<class T, unsigned D> class AntiSymTenzor;

//////////////////////////////////////////////////////////////////////
//
// Define the macro TSV_ELEMENTWISE_OPERATOR which will let
// Vektor, Tenzor and SymTenzor define their operators easily.
//
// The first argument of the macro is Vektor, Tenzor, SymTenzor, 
// or AntiSymTenzor.
// The second is the name of the operator (like operator+).
// The third is the PETE tag class for that operation.
//
//////////////////////////////////////////////////////////////////////


#define TSV_ELEMENTWISE_OPERATOR_WITH_SCALAR(TSV,SCA,OP,APP)		\
									\
template < class T1 , unsigned D >					\
inline TSV<typename PETEBinaryReturn<T1,SCA,APP>::type,D>		\
OP(const TSV<T1,D>& lhs, SCA sca)					\
{									\
  return TSV_MetaBinaryScalar< TSV<T1,D> , SCA , APP > :: apply(lhs,sca);\
}									\
									\
template < class T1 , unsigned D >					\
inline TSV<typename PETEBinaryReturn<T1,SCA,APP>::type,D>		\
OP(SCA sca, const TSV<T1,D>& rhs)					\
{									\
  return TSV_MetaBinaryScalar< SCA , TSV<T1,D> , APP > :: apply(sca,rhs);\
}

#define TSV_ELEMENTWISE_OPERATOR(TSV,OP,APP)				\
									\
template < class T1, class T2, unsigned D >				\
inline TSV<typename PETEBinaryReturn<T1,T2,APP>::type,D>		\
OP(const TSV<T1,D> &lhs, const TSV<T2,D> &rhs)				\
{									\
  return TSV_MetaBinary< TSV<T1,D> , TSV<T2,D> , APP > :: apply(lhs,rhs);\
}									\
									\
TSV_ELEMENTWISE_OPERATOR_WITH_SCALAR(TSV,short,OP,APP)			\
TSV_ELEMENTWISE_OPERATOR_WITH_SCALAR(TSV,int,OP,APP)			\
TSV_ELEMENTWISE_OPERATOR_WITH_SCALAR(TSV,unsigned int,OP,APP)		\
TSV_ELEMENTWISE_OPERATOR_WITH_SCALAR(TSV,long,OP,APP)			\
TSV_ELEMENTWISE_OPERATOR_WITH_SCALAR(TSV,float,OP,APP)			\
TSV_ELEMENTWISE_OPERATOR_WITH_SCALAR(TSV,double,OP,APP)			\
TSV_ELEMENTWISE_OPERATOR_WITH_SCALAR(TSV,std::complex<double>,OP,APP)

#define TSV_ELEMENTWISE_OPERATOR2(TSV1,TSV2,OP,APP)    		        \
									\
template < class T1, class T2, unsigned D >				\
inline typename PETEBinaryReturn<TSV1<T1,D>,TSV2<T2,D>,APP>::type       \
OP(const TSV1<T1,D> &lhs, const TSV2<T2,D> &rhs)			\
{									\
  return TSV_MetaBinary< TSV1<T1,D>, TSV2<T2,D>, APP >::                \
    apply(lhs,rhs);                                                     \
}

//////////////////////////////////////////////////////////////////////

#endif // TSV_META_H

/***************************************************************************
 * $RCSfile: TSVMeta.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: TSVMeta.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/

