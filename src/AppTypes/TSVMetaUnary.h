// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef TSV_META_UNARY_H
#define TSV_META_UNARY_H

//////////////////////////////////////////////////////////////////////
//
// Definition of the struct TSV_MetaUnary.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP> struct TSV_MetaUnary {};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP, unsigned D>
struct TSV_MetaUnary< Vektor<T1,D> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static Vektor<T0,D>
  apply(const Vektor<T1,D>& lhs) {
    Vektor<T0,D> ret;
    for (unsigned d=0; d<D; ++d)
      ret[d] = PETE_apply(OP(),lhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for Vektors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< Vektor<T1,1> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static Vektor<T0,1>
  apply(const Vektor<T1,1>& lhs) {
    return Vektor<T0,1>( PETE_apply( OP(), lhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for Vektors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< Vektor<T1,2> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static Vektor<T0,2>
  apply(const Vektor<T1,2>& lhs) {
    return Vektor<T0,2>( PETE_apply( OP(), lhs[0] ) ,
			 PETE_apply( OP(), lhs[1] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for Vektors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< Vektor<T1,3> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static Vektor<T0,3>
  apply(const Vektor<T1,3>& lhs) {
    return Vektor<T0,3>( PETE_apply( OP(), lhs[0] ) ,
			 PETE_apply( OP(), lhs[1] ) ,
			 PETE_apply( OP(), lhs[2] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP, unsigned D>
struct TSV_MetaUnary< Tenzor<T1,D> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static Tenzor<T0,D>
  apply(const Tenzor<T1,D>& lhs) {
    Tenzor<T0,D> ret;
    for (unsigned d=0; d<D*D; ++d)
      ret[d] = PETE_apply(OP(),lhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for Tenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< Tenzor<T1,1> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static Tenzor<T0,1>
  apply(const Tenzor<T1,1>& lhs) {
    return Tenzor<T0,1>( PETE_apply( OP(), lhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for Tenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< Tenzor<T1,2> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static Tenzor<T0,2>
  apply(const Tenzor<T1,2>& lhs) {
    return Tenzor<T0,2>( PETE_apply( OP(), lhs[0] ) ,
			 PETE_apply( OP(), lhs[1] ) ,
			 PETE_apply( OP(), lhs[2] ) ,
			 PETE_apply( OP(), lhs[3] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for Tenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< Tenzor<T1,3> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static Tenzor<T0,3>
  apply(const Tenzor<T1,3>& lhs) {
    return Tenzor<T0,3>( PETE_apply( OP(), lhs[0] ) ,
			 PETE_apply( OP(), lhs[1] ) ,
			 PETE_apply( OP(), lhs[2] ) ,
			 PETE_apply( OP(), lhs[3] ) ,
			 PETE_apply( OP(), lhs[4] ) ,
			 PETE_apply( OP(), lhs[5] ) ,
			 PETE_apply( OP(), lhs[6] ) ,
			 PETE_apply( OP(), lhs[7] ) ,
			 PETE_apply( OP(), lhs[8] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP, unsigned D>
struct TSV_MetaUnary< SymTenzor<T1,D> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static SymTenzor<T0,D>
  apply(const SymTenzor<T1,D>& lhs) {
    SymTenzor<T0,D> ret;
    for (unsigned d=0; d<D*(D+1)/2; ++d)
      ret[d] = PETE_apply(OP(),lhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for SymTenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< SymTenzor<T1,1> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static SymTenzor<T0,1>
  apply(const SymTenzor<T1,1>& lhs) {
    return SymTenzor<T0,1>( PETE_apply( OP(), lhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for SymTenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< SymTenzor<T1,2> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static SymTenzor<T0,2>
  apply(const SymTenzor<T1,2>& lhs) {
    return SymTenzor<T0,2>( PETE_apply( OP(), lhs[0] ) ,
			    PETE_apply( OP(), lhs[1] ) ,
			    PETE_apply( OP(), lhs[2] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for SymTenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< SymTenzor<T1,3> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static SymTenzor<T0,3>
  apply(const SymTenzor<T1,3>& lhs) {
    return SymTenzor<T0,3>( PETE_apply( OP(), lhs[0] ) ,
			    PETE_apply( OP(), lhs[1] ) ,
			    PETE_apply( OP(), lhs[2] ) ,
			    PETE_apply( OP(), lhs[3] ) ,
			    PETE_apply( OP(), lhs[4] ) ,
			    PETE_apply( OP(), lhs[5] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for AntiSymTenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP, unsigned D>
struct TSV_MetaUnary< AntiSymTenzor<T1,D> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static AntiSymTenzor<T0,D>
  apply(const AntiSymTenzor<T1,D>& lhs) {
    AntiSymTenzor<T0,D> ret;
    for (unsigned d=0; d<D*(D-1)/2; ++d)
      ret[d] = PETE_apply(OP(),lhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for AntiSymTenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< AntiSymTenzor<T1,1> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static AntiSymTenzor<T0,1>
  apply(const AntiSymTenzor<T1,1>& lhs) {
    return AntiSymTenzor<T0,1>( PETE_apply( OP(), lhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for AntiSymTenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< AntiSymTenzor<T1,2> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static AntiSymTenzor<T0,2>
  apply(const AntiSymTenzor<T1,2>& lhs) {
    return AntiSymTenzor<T0,2>( PETE_apply( OP(), lhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaUnary for AntiSymTenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class OP>
struct TSV_MetaUnary< AntiSymTenzor<T1,3> , OP >
{
  typedef typename PETEUnaryReturn<T1,OP>::type T0;
  inline static AntiSymTenzor<T0,3>
  apply(const AntiSymTenzor<T1,3>& lhs) {
    return AntiSymTenzor<T0,3>( PETE_apply( OP(), lhs[0] ) ,
				PETE_apply( OP(), lhs[1] ) ,
				PETE_apply( OP(), lhs[2] ) );
  }
};

//////////////////////////////////////////////////////////////////////

#endif // TSV_META_UNARY_H

/***************************************************************************
 * $RCSfile: TSVMetaUnary.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: TSVMetaUnary.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/

