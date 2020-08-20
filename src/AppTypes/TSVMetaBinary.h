// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef TSV_META_BINARY_H
#define TSV_META_BINARY_H

//////////////////////////////////////////////////////////////////////
//
// Definition of structs TSV_MetaBinary and TSV_MetaBinaryScalar
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP> struct TSV_MetaBinary {};
template<class T1, class T2, class OP> struct TSV_MetaBinaryScalar {};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinary< Vektor<T1,D> , Vektor<T2,D> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,D>
  apply(const Vektor<T1,D>& lhs, const Vektor<T2,D>& rhs) {
    Vektor<T0,D> ret;
    for (unsigned d=0; d<D; ++d)
      ret[d] = PETE_apply(OP(),lhs[d] , rhs[d]);
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinaryScalar< Vektor<T1,D> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,D>
  apply(const Vektor<T1,D>& lhs, T2 rhs) {
    Vektor<T0,D> ret;
    for (unsigned d=0; d<D; ++d)
      ret[d] = PETE_apply( OP(), lhs[d] , rhs );
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinaryScalar< T1, Vektor<T2,D> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,D>
  apply(T1 lhs, const Vektor<T2,D>& rhs) {
    Vektor<T0,D> ret;
    for (unsigned d=0; d<D; ++d)
      ret[d] = PETE_apply( OP(), lhs , rhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for Vektors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< Vektor<T1,1> , Vektor<T2,1> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,1>
  apply(const Vektor<T1,1>& lhs, const Vektor<T2,1>& rhs) {
    return Vektor<T0,1>( PETE_apply( OP(), lhs[0], rhs[0] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< Vektor<T1,1> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,1>
  apply(const Vektor<T1,1>& lhs, T2 rhs) {
    return Vektor<T0,1>( PETE_apply( OP(), lhs[0], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, Vektor<T2,1> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,1>
  apply(T1 lhs, const Vektor<T2,1>& rhs) {
    return Vektor<T0,1>( PETE_apply( OP(), lhs, rhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for Vektors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< Vektor<T1,2> , Vektor<T2,2> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,2>
  apply(const Vektor<T1,2>& lhs, const Vektor<T2,2>& rhs) {
    return Vektor<T0,2>( PETE_apply( OP(), lhs[0], rhs[0] ) ,
			 PETE_apply( OP(), lhs[1], rhs[1] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< Vektor<T1,2> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,2>
  apply(const Vektor<T1,2>& lhs, T2 rhs) {
    return Vektor<T0,2>( PETE_apply( OP(), lhs[0], rhs ) ,
			 PETE_apply( OP(), lhs[1], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, Vektor<T2,2> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,2>
  apply(T1 lhs, const Vektor<T2,2>& rhs) {
    return Vektor<T0,2>( PETE_apply( OP(), lhs, rhs[0] ) ,
			 PETE_apply( OP(), lhs, rhs[1] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for Vektors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< Vektor<T1,3> , Vektor<T2,3> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,3>
  apply(const Vektor<T1,3>& lhs, const Vektor<T2,3>& rhs) {
    return Vektor<T0,3>( PETE_apply( OP(), lhs[0], rhs[0] ) ,
			 PETE_apply( OP(), lhs[1], rhs[1] ) ,
			 PETE_apply( OP(), lhs[2], rhs[2] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< Vektor<T1,3> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,3>
  apply(const Vektor<T1,3>& lhs, T2 rhs) {
    return Vektor<T0,3>( PETE_apply( OP(), lhs[0], rhs ) ,
			 PETE_apply( OP(), lhs[1], rhs ) ,
			 PETE_apply( OP(), lhs[2], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, Vektor<T2,3> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Vektor<T0,3>
  apply(T1 lhs, const Vektor<T2,3>& rhs) {
    return Vektor<T0,3>( PETE_apply( OP(), lhs, rhs[0] ) ,
			 PETE_apply( OP(), lhs, rhs[1] ) ,
			 PETE_apply( OP(), lhs, rhs[2] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinary< Tenzor<T1,D> , Tenzor<T2,D> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,D>
  apply(const Tenzor<T1,D>& lhs, const Tenzor<T2,D>& rhs) {
    Tenzor<T0,D> ret;
    for (unsigned d=0; d<D*D; ++d)
      ret[d] = PETE_apply(OP(),lhs[d] , rhs[d]);
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinaryScalar< Tenzor<T1,D> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,D>
  apply(const Tenzor<T1,D>& lhs, T2 rhs) {
    Tenzor<T0,D> ret;
    for (unsigned d=0; d<D*D; ++d)
      ret[d] = PETE_apply( OP(), lhs[d] , rhs );
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinaryScalar< T1, Tenzor<T2,D> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,D>
  apply(T1 lhs, const Tenzor<T2,D>& rhs) {
    Tenzor<T0,D> ret;
    for (unsigned d=0; d<D*D; ++d)
      ret[d] = PETE_apply( OP(), lhs , rhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for Tenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< Tenzor<T1,1> , Tenzor<T2,1> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,1>
  apply(const Tenzor<T1,1>& lhs, const Tenzor<T2,1>& rhs) {
    return Tenzor<T0,1>( PETE_apply( OP(), lhs[0], rhs[0] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< Tenzor<T1,1> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,1>
  apply(const Tenzor<T1,1>& lhs, T2 rhs) {
    return Tenzor<T0,1>( PETE_apply( OP(), lhs[0], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, Tenzor<T2,1> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,1>
  apply(T1 lhs, const Tenzor<T2,1>& rhs) {
    return Tenzor<T0,1>( PETE_apply( OP(), lhs, rhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for Tenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< Tenzor<T1,2> , Tenzor<T2,2> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,2>
  apply(const Tenzor<T1,2>& lhs, const Tenzor<T2,2>& rhs) {
    return Tenzor<T0,2>( PETE_apply( OP(), lhs[0], rhs[0] ) ,
			 PETE_apply( OP(), lhs[1], rhs[1] ) ,
			 PETE_apply( OP(), lhs[2], rhs[2] ) ,
			 PETE_apply( OP(), lhs[3], rhs[3] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< Tenzor<T1,2> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,2>
  apply(const Tenzor<T1,2>& lhs, T2 rhs) {
    return Tenzor<T0,2>( PETE_apply( OP(), lhs[0], rhs ) ,
			 PETE_apply( OP(), lhs[1], rhs ) ,
			 PETE_apply( OP(), lhs[2], rhs ) ,
			 PETE_apply( OP(), lhs[3], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, Tenzor<T2,2> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,2>
  apply(T1 lhs, const Tenzor<T2,2>& rhs) {
    return Tenzor<T0,2>( PETE_apply( OP(), lhs, rhs[0] ) ,
			 PETE_apply( OP(), lhs, rhs[1] ) ,
			 PETE_apply( OP(), lhs, rhs[2] ) ,
			 PETE_apply( OP(), lhs, rhs[3] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for Tenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< Tenzor<T1,3> , Tenzor<T2,3> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,3>
  apply(const Tenzor<T1,3>& lhs, const Tenzor<T2,3>& rhs) {
    return Tenzor<T0,3>( PETE_apply( OP(), lhs[0], rhs[0] ) ,
			 PETE_apply( OP(), lhs[1], rhs[1] ) ,
			 PETE_apply( OP(), lhs[2], rhs[2] ) ,
			 PETE_apply( OP(), lhs[3], rhs[3] ) ,
			 PETE_apply( OP(), lhs[4], rhs[4] ) ,
			 PETE_apply( OP(), lhs[5], rhs[5] ) ,
			 PETE_apply( OP(), lhs[6], rhs[6] ) ,
			 PETE_apply( OP(), lhs[7], rhs[7] ) ,
			 PETE_apply( OP(), lhs[8], rhs[8] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< Tenzor<T1,3> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,3>
  apply(const Tenzor<T1,3>& lhs, T2 rhs) {
    return Tenzor<T0,3>( PETE_apply( OP(), lhs[0], rhs ) ,
			 PETE_apply( OP(), lhs[1], rhs ) ,
			 PETE_apply( OP(), lhs[2], rhs ) ,
			 PETE_apply( OP(), lhs[3], rhs ) ,
			 PETE_apply( OP(), lhs[4], rhs ) ,
			 PETE_apply( OP(), lhs[5], rhs ) ,
			 PETE_apply( OP(), lhs[6], rhs ) ,
			 PETE_apply( OP(), lhs[7], rhs ) ,
			 PETE_apply( OP(), lhs[8], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, Tenzor<T2,3> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,3>
  apply(T1 lhs, const Tenzor<T2,3>& rhs) {
    return Tenzor<T0,3>( PETE_apply( OP(), lhs, rhs[0] ) ,
			 PETE_apply( OP(), lhs, rhs[1] ) ,
			 PETE_apply( OP(), lhs, rhs[2] ) ,
			 PETE_apply( OP(), lhs, rhs[3] ) ,
			 PETE_apply( OP(), lhs, rhs[4] ) ,
			 PETE_apply( OP(), lhs, rhs[5] ) ,
			 PETE_apply( OP(), lhs, rhs[6] ) ,
			 PETE_apply( OP(), lhs, rhs[7] ) ,
			 PETE_apply( OP(), lhs, rhs[8] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinary< SymTenzor<T1,D> , SymTenzor<T2,D> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,D>
  apply(const SymTenzor<T1,D>& lhs, const SymTenzor<T2,D>& rhs) {
    SymTenzor<T0,D> ret;
    for (unsigned d=0; d<D*(D+1)/2; ++d)
      ret[d] = PETE_apply(OP(),lhs[d] , rhs[d]);
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinaryScalar< SymTenzor<T1,D> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,D>
  apply(const SymTenzor<T1,D>& lhs, T2 rhs) {
    SymTenzor<T0,D> ret;
    for (unsigned d=0; d<D*(D+1)/2; ++d)
      ret[d] = PETE_apply( OP(), lhs[d] , rhs );
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinaryScalar< T1, SymTenzor<T2,D> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,D>
  apply(T1 lhs, const SymTenzor<T2,D>& rhs) {
    SymTenzor<T0,D> ret;
    for (unsigned d=0; d<D*(D+1)/2; ++d)
      ret[d] = PETE_apply( OP(), lhs , rhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for SymTenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< SymTenzor<T1,1> , SymTenzor<T2,1> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,1>
  apply(const SymTenzor<T1,1>& lhs, const SymTenzor<T2,1>& rhs) {
    return SymTenzor<T0,1>( PETE_apply( OP(), lhs[0], rhs[0] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< SymTenzor<T1,1> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,1>
  apply(const SymTenzor<T1,1>& lhs, T2 rhs) {
    return SymTenzor<T0,1>( PETE_apply( OP(), lhs[0], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, SymTenzor<T2,1> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,1>
  apply(T1 lhs, const SymTenzor<T2,1>& rhs) {
    return SymTenzor<T0,1>( PETE_apply( OP(), lhs, rhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for SymTenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< SymTenzor<T1,2> , SymTenzor<T2,2> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,2>
  apply(const SymTenzor<T1,2>& lhs, const SymTenzor<T2,2>& rhs) {
    return SymTenzor<T0,2>( PETE_apply( OP(), lhs[0], rhs[0] ) ,
			    PETE_apply( OP(), lhs[1], rhs[1] ) ,
			    PETE_apply( OP(), lhs[2], rhs[2] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< SymTenzor<T1,2> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,2>
  apply(const SymTenzor<T1,2>& lhs, T2 rhs) {
    return SymTenzor<T0,2>( PETE_apply( OP(), lhs[0], rhs ) ,
			    PETE_apply( OP(), lhs[1], rhs ) ,
			    PETE_apply( OP(), lhs[2], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, SymTenzor<T2,2> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,2>
  apply(T1 lhs, const SymTenzor<T2,2>& rhs) {
    return SymTenzor<T0,2>( PETE_apply( OP(), lhs, rhs[0] ) ,
			    PETE_apply( OP(), lhs, rhs[1] ) ,
			    PETE_apply( OP(), lhs, rhs[2] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for SymTenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< SymTenzor<T1,3> , SymTenzor<T2,3> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,3>
  apply(const SymTenzor<T1,3>& lhs, const SymTenzor<T2,3>& rhs) {
    return SymTenzor<T0,3>( PETE_apply( OP(), lhs[0], rhs[0] ) ,
			    PETE_apply( OP(), lhs[1], rhs[1] ) ,
			    PETE_apply( OP(), lhs[2], rhs[2] ) ,
			    PETE_apply( OP(), lhs[3], rhs[3] ) ,
			    PETE_apply( OP(), lhs[4], rhs[4] ) ,
			    PETE_apply( OP(), lhs[5], rhs[5] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< SymTenzor<T1,3> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,3>
  apply(const SymTenzor<T1,3>& lhs, T2 rhs) {
    return SymTenzor<T0,3>( PETE_apply( OP(), lhs[0], rhs ) ,
			    PETE_apply( OP(), lhs[1], rhs ) ,
			    PETE_apply( OP(), lhs[2], rhs ) ,
			    PETE_apply( OP(), lhs[3], rhs ) ,
			    PETE_apply( OP(), lhs[4], rhs ) ,
			    PETE_apply( OP(), lhs[5], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, SymTenzor<T2,3> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static SymTenzor<T0,3>
  apply(T1 lhs, const SymTenzor<T2,3>& rhs) {
    return SymTenzor<T0,3>( PETE_apply( OP(), lhs, rhs[0] ) ,
			    PETE_apply( OP(), lhs, rhs[1] ) ,
			    PETE_apply( OP(), lhs, rhs[2] ) ,
			    PETE_apply( OP(), lhs, rhs[3] ) ,
			    PETE_apply( OP(), lhs, rhs[4] ) ,
			    PETE_apply( OP(), lhs, rhs[5] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specialization for SymTenzor OP Tenzor of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinary< SymTenzor<T1,D>, Tenzor<T2,D>, OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,D>
  apply(const SymTenzor<T1,D> &lhs, const Tenzor<T2,D> &rhs) {
    Tenzor<T0,D> ret;
    for (unsigned i = 0; i < D; i++)
      for (unsigned j = 0; j < D; j++)
	ret(i, j) = PETE_apply(OP(), lhs(i, j), rhs(i, j));
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specialization for Tenzor OP SymTenzor of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinary< Tenzor<T1,D>, SymTenzor<T2,D>, OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static Tenzor<T0,D>
  apply(const Tenzor<T1,D> &lhs, const SymTenzor<T2,D> &rhs) {
    Tenzor<T0,D> ret;
    for (unsigned i = 0; i < D; i++)
      for (unsigned j = 0; j < D; j++)
	ret(i, j) = PETE_apply(OP(), lhs(i, j), rhs(i, j));
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for AntiSymTenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinary< AntiSymTenzor<T1,D> , AntiSymTenzor<T2,D> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,D>
  apply(const AntiSymTenzor<T1,D>& lhs, const AntiSymTenzor<T2,D>& rhs) {
    AntiSymTenzor<T0,D> ret;
    for (unsigned d=0; d<D*(D-1)/2; ++d)
      ret[d] = PETE_apply(OP(),lhs[d] , rhs[d]);
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinaryScalar< AntiSymTenzor<T1,D> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,D>
  apply(const AntiSymTenzor<T1,D>& lhs, T2 rhs) {
    AntiSymTenzor<T0,D> ret;
    for (unsigned d=0; d<D*(D-1)/2; ++d)
      ret[d] = PETE_apply( OP(), lhs[d] , rhs );
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaBinaryScalar< T1, AntiSymTenzor<T2,D> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,D>
  apply(T1 lhs, const AntiSymTenzor<T2,D>& rhs) {
    AntiSymTenzor<T0,D> ret;
    for (unsigned d=0; d<D*(D-1)/2; ++d)
      ret[d] = PETE_apply( OP(), lhs , rhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for AntiSymTenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< AntiSymTenzor<T1,1> , AntiSymTenzor<T2,1> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,1>
  apply(const AntiSymTenzor<T1,1>& /*lhs*/, const AntiSymTenzor<T2,1>& /*rhs*/) {
    typedef typename AntiSymTenzor<T0,1>::DontInitialize T;
    return AntiSymTenzor<T0,1>( T() );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< AntiSymTenzor<T1,1> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,1>
  apply(const AntiSymTenzor<T1,1>& /*lhs*/, T2 /*rhs*/) {
    typedef typename AntiSymTenzor<T0,1>::DontInitialize T;
    return AntiSymTenzor<T0,1>( T() );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, AntiSymTenzor<T2,1> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,1>
  apply(T1 /*lhs*/, const AntiSymTenzor<T2,1>& /*rhs*/) {
    typedef typename AntiSymTenzor<T0,1>::DontInitialize T;
    return AntiSymTenzor<T0,1>( T() );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for AntiSymTenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< AntiSymTenzor<T1,2> , AntiSymTenzor<T2,2> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,2>
  apply(const AntiSymTenzor<T1,2>& lhs, const AntiSymTenzor<T2,2>& rhs) {
    return AntiSymTenzor<T0,2>( PETE_apply( OP(), lhs[0], rhs[0] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< AntiSymTenzor<T1,2> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,2>
  apply(const AntiSymTenzor<T1,2>& lhs, T2 rhs) {
    return AntiSymTenzor<T0,2>( PETE_apply( OP(), lhs[0], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, AntiSymTenzor<T2,2> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,2>
  apply(T1 lhs, const AntiSymTenzor<T2,2>& rhs) {
    return AntiSymTenzor<T0,2>( PETE_apply( OP(), lhs, rhs[0] ) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations of TSV_MetaBinary for AntiSymTenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaBinary< AntiSymTenzor<T1,3> , AntiSymTenzor<T2,3> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,3>
  apply(const AntiSymTenzor<T1,3>& lhs, const AntiSymTenzor<T2,3>& rhs) {
    return AntiSymTenzor<T0,3>( PETE_apply( OP(), lhs[0], rhs[0] ) ,
				PETE_apply( OP(), lhs[1], rhs[1] ) ,
				PETE_apply( OP(), lhs[2], rhs[2] ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< AntiSymTenzor<T1,3> , T2 , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,3>
  apply(const AntiSymTenzor<T1,3>& lhs, T2 rhs) {
    return AntiSymTenzor<T0,3>( PETE_apply( OP(), lhs[0], rhs ) ,
				PETE_apply( OP(), lhs[1], rhs ) ,
				PETE_apply( OP(), lhs[2], rhs ) );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaBinaryScalar< T1, AntiSymTenzor<T2,3> , OP >
{
  typedef typename PETEBinaryReturn<T1,T2,OP>::type T0;
  inline static AntiSymTenzor<T0,3>
  apply(T1 lhs, const AntiSymTenzor<T2,3>& rhs) {
    return AntiSymTenzor<T0,3>( PETE_apply( OP(), lhs, rhs[0] ) ,
				PETE_apply( OP(), lhs, rhs[1] ) ,
				PETE_apply( OP(), lhs, rhs[2] ) );
  }
};

//////////////////////////////////////////////////////////////////////

#endif // TSV_META_BINARY_H

/***************************************************************************
 * $RCSfile: TSVMetaBinary.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: TSVMetaBinary.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/
