// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef TSV_META_ASSIGN_H
#define TSV_META_ASSIGN_H

//////////////////////////////////////////////////////////////////////
//
// Definition of structs TSV_MetaAssign and TSV_MetaAssignScalar
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP> struct TSV_MetaAssign {};
template<class T1, class T2, class OP> struct TSV_MetaAssignScalar {};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaAssign< Vektor<T1,D> , Vektor<T2,D> , OP >
{
  inline static void
  apply( Vektor<T1,D>& lhs, const Vektor<T2,D>& rhs) {
    for (unsigned d=0; d<D; ++d)
      PETE_apply( OP(), lhs[d] , rhs[d]);
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaAssignScalar< Vektor<T1,D> , T2 , OP >
{
  inline static void
  apply( Vektor<T1,D>& lhs, T2 rhs ) {
    for (unsigned d=0; d<D; ++d)
      PETE_apply( OP(), lhs[d] , rhs);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< Vektor<T1,1> , Vektor<T2,1> , OP >
{
  inline static void
  apply( Vektor<T1,1>& lhs, const Vektor<T2,1>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< Vektor<T1,1> , T2 , OP >
{
  inline static void
  apply( Vektor<T1,1>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< Vektor<T1,2> , Vektor<T2,2> , OP >
{
  inline static void
  apply( Vektor<T1,2>& lhs, const Vektor<T2,2>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
    PETE_apply( OP(), lhs[1] , rhs[1] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< Vektor<T1,2> , T2 , OP >
{
  inline static void
  apply( Vektor<T1,2>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
    PETE_apply( OP(), lhs[1] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< Vektor<T1,3> , Vektor<T2,3> , OP >
{
  inline static void
  apply( Vektor<T1,3>& lhs, const Vektor<T2,3>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
    PETE_apply( OP(), lhs[1] , rhs[1] );
    PETE_apply( OP(), lhs[2] , rhs[2] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< Vektor<T1,3> , T2 , OP >
{
  inline static void
  apply( Vektor<T1,3>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
    PETE_apply( OP(), lhs[1] , rhs );
    PETE_apply( OP(), lhs[2] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// The default definitions for Tenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaAssign< Tenzor<T1,D> , Tenzor<T2,D> , OP >
{
  inline static void
  apply( Tenzor<T1,D>& lhs, const Tenzor<T2,D>& rhs) {
    for (unsigned d=0; d<D*D; ++d)
      PETE_apply( OP(), lhs[d] , rhs[d]);
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaAssignScalar< Tenzor<T1,D> , T2 , OP >
{
  inline static void
  apply( Tenzor<T1,D>& lhs, T2 rhs ) {
    for (unsigned d=0; d<D*D; ++d)
      PETE_apply( OP(), lhs[d] , rhs);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< Tenzor<T1,1> , Tenzor<T2,1> , OP >
{
  inline static void
  apply( Tenzor<T1,1>& lhs, const Tenzor<T2,1>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< Tenzor<T1,1> , T2 , OP >
{
  inline static void
  apply( Tenzor<T1,1>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< Tenzor<T1,2> , Tenzor<T2,2> , OP >
{
  inline static void
  apply( Tenzor<T1,2>& lhs, const Tenzor<T2,2>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
    PETE_apply( OP(), lhs[1] , rhs[1] );
    PETE_apply( OP(), lhs[2] , rhs[2] );
    PETE_apply( OP(), lhs[3] , rhs[3] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< Tenzor<T1,2> , T2 , OP >
{
  inline static void
  apply( Tenzor<T1,2>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
    PETE_apply( OP(), lhs[1] , rhs );
    PETE_apply( OP(), lhs[2] , rhs );
    PETE_apply( OP(), lhs[3] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< Tenzor<T1,3> , Tenzor<T2,3> , OP >
{
  inline static void
  apply( Tenzor<T1,3>& lhs, const Tenzor<T2,3>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
    PETE_apply( OP(), lhs[1] , rhs[1] );
    PETE_apply( OP(), lhs[2] , rhs[2] );
    PETE_apply( OP(), lhs[3] , rhs[3] );
    PETE_apply( OP(), lhs[4] , rhs[4] );
    PETE_apply( OP(), lhs[5] , rhs[5] );
    PETE_apply( OP(), lhs[6] , rhs[6] );
    PETE_apply( OP(), lhs[7] , rhs[7] );
    PETE_apply( OP(), lhs[8] , rhs[8] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< Tenzor<T1,3> , T2 , OP >
{
  inline static void
  apply( Tenzor<T1,3>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
    PETE_apply( OP(), lhs[1] , rhs );
    PETE_apply( OP(), lhs[2] , rhs );
    PETE_apply( OP(), lhs[3] , rhs );
    PETE_apply( OP(), lhs[4] , rhs );
    PETE_apply( OP(), lhs[5] , rhs );
    PETE_apply( OP(), lhs[6] , rhs );
    PETE_apply( OP(), lhs[7] , rhs );
    PETE_apply( OP(), lhs[8] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// The default definitions for SymTenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaAssign< SymTenzor<T1,D> , SymTenzor<T2,D> , OP >
{
  inline static void
  apply( SymTenzor<T1,D>& lhs, const SymTenzor<T2,D>& rhs) {
    for (unsigned d=0; d<D*(D+1)/2; ++d)
      PETE_apply( OP(), lhs[d] , rhs[d]);
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaAssignScalar< SymTenzor<T1,D> , T2 , OP >
{
  inline static void
  apply( SymTenzor<T1,D>& lhs, T2 rhs ) {
    for (unsigned d=0; d<D*(D+1)/2; ++d)
      PETE_apply( OP(), lhs[d] , rhs);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< SymTenzor<T1,1> , SymTenzor<T2,1> , OP >
{
  inline static void
  apply( SymTenzor<T1,1>& lhs, const SymTenzor<T2,1>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< SymTenzor<T1,1> , T2 , OP >
{
  inline static void
  apply( SymTenzor<T1,1>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< SymTenzor<T1,2> , SymTenzor<T2,2> , OP >
{
  inline static void
  apply( SymTenzor<T1,2>& lhs, const SymTenzor<T2,2>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
    PETE_apply( OP(), lhs[1] , rhs[1] );
    PETE_apply( OP(), lhs[2] , rhs[2] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< SymTenzor<T1,2> , T2 , OP >
{
  inline static void
  apply( SymTenzor<T1,2>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
    PETE_apply( OP(), lhs[1] , rhs );
    PETE_apply( OP(), lhs[2] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< SymTenzor<T1,3> , SymTenzor<T2,3> , OP >
{
  inline static void
  apply( SymTenzor<T1,3>& lhs, const SymTenzor<T2,3>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
    PETE_apply( OP(), lhs[1] , rhs[1] );
    PETE_apply( OP(), lhs[2] , rhs[2] );
    PETE_apply( OP(), lhs[3] , rhs[3] );
    PETE_apply( OP(), lhs[4] , rhs[4] );
    PETE_apply( OP(), lhs[5] , rhs[5] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< SymTenzor<T1,3> , T2 , OP >
{
  inline static void
  apply( SymTenzor<T1,3>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
    PETE_apply( OP(), lhs[1] , rhs );
    PETE_apply( OP(), lhs[2] , rhs );
    PETE_apply( OP(), lhs[3] , rhs );
    PETE_apply( OP(), lhs[4] , rhs );
    PETE_apply( OP(), lhs[5] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// The default definitions for AntiSymTenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaAssign< AntiSymTenzor<T1,D> , AntiSymTenzor<T2,D> , OP >
{
  inline static void
  apply( AntiSymTenzor<T1,D>& lhs, const AntiSymTenzor<T2,D>& rhs) {
    for (unsigned d=0; d<D*(D-1)/2; ++d)
      PETE_apply( OP(), lhs[d] , rhs[d]);
  }
};

template<class T1, class T2, class OP, unsigned D>
struct TSV_MetaAssignScalar< AntiSymTenzor<T1,D> , T2 , OP >
{
  inline static void
  apply( AntiSymTenzor<T1,D>& lhs, T2 rhs ) {
    for (unsigned d=0; d<D*(D-1)/2; ++d)
      PETE_apply( OP(), lhs[d] , rhs);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for AntiSymTenzors with D=1.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< AntiSymTenzor<T1,1> , AntiSymTenzor<T2,1> , OP >
{
  inline static void
  apply( AntiSymTenzor<T1,1>& /*lhs*/, const AntiSymTenzor<T2,1>& /*rhs*/) {
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< AntiSymTenzor<T1,1> , T2 , OP >
{
  inline static void
  apply( AntiSymTenzor<T1,1>& /*lhs*/, T2 /*rhs*/) {
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for AntiSymTenzors with D=2.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< AntiSymTenzor<T1,2> , AntiSymTenzor<T2,2> , OP >
{
  inline static void
  apply( AntiSymTenzor<T1,2>& lhs, const AntiSymTenzor<T2,2>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< AntiSymTenzor<T1,2> , T2 , OP >
{
  inline static void
  apply( AntiSymTenzor<T1,2>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for AntiSymTenzors with D=3.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP>
struct TSV_MetaAssign< AntiSymTenzor<T1,3> , AntiSymTenzor<T2,3> , OP >
{
  inline static void
  apply( AntiSymTenzor<T1,3>& lhs, const AntiSymTenzor<T2,3>& rhs) {
    PETE_apply( OP(), lhs[0] , rhs[0] );
    PETE_apply( OP(), lhs[1] , rhs[1] );
    PETE_apply( OP(), lhs[2] , rhs[2] );
  }
};

template<class T1, class T2, class OP>
struct TSV_MetaAssignScalar< AntiSymTenzor<T1,3> , T2 , OP >
{
  inline static void
  apply( AntiSymTenzor<T1,3>& lhs, T2 rhs ) {
    PETE_apply( OP(), lhs[0] , rhs );
    PETE_apply( OP(), lhs[1] , rhs );
    PETE_apply( OP(), lhs[2] , rhs );
  }
};

////////////////////////////////////////////////////////////////////////////

#endif // TSV_META_ASSIGN_H

/***************************************************************************
 * $RCSfile: TSVMetaAssign.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: TSVMetaAssign.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/

