// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef TSV_META_COMPARE_H
#define TSV_META_COMPARE_H

//////////////////////////////////////////////////////////////////////
//
// The definition for arrays of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaCompareArrays
{
  inline static bool
  apply(const T1* lhs, const T2* rhs) {
    for (unsigned int i = 0; i < D; i++)
      if (lhs[i] != rhs[i])
	return false;
    return true;
  }
};


//////////////////////////////////////////////////////////////////////
//
// The default definition for the template TSV_MetaCompare;
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2> struct TSV_MetaCompare {};

//////////////////////////////////////////////////////////////////////
//
// Specialization for Vektors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaCompare< Vektor<T1,D> , Vektor<T2,D> >
{
  inline static bool
  apply(const Vektor<T1,D>& lhs, const Vektor<T2,D>& rhs) {
    return TSV_MetaCompareArrays<T1,T2,D>::apply(&lhs[0],&rhs[0]);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specialization for Tenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaCompare< Tenzor<T1,D> , Tenzor<T2,D> >
{
  inline static bool
  apply(const Tenzor<T1,D>& lhs, const Tenzor<T2,D>& rhs) {
    return TSV_MetaCompareArrays<T1,T2,D*D>::apply(&lhs[0],&rhs[0]);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specialization for SymTenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaCompare< SymTenzor<T1,D> , SymTenzor<T2,D> >
{
  inline static bool
  apply(const SymTenzor<T1,D>& lhs, const SymTenzor<T2,D>& rhs) {
    return TSV_MetaCompareArrays<T1,T2,D*(D+1)/2>::apply(&lhs[0],&rhs[0]);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specialization for AntiSymTenzors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaCompare< AntiSymTenzor<T1,D> , AntiSymTenzor<T2,D> >
{
  inline static bool
  apply(const AntiSymTenzor<T1,D>& lhs, const AntiSymTenzor<T2,D>& rhs) {
    return TSV_MetaCompareArrays<T1,T2,D*(D-1)/2>::apply(&lhs[0],&rhs[0]);
  }
};

//////////////////////////////////////////////////////////////////////

#endif // TSV_META_COMPARE_H

/***************************************************************************
 * $RCSfile: TSVMetaCompare.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: TSVMetaCompare.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/

