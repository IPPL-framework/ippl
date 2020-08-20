// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef TSV_META_DOT_DOT_H
#define TSV_META_DOT_DOT_H

//////////////////////////////////////////////////////////////////////
//
// Definition of the struct TSV_MetaDotDot.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2> struct TSV_MetaDotDot {};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzor dot-dot Tenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDotDot< Tenzor<T1,D> , Tenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Tenzor<T1,D>& lhs, const Tenzor<T2,D>& rhs) {
    T0 sum = 0.0;
    for (unsigned int i=0; i<D; ++i)
      for (unsigned int j=0; j<D; ++j)
	sum += lhs(i,j) * rhs(i,j);

    return sum;
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< Tenzor<T1,1> , Tenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Tenzor<T1,1>& lhs, const Tenzor<T2,1>& rhs) {
    return lhs[0]*rhs[0];
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< Tenzor<T1,2> , Tenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Tenzor<T1,2>& lhs, const Tenzor<T2,2>& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + 
      lhs[2] * rhs[2] + lhs[3] * rhs[3];
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< Tenzor<T1,3> , Tenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Tenzor<T1,3>& lhs, const Tenzor<T2,3>& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + 
      lhs[2] * rhs[2] + lhs[3] * rhs[3] + lhs[4] * rhs[4] +
      lhs[5] * rhs[5] + lhs[6] * rhs[6] + lhs[7] * rhs[7] +
      lhs[8] * rhs[8];
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzor dot-dot SymTenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDotDot< SymTenzor<T1,D> , SymTenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const SymTenzor<T1,D>& lhs, const SymTenzor<T2,D>& rhs) {
    T0 sum = 0.0;
    for (unsigned int i=0; i<D; ++i)
      sum += lhs.HL(i, i) * rhs.HL(i, i);

    for (unsigned int i=0; i<D; ++i)
      for (unsigned int j=i+1; j<D; ++j) 
	sum += 2.0 * lhs.HL(j, i) * rhs.HL(j, i);

    return sum;
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< SymTenzor<T1,1> , SymTenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const SymTenzor<T1,1>& lhs, const SymTenzor<T2,1>& rhs) {
    return lhs[0] * rhs[0];
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< SymTenzor<T1,2> , SymTenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const SymTenzor<T1,2>& lhs, const SymTenzor<T2,2>& rhs) {
    return lhs(0,0) * rhs(0,0) + lhs(1,1) * rhs(1,1) + 
      2.0 * lhs(0,1) * rhs(0,1);
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< SymTenzor<T1,3> , SymTenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const SymTenzor<T1,3>& lhs, const SymTenzor<T2,3>& rhs) {
    return lhs(0,0) * rhs(0,0) + lhs(1,1) * rhs(1,1) + lhs(2,2) * rhs(2,2) +
      2.0 * (lhs(0,1) * rhs(0,1) + lhs(0,2) * rhs(0,2) + 
        lhs(1,2) * rhs(1,2));
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzor dot-dot Tenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDotDot< SymTenzor<T1,D> , Tenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const SymTenzor<T1,D>& lhs, const Tenzor<T2,D>& rhs) {
    T0 sum = 0.0;
    for (unsigned int i=0; i<D; ++i)
      for (unsigned int j=0; j<D; ++j)
	sum += lhs(i,j) * rhs(i,j);

    return sum;
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< SymTenzor<T1,1> , Tenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const SymTenzor<T1,1>& lhs, const Tenzor<T2,1>& rhs) {
    return lhs[0]*rhs[0];
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< SymTenzor<T1,2> , Tenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const SymTenzor<T1,2>& lhs, const Tenzor<T2,2>& rhs) {
    return lhs(0,0) * rhs(0,0) + lhs(0,1) * (rhs(0,1) + rhs(1,0)) +
      lhs(1,1) * rhs(1,1);
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< SymTenzor<T1,3> , Tenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const SymTenzor<T1,3>& lhs, const Tenzor<T2,3>& rhs) {
    return lhs(0,0) * rhs(0,0) + lhs(0,1) * (rhs(0,1) + rhs(1,0)) +
      + lhs(0,2) * (rhs(0,2) + rhs(2,0)) + lhs(1,1) * rhs(1,1) +
      lhs(1,2) * (rhs(1,2) + rhs(2,1)) + lhs(2,2) * rhs(2,2);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzor dot-dot SymTenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDotDot< Tenzor<T1,D> , SymTenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Tenzor<T1,D>& lhs, const SymTenzor<T2,D>& rhs) {
    T0 sum = 0.0;
    for (unsigned int i=0; i<D; ++i)
      for (unsigned int j=0; j<D; ++j)
	sum += lhs(i,j) * rhs(j,j);

    return sum;
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< Tenzor<T1,1> , SymTenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Tenzor<T1,1>& lhs, const SymTenzor<T2,1>& rhs) {
    return lhs[0]*rhs[0];
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< Tenzor<T1,2> , SymTenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Tenzor<T1,2>& lhs, const SymTenzor<T2,2>& rhs) {
    return lhs(0,0) * rhs(0,0) + (lhs(0,1) + lhs(1,0)) * rhs(0,1) +
      lhs(1,1) * rhs(1,1);
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< Tenzor<T1,3> , SymTenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Tenzor<T1,3>& lhs, const SymTenzor<T2,3>& rhs) {
    return lhs(0,0) * rhs(0,0) + (lhs(0,1) + lhs(1,0)) * rhs(0,1) +
      lhs(1,1) * rhs(1,1) + (lhs(0,2) + lhs(2,0)) * rhs(0,2) +
      (lhs(1,2) + lhs(2,1)) * rhs(1,2) + lhs(2,2) * rhs(2,2);
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for AntiSymTenzor dot-dot AntiSymTenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDotDot< AntiSymTenzor<T1,D> , AntiSymTenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const AntiSymTenzor<T1,D>& lhs, const AntiSymTenzor<T2,D>& rhs) {
    T0 sum = lhs[0]*rhs[0];
    for ( int i=1; i<D*(D-1)/2; ++i)
      sum += lhs[i]*rhs[i];
    return sum+sum;
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< AntiSymTenzor<T1,2> , AntiSymTenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const AntiSymTenzor<T1,2>& lhs, const AntiSymTenzor<T2,2>& rhs) {
    T0 sum = lhs[0]*rhs[0];
    return sum+sum;
  }
};

template<class T1, class T2>
struct TSV_MetaDotDot< AntiSymTenzor<T1,3> , AntiSymTenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const AntiSymTenzor<T1,3>& lhs, const AntiSymTenzor<T2,3>& rhs) {
    T0 sum = lhs[0]*rhs[0]+lhs[1]*rhs[1]+lhs[2]*rhs[2];
    return sum+sum;
  }
};

//////////////////////////////////////////////////////////////////////

#endif // TSV_META_DOT_DOT_H

/***************************************************************************
 * $RCSfile: TSVMetaDotDot.h,v $
 * $Revision: 1.1.1.1 $
 * IPPL_VERSION_ID: $Id: TSVMetaDotDot.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $
 ***************************************************************************/

