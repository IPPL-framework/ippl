// -*- C++ -*-
//-----------------------------------------------------------------------------
// The IPPL Framework - Visit http://people.web.psi.ch/adelmann/ for more details
//
// This program was prepared by the Regents of the University of California at
#ifndef TSV_META_CROSS_H
#define TSV_META_CROSS_H

//////////////////////////////////////////////////////////////////////
//
// Definition of the struct TSV_MetaCross.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2> struct TSV_MetaCross {};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektor cross Vektor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaCross< Vektor<T1,D> , Vektor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,D>
  apply(const Vektor<T1,D>& /*a*/, const Vektor<T2,D>& /*b*/) {
    ERRORMSG("Cross-product *only* implemented for 3D; you're trying to"
	     << " do it for " << D << "D." << endl);
    Ippl::abortAllNodes("...aborting from cross()");
    Vektor<T0,D> bogusCross(-99999);
    return bogusCross;
  }
};

template<class T1, class T2>
struct TSV_MetaCross< Vektor<T1,3> , Vektor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,3>
  apply(const Vektor<T1,3>& a, const Vektor<T2,3>& b) {
    Vektor<T0,3> cross;
    cross[0] = a[1]*b[2] - a[2]*b[1];
    cross[1] = a[2]*b[0] - a[0]*b[2];
    cross[2] = a[0]*b[1] - a[1]*b[0];
    return cross;
  }
};

//////////////////////////////////////////////////////////////////////

#endif // TSV_META_CROSS_H

// $(Id)

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $ 
 ***************************************************************************/

