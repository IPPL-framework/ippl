// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

// include files
#include "Field/BareField.h"
#include "Field/LField.h"
#include "Field/CompressedBrickIterator.h"
#include "Index/NDIndex.h"
#include "AppTypes/Vektor.h"
#include "Utility/IpplInfo.h"
#include "Utility/IpplException.h"

#include <iostream>
#include <vector>
#include <utility>
#include <cmath>

// Helper class and functions for finding nearest grid point given centering

// A tag indicating the Field centering type
template <class C>
class CenteringTag {
};

// Return NDIndex referring to the nearest Field element to the given position
template <class PT, unsigned Dim, class M>
inline
NDIndex<Dim> FindNGP(const M& mesh, const Vektor<PT,Dim>& ppos,
                     CenteringTag<Cell>) {
  return mesh.getCellContaining(ppos);
}

template <class PT, unsigned Dim, class M>
inline
NDIndex<Dim> FindNGP(const M& mesh, const Vektor<PT,Dim>& ppos,
                     CenteringTag<Vert>) {
  return mesh.getNearestVertex(ppos);
}

template <class PT, unsigned Dim, class M>
inline
std::vector<NDIndex<Dim> > FindNGP(const M& mesh, const Vektor<PT,Dim>&ppos,
                                   CenteringTag<Edge>) {
    std::vector<NDIndex<Dim> > ngp;
    ngp.push_back(mesh.getCellContaining(ppos));
    ngp.push_back(mesh.getNearestVertex(ppos));
    return ngp;
}

// Return position of element indicated by NDIndex
template <class PT, unsigned Dim, class M>
inline
void FindPos(Vektor<PT,Dim>& pos, const M& mesh, const NDIndex<Dim>& indices,
             CenteringTag<Cell>) {
  pos = mesh.getCellPosition(indices);
  return;
}

template <class PT, unsigned Dim, class M>
inline
void FindPos(Vektor<PT,Dim>& pos, const M& mesh, const NDIndex<Dim>& indices,
             CenteringTag<Vert>) {
  pos = mesh.getVertexPosition(indices);
  return;
}

template <class PT, unsigned Dim, class M>
inline
void FindPos(std::vector<Vektor<PT,Dim> >& pos, const M& mesh,
             const std::vector<NDIndex<Dim> >& indices, CenteringTag<Edge>) {
  pos.resize(Dim);
  Vektor<PT,Dim> cell_pos = mesh.getCellPosition(indices[0]);
  Vektor<PT,Dim> vert_pos = mesh.getVertexPosition(indices[1]);

  for (unsigned int d = 0; d < Dim; ++ d) {
    pos[d] = vert_pos;
    pos[d](d) = cell_pos(d);
  }

  return;
}

// Find sizes of next mesh element
template <class PT, unsigned Dim, class M>
inline
void FindDelta(Vektor<PT,Dim>& delta, const M& mesh, const NDIndex<Dim>& gp,
                         CenteringTag<Cell>) {
  NDIndex<Dim> vp;
  for (unsigned d=0; d<Dim; ++d) vp[d] = gp[d] + 1;
  delta = mesh.getDeltaCell(vp);
  return;
}

template <class PT, unsigned Dim, class M>
inline
void FindDelta(Vektor<PT,Dim>& delta, const M& mesh, const NDIndex<Dim>& gp,
                         CenteringTag<Vert>) {
  delta = mesh.getDeltaVertex(gp);
  return;
}

template <class PT, unsigned Dim, class M>
inline
void FindDelta(std::vector<Vektor<PT,Dim> >& delta, const M& mesh,
               const std::vector<NDIndex<Dim> >& gp, CenteringTag<Edge>) {
    NDIndex<Dim> vp;
    for (unsigned d=0; d<Dim; ++d) vp[d] = (gp[0])[d] + 1;
    delta.resize(2U);
    delta[0] = mesh.getDeltaCell(vp);
    delta[1] = mesh.getDeltaVertex(gp[1]);
    return;
}


// InterpolatorTraits struct -- used to specify type of mesh info to cache
//                              (specialized by each interpolator subclass)

template <class T, unsigned Dim, class InterpolatorType>
struct InterpolatorTraits {};

// define struct for cached mesh info for 1st-order interpolators
template <class T, unsigned Dim>
struct CacheData1 {
  NDIndex<Dim> Index_m;
  Vektor<T,Dim> Delta_m;
};

template <class T, unsigned Dim>
inline std::ostream &operator<<(std::ostream &o, const CacheData1<T,Dim> &c)
{
  o << "(" << c.Index_m << "," << c.Delta_m << ")";
  return o;
}


// define struct for cached mesh info for CIC interpolator
template <class T, unsigned Dim>
struct CacheDataCIC {
  NDIndex<Dim> Index_m;
  int Offset_m[Dim];
  Vektor<T,Dim> Delta_m;
};

//BENI:
// define struct for cached mesh info for TSC interpolator
template <class T, unsigned Dim>
struct CacheDataTSC {
  NDIndex<Dim> Index_m;
  int Offset_m[Dim];
  Vektor<T,Dim> Delta_m;
};

template <class T, unsigned Dim>
inline std::ostream &operator<<(std::ostream &o, const CacheDataCIC<T,Dim> &c)
{
  Vektor<int,Dim> offset;
  for (unsigned int i=0; i < Dim; ++i)
    offset[i] = c.Offset_m[i];
  o << "(" << c.Index_m << "," << c.Delta_m << "," << offset << ")";
  return o;
}


/* Interpolator -- Definition of base class for interpolation of data
                   for a single particle to or from a IPPL Field.         */

class Interpolator {

protected:

  // helper function, similar to BareField::localElement, but return iterator
  template <class T, unsigned Dim>
  static CompressedBrickIterator<T,Dim>
  getFieldIter(const BareField<T,Dim>& f, const NDIndex<Dim>& pt) {

    typename BareField<T,Dim>::const_iterator_if lf_i, lf_end = f.end_if();
    for (lf_i = f.begin_if(); lf_i != lf_end; ++lf_i) {
      LField<T,Dim>& lf(*(*lf_i).second);
      if ( lf.getOwned().contains(pt) ) {
	// found it ... get iterator for requested element
	return lf.begin(pt);
      }
    }

    // if not found ... try examining guard cell layers
    for (lf_i = f.begin_if(); lf_i != lf_end; ++lf_i) {
      LField<T,Dim>& lf(*(*lf_i).second);
      if ( lf.getAllocated().contains(pt) ) {
	// found it ... get iterator for requested element
	return lf.begin(pt);
      }
    }

    //    throw ("Interploator:getFieldIter: attempt to access non-local index");


    // if we're here, we did not find it ... it must not be local
    ERRORMSG("Interpolator::getFieldIter: attempt to access non-local index");
    ERRORMSG(pt << " on node " << Ippl::myNode() << endl);
    ERRORMSG("Dumping local owned and allocated domains:" << endl);
    int lfc = 0;
    for ( lf_i = f.begin_if(); lf_i != lf_end ; ++lf_i, ++lfc ) {
      LField<T,Dim>& lf(*(*lf_i).second);
      ERRORMSG(lfc << ": owned = " << lf.getOwned());
      ERRORMSG(", allocated = " << lf.getAllocated() << endl);
    }
    ERRORMSG("Error occurred for BareField with layout = " << f.getLayout());
    ERRORMSG(endl);
    ERRORMSG("Calling abort ..." << endl);
    Ippl::abort();
    return (*(*(f.begin_if())).second).begin();

  }

public:
  // constructor/destructor
  Interpolator() {}
  ~Interpolator() {}

  // gather/scatter function interfaces (implemented in derived classes)
  /*

  // scatter particle data into Field using particle position and mesh
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh);

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh,
               InterpolatorTraits<PT,Dim,Interpolator>::Cache_t& cache);

  // scatter particle data into Field using cached mesh information
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const InterpolatorTraits<PT,Dim,Interpolator>::Cache_t& cache);


  // gather particle data from Field using particle position and mesh
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh);

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh,
              InterpolatorTraits<PT,Dim,Interpolator>::Cache_t& cache);

  // gather particle data from Field using cached mesh information
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const InterpolatorTraits<PT,Dim,Interpolator>::Cache_t& cache);

  */

};

#endif // INTERPOLATOR_H

/***************************************************************************
 * $RCSfile: Interpolator.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: Interpolator.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $
 ***************************************************************************/
