// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INT_NGP_H
#define INT_NGP_H

/* IntNGP.h -- Definition of simple class to perform nearest-grid-point
   interpolation of data for a single particle to or from IPPL Field. */

// include files
#include "Particle/Interpolator.h"
#include "Field/Field.h"


// forward declaration
class IntNGP;

// specialization of InterpolatorTraits

template <class T, unsigned Dim>
struct InterpolatorTraits<T,Dim,IntNGP> {
  typedef NDIndex<Dim> Cache_t;
};


// IntNGP class definition
class IntNGP : public Interpolator {

public:
  // constructor/destructor
  IntNGP() {}
  ~IntNGP() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh) {
    // find nearest-grid-point for particle position, store in NDIndex obj
    NDIndex<Dim> ngp = FindNGP(mesh, ppos, CenteringTag<C>());
    // scatter data value to Field ... this assumes that the Field
    // data point is local to this processor, if not an error will be printed.

    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    *fiter += pdata;

    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh,
               NDIndex<Dim>& ngp) {
    // find nearest-grid-point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, CenteringTag<C>());
    // scatter data value to Field ... this assumes that the Field
    // data point is local to this processor, if not an error will be printed.

    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    *fiter += pdata;

    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, unsigned Dim, class M, class C>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const NDIndex<Dim>& ngp) {
    // scatter data value to Field ... this assumes that the Field
    // data point is local to this processor, if not an error will be printed.

    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    *fiter += pdata;

    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh) {
    // find nearest-grid-point for particle position, store in NDIndex obj
    NDIndex<Dim> ngp = FindNGP(mesh, ppos, CenteringTag<C>());
    // gather Field value to particle data ... this assumes that the Field
    // data point is local to this processor, if not an error will be printed.

    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    pdata = *fiter;

    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh,
              NDIndex<Dim>& ngp) {
    // find nearest-grid-point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, CenteringTag<C>());
    // gather Field value to particle data ... this assumes that the Field
    // data point is local to this processor, if not an error will be printed.

    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    pdata = *fiter;

    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, unsigned Dim, class M, class C>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const NDIndex<Dim>& ngp) {
    // gather Field value to particle data ... this assumes that the Field
    // data point is local to this processor, if not an error will be printed.

    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    pdata = *fiter;

    return;
  }

};

#endif // INT_NGP_H

/***************************************************************************
 * $RCSfile: IntNGP.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: IntNGP.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/

