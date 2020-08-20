// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INT_SUDS_H
#define INT_SUDS_H

/* IntSUDS.h -- Definition of simple class to perform subtracted dipole
   interpolation of data for a single particle to or from a IPPL Field.   */

// include files
#include "Particle/Interpolator.h"
#include "Field/Field.h"


// forward declaration
class IntSUDS;

// specialization of InterpolatorTraits
template <class T, unsigned Dim>
struct InterpolatorTraits<T,Dim,IntSUDS> {
  typedef CacheData1<T,Dim> Cache_t;
};


// IntSUDSImpl class definition
template <unsigned Dim>
class IntSUDSImpl : public Interpolator {

public:
  // constructor/destructor
  IntSUDSImpl() {}
  ~IntSUDSImpl() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& /*pdata*/, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,Dim> gpos, dpos, delta;
    //    unsigned int d;
    // find nearest grid point for particle position, store in NDIndex obj
    NDIndex<Dim> ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    ERRORMSG("IntSUDS::scatter: not implemented for Dim>3!!"<<endl);
    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& /*pdata*/, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh,
               NDIndex<Dim>& ngp, Vektor<PT,Dim>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,Dim> gpos, delta;
    //    unsigned int d;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    ERRORMSG("IntSUDS::scatter: not implemented for Dim>3!!"<<endl);
    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& /*pdata*/, Field<FT,Dim,M,C>& f,
	       const NDIndex<Dim>& ngp, const Vektor<PT,Dim>&) {
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    ERRORMSG("IntSUDS::scatter: not implemented for Dim>3!!"<<endl);
    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& /*pdata*/, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,Dim> gpos, dpos, delta;
    //    unsigned int d;
    // find nearest grid point for particle position, store in NDIndex obj
    NDIndex<Dim> ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    ERRORMSG("IntSUDS::gather: not implemented for Dim>3!!"<<endl);
    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& /*pdata*/, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh,
              NDIndex<Dim>& ngp, Vektor<PT,Dim>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,Dim> gpos, delta;
    //    unsigned int d;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    ERRORMSG("IntSUDS::gather: not implemented for Dim>3!!"<<endl);
    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& /*pdata*/, const Field<FT,Dim,M,C>& f,
	      const NDIndex<Dim>& ngp, const Vektor<PT,Dim>&) {
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,Dim> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    ERRORMSG("IntSUDS::gather: not implemented for Dim>3!!"<<endl);
    return;
  }

};


template <>
class IntSUDSImpl<1U> : public Interpolator {

public:
  // constructor/destructor
  IntSUDSImpl() {}
  ~IntSUDSImpl() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,1U,M,C>& f,
	       const Vektor<PT,1U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,1U> gpos, dpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    NDIndex<1U> ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,1U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1) -= 0.5 * dpos(0) * pdata;
    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,1U,M,C>& f,
	       const Vektor<PT,1U>& ppos, const M& mesh,
               NDIndex<1U>& ngp, Vektor<PT,1U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,1U> gpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,1U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1) -= 0.5 * dpos(0) * pdata;
    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,1U,M,C>& f,
	       const NDIndex<1U>& ngp, const Vektor<PT,1U>& dpos) {
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,1U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1) -= 0.5 * dpos(0) * pdata;
    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,1U,M,C>& f,
	      const Vektor<PT,1U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,1U> gpos, dpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    NDIndex<1U> ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,1U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1) - fiter.offset(-1));
    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,1U,M,C>& f,
	      const Vektor<PT,1U>& ppos, const M& mesh,
              NDIndex<1U>& ngp, Vektor<PT,1U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,1U> gpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,1U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1) - fiter.offset(-1));
    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,1U,M,C>& f,
	      const NDIndex<1U>& ngp, const Vektor<PT,1U>& dpos) {
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,1U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1) - fiter.offset(-1));
    return;
  }

};


template <>
class IntSUDSImpl<2U> : public Interpolator {

public:
  // constructor/destructor
  IntSUDSImpl() {}
  ~IntSUDSImpl() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,2U,M,C>& f,
	       const Vektor<PT,2U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,2U> gpos, dpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    NDIndex<2U> ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,2U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1,0) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1,0) -= 0.5 * dpos(0) * pdata;
    fiter.offset(0, 1) += 0.5 * dpos(1) * pdata;
    fiter.offset(0,-1) -= 0.5 * dpos(1) * pdata;
    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,2U,M,C>& f,
	       const Vektor<PT,2U>& ppos, const M& mesh,
               NDIndex<2U>& ngp, Vektor<PT,2U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,2U> gpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,2U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1,0) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1,0) -= 0.5 * dpos(0) * pdata;
    fiter.offset(0, 1) += 0.5 * dpos(1) * pdata;
    fiter.offset(0,-1) -= 0.5 * dpos(1) * pdata;
    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,2U,M,C>& f,
	       const NDIndex<2U>& ngp, const Vektor<PT,2U>& dpos) {
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,2U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1,0) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1,0) -= 0.5 * dpos(0) * pdata;
    fiter.offset(0, 1) += 0.5 * dpos(1) * pdata;
    fiter.offset(0,-1) -= 0.5 * dpos(1) * pdata;
    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,2U,M,C>& f,
	      const Vektor<PT,2U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,2U> gpos, dpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    NDIndex<2U> ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,2U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1,0) - fiter.offset(-1,0)) +
            0.5 * dpos(1) * (fiter.offset(0,1) - fiter.offset(0,-1));
    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,2U,M,C>& f,
	      const Vektor<PT,2U>& ppos, const M& mesh,
              NDIndex<2U>& ngp, Vektor<PT,2U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,2U> gpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,2U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1,0) - fiter.offset(-1,0)) +
            0.5 * dpos(1) * (fiter.offset(0,1) - fiter.offset(0,-1));
    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,2U,M,C>& f,
	      const NDIndex<2U>& ngp, const Vektor<PT,2U>& dpos) {
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,2U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1,0) - fiter.offset(-1,0)) +
            0.5 * dpos(1) * (fiter.offset(0,1) - fiter.offset(0,-1));
    return;
  }

};


template <>
class IntSUDSImpl<3U> : public Interpolator {

public:
  // constructor/destructor
  IntSUDSImpl() {}
  ~IntSUDSImpl() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,3U,M,C>& f,
	       const Vektor<PT,3U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,3U> gpos, dpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    NDIndex<3U> ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,3U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1,0,0) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1,0,0) -= 0.5 * dpos(0) * pdata;
    fiter.offset(0, 1,0) += 0.5 * dpos(1) * pdata;
    fiter.offset(0,-1,0) -= 0.5 * dpos(1) * pdata;
    fiter.offset(0,0, 1) += 0.5 * dpos(2) * pdata;
    fiter.offset(0,0,-1) -= 0.5 * dpos(2) * pdata;
    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,3U,M,C>& f,
	       const Vektor<PT,3U>& ppos, const M& mesh,
               NDIndex<3U>& ngp, Vektor<PT,3U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,3U> gpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,3U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1,0,0) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1,0,0) -= 0.5 * dpos(0) * pdata;
    fiter.offset(0, 1,0) += 0.5 * dpos(1) * pdata;
    fiter.offset(0,-1,0) -= 0.5 * dpos(1) * pdata;
    fiter.offset(0,0, 1) += 0.5 * dpos(2) * pdata;
    fiter.offset(0,0,-1) -= 0.5 * dpos(2) * pdata;
    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,3U,M,C>& f,
	       const NDIndex<3U>& ngp, const Vektor<PT,3U>& dpos) {
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,3U> fiter = getFieldIter(f,ngp);
    // accumulate into local elements
    *fiter += pdata;
    fiter.offset( 1,0,0) += 0.5 * dpos(0) * pdata;
    fiter.offset(-1,0,0) -= 0.5 * dpos(0) * pdata;
    fiter.offset(0, 1,0) += 0.5 * dpos(1) * pdata;
    fiter.offset(0,-1,0) -= 0.5 * dpos(1) * pdata;
    fiter.offset(0,0, 1) += 0.5 * dpos(2) * pdata;
    fiter.offset(0,0,-1) -= 0.5 * dpos(2) * pdata;
    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,3U,M,C>& f,
	      const Vektor<PT,3U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,3U> gpos, dpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    NDIndex<3U> ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,3U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1,0,0) - fiter.offset(-1,0,0)) +
            0.5 * dpos(1) * (fiter.offset(0,1,0) - fiter.offset(0,-1,0)) +
            0.5 * dpos(2) * (fiter.offset(0,0,1) - fiter.offset(0,0,-1));
    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,3U,M,C>& f,
	      const Vektor<PT,3U>& ppos, const M& mesh,
              NDIndex<3U>& ngp, Vektor<PT,3U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,3U> gpos, delta;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // normalize dpos by mesh spacing
    dpos /= delta;
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,3U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1,0,0) - fiter.offset(-1,0,0)) +
            0.5 * dpos(1) * (fiter.offset(0,1,0) - fiter.offset(0,-1,0)) +
            0.5 * dpos(2) * (fiter.offset(0,0,1) - fiter.offset(0,0,-1));
    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,3U,M,C>& f,
	      const NDIndex<3U>& ngp, const Vektor<PT,3U>& dpos) {
    // Try to find ngp in local fields and get iterator
    CompressedBrickIterator<FT,3U> fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = (*fiter) +
            0.5 * dpos(0) * (fiter.offset(1,0,0) - fiter.offset(-1,0,0)) +
            0.5 * dpos(1) * (fiter.offset(0,1,0) - fiter.offset(0,-1,0)) +
            0.5 * dpos(2) * (fiter.offset(0,0,1) - fiter.offset(0,0,-1));
    return;
  }

};


// IntSUDS class -- what the user sees
class IntSUDS {

public:
  // constructor/destructor
  IntSUDS() {}
  ~IntSUDS() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh) {
    IntSUDSImpl<Dim>::scatter(pdata,f,ppos,mesh);
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh,
               CacheData1<PT,Dim>& cache) {
    IntSUDSImpl<Dim>::scatter(pdata,f,ppos,mesh,cache.Index_m,cache.Delta_m);
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const CacheData1<PT,Dim>& cache) {
    IntSUDSImpl<Dim>::scatter(pdata,f,cache.Index_m,cache.Delta_m);
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh) {
    IntSUDSImpl<Dim>::gather(pdata,f,ppos,mesh);
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh,
              CacheData1<PT,Dim>& cache) {
    IntSUDSImpl<Dim>::gather(pdata,f,ppos,mesh,cache.Index_m,cache.Delta_m);
  }

  // gather particle data from Field using cached mesh information
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const CacheData1<PT,Dim>& cache) {
    IntSUDSImpl<Dim>::gather(pdata,f,cache.Index_m,cache.Delta_m);
  }

};

#endif // INT_SUDS_H

/***************************************************************************
 * $RCSfile: IntSUDS.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: IntSUDS.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/

