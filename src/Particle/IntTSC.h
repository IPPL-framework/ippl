// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INT_TSC_H
#define INT_TSC_H

/* IntTSC.h -- Definition of simple class to perform cloud-in-cell
   interpolation of data for a single particle to or from a IPPL Field.
   This interpolation scheme is also referred to as linear interpolation,
   area-weighting (in 2D), or volume-weighting (in 3D).                    */

// include files
#include "Particle/Interpolator.h"
#include "Field/Field.h"


// forward declaration
class IntTSC;

// specialization of InterpolatorTraits
template <class T, unsigned Dim>
struct InterpolatorTraits<T,Dim,IntTSC> {
  typedef CacheDataTSC<T,Dim> Cache_t;
};



// IntTSCImpl class definition
template <unsigned Dim>
class IntTSCImpl : public Interpolator {

public:
  // constructor/destructor
  IntTSCImpl() {}
  ~IntTSCImpl() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& /*pdata*/, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,Dim> gpos, dpos, delta;
    NDIndex<Dim> ngp;
    CompressedBrickIterator<FT,Dim> fiter;
    //BENI: offset not needed, since NGP is used as center
	//int lgpoff[Dim];
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // Now we find the offset from ngp to next-lowest grip point (lgp)
    /* //BENI: we take nearest grid point as it is as the center. No modification needed.
	for (d=0; d<Dim; ++d) {
      if (gpos(d) > ppos(d)) {
        lgpoff[d] = -1;                // save the offset
        gpos(d) = gpos(d) - delta(d);  // adjust gpos to lgp position
      }
      else {
        lgpoff[d] = 0;                 // save the offset
      }
    }
    // adjust position of Field iterator to lgp position
    fiter.moveBy(lgpoff);
	*/
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into local elements
    ERRORMSG("IntTSC::scatter: not implemented for Dim>3!!"<<endl);
    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& /*pdata*/, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh,
               NDIndex<Dim>& ngp, int [Dim], Vektor<PT,Dim>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,Dim> gpos, delta;
    CompressedBrickIterator<FT,Dim> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // Now we find the offset from ngp to next-lowest grip point (lgp)
	/*
    for (d=0; d<Dim; ++d) {
      if (gpos(d) > ppos(d)) {
        lgpoff[d] = -1;                // save the offset
        gpos(d) = gpos(d) - delta(d);  // adjust gpos to lgp position
      }
      else {
        lgpoff[d] = 0;                 // save the offset
      }
    }
    // adjust position of Field iterator to lgp position
    fiter.moveBy(lgpoff);
    // get distance from ppos to gpos
	*/
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into local elements
    ERRORMSG("IntTSC::scatter: not implemented for Dim>3!!"<<endl);
    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& /*pdata**/, Field<FT,Dim,M,C>& f,
	       const NDIndex<Dim>& ngp, const int /*lgpoff*/ [Dim],
               const Vektor<PT,Dim>& /*dpos*/) {
    CompressedBrickIterator<FT,Dim> fiter;
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // adjust position of Field iterator to lgp position
    //fiter.moveBy(lgpoff);
    // accumulate into local elements
    ERRORMSG("IntTSC::scatter: not implemented for Dim>3!!"<<endl);
    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& /*pdata*/, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,Dim> gpos, dpos, delta;
    NDIndex<Dim> ngp;
    CompressedBrickIterator<FT,Dim> fiter;
    int lgpoff[Dim];
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
	/*
    // Now we find the offset from ngp to next-lowest grip point (lgp)
    for (d=0; d<Dim; ++d) {
      if (gpos(d) > ppos(d)) {
        lgpoff[d] = -1;                // save the offset
        gpos(d) = gpos(d) - delta(d);  // adjust gpos to lgp position
      }
      else {
        lgpoff[d] = 0;                 // save the offset
      }
    }
    // adjust position of Field iterator to lgp position
    fiter.moveBy(lgpoff);
	*/
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into particle attrib
    ERRORMSG("IntTSC::gather: not implemented for Dim>3!!"<<endl);
    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& /*pdata*/, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh,
              NDIndex<Dim>& ngp, int /*lgpoff*/[Dim], Vektor<PT,Dim>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,Dim> gpos, delta;
    CompressedBrickIterator<FT,Dim> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
	/*
    // Now we find the offset from ngp to next-lowest grip point (lgp)
    for (d=0; d<Dim; ++d) {
      if (gpos(d) > ppos(d)) {
        lgpoff[d] = -1;                // save the offset
        gpos(d) = gpos(d) - delta(d);  // adjust gpos to lgp position
      }
      else {
        lgpoff[d] = 0;                 // save the offset
      }
    }
    // adjust position of Field iterator to lgp position
    fiter.moveBy(lgpoff);
	*/
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into particle attrib
    ERRORMSG("IntTSC::gather: not implemented for Dim>3!!"<<endl);
    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& /*pdata*/, const Field<FT,Dim,M,C>& f,
	      const NDIndex<Dim>& ngp, const int /*lgpoff*/[Dim],
              const Vektor<PT,Dim>& /*dpos*/) {
    CompressedBrickIterator<FT,Dim> fiter;
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // adjust position of Field iterator to lgp position
    //fiter.moveBy(lgpoff);
    // accumulate into particle attrib
    ERRORMSG("IntTSC::gather: not implemented for Dim>3!!"<<endl);
    return;
  }

};


template <>
class IntTSCImpl<1U> : public Interpolator {

public:
  // constructor/destructor
  IntTSCImpl() {}
  ~IntTSCImpl() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,1U,M,C>& f,
	       const Vektor<PT,1U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,1U> gpos, dpos, delta;
    NDIndex<1U> ngp;
    CompressedBrickIterator<FT,1U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into local elements
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        fiter.offset(p0) += W(p0,0) * pdata;
    }
    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,1U,M,C>& f,
	       const Vektor<PT,1U>& ppos, const M& mesh,
               NDIndex<1U>& ngp, int /*lgpoff*/ [1U], Vektor<PT,1U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,1U> gpos, delta;
    CompressedBrickIterator<FT,1U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into local elements
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        fiter.offset(p0) += W(p0,0) * pdata;
    }
    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,1U,M,C>& f,
	       const NDIndex<1U>& ngp, const int /*lgpoff*/ [1U],
               const Vektor<PT,1U>& dpos) {
    CompressedBrickIterator<FT,1U> fiter;
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // adjust position of Field iterator to lgp position
    //fiter.moveBy(lgpoff);
    // accumulate into local elements
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        fiter.offset(p0) += W(p0,0) * pdata;
    }

    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,1U,M,C>& f,
	      const Vektor<PT,1U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,1U> gpos, dpos, delta;
    NDIndex<1U> ngp;
    CompressedBrickIterator<FT,1U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into particle attrib
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    pdata = 0;
    for (int p0 = -1; p0 <= 1; ++p0) {
        pdata += W(p0,0) * fiter.offset(p0);
    }
    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,1U,M,C>& f,
	      const Vektor<PT,1U>& ppos, const M& mesh,
              NDIndex<1U>& ngp, int /*lgpoff*/ [1U], Vektor<PT,1U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,1U> gpos, delta;
    CompressedBrickIterator<FT,1U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into particle attrib
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    pdata = 0;
    for (int p0 = -1; p0 <= 1; ++p0) {
        pdata += W(p0,0) * fiter.offset(p0);
    }

    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,1U,M,C>& f,
	      const NDIndex<1U>& ngp, const int /*lgpoff*/[1U],
              const Vektor<PT,1U>& dpos) {
    CompressedBrickIterator<FT,1U> fiter;
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
     // accumulate into particle attrib
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    pdata = 0;
    for (int p0 = -1; p0 <= 1; ++p0) {
        pdata += W(p0,0) * fiter.offset(p0);
    }

    return;
  }

};


template <>
class IntTSCImpl<2U> : public Interpolator {

public:
  // constructor/destructor
  IntTSCImpl() {}
  ~IntTSCImpl() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,2U,M,C>& f,
	       const Vektor<PT,2U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,2U> gpos, dpos, delta;
    NDIndex<2U> ngp;
    CompressedBrickIterator<FT,2U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into local elements
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            fiter.offset(p0,p1) += W(p0,0) * W(p1,1) * pdata;
        }
    }
    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,2U,M,C>& f,
	       const Vektor<PT,2U>& ppos, const M& mesh,
               NDIndex<2U>& ngp, int /*lgpoff*/[2U], Vektor<PT,2U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,2U> gpos, delta;
    CompressedBrickIterator<FT,2U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into local elements
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            fiter.offset(p0,p1) += W(p0,0) * W(p1,1) * pdata;
        }
    }

    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,2U,M,C>& f,
	       const NDIndex<2U>& ngp, const int /*lpgoff*/[2U],
               const Vektor<PT,2U>& dpos) {
    CompressedBrickIterator<FT,2U> fiter;
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // adjust position of Field iterator to lgp position
   // fiter.moveBy(lgpoff);
    // accumulate into local elements
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            fiter.offset(p0,p1) += W(p0,0) * W(p1,1) * pdata;
        }
    }

    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,2U,M,C>& f,
	      const Vektor<PT,2U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,2U> gpos, dpos, delta;
    NDIndex<2U> ngp;
    CompressedBrickIterator<FT,2U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into particle attrib
    pdata = 0;
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            pdata += W(p0,0) * W(p1,1) * fiter.offset(p0,p1);
        }
    }
    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,2U,M,C>& f,
	      const Vektor<PT,2U>& ppos, const M& mesh,
              NDIndex<2U>& ngp, int /*lgpoff*/[2U], Vektor<PT,2U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,2U> gpos, delta;
    CompressedBrickIterator<FT,2U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into particle attrib
    pdata = 0;
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            pdata += W(p0,0) * W(p1,1) * fiter.offset(p0,p1);
        }
    }
    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,2U,M,C>& f,
	      const NDIndex<2U>& ngp, const int /*lgpoff*/[2U],
              const Vektor<PT,2U>& dpos) {
    CompressedBrickIterator<FT,2U> fiter;
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // accumulate into particle attrib
    pdata = 0;
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            pdata += W(p0,0) * W(p1,1) * fiter.offset(p0,p1);
        }
    }
    return;
  }

};


template <>
class IntTSCImpl<3U> : public Interpolator {

public:
  // constructor/destructor
  IntTSCImpl() {}
  ~IntTSCImpl() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,3U,M,C>& f,
	       const Vektor<PT,3U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,3U> gpos, dpos, delta;
    NDIndex<3U> ngp;
    CompressedBrickIterator<FT,3U> fiter;
    //int lgpoff[3U];
    //unsigned int d;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into local elements
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            for (int p2 = -1; p2 <= 1; ++p2) {
                fiter.offset(p0,p1,p2) += W(p0,0) * W(p1,1) * W(p2,2) * pdata;
            }
        }
    }
    return;
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,3U,M,C>& f,
	       const Vektor<PT,3U>& ppos, const M& mesh,
               NDIndex<3U>& ngp, int /*lgpoff*/[3U], Vektor<PT,3U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,3U> gpos, delta;
    CompressedBrickIterator<FT,3U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into local elements
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            for (int p2 = -1; p2 <= 1; ++p2) {
                fiter.offset(p0,p1,p2) += W(p0,0) * W(p1,1) * W(p2,2) * pdata;
            }
        }
    }
    return;
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,3U,M,C>& f,
	       const NDIndex<3U>& ngp, const int /*lgpoff*/[3U],
               const Vektor<PT,3U>& dpos) {
    CompressedBrickIterator<FT,3U> fiter;
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);

    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            for (int p2 = -1; p2 <= 1; ++p2) {
                fiter.offset(p0,p1,p2) += W(p0,0) * W(p1,1) * W(p2,2) * pdata;
            }
        }
    }
    return;
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,3U,M,C>& f,
	      const Vektor<PT,3U>& ppos, const M& mesh) {
    CenteringTag<C> ctag;
    Vektor<PT,3U> gpos, dpos, delta;
    NDIndex<3U> ngp;
    CompressedBrickIterator<FT,3U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into particle attrib
    pdata = 0;
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            for (int p2 = -1; p2 <= 1; ++p2) {
                pdata += W(p0,0) * W(p1,1) * W(p2,2) * fiter.offset(p0,p1,p2);
            }
        }
    }
    return;
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,3U,M,C>& f,
	      const Vektor<PT,3U>& ppos, const M& mesh,
              NDIndex<3U>& ngp, int /*lgpoff*/[3U], Vektor<PT,3U>& dpos) {
    CenteringTag<C> ctag;
    Vektor<PT,3U> gpos, delta;
    CompressedBrickIterator<FT,3U> fiter;
    // find nearest grid point for particle position, store in NDIndex obj
    ngp = FindNGP(mesh, ppos, ctag);
    // get position of ngp
    FindPos(gpos, mesh, ngp, ctag);
    // get mesh spacings
    FindDelta(delta, mesh, ngp, ctag);
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // get distance from ppos to gpos
    dpos = ppos - gpos;
    // normalize dpos by mesh spacing
    dpos /= delta;
    // accumulate into particle attrib
    pdata = 0;
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            for (int p2 = -1; p2 <= 1; ++p2) {
                pdata += W(p0,0) * W(p1,1) * W(p2,2) * fiter.offset(p0,p1,p2);
            }
        }
    }
    return;
  }

  // gather particle data from Field using cached mesh information
  template <class FT, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,3U,M,C>& f,
	      const NDIndex<3U>& ngp, const int /*lgpoff*/[3U],
              const Vektor<PT,3U>& dpos) {
    CompressedBrickIterator<FT,3U> fiter;
    // Try to find ngp in local fields and get iterator
    fiter = getFieldIter(f,ngp);
    // adjust position of Field iterator to lgp position
    //fiter.moveBy(lgpoff);
    // accumulate into particle attrib
    pdata = 0;
    auto W = [dpos](int p, unsigned i) {
        if      (p==-1) return .125 * (1 - 4 * dpos(i) + 4 * dpos(i) * dpos(i));
        else if (p==0)  return .25  * (3 - 4 * dpos(i) * dpos(i));
        else if (p==+1) return .125 * (1 + 4 * dpos(i) + 4 * dpos(i) * dpos(i)); };

    for (int p0 = -1; p0 <= 1; ++p0) {
        for (int p1 = -1; p1 <= 1; ++p1) {
            for (int p2 = -1; p2 <= 1; ++p2) {
                pdata += W(p0,0) * W(p1,1) * W(p2,2) * fiter.offset(p0,p1,p2);
            }
        }
    }
    return;
  }

};


// IntTSC class -- what the user sees
class IntTSC {

public:
  // constructor/destructor
  IntTSC() {}
  ~IntTSC() {}

  // gather/scatter functions

  // scatter particle data into Field using particle position and mesh
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh) {
    IntTSCImpl<Dim>::scatter(pdata,f,ppos,mesh);
  }

  // scatter particle data into Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const Vektor<PT,Dim>& ppos, const M& mesh,
               CacheDataTSC<PT,Dim>& cache) {
    IntTSCImpl<Dim>::scatter(pdata,f,ppos,mesh,cache.Index_m,
                             cache.Offset_m,cache.Delta_m);
  }

  // scatter particle data into Field using cached mesh information
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void scatter(const FT& pdata, Field<FT,Dim,M,C>& f,
	       const CacheDataTSC<PT,Dim>& cache) {
    IntTSCImpl<Dim>::scatter(pdata,f,cache.Index_m,
                             cache.Offset_m,cache.Delta_m);
  }

  // gather particle data from Field using particle position and mesh
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh) {
    IntTSCImpl<Dim>::gather(pdata,f,ppos,mesh);
  }

  // gather particle data from Field using particle position and mesh
  // and cache mesh information for reuse
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const Vektor<PT,Dim>& ppos, const M& mesh,
              CacheDataTSC<PT,Dim>& cache) {
    IntTSCImpl<Dim>::gather(pdata,f,ppos,mesh,cache.Index_m,
                            cache.Offset_m,cache.Delta_m);
  }

  // gather particle data from Field using cached mesh information
  template <class FT, unsigned Dim, class M, class C, class PT>
  static
  void gather(FT& pdata, const Field<FT,Dim,M,C>& f,
	      const CacheDataTSC<PT,Dim>& cache) {
    IntTSCImpl<Dim>::gather(pdata,f,cache.Index_m,
                            cache.Offset_m,cache.Delta_m);
  }

};

#endif // INT_TSC_H

/***************************************************************************
 * $RCSfile: IntTSC.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: IntTSC.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $
 ***************************************************************************/
