// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_BCONDS_H
#define PARTICLE_BCONDS_H

/***************************************************************************
 * ParticleBConds is a container for a set of particle boundary condition
 * functions.  Boundary conditions for particles are not objects, but just
 * functions which map a position X -> X', given the minimum and maximum
 * values of the spatial domain.
 ***************************************************************************/

// include files
#include "Region/NDRegion.h"
#include "Index/NDIndex.h"



//////////////////////////////////////////////////////////////////////
// particle boundary condition functions ...

// null BC; value is not changed
template<class T>
inline T ParticleNoBCond(const T t, const T /* minval */, const T /* maxval */) {
  return t;
}

// periodic BC; values wrap around at endpoints of the interval
template<class T>
inline T ParticlePeriodicBCond(const T t, const T minval, const T maxval) {
  if (t < minval)
    return (maxval - (minval - t));
  else if (t >= maxval)
    return (minval + (t - maxval));
  else
    return t;
}

// reflective BC; values bounce back from endpoints
template<class T>
inline T ParticleReflectiveBCond(const T t, const T minval, const T maxval) {
  if (t < minval)
    return (minval + (minval - t));
  else if (t >= maxval)
    return (maxval - (t - maxval));
  else
    return t;
}

// sink BC; particles stick to the selected face
template<class T>
inline T ParticleSinkBCond(const T t, const T minval, const T maxval) {
  if (t < minval)
    return minval;
  else if (t >= maxval)
    return maxval;
  else
    return t;
}


//////////////////////////////////////////////////////////////////////
// general container for a set of particle boundary conditions
template<class T, unsigned Dim>
class ParticleBConds {

public:
  // typedef for a pointer to boundary condition function
  typedef T (*ParticleBCond)(const T, const T, const T);

public:
  // constructor: initialize all BC's to null ones, which do not change
  // the value of the data any
  ParticleBConds() {
    for (int d=(2*Dim - 1); d >= 0; --d)
      BCList[d] = ParticleNoBCond;
  }

  // operator= to copy values from another container
  ParticleBConds<T,Dim>& operator=(const ParticleBConds<T,Dim>& pbc) {
    for (int d=(2*Dim - 1); d >= 0; --d)
      BCList[d] = pbc.BCList[d];
    return *this;
  }

  // operator[] to get value of Nth boundary condition
  ParticleBCond& operator[](unsigned d) { return BCList[d]; }

  // for the given value in the given dimension over the given NDRegion,
  // apply the proper BC and return back the new value
  T apply(const T t, const unsigned d, const NDRegion<T,Dim>& nr) const {
    return apply(t, d, nr[d].min(), nr[d].max());
  }

  // for the given value in the given dimension over the given NDIndex,
  // apply the proper BC and return back the new value.  The extra +1
  // added to the max value is due to the assumption of a cell-centered
  // field.
  T apply(const T t, const unsigned d, const NDIndex<Dim>& ni) const {
    return apply(t, d, ni[d].first(), ni[d].last() + 1);
  }

  // a different version of apply, where the user just specifies the min
  // and max values of the given dimension
  T apply(const T t, const unsigned d, const T m1, const T m2) const {
    if (t < m1)
      return (BCList[d+d])(t, m1, m2);
    else if (t >= m2)		           // here we take into account that
      return (BCList[d+d+1])(t, m1, m2);   // Region's store intervals [A,B)
    else
      return t;
  }

private:
  // array storing the function pointers
  ParticleBCond BCList[2*Dim];

};

#endif // PARTICLE_BCONDS_H

/***************************************************************************
 * $RCSfile: ParticleBConds.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: ParticleBConds.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
