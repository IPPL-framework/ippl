// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef BINARY_BALANCER_H
#define BINARY_BALANCER_H

//////////////////////////////////////////////////////////////////////
/*

  A fairly simple load balancer inspired by Dan Quinlan's MLB.

  It does recursive binary subdivision of a FieldLayout domain,
  restricting the cuts to coordinate directions, so as to balance the
  workload.  The "workload" is given by a Field of weights passed in.
  It decides on the cut axis by cutting the longest axis of a brick,
  and the location of that cut by balancing the weights on each side
  of the cut.  The resulting distribution has one vnode per processor.

  This is restricted to a processor number that is a power of two. 

  It performs log(P) parallel reductions.

  It does nothing fancy when deciding on the splits to try to make the
  new paritioning close to the previous.  The same set of weights will
  always give the same repartitioning, but similar sets of weights
  could result in quite different partitionings.

  There are two functions defined here:

  NDIndex<Dim>
  CalcBinaryRepartion(FieldLayout<Dim>&, BareField<double,Dim>&);

  Given a FieldLayout and a Field of weights, find the domain for this
  processor.  This does not repartition the FieldLayout, it just
  calculates the domain.  If you want to further subdivide these
  domains, just cut up what this function returns.

  void
  BinaryRepartition(FieldLayout<Dim>&, BareField<double,Dim>&);

  Just call the above function and then repartition the FieldLayout
  (and all the Fields defined on it).

 */
//////////////////////////////////////////////////////////////////////

// forward declarations
template<unsigned Dim> class FieldLayout;
template<class T, unsigned Dim> class BareField;

class BinaryRepartitionFailed {  };

// Calculate the local domain for a binary repartition.
template<unsigned Dim>
NDIndex<Dim>
CalcBinaryRepartition(FieldLayout<Dim>&, BareField<double,Dim>&);

// Calculate and apply a local domain for a binary repartition.
template<unsigned Dim>
inline void
BinaryRepartition(FieldLayout<Dim>& layout, BareField<double,Dim>& weights)
{
  layout.Repartition( CalcBinaryRepartition(layout,weights) );
}

//////////////////////////////////////////////////////////////////////

#include "FieldLayout/BinaryBalancer.hpp"

#endif // BINARY_BALANCER_H

/***************************************************************************
 * $RCSfile: BinaryBalancer.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: BinaryBalancer.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
