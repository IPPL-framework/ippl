// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef VRB_H
#define VRB_H

//////////////////////////////////////////////////////////////////////
//
// Vnode Recursive Bisection package
//
// This package figures out how to distribute rectangular arrays of 
// vnodes onto processors with an attempt to minimize communication
// and unbalance.
//
// There one function with external linkage in this package:
//
// VnodeRecurseiveBisection(int dim, const int* sizes, int nprocs, int *vprocs);
//
// Input:
//   dim  : The number of dimensions.
//   sizes: The number of vnodes in each dimension.
//   nprocs: The number of procs to distribute them over.
// 
// Output:
//   procs: the proc number for each vnode, in the range 0..nprocs-1.
//
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//
// The problem:
//
// Given a set of vnodes in D dimensions, we need to allocate them
// to processors in a way that satisfies three criteria:
//
// 1. Layouts with the same number of vnodes in each direction should
//    be aligned independent of the size of the array in each dimension.
//    That means that the distribution is independent of whether it is 
//    for a vert or cell centered Field.
//
//    We do this by considering two things:
//
//    a. Consider each vnode to have the same weight.  That is, don't 
//       try to consider the fact that different vnodes may have slightly
//       different numbers of cells in them.
//
//    b. Each face between vnodes has the same computational cost. That is,
//       don't try to consider the fact that the faces between vnodes will
//       have different sizes if the vnodes are not square.
//
//
// 2. Minimize communication.
//
//    This means that you try to minimize the number of vnode faces 
//    that must be communicated.
//
// 3. Minimize unbalance
//
//    This means that you try to put the same number of vnodes on
//    each processor.
//
//////////////////////////////////////////////////////////////////////
//
// The Solution: Generalized Recursive Bisection
//
// The algorithm operates in a series of stages.
// In each stage you split the work to be done approximately in half
// and the processor pool approximately in half, and giving the split 
// work to the split processor pools.
//
// There are multipple ways of doing this.
// 1. You can split along coordinate directions.
// 2. You can renumber the vnodes with some space filling curve, and then
//    split along that curve.
//
// After trying both, we use number 1.
//
//////////////////////////////////////////////////////////////////////

void vnodeRecursiveBisection(int dim, const int* sizes, int nprocs, int *vprocs);

#endif // VRB_H

/***************************************************************************
 * $RCSfile: VRB.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: VRB.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
