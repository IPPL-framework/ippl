// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "VRB.h"
#include "Utility/PAssert.h"

#ifdef __VRB_DIAGNOSTIC__
#include <cstdio>
#include <cstdlib>
#endif

//
// A local function we use for the recursive bisection.
//

static void
recurseCoordinateVRB(int dim,
                     const int *strides, const int* sizes, const int *offsets,
                     int nprocs, int firstProc, int *procs );

//////////////////////////////////////////////////////////////////////

//
// vnodeRecurseiveBisection
//
// This is the user visible function for decomposing along coordinate
// directions.
//
// This function just sets up the arguments for the recursive
// algorithm.  That algorithm works by subdividing blocks of
// vnodes and blocks of processors.
//
// The blocks are identified with the offsets of the lower left
// corner and the sizes of the block in each dimension.
//
// The processors are identified by a continuous range from
// a firstProc and a number of procs nprocs.
//
// In addition to the input arguments, it calculates for the recursion:
//
// strides: The strides to go from vnode to its neighbors in N dimensions.
// offsets: The lower left corner of the block of vnodes.
//

void
vnodeRecursiveBisection(int dim, const int *sizes, int nprocs, int *procs)
{
  int i;

  // Initialize the lower left corner of the vnodes to the origin.
  int *offsets = new int[dim];
  for (i=0; i<dim; ++i)
    offsets[i] = 0;

  // Initialize the strides with C semantics (last index varies fastest).
  int *strides = new int[dim];
  strides[dim-1] = 1;
  for (i=dim-2; i>=0; --i)
    strides[i] = strides[i+1]*sizes[i+1];

  // Dive into the recursion.
  recurseCoordinateVRB(dim,strides,sizes,offsets,nprocs,0,procs);

  // Clean up after ourselves.
  delete [] offsets;
  delete [] strides;
}

//////////////////////////////////////////////////////////////////////

//
// assign
//
// Given a description of a block of vnodes, fill the proc array
// with a processor number x.
//
// The block of vnodes is described by:
// dim:     The dimension of space.
// strides: The strides to find neighbors of a vnode.
// offsets: The lower left corner of the block of vnodes
// sizes  : the size of the block in each dimension.
//

static void
assign(int dim, const int *strides, const int *sizes, const int *offsets,
       int x, int *procs)
{
  // Make sure the input is sensible.
  PAssert_GT(dim, 0);
  PAssert(sizes);
  PAssert(offsets);
  PAssert(procs);

  int i;

#ifdef __VRB__DIAGNOSTIC__
  printf("---------- assign ----------\n");
  printf("dim=%d, sizes=",dim);
  for (i=0; i<dim; ++i)
    printf(" %d",sizes[i]);
  printf(", offsets=");
  for (i=0; i<dim; ++i)
    printf(" %d",offsets[i]);
  printf(", strides=");
  for (i=0; i<dim; ++i)
    printf(" %d",strides[i]);
  printf(", procs=%lx\n",procs);
  printf("----------------------------\n");
#endif

  // Termination condition: one dimension.
  if ( dim==1 )
    // Fill the one row with x.
    for (i=0; i<sizes[0]; ++i)
      procs[offsets[0]+i] = x;

  // Otherwise, loop over the outermost dimension and recurse.
  else
    {
      // For each slab, fill it.
      for (i=0; i<sizes[0]; ++i)
        assign(dim-1,strides+1,sizes+1,offsets+1,x,procs+(offsets[0]+i)*strides[0]);
    }
}

//////////////////////////////////////////////////////////////////////

//
// RecurseCoordinateVRB
//
// Perform the recursion, finding the procs for each vnode.
//
// Inputs:
// dim     : The dimension of the breakdown.
// strides : The strides between the slabs of the proc array.
// sizes   : The size of the block of vnodes.
// offsets : The lower left corner of the block of vnodes.
// nprocs  : The number of processors to put these vnodes on.
// firstProc : The first proc we are putting these vnodes on.
//
// Output:
// procs: Array to store the processor for each vnode.
//

static void
recurseCoordinateVRB(int dim,
                     const int* strides, const int* sizes, const int *offsets,
                     int nprocs, int firstProc, int *procs )
{
  // Make sure the input is sensible.
  PAssert_GT(dim, 0);
  PAssert(sizes);
  PAssert_GT(nprocs, 0);
  PAssert(procs);
  PAssert_GE(firstProc, 0);
  for (int i=0; i<dim; ++i)
    {
      PAssert_GT(sizes[i], 0);
      PAssert_GE(offsets[i], 0);
    }

#ifdef __VRB_DIAGNOSTIC__
  printf("---------- RecurseCoordinateVRB ----------\n");
  printf("dim= %d, sizes=",dim);
  for (i=0; i<dim; ++i)
    printf(" %d",sizes[i]);
  printf(", offsets=");
  for (i=0; i<dim; ++i)
    printf(" %d",offsets[i]);
  printf(", nprocs= %d, firstProc= %d\n",nprocs,firstProc);
  printf("------------------------------------------\n");
#endif

  // If we just have one proc, all the vnodes are on this proc.
  if ( nprocs == 1 )
    {
      // Fill the hypercube defined by sizes,offsets with
      // the value firstProc.
      assign(dim,strides,sizes,offsets,firstProc,procs);
    }

  // If there is more than one processor left,
  // recurse by splitting the procs into two groups and
  // the work into two groups, and allocating work to procs.
  else
    {
      int d;

      // Calculate the total number of vnodes.
      int totalVnodes = sizes[0];
      for (d=1; d<dim; ++d)
        totalVnodes *= sizes[d];
      PAssert_GE(totalVnodes, nprocs);

      // Find the number of processors on each side.
      int leftProcs = nprocs/2;
      int rightProcs = nprocs-leftProcs;

      // Decide which dimension to split.
      // Just loop over the dimensions and pick the biggest.
      int splitDim = 0;
      for (d=1; d<dim; ++d)
        if ( sizes[d] > sizes[splitDim] )
          splitDim = d;

      // Get the number of vnodes in that dimension.
      int splitSize = sizes[splitDim];

      // Find where along that dimension to split.
      // Balance the work between the two procs as well as possible.

      // The number of vnodes on the left.
      // Start with none on the left.
      int leftVnodes = 0;

      // The degree to which things are out of balance.
      double outOfBalance = splitSize/(double)rightProcs;

      // The number of vnodes in a hyperslab perpendicular
      // to the split direction.
      int crossSize = 1;
      for (d=0; d<dim; ++d)
	if ( d != splitDim )
	  crossSize *= sizes[d];

      // Consider all possible split locations and pick the best.
      for (int l=1; l<splitSize; ++l)
        {
          // How far out of balance is this?
	  int r = splitSize - l;
          double b = l/(double)leftProcs - r/(double)rightProcs;

          // Get the absolute value of the unbalance.
          if ( b<0 )
            b=-b;

          // Compare to the best so far.
	  // If is better balance and we have at least as many
	  // procs on each side of the divide as we have vnodes,
	  // then keep this split.
          if ( (b < outOfBalance) &&
	       (l*crossSize>=leftProcs) &&
	       (r*crossSize>=rightProcs) )
            {
              // It is better, keep it.
              leftVnodes = l;
              outOfBalance = b;
            }
        }

      // If we couldn't find a good split, die.
      PAssert_GT(leftVnodes, 0);

      // We now know what dimension to split on, and where in
      // that dimension to split.  Recurse.

      // Make a copy of the sizes array.
      int *newSizes = new int[dim];
      for (d=0; d<dim; ++d)
        newSizes[d] = sizes[d];

      // Make a copy of the offsets array.
      int *newOffsets = new int[dim];
      for (d=0; d<dim; ++d)
        newOffsets[d] = offsets[d];

      // Get the sizes for the left.
      newSizes[splitDim] = leftVnodes;

      // Recurse for the left.
      recurseCoordinateVRB(dim,strides,newSizes,newOffsets,leftProcs,firstProc,procs);

      // Get the sizes and offsets for the right.
      newSizes[splitDim] = splitSize - leftVnodes;
      newOffsets[splitDim] += leftVnodes;

      // Recurse for the right.
      recurseCoordinateVRB(dim,strides,newSizes,newOffsets,rightProcs,firstProc+leftProcs,procs);

      // Delete the memory.
      delete [] newSizes;
      delete [] newOffsets;
    }
}

//////////////////////////////////////////////////////////////////////

#ifdef __VRB_DIAGNOSTIC__

//
// print out a hypercube of proc data.
//

static void
print(int dim, const int *sizes, const int *procs)
{
  if ( dim == 1 )
    {
      for ( int i=0; i<*sizes; ++i)
        printf("%4d",procs[i]);
    }
  else
    {
      int skip = 1;
      int i;
      for (i=1; i<dim; ++i)
        skip *= sizes[i];
      for (i=0; i<sizes[0]; ++i)
        {
          print(dim-1,sizes+1,procs+skip*i);
          printf("\n");
        }
      printf("\n");
    }
}

//////////////////////////////////////////////////////////////////////

int
main(int argc, char *argv[])
{
  // The number of dimensions is the number of args to the program.
  int dim = argc-2;
  PAssert_GT(dim, 0);

  // Get the number of procs.
  int nprocs = atoi(argv[1]);
  PAssert_GT(nprocs, 0);

  // Get the size of each dimension.
  int *sizes = new int[dim];
  int totalVnodes = 1;
  for ( int d = 0; d<dim; ++d )
    {
      sizes[d] = atoi(argv[d+2]);
      totalVnodes *= sizes[d];
    }
  int *procs = new int[totalVnodes];

  // Do it.
  vnodeRecursiveBisection(dim,sizes,nprocs,procs);

  // Print it out.
  print(dim,sizes,procs);

  return 0;
}

#endif

/***************************************************************************
 * $RCSfile: VRB.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: VRB.cpp,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $
 ***************************************************************************/