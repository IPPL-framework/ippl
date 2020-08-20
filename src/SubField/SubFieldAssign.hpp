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
#include "SubField/SubFieldAssign.h"
#include "SubField/SubFieldAssignDefs.h"
#include "SubField/SubBareField.h"
#include "SubField/SubFieldTraits.h"
#include "SubField/SubFieldIter.h"
#include "Field/BareField.h"
#include "Field/BrickExpression.h"
#include "Field/LField.h"
#include "Field/Assign.h"
#include "Field/AssignDefs.h"

#include "Utility/IpplInfo.h"


//////////////////////////////////////////////////////////////////////
// assign an arbitrary expression to a SubBareField, with aligned Fields
template<class T1, unsigned Dim, class S, class RHS, class OP>
void assign(SubBareField<T1,Dim,S> a, RHS b, OP op, ExprTag<true>) {

  // Get the object wrapped in the PETE expression object
  typename RHS::Wrapped& bb = b.PETE_unwrap();

  // Inform msg("SubField::assign");

  // Check to see if any of the terms on the rhs
  // are the field on the lhs.  If so we'll have to make temporaries.
  int lhs_id = a.getBareField().get_Id();
  bool both_sides = for_each(bb,SameFieldID(lhs_id),PETE_OrCombiner());

  // Check to see if the same type of indexing object is used by all
  // the Field's involved in the expression.  If any are different, we
  // must flag an error.  This also checks to make sure that all
  // indexed fields have the proper number of brackets.
  PInsist(for_each(bb,SameSubsetType(a.getSubsetType()),PETE_AndCombiner()),
	  "Sparse-Indexed expressions must use consistent indexing.");

  // Find the 'bounding box' of the LHS domain
  NDIndex<Dim> boundBox;
  a.makeNDIndex(boundBox);

  // Fill guard cells, if necessary ... this is done if the RHS domains
  // represent a stencil, and the guard cells for the corresponding fields
  // actually need to be swapped.
  //tjw3/3/99  for_each(bb, FillGCIfNecessary(boundBox), PETE_NullCombiner());

  // This weird tag is needed, because using a no-argument FillGCIFNEcessary()
  // function didn't work for some reason, at least with some compilers like
  // the pre-7.3 SGI compiler and CodeWarrior Pro4. Once the global function
  // invocation syntax FillGCIfNecessary<T,D>() is supported by all our
  // compilers (it's not there yet in the non-beta SGI compiler), should be
  // able to eliminate this FGCINTag business. --tjw 3/3/1999
  for_each(bb, FillGCIfNecessary(FGCINTag<Dim,T1>()), PETE_NullCombiner());

  // Iterators for the LHS components and LField's
  typename SubBareField<T1,Dim,S>::iterator sba    = a.begin();
  typename BareField<T1,Dim>::iterator_if   la     = sba.getLFieldIter();
  typename BareField<T1,Dim>::iterator_if   aend   = a.end().getLFieldIter();

  // Do any initial work on the expression it needs before starting.
  // E.g., for single-value subsets, this will distribute the values to nodes
  sba.initialize();
  for_each(bb, SubsetInit(), PETE_NullCombiner());

  // Loop over all the local fields of the left hand side.
  while (la != aend) {
    // Some typedefs to make the lines shorter...
    // types for the left hand side, right hand side and
    // the whole expression.
    typedef typename SubBareField<T1,Dim,S>::iterator LHS;
    typedef BrickExpression<LHS::ExprDim_u,LHS,typename RHS::Wrapped,OP> ExprT;

    // The pointer to the current lhs local field.
    // If it is on the rhs somewhere, make a copy of it, and tell the
    // subfield iterator about it.
    LField<T1,Dim> *lf = (*la).second.get();
    if ( both_sides )
      lf = new LField<T1,Dim>( *lf );

    //msg << "  Now doing LField pointer = " << &(*lf);
    //msg << ", id = " << lf->getVnode();
    //msg << ", data pointer = " << &(*(lf->getP()));
    //msg << endl;

    // Find the local domain, if any.  To do so, intersect
    // with the indexes used to see how much we will actually use.  If
    // findIntersect returns false, then we know there is nothing to do.
    // The domain over which we'll be working (and which will be used
    // in BrickExpression) is returned in local_domain.
    NDIndex<Dim> local_domain;
    if (sba.findIntersection(lf->getOwned(), local_domain)) {
      // First look and see if the arrays are sufficiently aligned
      // to do this in one shot.
      // We do this by trying to do a plugBase and seeing if it worked.
      if (for_each(bb,PlugBase<Dim>(local_domain), PETE_AndCombiner())) {

	// Check and see if the lhs can be compressed.
        if (sba.DomainCompressed() &&
            for_each(bb, DomainCompressed(), PETE_AndCombiner()) &&
	    a.getBareField().compressible() &&
	    TryCompressLHS(*lf,bb,op,local_domain)) {

	  // Compressed assign.
	  PETE_apply(op, *(lf->begin()), for_each(bb,EvalFunctor_0()));

	} else {

	  // Loop assign.
	  lf->Uncompress();
	  sba.setLFieldData(lf, local_domain);
	  ExprT(sba,bb).apply();

	  // Try to compress this LField since we did an uncompressed
	  // assignment, if the user has requested this kind of
	  // check right after computation on the LField.
	  if (IpplInfo::extraCompressChecks)
	    lf->TryCompress();
	}
      } else {
        ERRORMSG("All Fields in an expression must be aligned.  ");
        ERRORMSG("(Do you have enough guard cells?)" << endl);
        ERRORMSG("This error occurred while evaluating an SIndex-expression ");
        ERRORMSG("for an LField with domain " << lf->getOwned() << endl);
        Ippl::abort();
      }
    }

    // If we had to make a copy of the current LField,
    // swap the pointers and delete the old memory.
    if (both_sides) {
      (*la).second->swapData( *lf );
      delete lf;
    }

    // move iterators on to the next LField
    la = sba.nextLField();
    for_each(bb, SubsetNextLField(), PETE_NullCombiner());
  }

  // Fill the guard cells on the left hand side.
  a.getBareField().setDirtyFlag();
  a.getBareField().fillGuardCellsIfNotDirty();

  // Compress the LHS, if necessary.
  if (!IpplInfo::extraCompressChecks)
    a.getBareField().Compress();

  INCIPPLSTAT(incExpressions);
  INCIPPLSTAT(incSubEqualsExpression);
}


/***************************************************************************
 * $RCSfile: SubFieldAssign.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubFieldAssign.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/

