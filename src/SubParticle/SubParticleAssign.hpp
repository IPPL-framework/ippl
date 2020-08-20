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
#include "SubParticle/SubParticleAssign.h"
#include "SubParticle/SubParticleAssignDefs.h"
#include "SubParticle/SubParticleAttrib.h"
#include "Particle/ParticleAttrib.h"
#include "Index/SIndex.h"
#include "Index/LSIndex.h"
#include "Field/BrickExpression.h"
#include "Utility/IpplStats.h"



//////////////////////////////////////////////////////////////////////
// simple struct used to indicate whether an operation is just an
// assign, or an accumulation
template<class OP>
struct PAIsAssign { enum { assign = 0 }; };

template<>
struct PAIsAssign<OpAssign> { enum { assign = 1 }; };


//////////////////////////////////////////////////////////////////////
// assign an arbitrary expression to a SubParticleAttrib, with an
// aligned expression.  Note that the ParticleAttrib length may change,
// as it will be adjusted to have the same length as the number of local
// points in the SIndex indexing the SubParticleAttrib.
template<class PA, class T, unsigned Dim, class RHS, class OP>
void assign(SubParticleAttrib<PA,T,Dim> a, RHS b, OP /*op*/, ExprTag<true>) {

  // Make sure the LHS ParticleAttrib has the proper length.  It should
  // have the same length as the number of LOCAL sparse index points.
  // If it does not, we readjust the length.  Note that for accumulations,
  // like +=, -=, etc., the size must already be OK.
  if (PAIsAssign<OP>::assign == 1)
    a.adjustSize();
  else
    PInsist(a.getDomain().size() == a.getAttrib().size(),
	    "ParticleAttrib[SIndex] accumulation with wrong-sized attribute.");

  typedef typename RHS::Wrapped Wrapped;

  // Get the object wrapped in the PETE expression object
  // RHS::Wrapped& bb = b.PETE_unwrap();
  Wrapped& bb = b.PETE_unwrap();
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
  //tjw  for_each(bb, FillGCIfNecessary(boundBox), PETE_NullCombiner());

  // This weird tag is needed, because using a no-argument FillGCIFNEcessary()
  // function didn't work for some reason, at least with some compilers like
  // the pre-7.3 SGI compiler and CodeWarrior Pro4. Once the global function
  // invocation syntax FillGCIfNecessary<T,D>() is supported by all our
  // compilers (it's not there yet in the non-beta SGI compiler), should be
  // able to eliminate this FGCINTag business. --tjw 3/3/1999
  for_each(bb, FillGCIfNecessary(FGCINTag<Dim,T>()), PETE_NullCombiner());

  // Iterators for the LHS components and LField's
  typename SubParticleAttrib<PA,T,Dim>::iterator sba = a.begin();
  typename SIndex<Dim>::const_iterator_iv la = sba.getLFieldIter();
  typename SIndex<Dim>::const_iterator_iv aend = a.end().getLFieldIter();

  // Do any initial work on the expression it needs before starting.
  // E.g., for single-value subsets, this will distribute the values to nodes
  sba.initialize();
  for_each(bb, SubsetInit(), PETE_NullCombiner());

  // Loop over all the local fields of the left hand side.
  while (la != aend) {
    // Some typedefs to make the lines shorter...
    // types for the left hand side, right hand side and
    // the whole expression.
    typedef typename SubParticleAttrib<PA,T,Dim>::iterator LHS;
    typedef BrickExpression<1,LHS,typename RHS::Wrapped,OP> ExprT;

    // If there are any points in the local SIndex object (for the current
    // local vnode, that is), there is some work to do here.
    if ((*la)->size() > 0) {
      // Now look and see if the RHS arrays are sufficiently aligned.
      // We do this by trying to do a plugBase and seeing if it worked.
      if (for_each(bb,PlugBase<Dim>((*la)->getDomain()),PETE_AndCombiner())) {
	// Loop assign.
	ExprT(sba,bb).apply();
      } else {
        ERRORMSG("All items in an expression must be aligned.  ");
        ERRORMSG("(Do you have enough guard cells?)" << endl);
        ERRORMSG("This error occurred while evaluating a ");
	ERRORMSG("SubParticleAttrib-expression ");
        ERRORMSG("for a vnode with domain " << (*la)->getDomain() << endl);
        Ippl::abort();
      }
    }

    // move iterators on to the next LField
    la = sba.nextLField();
    for_each(bb, SubsetNextLField(), PETE_NullCombiner());
  }

  INCIPPLSTAT(incParticleExpressions);
}


/***************************************************************************
 * $RCSfile: SubParticleAssign.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubParticleAssign.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/

