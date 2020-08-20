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
#include "Index/SIndexAssign.h"
#include "Index/SIndex.h"
#include "Field/BrickIterator.h"
#include "Field/Field.h"
#include "Field/IndexedField.h"
#include "Field/Assign.h"
#include "Utility/IpplInfo.h"



//////////////////////////////////////////////////////////////////////
// a simple class to apply the assign function to one point of the
// assignment domain.
template<unsigned int Dim, class OP>
class SIndexAssignTraits { };


//////////////////////////////////////////////////////////////////////
// a specialization of the simple class to apply the regular assign
// function to one point of the assignment domain.

template<unsigned int Dim>
class SIndexAssignTraits<Dim, OpAssign> {
public:
  // initially, the sparse index container must be cleared out
  static void initialize(SIndex<Dim>& s) { s.clear(); }

  // S = exp means we cleared out all the points previously, so we know
  // we do not need to worry about repeated points
  static void apply(SIndex<Dim>& /*SI*/,
		    typename SIndex<Dim>::iterator_iv& LSI,
		    const SOffset<Dim>& SO,
		    bool result) {
    if (result) {
      // we can just add the index to the LSIndex's list here, for the
      // following reasons:
      // 1. We did a 'clear' before the expression started (when
      //    initialize was called), so all the LSIndex objects in this
      //    SIndex are separate, empty items, not references to other
      //    SIndex lists.
      // 2. We know this point is unique, since we did a clear and then
      //    looped over completely different points in the expression
      //    evaluation.
      // 3. We are sure this point belongs to this LSIndex, since we
      //    iterate over the vnodes of the SIndex.
      (*LSI)->addIndex(SO);
    }
  }
};


//////////////////////////////////////////////////////////////////////
// a specialization of the simple class to apply the bitwise AND assign
// function to one point of the assignment domain.

template<unsigned int Dim>
class SIndexAssignTraits<Dim, OpBitwiseAndAssign> {
public:
  // since we do an intersection, we must keep the existing data
  static void initialize(SIndex<Dim>&) { }

  // S &= expr means 'intersection' ... only keep the point if result is
  // true, remove the point if result is false
  static void apply(SIndex<Dim>& SI,
		    typename SIndex<Dim>::iterator_iv& LSI,
		    const SOffset<Dim>& SO,
		    bool result) {
    if (!result && (*LSI)->hasIndex(SO)) {
      // we must call removeIndex in SIndex (and not in LSIndex) here
      // because we might need to make a copy of the LSIndex data
      // first (copy-on-write).  This will be slower than doing some
      // form of direct assignment.
      SI.removeIndex(LSI, SO);
    }
  }
};


//////////////////////////////////////////////////////////////////////
// a specialization of the simple class to apply the bitwise OR assign
// function to one point of the assignment domain.

template<unsigned int Dim>
class SIndexAssignTraits<Dim, OpBitwiseOrAssign> {
public:
  // since we do a union, we must keep the existing data
  static void initialize(SIndex<Dim>&) { }

  // S |= expr means 'union' ... we add points as normal, but we did not
  // clear out the points that were there previously
  static void apply(SIndex<Dim>& SI,
		    typename SIndex<Dim>::iterator_iv& LSI,
		    const SOffset<Dim>& SO,
		    bool result) {

    if (result && ! (*LSI)->hasIndex(SO)) {
      // we must call addIndex in SIndex (and not in LSIndex) here
      // because we need to check if this point SO is already in the
      // list of points.  This will be slower than doing some form
      // of direct assignment.
      SI.addIndex(LSI, SO);
    }
  }
};


//////////////////////////////////////////////////////////////////////
// a simple class to do different dimension-specific loops in an
// SIndex expression.
template<class OP, unsigned int Dim>
class SIndexExpLoop {
public:
  // the general loop; this is done for 4D or higher dimensions
  template<class RHS>
  static void evaluate(SIndex<Dim>& si, typename SIndex<Dim>::iterator_iv& lsi,
		       NDIndex<Dim>& dom, RHS& Rhs) {
    
    int n0 = dom[0].length();
    int n1 = dom[1].length();
    int n2 = dom[2].length();
    if (n0 > 0 && n1 > 0 && n2 > 0) {
        BrickCounter<Dim> count(dom);   // ada: changed from cdom to dom, cdom does not make sense!
        ERRORMSG("changed from cdom to dom, cdom does not make sense! (SindexAssign.cpp)" << endl);
        
        unsigned d;
      SOffset<Dim> so;
      for (d=3; d < Dim; ++d)
	so[d] = dom[d].first();
      do {
	Index::iterator x2 = dom[2].begin();
	for (int i2=0; i2<n2; ++i2, ++x2) {
	  so[2] = *x2;
	  Index::iterator x1 = dom[1].begin();
	  for (int i1=0; i1<n1; ++i1, ++x1) {
	    so[1] = *x1;
	    Index::iterator x0 = dom[0].begin();
	    for (int i0=0; i0<n0; ++i0, ++x0) {
	      so[0] = *x0;
	      bool result = (for_each(Rhs, EvalFunctor_3(i0,i1,i2)) != false);
	      SIndexAssignTraits<3U,OP>::apply(si, lsi, so, result);
	    }
	  }
	}

	for (d=3; d<Dim; ++d) {
	  count.step(d);
	  so[d] += dom[d].stride();
	  for_each(Rhs,StepFunctor(d),PETE_NullCombiner());
	  if ( ! count.done(d) )
	    break;
	  count.rewind(d);
	  so[d] = dom[d].first();
	  for_each(Rhs,RewindFunctor(d),PETE_NullCombiner());
	}

      } while (d<Dim);
    }
  }
};


//////////////////////////////////////////////////////////////////////
//a specialization of SIndexAssignLoop for a 1D loop evaluation NOTE:
//dom must be within the local domain of the given LSIndex
template<class OP>
class SIndexExpLoop<OP,1U> {
public:
  template<class RHS>
  static void evaluate(SIndex<1U>& si, SIndex<1U>::iterator_iv& lsi,
		       NDIndex<1U>& dom, RHS& Rhs) {
    
    int n0 = dom[0].length();
    if (n0 > 0) {
      Index::iterator x0 = dom[0].begin();
      for (int i0 = 0; i0 < n0; ++i0, ++x0) {
	bool result = (for_each(Rhs, EvalFunctor_1(i0)) != false);
	SIndexAssignTraits<1U,OP>::apply(si, lsi, SOffset<1>(*x0), result);
      }
    }
  }
};


//////////////////////////////////////////////////////////////////////
// a specialization of SIndexAssignLoop for a 2D loop evaluation
// NOTE: dom must be within the local domain of the given LSIndex
template<class OP>
class SIndexExpLoop<OP,2U> {
public:
  template<class RHS>
  static void evaluate(SIndex<2U>& si, SIndex<2U>::iterator_iv& lsi,
		       NDIndex<2U>& dom, RHS& Rhs) {
    
    int n0 = dom[0].length();
    int n1 = dom[1].length();
    if (n0 > 0 && n1 > 0) {
      Index::iterator x1 = dom[1].begin();
      for (int i1 = 0; i1 < n1; ++i1, ++x1) {
	Index::iterator x0 = dom[0].begin();
	for (int i0 = 0; i0 < n0; ++i0, ++x0) {
	  bool result = (for_each(Rhs, EvalFunctor_2(i0,i1)) != false);
	  SIndexAssignTraits<2U,OP>::apply(si, lsi, SOffset<2>(*x0, *x1),
					   result);
	}
      }
    }
  }
};


//////////////////////////////////////////////////////////////////////
// a specialization of SIndexAssignLoop for a 3D loop evaluation
// NOTE: dom must be within the local domain of the given LSIndex
template<class OP>
class SIndexExpLoop<OP,3U> {
public:
  template<class RHS>
  static void evaluate(SIndex<3U>& si, SIndex<3U>::iterator_iv& lsi,
		       NDIndex<3U>& dom, RHS& Rhs) {
    
    int n0 = dom[0].length();
    int n1 = dom[1].length();
    int n2 = dom[2].length();
    if (n0 > 0 && n1 > 0 && n2 > 0) {
      Index::iterator x2 = dom[2].begin();
      for (int i2 = 0; i2 < n2; ++i2, ++x2) {
	Index::iterator x1 = dom[1].begin();
	for (int i1 = 0; i1 < n1; ++i1, ++x1) {
	  Index::iterator x0 = dom[0].begin();
	  for (int i0 = 0; i0 < n0; ++i0, ++x0) {
	    bool result = (for_each(Rhs, EvalFunctor_3(i0,i1,i2)) != false);
	    SIndexAssignTraits<3U,OP>::apply(si, lsi,
					     SOffset<3>(*x0, *x1, *x2),
					     result);
	  }
	}
      }
    }
  }
};


//////////////////////////////////////////////////////////////////////
// The 'assign' function for SIndex objects must handle two situations,
// one where the RHS is a simple, full-field expression, and one where
// the RHS is a complex, indexed expression.  The 'AssignActions'
// struct is specialized on the 'SiExprTag<bool>' class to perform
// different setup and cleaup operations for the RHS objects based on
// whether the expression is simple (bool=false) or indexed (bool=true).

template<unsigned Dim, class T>
struct AssignActions { };

template<unsigned Dim>
struct AssignActions<Dim, SIExprTag<true> > {
  template<class RHS>
  static void fillgc(RHS &bb, const NDIndex<Dim> & /*domain*/) {
    // ask each field on the RHS to fill its guard cells, if necessary
    //tjw3/3/99    for_each(bb, FillGCIfNecessary(domain), PETE_NullCombiner());
    for_each(bb, FillGCIfNecessary(FGCINTag<Dim,double>()), PETE_NullCombiner());
  }

  template<class RHS>
  static bool plugbase(RHS &bb, const NDIndex<Dim> &domain) {
    // ask each RHS field to set itself up to iterate over the given locdomain.
    // if any cannot, this will return false.
    return for_each(bb, PlugBase<Dim>(domain), PETE_AndCombiner());
  }

  template<class RHS>
  static void nextLField(RHS &) {
    // in this case, there is nothing to do, since PlugBase always ends
    // up setting which LField to use
  }
};

template<unsigned Dim>
struct AssignActions<Dim, SIExprTag<false> > {
  template<class RHS>
  static void fillgc(RHS &bb, const NDIndex<Dim> &) {
    // ask each field on the RHS to fill its guard cells, if necessary
    // here, it cannot be a stencil, so not necessary
    //tjw 3/3/99, add:
    for_each(bb, FillGCIfNecessary(FGCINTag<Dim,double>()), PETE_NullCombiner());
    //tjw 3/3/99, add.
  }

  template<class RHS>
  static bool plugbase(RHS &bb, const NDIndex<Dim> &) {
    // tell each RHS field to reset to the beginning of it's current LField
    for_each(bb, BeginLField(), PETE_NullCombiner());
    return true;
  }

  template<class RHS>
  static void nextLField(RHS &bb) {
    // tell each RHS field to move on to the next LField
    for_each(bb, NextLField(), PETE_NullCombiner());
  }
};


//////////////////////////////////////////////////////////////////////
// assign values to an SIndex: evaluate the RHS at all the points in
// the given domain, find where it evaluates to true, and store the
// value of that index point.
template<unsigned Dim, class RHS, class Op, bool IsExpr>
void assign(SIndex<Dim>& a, RHS b, Op, const NDIndex<Dim> &domain, 
	    SIExprTag<IsExpr> /*isexpr*/) {
  
   
  

  // Inform dbgmsg("SIndex assign", INFORM_ALL_NODES);
  // dbgmsg << "Computing on total domain = " << domain << endl;

  // unwrap the PETE expression object to get what we're computing with
  typename RHS::Wrapped& bb = b.PETE_unwrap();

  // initialize the SIndex for this operation
  SIndexAssignTraits<Dim,Op>::initialize(a);

  // Fill GC's, if necessary
  AssignActions<Dim,SIExprTag<IsExpr> >::fillgc(bb, domain);

  // iterate through all the local vnodes in the FieldLayout used
  // by the SIndex, and look at the points in these domains
  typename SIndex<Dim>::iterator_iv la   = a.begin_iv();
  typename SIndex<Dim>::iterator_iv aend = a.end_iv();
  for ( ; la != aend; ++la ) {
    // check if we have any points in this LField to process
    NDIndex<Dim> locdomain = domain.intersect((*la)->getDomain());
    //dbgmsg << "Doing LField " << (*la)->getDomain() << " with intersection ";
    //dbgmsg << locdomain << endl;

    // only proceed if we have any intersecting points
    if (!locdomain.empty()) {
      // First look and see if the arrays are sufficiently aligned to do
      // this in one shot.  We do this by trying to do a plugBase and
      // seeing if it worked.
      if (AssignActions<Dim,SIExprTag<IsExpr> >::plugbase(bb, locdomain)) {

	// check if the RHS is compressed and we can do a compressed assign
	if (OperatorTraits<Op>::IsAssign && locdomain == (*la)->getDomain() &&
	    for_each(bb, IsCompressed(), PETE_AndCombiner())) {
	  // why yes we can ... tell the SIndex to store the results for
	  // the entire vnode in one shot
	  bool result = (for_each(bb, EvalFunctor_0()) != false);
	  (*la)->Compress(result);

	  //NDIndex<Dim>& dom = (NDIndex<Dim>&) ((*la)->getDomain());
	  //dbgmsg << "compressed in dom = " << dom << ", result = " << result;
	  //dbgmsg << endl;
	} else { 
	  // perform actions to find necessary points in the current Vnode
	  // of the SIndex, by looping over the domain and seeing where
	  // the RHS == true
	  SIndexExpLoop<Op,Dim>::evaluate(a, la, locdomain, bb);

	  //NDIndex<Dim>& dom = (NDIndex<Dim>&) ((*la)->getDomain());
	  //dbgmsg << "uncompressed in dom = " << dom << endl;
	}
      } else {
	ERRORMSG("All Fields in an expression must be aligned.  ");
	ERRORMSG("(Do you have enough guard cells?)" << endl);
	ERRORMSG("This error occurred while evaluating an SIndex expression ");
	ERRORMSG("for an LField with domain " << (*la)->getDomain() << endl);
	Ippl::abort();
      }
      //    } else {
      //      dbgmsg << " ... intersection is empty." << endl;
    }

    // move the RHS Field's on to the next LField, if necessary
    AssignActions<Dim,SIExprTag<IsExpr> >::nextLField(bb);
  }

  // at the end, tell the SIndex we used this domain, so that should be
  // its new bounding box
  a.setDomain(domain);
}


/***************************************************************************
 * $RCSfile: SIndexAssign.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: SIndexAssign.cpp,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
