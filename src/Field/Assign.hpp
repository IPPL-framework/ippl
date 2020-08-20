/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

//////////////////////////////////////////////////////////////////////
//
// This file contains the versions of assign() that work with expressions
// on the RHS.  They do not handle general communications, and require
// sufficient guard cells to cover any stencil-like access.
//
//////////////////////////////////////////////////////////////////////

// include files
#include "Field/Assign.h"
#include "Field/AssignDefs.h"
#include "Field/BareField.h"
#include "Field/BrickExpression.h"
#include "Field/IndexedBareField.h"
#include "Field/LField.h"
#include "Message/Communicate.h"
#include "Message/Message.h"
#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"

#include "PETE/IpplExpressions.h"

#include <map>
#include <vector>
#include <functional>
#include <utility>
#include <iostream>
#include <typeinfo>

//////////////////////////////////////////////////////////////////////
//
// TryCompressLhs.
//
// Encodes the logic for whether to compress or uncompress the
// left hand side given information about the expression.
//
//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim, class A, class Op>
bool
TryCompressLHS(LField<T,Dim>& lf, A& rhs, Op op, const NDIndex<Dim>& domain)
{

  // just skip this if we can
  if (IpplInfo::noFieldCompression)
    return(false);

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("TryCompressLHS", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Checking for compressibility of LField with domain = ");
  ASSIGNMSG(msg << lf.getOwned() << " over assignment domain = " << domain);
  ASSIGNMSG(msg << endl);

  // If the right hand side is compressed and we are looking at
  // the whole domain for the lhs, we have a chance of
  // being able to compress the left hand side.
  // Then if it is simple assign or if the lhs is already compressed
  // we can do a compressed assign.
  bool c1 = for_each(rhs,IsCompressed(),PETE_AndCombiner());
  bool c2 = domain.containsAllPoints(lf.getOwned());
  bool c3 = OperatorTraits<Op>::IsAssign;
  bool c4 = lf.IsCompressed();
  bool compress = c1 && c2 && ( c3 || c4 );

  ASSIGNMSG(msg << "  RHS IsCompressed() = " << c1 << endl);
  ASSIGNMSG(msg << "  LHS IsCompressed() = " << c4 << endl);
  ASSIGNMSG(msg << "domain.contains(lhs) = " << c2 << endl);
  ASSIGNMSG(msg << "            IsAssign = " << c3 << endl);
  ASSIGNMSG(msg << "              result = " << compress << endl);

  // If we decide it can be compressed, do it, otherwise undo it.
  if (compress)
    {
      // We can compress this, so compress it down using first element
      // as the compression value.
      ASSIGNMSG(msg << "Yes we CAN compress, so do so now ... ");
      lf.Compress();
      ASSIGNMSG(msg << "now, compressed value = " << *lf.begin() << endl);
      return true;
    }

  // We can't compress the LHS.  Check if both sides are compressed already
  // and have the same value, and we're doing assignment (that is, check
  // if we're trying to assign the same value to a portion of an already
  // compressed region).  Note that if this is true, we know that the op
  // is for assignment.
  if (c1 && c3 && c4)
    {
      T tmpval{};
      PETE_apply(op, tmpval, for_each(rhs, EvalFunctor_0()));
      if (*lf.begin() == tmpval)
	{
	  // Both sides are compressed, and we're doing assignment, but the
	  // domains don't fully intersect.  We can still deal with this as
	  // a compressed entity if the LHS compressed value equals the RHS
	  // compressed value.
	  ASSIGNMSG(msg << "LHS and RHS are compressed, doing assign, and ");
	  ASSIGNMSG(msg << *lf.begin() << " == " <<tmpval<<", so result = 1");
	  ASSIGNMSG(msg << endl);
	  return true;
	}
    }

  // OK we need to uncompress the LHS, but we might not need to keep the data.
  // If we are doing assignment, or are not going to use all of the
  // LHS domain, we will need to copy the compressed value into the
  // uncompressed storage.  Otherwise, we know we'll just reset all the
  // values during the upcoming assignment, so it is a waste to do it
  // now.  If the arguments is true, then copy in the compressed value,
  // if it is false then allocate storage but do not do anything more
  // now to initialize it.
  ASSIGNMSG(msg << "No we cannot compress, so make sure we're ");
  ASSIGNMSG(msg << "uncompressed. Fill domain? " << !(c3&&c2) << endl);
  lf.Uncompress( !(c3 && c2) );
  return false;
}


//////////////////////////////////////////////////////////////////////
//
// A class with an interface like BrickIterator
// that applies the parens operator to some expression of type Expr.
// It passes most operations to the 'Child' item, but retrieves values
// by applying operator() to the Child or the Child's returned values.
// This is used by the version of assign that lets you assign values
// to just one component of a Field on the LHS.
//
//////////////////////////////////////////////////////////////////////

template<class Expr>
class ParensIterator : public Expr
{
public:
  typedef typename Expr::PETE_Return_t PETE_Return_t;

  ParensIterator( const Expr& e ) : Expr(e) {}
  PETE_Return_t& operator*() const
  {
    return (*Expr::Child)(Expr::Value.Arg);
  }
  PETE_Return_t& offset(int i) const
  {
    return Expr::Child.offset(i)(Expr::Value.Arg);
  }
  PETE_Return_t& offset(int i, int j) const
  {
    return Expr::Child.offset(i,j)(Expr::Value.Arg);
  }
  PETE_Return_t& offset(int i, int j, int k) const
  {
    return Expr::Child.offset(i,j,k)(Expr::Value.Arg);
  }

  PETE_Return_t& operator*() 
  {
    return (*Expr::Child)(Expr::Value.Arg);
  }
  PETE_Return_t& offset(int i) 
  {
    return Expr::Child.offset(i)(Expr::Value.Arg);
  }
  PETE_Return_t& offset(int i, int j) 
  {
    return Expr::Child.offset(i,j)(Expr::Value.Arg);
  }
  PETE_Return_t& offset(int i, int j, int k) 
  {
    return Expr::Child.offset(i,j,k)(Expr::Value.Arg);
  }
  PETE_Return_t& unit_offset(int i) 
  {
    return Expr::Child.unit_offset(i)(Expr::Value.Arg);
  }
  PETE_Return_t& unit_offset(int i, int j) 
  {
    return Expr::Child.unit_offset(i,j)(Expr::Value.Arg);
  }
  PETE_Return_t& unit_offset(int i, int j, int k) 
  {
    return Expr::Child.unit_offset(i,j,k)(Expr::Value.Arg);
  }

  void step(unsigned d)
  {
    Expr::Child.step(d);
  }
  void rewind(unsigned d)
  {
    Expr::Child.rewind(d);
  }
  int size(unsigned d) const
  {
    return Expr::Child.size(d);
  }
  int done(unsigned d) const
  {
    return Expr::Child.done(d);
  }
  int Stride(int d) const
  {
    return Expr::Child.Stride(d);
  }
};


//////////////////////////////////////////////////////////////////////
//
// IndexedBareField = expression assignment.
//
// This is the specialization with ExprTag<true>, meaning the RHS
// is an expression, not just a simple IndexedBareField.  This version
// only works if the LHS and RHS terms all agree in their parallel
// layout within guard-cell tolerances.  If they do not, it is
// an error and IPPL will report it and die.
// Since this is for IndexedBareField, extra checks are done to
// make sure you only process the relevant domain, and that you keep
// track of how you are indexing the values (using plugbase).
//
//////////////////////////////////////////////////////////////////////

template<class T1, unsigned Dim, class RHS, class OP>
void
assign(const IndexedBareField<T1,Dim,Dim> &aa, RHS b, OP op, ExprTag<true>, 
       bool fillGC)
{
  IndexedBareField<T1,Dim,Dim> &a = 
    const_cast<IndexedBareField<T1,Dim,Dim>&>(aa);

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("assign IBF(t)", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Computing assignment to IBF[" << aa.getDomain());
  ASSIGNMSG(msg << "] ..." << endl);

  // First check to see if any of the terms on the rhs
  // are the field on the lhs.  If so we'll have to make temporaries.
  int lhs_id = a.getBareField().get_Id();
  typename RHS::Wrapped& bb = b.PETE_unwrap();
  bool both_sides = for_each(bb,SameFieldID(lhs_id),PETE_OrCombiner());

  // Fill guard cells if necessary
  ASSIGNMSG(msg << "Checking whether to fill GC's on RHS ..." << endl);
  for_each(bb, FillGCIfNecessary(a.getBareField()), PETE_NullCombiner());

  // Begin and end iterators for the local fields in the left hand side.
  typename BareField<T1,Dim>::iterator_if la = a.getBareField().begin_if();
  typename BareField<T1,Dim>::iterator_if aend = a.getBareField().end_if();

  // Set the dirty flag indicating this field should have guard cells
  // filled next time if we are doing deferred GC fills, since
  // we will be modifying at least one LField of this BareField.
  // We need to set this here so that our compression checks on each
  // LField take the dirty flag setting into account.
  a.getBareField().setDirtyFlag();

  // Loop over all the local fields of the left hand side.
  
  int lfcount=0;
  bool needFinalCompressCheck = false;
  while (la != aend)
    {
      // The pointer to the current lhs local field.
      LField<T1,Dim> *lf = (*la).second.get();

      // If it is on the rhs somewhere, make a copy of it.
      if ( both_sides ) {
      	lf = new LField<T1,Dim>( *lf );
        ASSIGNMSG(msg << "For lf " << lfcount << ": making lfield copy.");
	ASSIGNMSG(msg << endl);
      }

      // Find the local domain.
      // Intersect with the indexes used to see how much we will actually use.
      NDIndex<Dim> local_domain = a.getDomain().intersect( lf->getOwned() );

      // If there is something there...
      if ( ! local_domain.empty() )
	{
	  // Some typedefs to make the lines shorter...
	  // types for the left hand side, right hand side and
	  // the whole expression.
	  typedef typename LField<T1,Dim>::iterator LHS;
	  typedef BrickExpression<Dim,LHS,typename RHS::Wrapped,OP> ExprT;

	  // First look and see if the arrays are sufficiently aligned
	  // to do this in one shot.
	  // We do this by trying to do a plugBase and seeing if it worked.

	  ASSIGNMSG(msg << "For lf " << lfcount << ": plugbase on ");
	  ASSIGNMSG(msg << local_domain << endl);
	  if ( for_each(bb,PlugBase<Dim>( local_domain ), PETE_AndCombiner()) )
            {
	      ASSIGNMSG(msg << "For lf " << lfcount << " with owned domain ");
	      ASSIGNMSG(msg << lf->getOwned() << " assigned intersection ");
	      ASSIGNMSG(msg << local_domain << " : ");

	      if (a.getBareField().compressible() &&
		  TryCompressLHS(*lf,bb,op,local_domain) ) {
                // Compressed assign.
		ASSIGNMSG(msg << "compressed assign, changing ");
		ASSIGNMSG(msg << *(lf->begin()));

		// Just apply the operator to the single compressed value
		// on the LHS.  If we're here, we know the RHS is compressed
		// so we can just evaluate it at its first position.
                PETE_apply(op, *(lf->begin()), for_each(bb,EvalFunctor_0()));
		ASSIGNMSG(msg << " to " << *(lf->begin()) << endl);
              } else {
		// Loop assign.
		ASSIGNMSG(msg << "loop assign." << endl);

		// Create the expression object.
		ExprT expr(lf->begin(local_domain), bb);
		expr.apply();

		// Try to compress this LField since we did an uncompressed
		// assignment, if the user has requested this kind of
		// check right after computation on the LField.  If this
		// is not selected, then we'll need to do some end-of-loop
		// compression checks.
		if (IpplInfo::extraCompressChecks) {
		  ASSIGNMSG(msg << "For lf " << lfcount);
		  ASSIGNMSG(msg << ": doing extra post-compute ");
		  ASSIGNMSG(msg << "compression check ..." << endl);
		  lf->TryCompress(a.getBareField().isDirty());
		} else {
		  needFinalCompressCheck = true;
		}
	      }
            }
	  else
	    {
	      ERRORMSG("All Fields in an expression must be aligned.  ");
	      ERRORMSG("(Do you have enough guard cells?)" << endl);
	      ERRORMSG("This error occurred while evaluating an expression ");
	      ERRORMSG("for an LField with domain " << lf->getOwned() << endl);
	      Ippl::abort();
	    }
	}

      // If we had to make a copy of the current LField,
      // swap the pointers and delete the old memory.
      if ( both_sides )
	{
	  ASSIGNMSG(msg << "For lf " << lfcount << ": swapping lfield data.");
	  ASSIGNMSG(msg << endl);
	  ASSIGNMSG(msg << "For lf " << lfcount << ": at beg, lfield=" << *lf);
	  ASSIGNMSG(msg << endl);
	  (*la).second->swapData( *lf );
	  delete lf;
	  ASSIGNMSG(msg << "For lf " << lfcount << ": at end, lfield=");
	  ASSIGNMSG(msg << *((*la).second) << endl);
	}

      ++la;
      ++lfcount;
    }
  

  // If we are not deferring guard cell fills, and we need to do this
  // now, fill the guard cells.  This will also apply any boundary
  // conditions after the guards have been updated.
  if (fillGC) {
    ASSIGNMSG(msg << "Filling GC's at end if necessary ..." << endl);
    
    a.getBareField().fillGuardCellsIfNotDirty();
    
  }

  // Try to compress the result.
  if (fillGC && needFinalCompressCheck) {
    ASSIGNMSG(msg << "Trying to compress BareField at end ..." << endl);
    a.getBareField().Compress(); // tjw added fillGC 12/16/1997
  }

  //INCIPPLSTAT(incExpressions);
  //INCIPPLSTAT(incIBFEqualsExpression);
}


//////////////////////////////////////////////////////////////////////
//
// ParensExpression = expression assignment.
//
// A version of assign() that handles assignment to just a component of
// a Field that has been selected via operator().  This version
// only works if the LHS and RHS terms all agree in their parallel
// layout within guard-cell tolerances.  If they do not, it is
// an error and IPPL will report it and die.  The item having operator()
// applied can be a BareField or an IndexedBareField.
//
//////////////////////////////////////////////////////////////////////

template<class A, class RHS, class OP, class Tag, class TP>
void
assign(PETE_TUTree<OpParens<TP>,A> lhs, RHS wrhs, OP op, Tag,
       bool fillGC)
{

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("assign Parens", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Computing assignment to IBF[" << lhs.Child.getDomain());
  ASSIGNMSG(msg << "](" << lhs.Value.Arg << ") ..." << endl);

  enum { Dim = A::Dim_u };
  typedef typename A::return_type T1;

  typedef typename Expressionize<RHS>::type::Wrapped RHS_Wrapped;
  typename Expressionize<RHS>::type expr = Expressionize<RHS>::apply(wrhs);
  RHS_Wrapped & rhs = expr.PETE_unwrap();
  
  // Get a reference to the BareField on the left hand side, and the
  // total domain we are modifying.
  BareField<T1,Dim>& bare = lhs.Child.getBareField();
  const NDIndex<Dim> &total_domain = lhs.Child.getDomain();

  // Fill guard cells if necessary
  ASSIGNMSG(msg << "Checking whether to fill GC's on RHS ..." << endl);
  for_each(rhs, FillGCIfNecessary(bare), PETE_NullCombiner());

  // Begin and end iterators for the local fields in the left hand side.
  typename BareField<T1,Dim>::iterator_if la = bare.begin_if();
  typename BareField<T1,Dim>::iterator_if aend = bare.end_if();

  // Set the dirty flag indicating this field should have guard cells
  // filled next time if we are doing deferred GC fills.
  // We need to set this here so that our compression checks on each
  // LField take the dirty flag setting into account.
  bare.setDirtyFlag();

  // Loop over all the local fields of the left hand side.
  
  bool needFinalCompressCheck = false;
  while (la != aend)
    {
      // The pointer to the current lhs local field, and the owned domain.
      LField<T1,Dim> *lf = (*la).second.get();
      const NDIndex<Dim> &lo = lf->getOwned();

      // Find the local domain.
      // Intersect with the indexes used to see how much we will actually use.
      NDIndex<Dim> local_domain = total_domain.intersect(lo);

      // If there is something there...
      if (!local_domain.empty())
	{
	  // Some typedefs to make the lines shorter...
	  // types for the left hand side, right hand side and
	  // the whole expression.
	  typedef typename LField<T1,Dim>::iterator LA;
	  typedef PETE_TUTree<OpParens<TP>,LA> LHS;
	  typedef BrickExpression<Dim,ParensIterator<LHS>,RHS_Wrapped,OP> 
	    ExprT;

	  // First look and see if the arrays are sufficiently aligned
	  // to do this in one shot.
	  // We do this by trying to do a plugBase and seeing if it worked.
	  ASSIGNMSG(msg << "For lf " << lo << ": plugbase on ");
	  ASSIGNMSG(msg << local_domain << endl);
	  if (for_each(rhs, PlugBase<Dim>(local_domain), PETE_AndCombiner()))
            {
              // Check and see if the lhs is already compressed, and if so,
	      // if the RHS is compressed as well.  If so, then we can
	      // just do a compressed assign to modify the Nth element
	      // of the compressed value.
	      bool c1 = for_each(rhs, IsCompressed(), PETE_AndCombiner());
	      bool c2 = local_domain.containsAllPoints(lo);
	      bool c3 = lf->IsCompressed();
              if (bare.compressible() && c1 && c2 && c3) {
		// We can do a compressed assign, and we know the LHS is
		// already compressed.  So we can just assign to the first
		// value, which will modify the selected element of that
		// value.
		ASSIGNMSG(msg << "Compressed assign on ");
		ASSIGNMSG(msg << local_domain << ", changing "<< *lf->begin());
		const ParensIterator<LHS> ilhs = LHS(lhs.Value, lf->begin());
                PETE_apply(op, *ilhs, for_each(rhs, EvalFunctor_0()));
		ASSIGNMSG(msg << " to " << *lf->begin() << endl);

	      } else {
		ASSIGNMSG(msg << "Loop assign on " << local_domain << endl);

		// We must uncompress, and since we will only be writing
		// to a portion of each element, we must definitely fill
		// the domain with the compressed value if it is currently
		// compressed.
		lf->Uncompress();

		// Build an object that will carry out the expression.
		const ParensIterator<LHS> ilhs =
		  LHS(lhs.Value, lf->begin(local_domain));
		ExprT expr2(ilhs, rhs, op);
		expr2.apply();

		// Try to compress this LField right after we've modified it,
		// if the user wants us to do this now.
		if (IpplInfo::extraCompressChecks) {
		  ASSIGNMSG(msg << "Doing extra post-compute ");
		  ASSIGNMSG(msg << "compression check ..." << endl);
		  lf->TryCompress(bare.isDirty());
		} else {
		  needFinalCompressCheck = true;
		}
	      }
            }
	  else
	    {
	      ERRORMSG("All Fields in an expression must be aligned.  ");
	      ERRORMSG("(Do you have enough guard cells?)" << endl);
	      ERRORMSG("This error occurred while evaluating an expression ");
	      ERRORMSG("for an LField with domain ");
	      ERRORMSG((*la).second->getOwned() << endl);
	      Ippl::abort();
	    }
	}
      ++la;
    }
  

  // Fill the guard cells on the left hand side, if we are deferring
  // this operation until the next time it is needed.
  ASSIGNMSG(msg << "Filling GC's at end if necessary ..." << endl);
  if (fillGC) {
    
    bare.fillGuardCellsIfNotDirty();
    
  }

  // Compress the LHS.
  if (fillGC && needFinalCompressCheck) {
    ASSIGNMSG(msg << "Trying to compress BareField at end ..." << endl);
    bare.Compress();
  }

  //INCIPPLSTAT(incExpressions);
  //INCIPPLSTAT(incParensEqualsExpression);
}


//////////////////////////////////////////////////////////////////////
//
// BareField = expression assignment.
//
// This is the specialization with ExprTag<true>, meaning the RHS
// is an expression, not just a simple BareField.  This version
// only works if the LHS and RHS terms all agree in their parallel
// layout within guard-cell tolerances.  If they do not, it is
// an error and IPPL will report it and die.
// Since this is for BareField, the entire domain of the LHS is
// used to index the RHS.  The RHS terms cannot be IndexedBareFields,
// they must be BareFields as well (or other simpler items).
//
//////////////////////////////////////////////////////////////////////

template<class T1, unsigned Dim, class RHS, class OP>
void
assign(const BareField<T1,Dim>& ca, RHS b, OP op, ExprTag<true>)
{

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("assign BF(t)", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Computing assignment to BF[" << ca.getDomain());
  ASSIGNMSG(msg << "] ..." << endl);

  // cast away const here for lhs ... unfortunate but necessary.
  // Also get the item wrapped within the PETE expression.
  BareField<T1,Dim>& a = const_cast<BareField<T1,Dim>&>(ca);
  typename RHS::Wrapped& bb = b.PETE_unwrap();

  // Create iterators over the LHS's LFields.
  typedef typename LField<T1,Dim>::iterator It;
  typedef BrickExpression<Dim,It,typename RHS::Wrapped,OP> ExprT;
  typename BareField<T1,Dim>::iterator_if la = a.begin_if();
  typename BareField<T1,Dim>::iterator_if aend = a.end_if();

  // Set the dirty flag indicating this field should have guard cells
  // filled next time if we are doing deferred GC fills.
  // We need to set this here so that our compression checks on each
  // LField take the dirty flag setting into account.
  a.setDirtyFlag();

  // Loop over the LHS LFields, and assign from RHS LFields
  
  int lfcount = 0;
  bool needFinalCompressCheck = false;
  while (la != aend)
  {
    // Get the current LHS and set up the RHS to point to the beginning
    // of its current LField.  This second step is done in lieu of doing
    // a "plugbase", since it is faster and we know we're dealing with
    // a whole BareField here, not an indexed BareField.
    LField<T1,Dim>& lf = *(*la).second;
    for_each(bb, BeginLField(), PETE_NullCombiner());

    ASSIGNMSG(msg << "For lf " << lfcount << " with domain ");
    ASSIGNMSG(msg << lf.getOwned() << " : ");

    // Check to see if we can compress here.  If so, we can avoid a lot
    // of work.
    if (a.compressible() && TryCompressLHS(lf, bb, op, a.getDomain())) {
      ASSIGNMSG(msg << "compressed assign, changing " << *lf.begin());
      PETE_apply(op,*lf.begin(),for_each(bb,EvalFunctor_0()));
      ASSIGNMSG(msg << " to " << *lf.begin() << endl);
    } else {
      ASSIGNMSG(msg << "loop assign." << endl);

      // Create the expression object.
      ExprT expr(lf.begin(),bb,op);
      expr.apply();

      // Try to compress this LField since we did an uncompressed
      // assignment, if the user has requested this kind of
      // check right after computation on the LField.  If this kind
      // of request has not been made, then we'll need to do a compression
      // check at the end.
      if (IpplInfo::extraCompressChecks) {
	ASSIGNMSG(msg << "For lf " << lfcount);
	ASSIGNMSG(msg << ": doing extra post-compute ");
	ASSIGNMSG(msg << "compression check ..." << endl);
	lf.TryCompress(a.isDirty());
      } else {
	needFinalCompressCheck = true;
      }
    }

    ++la;
    for_each(bb,NextLField(),PETE_NullCombiner());
    ++lfcount;
  }
  

  // Fill the guard cells on the left hand side, if we are deferring
  // this operation until the next time it is needed.
  ASSIGNMSG(msg << "Filling GC's at end if necessary ..." << endl);
  
  a.fillGuardCellsIfNotDirty();
  

  // Compress the LHS, if necessary
  if (needFinalCompressCheck) {
    ASSIGNMSG(msg << "Trying to compress BareField at end ..." << endl);
    a.Compress();
  }

  // Compress the LHS.
  //INCIPPLSTAT(incExpressions);
  //INCIPPLSTAT(incBFEqualsExpression);
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
