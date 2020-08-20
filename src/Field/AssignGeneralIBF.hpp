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

//////////////////////////////////////////////////////////////////////
//
// This file contains the version of assign() that work with 
// general IndexedBareField = IndexedBareField expressions.  It will perform
// general communication to exchange data if the LHS and RHS layouts
// do not match.  It will also handle assignments between sliced fields,
// permuted indices, etc.
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
// Send out messages needed to do an IBF = IBF assignment.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D1, unsigned D2>
inline void
IndexedSend(IndexedBareField<T1,D1,D1>& ilhs,
	    IndexedBareField<T2,D2,D2>& irhs,
	    int tag)
{
  
  

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("IndexedSend", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Sending out messages for IBF[" << ilhs.getDomain());
  ASSIGNMSG(msg << "] = IBF[" << irhs.getDomain() << "] ..." << endl);

  // Get the BareField for the left and right hand sides.
  BareField<T1,D1> &lhs = ilhs.getBareField();
  BareField<T2,D2> &rhs = irhs.getBareField();
  typename BareField<T2,D2>::iterator_if rf_i, rf_e = rhs.end_if();
  T2 compressed_value;

  // set up messages to be sent
  int nprocs = Ippl::getNodes();
  Message** mess = new Message*[nprocs];
  int iproc;
  for (iproc=0; iproc<nprocs; ++iproc) 
    mess[iproc] = 0;

  // Loop over all the local nodes of the right hand side and
  // send data to the remote ones they overlap on the left hand side.
  for (rf_i = rhs.begin_if(); rf_i != rf_e; ++rf_i) {
    // Cache some information about this local array.
    LField<T2,D2> &rf = *((*rf_i).second);
    const NDIndex<D2>& ro = rf.getOwned();

    // Is this local field in the given right hand side domain?
    if ( ro.touches( irhs.getDomain() ) ) {
      // They touch, find the intersection.
      NDIndex<D2> rt = irhs.getDomain().intersect( ro );

      // Find the lhs domain where this is going to go.
      NDIndex<D1> lt = ilhs.getDomain().plugBase( rt );

      // Loop over the remote parts of lhs to find where to send stuff.
      typename FieldLayout<D1>::touch_range_dv
	range(lhs.getLayout().touch_range_rdv(lt,lhs.getGuardCellSizes()));
      typename FieldLayout<D1>::touch_iterator_dv remote_i;
      for (remote_i = range.first; remote_i != range.second; ++remote_i) {
	// Find the intersection.
	NDIndex<D1> left_intersect = lt.intersect( (*remote_i).first );

        // Find out who owns this remote domain
        int rnode = (*remote_i).second->getNode();

	// Forward substitute to get the domain in the rhs.
	NDIndex<D2> right_intersect =
          irhs.getDomain().plugBase(left_intersect);

	// Build the iterator for the data.
	typename LField<T2,D2>::iterator rhs_i =
	  rf.begin(right_intersect, compressed_value);

        ASSIGNMSG(msg << "Sending IndexedField data from domain ");
	ASSIGNMSG(msg << right_intersect << " to domain " << left_intersect);
	ASSIGNMSG(msg << endl);

        // Permute the loop order so that they agree.
	CompressedBrickIterator<T2,D1> prhs_i =
	  rhs_i.permute(right_intersect,left_intersect);

	// Try to compress it.
	prhs_i.TryCompress();

        // put data into proper message
	if (!mess[rnode]) mess[rnode] = new Message;
        PAssert(mess[rnode]);
        left_intersect.putMessage(*mess[rnode]);
        prhs_i.putMessage(*mess[rnode]);
      }  // loop over touching remote nodes
    }
  }  // loop over LFields

  // send all the messages
  for (iproc=0; iproc<nprocs; ++iproc) {
    if (mess[iproc])
      Ippl::Comm->send(mess[iproc],iproc,tag);
  }

  delete [] mess;
  return;
}


//////////////////////////////////////////////////////////////////////
//
// Calculate what messages we expect to receiving during an IBF = IBF assign.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D1, unsigned D2, class Container>
inline void
CalcIndexedReceive(IndexedBareField<T1,D1,D1>& ilhs,
		   IndexedBareField<T2,D2,D2>& irhs,
		   Container& recv_ac, int& msgnum)
{

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("CalcIndexedReceive", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Computing receive messages for IBF[" << ilhs.getDomain());
  ASSIGNMSG(msg << "] = IBF[" << irhs.getDomain() << "] ..." << endl);

  // Get the BareField for the left and right hand sides.
  BareField<T1,D1> &lhs = ilhs.getBareField();
  BareField<T2,D2> &rhs = irhs.getBareField();
  typename BareField<T1,D1>::iterator_if lf_i, lf_e = lhs.end_if();

  int nprocs = Ippl::getNodes();
  bool* recvmsg = new bool[nprocs];
  int iproc;
  for (iproc=0; iproc<nprocs; ++iproc)
    recvmsg[iproc] = false;

  // Loop over the locals.
  for (lf_i = lhs.begin_if(); lf_i != lf_e; ++lf_i) {
    // Cache some information about this LField.
    LField<T1,D1> &lf = *((*lf_i).second);
    const NDIndex<D1>& la = lf.getAllocated();
    // Is this local field in the domain in question.
    if ( la.touches( ilhs.getDomain() ) ) {
      // They touch.  Find the intersection.
      NDIndex<D1> lt = ilhs.getDomain().intersect( la );
      // Find the rhs domain this is coming from.
      NDIndex<D2> rt = irhs.getDomain().plugBase( lt );
      // Find the remote ones that that touch this.
      typename FieldLayout<D2>::touch_range_dv
	range( rhs.getLayout().touch_range_rdv(rt) );
      // Loop over them.
      typename FieldLayout<D2>::touch_iterator_dv rv_i;
      for (rv_i = range.first; rv_i != range.second; ++rv_i) {
	// Save the intersection and the LField it is for.
	NDIndex<D2> ri = rt.intersect((*rv_i).first);
	NDIndex<D1> li = ilhs.getDomain().plugBase( ri );

        ASSIGNMSG(msg << "Expecting IndexedField data from domain " << ri);
	ASSIGNMSG(msg << " for domain " << li << endl);

	typedef typename Container::value_type value_type;
	recv_ac.insert(value_type(li,&lf));
        // note who will be sending this data
        int rnode = (*rv_i).second->getNode();
        recvmsg[rnode] = true;
      }  // loop over remote nodes
    }
  }  // loop over LFields

  msgnum = 0;
  for (iproc=0; iproc<nprocs; ++iproc)
    if (recvmsg[iproc]) ++msgnum;
  delete [] recvmsg;
  return;
}


//////////////////////////////////////////////////////////////////////
//
// Assign between just the local blocks for an IBF = IBF assign.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D1, unsigned D2, class Op>
inline void
IndexedLocalAssign(IndexedBareField<T1,D1,D1>& ilhs,
		   IndexedBareField<T2,D2,D2>& irhs,
		   Op& op)
{

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("IndexedLocalAssign-IBF", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Computing general local assignment to IBF[");
  ASSIGNMSG(msg << ilhs.getDomain() << "] = IBF[");
  ASSIGNMSG(msg << irhs.getDomain() << "] ..." << endl);

  // Get the BareField for the left and right hand sides.
  BareField<T1,D1> &lhs = ilhs.getBareField();
  BareField<T2,D2> &rhs = irhs.getBareField();

  // Loop over all the local Fields of the lhs and all the local
  // fields in the rhs.
  // This is an N*N operation, but the expectation is that there won't
  // be TOO many Vnodes on a given processor.
  for (typename BareField<T1,D1>::iterator_if lf_i=lhs.begin_if();
       lf_i!=lhs.end_if(); ++lf_i)
    {
      // Cache some information about this LField.
      LField<T1,D1> &lf = *(*lf_i).second;
      const NDIndex<D1>& lo = lf.getOwned();
      const NDIndex<D1>& la = lf.getAllocated();

      // See if it touches the given domain.
      if ( lo.touches( ilhs.getDomain() ) )
	{
	  ASSIGNMSG(msg << "----------------" << endl);
	  ASSIGNMSG(msg << "Assigning to local LField with owned = " << lo);
	  ASSIGNMSG(msg << endl);

	  // Get the intersection.
	  NDIndex<D1> lt = ilhs.getDomain().intersect( lo );
	  ASSIGNMSG(msg << "Intersection lhs domain = " << lt << endl);

	  // Transform into right hand side space.
	  NDIndex<D2> rp = irhs.getDomain().plugBase( lt );
	  ASSIGNMSG(msg << "Plugbase domain = " << rp << endl);

	  // Loop over all the local lfields on the right.
	  for (typename BareField<T2,D2>::iterator_if rf_i = rhs.begin_if();
	       rf_i != rhs.end_if(); ++rf_i)
	    {
	      // Cache the domain for this local field.
	      LField<T2,D2> &rf = *(*rf_i).second;
	      const NDIndex<D2>& ra = rf.getAllocated();
	      const NDIndex<D2>& ro = rf.getOwned();

	      // Two cases:
	      // 1. rhs allocated contains lhs owned
	      //    Then this is a stencil and we should fill from allocated.
	      //    Assume that nobody else will fill the owned spot.
	      // 2. rhs allocated does not contain lhs owned.
	      //    Then this is general communication and we
	      //    should only fill from rhs owned.
	      // I can construct cases for which this gives wrong answers
	      // on += type operations.  We'll be able to fix this much much
	      // easier with one sided communication.

	      // If this local field touches the domain, we have work to do.
	      // ra has unit stride so end-point "contains" is OK.
	      const NDIndex<D2> &rd = ( ra.contains(rp) ? ra : ro );
	      if ( rd.touches( rp ) )
		{
		  // Get the intersection.  We'll copy out of this.
		  NDIndex<D2> rhsDomain = rp.intersect(rd);

		  // Transform that back.  We'll copy into this.
		  NDIndex<D1> lhsDomain = lt.plugBase(rhsDomain);

		  ASSIGNMSG(msg << "Found touching rhs field: assigning ");
		  ASSIGNMSG(msg << lhsDomain << " = " << rhsDomain << endl);

		  // Can we compress the left?
		  bool c1 = rf.IsCompressed();

		  // Not clear that lhs is unit stride, so call
		  // containsAllPoints...
		  bool c2 = lhsDomain.containsAllPoints(la);
		  bool c3 = OperatorTraits<Op>::IsAssign;
		  bool c4 = lf.IsCompressed();
		  ASSIGNMSG(msg << "Checking for possible compressed-assign:");
		  ASSIGNMSG(msg << "\n  rf.IsCompressed = " << c1);
		  ASSIGNMSG(msg << "\n  lhs.contains(allocatd) = " << c2);
		  ASSIGNMSG(msg << "\n  lf.IsCompressed = " << c4);
		  ASSIGNMSG(msg << "\n  Doing assignment = " << c3);
		  ASSIGNMSG(msg << "\n      Result = " << (c1&&c2&&(c3||c4)));
		  ASSIGNMSG(msg << endl);
		  if ( lhs.compressible() && c1 && c2 && ( c3 || c4 ) )
		    {
		      ASSIGNMSG(msg << "Can do compressed assign from rhs ");
		      ASSIGNMSG(msg << "to lhs." << endl);

		      // We can compress the left!
		      lf.Compress();

		      // Set the constant value.
		      PETE_apply(op, *lf.begin() , *rf.begin() );
		      ASSIGNMSG(msg << "After compress, " << *lf.begin());
		      ASSIGNMSG(msg << " = ");
		      ASSIGNMSG(msg << *rf.begin() << endl);
		    }
		  else
		    {
		      // Can't leave the left compressed.
		      // Only fill in the data if you have to.
		      bool c22 = lhsDomain.containsAllPoints(lo);
		      bool c32 = OperatorTraits<Op>::IsAssign;
		      bool filldom = ((!c22) || (!c32));
		      ASSIGNMSG(msg << "Must uncompress, filldom = ");
		      ASSIGNMSG(msg << filldom << endl);
		      lf.Uncompress(filldom);

		      // Types for the assignment.
		      typedef typename LField<T1,D1>::iterator LFI;
		      typedef typename LField<T2,D1>::iterator RFI;
		      typedef BrickExpression<D1,LFI,RFI,Op> Expr;

		      // Build iterators for the left and right hand sides.
		      RFI rhs_i =
			rf.begin(rhsDomain).permute(rhsDomain,lhsDomain) ;
		      LFI lhs_i =
			lf.begin(lhsDomain) ;

		      // And do the assignment.
		      ASSIGNMSG(msg << "Doing full loop assignment." << endl);
		      Expr(lhs_i,rhs_i,op).apply();
		    }
		}
	    }
	}
    }
}


//////////////////////////////////////////////////////////////////////
//
// Receive in needed messages and assign them to our storage
// for an IBF = IBF assign.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D1, unsigned D2,
         class Op, class Container>
inline void
IndexedReceive(IndexedBareField<T1,D1,D1>& ilhs,
               IndexedBareField<T2,D2,D2>& ,
	       Op& op,
	       Container& recv_ac, int msgnum,
	       int tag)
{ 

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("IndexedReceive", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Receiving messages for IBF[" << ilhs.getDomain());
  ASSIGNMSG(msg << "] = IBF[" << irhs.getDomain() << "] ..." << endl);

  // Get the BareField for the left hand side.
  BareField<T1,D1> &lhs = ilhs.getBareField();

  // Build iterators within local fields on the left and right hand side.
  typedef typename LField<T1,D1>::iterator LFI;
  typedef CompressedBrickIterator<T2,D1> RFI;

  // The type for the expression.
  typedef BrickExpression<D1,LFI,RFI,Op> Expr;

  // Loop until we've received all the messages.
  while (msgnum>0) {
    // Receive the next message.
    int any_node = COMM_ANY_NODE;
    Message *mess = Ippl::Comm->receive_block(any_node,tag);
    PAssert(mess != 0);
    --msgnum;
    // determine the number of domains being sent
    int ndoms = mess->size() / (D1 + 3);
    for (int idom=0; idom<ndoms; ++idom) {
      // extract the next domain from the message
      NDIndex<D1> domain;
      domain.getMessage(*mess);
      // Extract the rhs iterator from it.
      T2 compressed_data;
      RFI rhs_i(compressed_data);
      rhs_i.getMessage(*mess);

      ASSIGNMSG(msg << "Received IndexedField data for domain " << domain);
      ASSIGNMSG(msg << endl);

      // Find the LField it is destined for.
      typename Container::iterator hit = recv_ac.find( domain );
      PAssert( hit != recv_ac.end() );
      LField<T1,D1> &lf = *(*hit).second;

      // Can we compress the left?
      bool c1 = rhs_i.IsCompressed();
      bool c2 = domain.containsAllPoints( lf.getAllocated() );
      bool c3 = OperatorTraits<Op>::IsAssign;
      bool c4 = lf.IsCompressed();
      if ( lhs.compressible() && c1 && c2 && ( c3 || c4 ) ) {
	// We can compress the left!
	lf.Compress();
	// Set the constant value.
	PETE_apply(op, *lf.begin() , *rhs_i);
      }
      else {
	// Can't leave the left compressed.
	// Only fill in the data if you have to.
        c2 = domain.containsAllPoints(lf.getOwned());
	c3 = OperatorTraits<Op>::IsAssign;
	lf.Uncompress( !(c2&&c3) );
	// Build iterators for the left and right hand sides.
	LFI lhs_i = lf.begin(domain);
	// And do the assignment.
	Expr(lhs_i,rhs_i,op).apply();
      }

      // Take that entry out of the receive list.
      recv_ac.erase( hit );
    }
    delete mess;
  }  // loop over messages to receive
  return;
}


//////////////////////////////////////////////////////////////////////
//
// A simple utility struct used to do touching calculations between
// domains of different dimensions.  They only can possibly touch
// if their dimensions are the same.
//
//////////////////////////////////////////////////////////////////////

template<unsigned int D1, unsigned int D2>
struct AssignTouches
{
  static bool apply(const NDIndex<D1>&, const NDIndex<D2>&)
  {
    return false;
  }
};

template<unsigned int D1>
struct AssignTouches< D1, D1 >
{
  static bool apply(const NDIndex<D1>& x,const NDIndex<D1>& y)
  {
    return x.touches(y);
  }
};


//////////////////////////////////////////////////////////////////////
//
// Assign one IndexedBareField to another.  This works even if the two
// are on different layouts and are different dimensionalities.
//
//////////////////////////////////////////////////////////////////////

template<class T1, unsigned D1, class RHS, class Op>
void
assign(IndexedBareField<T1,D1,D1> lhs,
       RHS rhsp,
       Op op, ExprTag<false>)
{

  typedef typename RHS::PETE_Return_t T2;
  enum { D2=RHS::Dim_u };
  IndexedBareField<T2,D2,D2> rhs ( rhsp.getBareField()[ rhsp.getDomain() ] );

  // Make sure we aren't assigning into overlapping domains in the same array.
  // First test to see if they are the same array.
  if ( lhs.getBareField().get_Id() == rhs.getBareField().get_Id() )
    {
      // Now check to see if the domains overlap.
      if ( AssignTouches<D1,D2>::apply(lhs.getDomain(),rhs.getDomain()))
	{
	  // They overlap.  Scream and die.
	  ERRORMSG("Overlapping domains in indexed assignment!"<<endl);
	  PAssert(0);
	}
    }
  
  // Get a unique tag for the messages here.
  int tag = Ippl::Comm->next_tag( F_GEN_ASSIGN_TAG , F_TAG_CYCLE );

  // Fill guard cells if necessary.

  for_each(rhs.begin(), FillGCIfNecessary(lhs.getBareField()), 
     PETE_NullCombiner());

  // ----------------------------------------
  // Send all the data from the right hand side
  // the the parts of the left hand side that need them.
  if (Ippl::getNodes() > 1) {
    
    IndexedSend(lhs,rhs,tag);
    
  }

  // ----------------------------------------
  // Build a map of the messages we expect to receive.
  std::multimap< NDIndex<D1> , LField<T1,D1>* , std::less< NDIndex<D1> > >  recv_ac;
  int msgnum = 0;
  if (Ippl::getNodes() > 1) {
    
    CalcIndexedReceive(lhs,rhs,recv_ac,msgnum);
    
  }

  // ----------------------------------------
  // Handle the local fills.
  
  IndexedLocalAssign(lhs,rhs,op);
  

  // ----------------------------------------
  // Receive all the messages.
  if (Ippl::getNodes() > 1) {
    
    IndexedReceive(lhs,rhs,op,recv_ac,msgnum,tag);
    
  }

  lhs.getBareField().setDirtyFlag();

  // Update the guard cells.
  
  lhs.getBareField().fillGuardCellsIfNotDirty();
  

  // Compress the LHS.
  lhs.getBareField().Compress();

  //INCIPPLSTAT(incExpressions);
  //INCIPPLSTAT(incIBFEqualsIBF);
}

/***************************************************************************
 * $RCSfile: AssignGeneralIBF.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: AssignGeneralIBF.cpp,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
