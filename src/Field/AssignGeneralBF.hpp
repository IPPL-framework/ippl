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
// general BareField = BareField expressions.  It will perform
// general communication to exchange data if the LHS and RHS layouts
// do not match.
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
// Assign one BareField to another.
// Unlike the above, this works even if the two BareFields are 
// on different layouts.
//
//////////////////////////////////////////////////////////////////////

template<class T1, unsigned Dim, class RHS, class Op>
void
assign(const BareField<T1,Dim>& clhs, RHS rhsp, Op op, ExprTag<false>)
{

  // debugging output macros.  these are only enabled if DEBUG_ASSIGN is
  // defined.
  ASSIGNMSG(Inform msg("NEW assign BF(f)", INFORM_ALL_NODES));
  ASSIGNMSG(msg << "Computing general assignment to BF[" << clhs.getDomain());
  ASSIGNMSG(msg << "] ..." << endl);

  // cast away const here for lhs ... unfortunate but necessary.
  // also, retrieve the BareField from the rhs iterator.  We know we can
  // do this since this is the ExprTag<false> specialization, only called if
  // the rhs is also a BareField.
  BareField<T1,Dim>& lhs = (BareField<T1,Dim>&) clhs ;
  typedef typename RHS::PETE_Return_t T2;
  const BareField<T2,Dim>& rhs( rhsp.GetBareField() );

  // Build iterators within local fields on the left and right hand side.
  typedef typename LField<T1,Dim>::iterator LFI;
  typedef typename LField<T2,Dim>::iterator RFI;
  T1 lhs_compressed_data;
  T2 rhs_compressed_data;
  LFI lhs_i(lhs_compressed_data);
  RFI rhs_i(rhs_compressed_data);

  // Build an iterator for the local fields on the left and right hand side.
  typename BareField<T1,Dim>::iterator_if lf_i, lf_e = lhs.end_if();
  typename BareField<T2,Dim>::const_iterator_if rf_i, rf_e = rhs.end_if();

  // Get message tag
  int tag = Ippl::Comm->next_tag( F_GEN_ASSIGN_TAG , F_TAG_CYCLE );
  int remaining = 0;  // counter for messages to receive

  // Build a map of the messages we expect to receive.
  typedef std::multimap< NDIndex<Dim> , LField<T1,Dim>* , std::less<NDIndex<Dim> > >
    ac_recv_type;
  ac_recv_type recv_ac;

  // ----------------------------------------
  // First the send loop.
  // Loop over all the local nodes of the right hand side and
  // send data to the remote ones they overlap on the left hand side.
  int nprocs = Ippl::getNodes();
  ASSIGNMSG(msg << "Processing send-loop for " << nprocs << " nodes." << endl);
  if (nprocs > 1) {

    // set up messages to be sent
    
    Message** mess = new Message*[nprocs];
    bool* recvmsg = new bool[nprocs]; // receive msg from this node?
    int iproc;
    for (iproc=0; iproc<nprocs; ++iproc) {
      mess[iproc] = 0;
      recvmsg[iproc] = false;
    }
    

    // now loop over LFields of lhs, building receive list
    
    for (lf_i = lhs.begin_if(); lf_i != lf_e; ++lf_i) {
      // Cache some information about this LField.
      LField<T1,Dim> &lf = *((*lf_i).second);
      const NDIndex<Dim> &la = lf.getAllocated();

      // Find the remote ones that have owned cells that touch this.
      typename FieldLayout<Dim>::touch_range_dv
	range( rhs.getLayout().touch_range_rdv(la) );
      typename FieldLayout<Dim>::touch_iterator_dv rv_i;
      for (rv_i = range.first; rv_i != range.second; ++rv_i) {
	// Save the intersection and the LField it is for.
	// It is a multimap so we can't use operator[].
        NDIndex<Dim> sub = la.intersect((*rv_i).first);
	typedef typename ac_recv_type::value_type value_type;
	recv_ac.insert( value_type(sub,&lf) );

        // Note who will be sending this data
        int rnode = (*rv_i).second->getNode();
        recvmsg[rnode] = true;
      }
    }
    

    // now loop over LFields of rhs, packing overlaps into proper messages
    
    for (rf_i = rhs.begin_if(); rf_i != rf_e; ++rf_i) {
      // Cache some information about this local array.
      LField<T2,Dim> &rf = *((*rf_i).second);
      const NDIndex<Dim>& ro = rf.getOwned();

      // Loop over the local ones that have allocated cells that this
      // remote one touches.
      typename FieldLayout<Dim>::touch_range_dv
	range( lhs.getLayout().touch_range_rdv(ro,lhs.getGuardCellSizes()) );
      typename FieldLayout<Dim>::touch_iterator_dv remote_i;
      for (remote_i = range.first; remote_i != range.second; ++remote_i) {
	// Find the intersection.
	NDIndex<Dim> intersection = ro.intersect( (*remote_i).first );

        // Find out who owns this domain
        int rnode = (*remote_i).second->getNode();

	// Construct an iterator for use in sending out the data
	rhs_i = rf.begin(intersection, rhs_compressed_data);
	rhs_i.TryCompress();

	// Put intersection domain and field data into message
	if (mess[rnode] == 0)
	  mess[rnode] = new Message;
	intersection.putMessage(*mess[rnode]);
	rhs_i.putMessage(*mess[rnode]);
      }
    }
    

    // tally number of messages to receive
    
    for (iproc=0; iproc<nprocs; ++iproc)
      if (recvmsg[iproc]) ++remaining;
    delete [] recvmsg;
    

    // send the messages
    
    for (iproc=0; iproc<nprocs; ++iproc) {
      if (mess[iproc] != 0)
        Ippl::Comm->send(mess[iproc],iproc,tag);
    }
    delete [] mess;
    
  }

  // ----------------------------------------
  // Handle the local fills.
  // Loop over all the local Fields of the lhs and all the local
  // fields in the rhs.
  // This is an N*N operation, but the expectation is that there won't
  // be TOO many Vnodes on a given processor.
  ASSIGNMSG(msg << "Doing local fills for " << lhs.size_if());
  ASSIGNMSG(msg << " local lhs blocks and ");
  ASSIGNMSG(msg << rhs.size_if() << " local rhs blocks." << endl);
  
  for (lf_i = lhs.begin_if(); lf_i != lf_e; ++lf_i) {
    // Cache some information about this LField.
    LField<T1,Dim> &lf = *(*lf_i).second;
    const NDIndex<Dim> &lo = lf.getOwned();
    const NDIndex<Dim> &la = lf.getAllocated();

    ASSIGNMSG(msg << "----------------" << endl);
    ASSIGNMSG(msg << "Assigning to local LField with owned = " << lo);
    ASSIGNMSG(msg << ", allocated = " << la << endl);

    // Loop over the ones it touches on the rhs.
    for (rf_i = rhs.begin_if(); rf_i != rf_e; ++rf_i) {
      // Cache some info about this LField.
      LField<T2,Dim> &rf = *(*rf_i).second;
      const NDIndex<Dim> &ro = rf.getOwned();

      // If the remote has info we want, then get it.
      if (la.touches(ro)) {
	ASSIGNMSG(msg << "Computing assignment of portion of rhs " << ro);
        ASSIGNMSG(msg << " to lhs " << la << endl);

	// Can we compress the left?
	// End point "contains" works here since ro has unit stride.
	bool c1 = rf.IsCompressed();
	bool c2 = lf.IsCompressed();
	bool c3 = ro.contains(lo);
	ASSIGNMSG(msg << "Checking for possible compressed-assign:");
	ASSIGNMSG(msg << "\n  rf.IsCompressed = " << c1);
	ASSIGNMSG(msg << "\n  lf.IsCompressed = " << c2);
	ASSIGNMSG(msg << "\n  ro.contains(lo) = " << c3);
	ASSIGNMSG(msg << endl);

	// If these are compressed we might not have to do any work.
	if (c1 && c2 && c3) {
	  ASSIGNMSG(msg << "LHS, RHS both compressed, and rhs contains lhs, ");
	  ASSIGNMSG(msg << "compress." << endl);
	  PETE_apply(op,*lf.begin(),*rf.begin());
	  ASSIGNMSG(msg << "Now " << *lf.begin() << " == " << *rf.begin());
	  ASSIGNMSG(msg << endl);
	} else {
	  // Find the intersection.
	  NDIndex<Dim> intersection = la.intersect(ro);
	  ASSIGNMSG(msg << "Intersection is " << intersection << endl);

	  // Build an iterator for the rhs.
	  RFI rhs_i2 = rf.begin(intersection);

	  // Could we compress that rhs iterator, and if so,
	  // Are we assigning the whole LField on the left?
	  // If both of these are true, we can compress the whole thing.
	  // Otherwise, we have to uncompress the LHS and do a full assign.
	  if (rhs_i2.CanCompress(*rf.begin(intersection)) &&
	      lhs.compressible() && intersection.containsAllPoints(la) &&
	      OperatorTraits<Op>::IsAssign) {

	    // Compress the whole LField to the value on the right:
	    ASSIGNMSG(msg << "LHS BF is compressible, rhs_i2 compressed, ");
	    ASSIGNMSG(msg << "intersection contains ");
	    ASSIGNMSG(msg << la << ", assignment ==> compress assign.");
	    ASSIGNMSG(msg << endl);
	    lf.Compress((T1)(*rf.begin(intersection)));
	    ASSIGNMSG(msg << "Now " << *lf.begin() << " == ");
	    ASSIGNMSG(msg << *rf.begin(intersection) << endl);

	  } else {
	    // Assigning only part of LField on the left.
	    // Must uncompress lhs, if not already uncompressed
	    // If the argument is true, we are not assigning to the whole
	    // allocated domain, and thus must fill in the uncompressed
	    // storage with the compressed value.  If it is false, then
	    // we're assigning to the whole allocated domain, so we don't
	    // have to fill (it would just all get overwritten in the
	    // BrickExpression::apply).
	    ASSIGNMSG(msg << "Cannot do compressed assign, so do loop."<<endl);
	    ASSIGNMSG(msg << "First uncompress LHS LF ..." << endl);
	    lf.Uncompress(!intersection.containsAllPoints(la));

	    // Get the iterator for it.
	    ASSIGNMSG(msg << "Get iterator for LHS ..." << endl);
	    LFI lhs_i2 = lf.begin(intersection);

	    // And do the assignment.
	    ASSIGNMSG(msg << "And do expression evaluation." << endl);
	    BrickExpression<Dim,LFI,RFI,Op>(lhs_i2,rhs_i2,op).apply();
	  }
	}
      }
    }
  }
  

  // ----------------------------------------
  // Receive all the messages.
  ASSIGNMSG(msg << "Processing receive-loop for " << nprocs<<" nodes."<<endl);
  if (nprocs > 1) {
    
    while (remaining>0) {
      // Receive the next message.
      int any_node = COMM_ANY_NODE;
      Message *rmess = Ippl::Comm->receive_block(any_node,tag);
      PAssert(rmess != 0);
      --remaining;

      // Determine the number of domains being sent
      int ndoms = rmess->size() / (Dim+3);
      for (int idom=0; idom<ndoms; ++idom) {
        // extract the next domain from the message
        NDIndex<Dim> intersection;
        intersection.getMessage(*rmess);

        // Extract the rhs iterator from it.
        T2 rhs_compressed_data2;
        RFI rhs_i2(rhs_compressed_data2);
        rhs_i2.getMessage(*rmess);

        // Find the LField it is destined for.
        typename ac_recv_type::iterator hit = recv_ac.find( intersection );
        PAssert( hit != recv_ac.end() );

        // Build the lhs brick iterator.
        LField<T1,Dim> &lf = *(*hit).second;
	const NDIndex<Dim> &lo = lf.getOwned();

        // Check and see if we really have to do this.
        if ( !(rhs_i2.IsCompressed() && lf.IsCompressed() &&
	     (*rhs_i2 == *lf.begin())) )
	{
	  // Yep. gotta do it.
	  // Only fill in the data if you have to.
	  bool c2 = intersection.containsAllPoints(lo);
	  bool c3 = OperatorTraits<Op>::IsAssign;
	  lf.Uncompress( !(c2&&c3) );
	  LFI lhs_i2 = lf.begin(intersection);

	  // Do the assignment.
	  BrickExpression<Dim,LFI,RFI,Op>(lhs_i2,rhs_i2,op).apply();
	}

        // Take that entry out of the receive list.
        recv_ac.erase( hit );
      }
      delete rmess;
    }
    
  }

  // Update the guard cells.
  ASSIGNMSG(msg << "Filling GC's at end if necessary ..." << endl);
  
  lhs.setDirtyFlag();
  lhs.fillGuardCellsIfNotDirty();
  

  // Compress the LHS.
  ASSIGNMSG(msg << "Trying to compress BareField at end ..." << endl);
  lhs.Compress();

  //INCIPPLSTAT(incExpressions);
  //INCIPPLSTAT(incBFEqualsBF);
}

/***************************************************************************
 * $RCSfile: AssignGeneralBF.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: AssignGeneralBF.cpp,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
