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
#include "Field/BareField.h"
#include "Field/BrickExpression.h"
#include "FieldLayout/FieldLayout.h"
#include "Message/Communicate.h"
#include "Message/GlobalComm.h"
#include "Message/Tags.h"
#include "Utility/Inform.h"
#include "Utility/Unique.h"
#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"

#include <map>
#include <utility>
#include <cstdlib>


//////////////////////////////////////////////////////////////////////
//
// Copy ctor.
//

template< class T, unsigned Dim >
BareField<T,Dim>::BareField(const BareField<T,Dim>& a)
: Layout( a.Layout ),		// Copy the layout.
  Gc( a.Gc ),			// Copy the number of guard cells.
  compressible_m( a.compressible_m ),
  pinned(false) //UL: for pinned memory allocation
{
  
  

  // We assume the guard cells are peachy so clear the dirty flag.
  clearDirtyFlag();

  // Reserve space for the pointers to the LFields.
  Locals_ac.reserve( a.size_if() );
  // Loop over the LFields of the other array, creating LFields as we go.
  for ( const_iterator_if a_i = a.begin_if(); a_i != a.end_if(); ++a_i )
    {
      // Create an LField with copy ctor.
      LField<T,Dim> *lf = new LField<T,Dim>( *((*a_i).second) );
      // Insert in the local list, hinting that it should go at the end.
      Locals_ac.insert( Locals_ac.end(),
			typename ac_id_larray::value_type((*a_i).first,lf) );
    }
  // Tell the layout we are here.
  getLayout().checkin( *this , Gc );
  //INCIPPLSTAT(incBareFields);
}


//////////////////////////////////////////////////////////////////////
//
// Destructor
//

template< class T, unsigned Dim >
BareField<T,Dim>::~BareField() {
  
  
  // must check out from our layout
  if (Layout != 0) {
    Layout->checkout(*this);
  }
}


//////////////////////////////////////////////////////////////////////
// Initialize the field, if it was constructed from the default constructor.
// This should NOT be called if the field was constructed by providing
// a FieldLayout.

template< class T, unsigned Dim >
void
BareField<T,Dim>::initialize(Layout_t & l) {
  
  

  // if our Layout has been previously set, we just ignore this request
  if (Layout == 0) {
    Layout = &l;
    setup();
  }
}

//UL: for pinned memory allocation
template< class T, unsigned Dim >
void
BareField<T,Dim>::initialize(Layout_t & l, const bool p) {
  
  

  // if our Layout has been previously set, we just ignore this request
  if (Layout == 0) {
    Layout = &l;
    pinned = p;
    setup();
  }
}

template< class T, unsigned Dim >
void
BareField<T,Dim>::initialize(Layout_t & l,
			     const GuardCellSizes<Dim>& gc) {
  
  

  // if our Layout has been previously set, we just ignore this request
  if (Layout == 0) {
    Layout = &l;
    Gc = gc;
    setup();
  }
}


//////////////////////////////////////////////////////////////////////
//
// Using the data that has been initialized by the ctors,
// complete the construction by allocating the LFields.
//

template< class T, unsigned Dim >
void
BareField<T,Dim>::setup()
{
  
  

  // Make sure this FieldLayout can handle the number of GuardCells
  // which we have here
  if ( ! getLayout().fitsGuardCells(Gc)) {
    ERRORMSG("Cannot construct IPPL BareField:" << endl);
    ERRORMSG("  One or more vnodes is too small for the guard cells." << endl);
    ERRORMSG("  Guard cell sizes = " << Gc << endl);
    ERRORMSG("  FieldLayout = " << getLayout() << endl);
    Ippl::abort();
  }

  // We assume the guard cells are peachy so clear the dirty flag.
  clearDirtyFlag();

  // Reserve space for the pointers to the LFields.
  Locals_ac.reserve( getLayout().size_iv() );

  // Loop over all the Vnodes, creating an LField in each.
  for (typename Layout_t::iterator_iv v_i=getLayout().begin_iv();
       v_i != getLayout().end_iv();
       ++v_i)
    {
      // Get the owned and guarded sizes.
      const NDIndex<Dim> &owned = (*v_i).second->getDomain();
      NDIndex<Dim> guarded = AddGuardCells( owned , Gc );

      // Get the global vnode number (ID number, value from 0 to nvnodes-1):
      int vnode = (*v_i).second->getVnode();

      // Construct the LField for this Vnode.
      //UL: for pinned memory allocation
      LField<T, Dim> *lf;
      if (pinned)
	lf = new LField<T,Dim>( owned, guarded, vnode, pinned );
      else
	lf = new LField<T,Dim>( owned, guarded, vnode );

      // Put it in the list.
      Locals_ac.insert( end_if(), 
                        typename ac_id_larray::value_type((*v_i).first,lf));
    }
  // Tell the layout we are here.
  getLayout().checkin( *this , Gc );
  //INCIPPLSTAT(incBareFields);
}

//////////////////////////////////////////////////////////////////////

//
// Print a BareField out.
//

template< class T, unsigned Dim>
void 
BareField<T,Dim>::write(std::ostream& out)
{
  
  

  // Inform dbgmsg(">>>>>>>> BareField::write", INFORM_ALL_NODES);
  // dbgmsg << "Printing values for field at address = " << &(*this) << endl;

  // on remote nodes, we must send the subnodes LField's to node 0
  int tag = Ippl::Comm->next_tag(F_WRITE_TAG, F_TAG_CYCLE);
  if (Ippl::myNode() != 0) {
    for ( iterator_if local = begin_if(); local != end_if() ; ++local) {
      // Cache some information about this local field.
      LField<T,Dim>&  l = *((*local).second);
      NDIndex<Dim>&  lo = (NDIndex<Dim>&) l.getOwned();
      typename LField<T,Dim>::iterator rhs(l.begin());

      // Build and send a message containing the owned LocalField data
      if (Ippl::myNode() != 0) {
	Message *mess = new Message();
	lo.putMessage(*mess);	      // send the local domain of the LField
	rhs.putMessage(*mess);          // send the data itself
	// dbgmsg << "Sending domain " << lo << " to node 0" << endl;
	Ippl::Comm->send(mess, 0, tag);
      }
    }
  } else {    // now, on node 0, receive the remaining LField's ...
    // put all the LField's in a big, uncompressed LField
    LField<T,Dim> data(getDomain(), getDomain());
    data.Uncompress();

    // first put in our local ones
    for ( iterator_if local = begin_if(); local != end_if() ; ++local) {
      // Cache some information about this local field.
      LField<T,Dim>&  l = *((*local).second);
      NDIndex<Dim>&  lo = (NDIndex<Dim>&) l.getOwned();
      typename LField<T,Dim>::iterator rhs(l.begin());
      
      // put the local LField in our big LField
      // dbgmsg << "  Copying local domain " << lo << " from Lfield at ";
      // dbgmsg << &l << ":" << endl;
      typename LField<T,Dim>::iterator putloc = data.begin(lo);
      typename LField<T,Dim>::iterator getloc = l.begin(lo);
      for ( ; getloc != l.end() ; ++putloc, ++getloc ) {
	// dbgmsg << "    from " << &(*getloc) << " to " << &(*putloc);
	// dbgmsg << ": " << *putloc << " = " << *getloc << endl;
	*putloc = *getloc;
      }
    }

    // we expect to receive one message from each remote vnode
    int remaining = getLayout().size_rdv();

    // keep receiving messages until they're all here
    for ( ; remaining > 0; --remaining) {
      // Receive the generic message.
      int any_node = COMM_ANY_NODE;
      Message *mess = Ippl::Comm->receive_block(any_node, tag);

      // Extract the domain size and LField iterator from the message
      NDIndex<Dim> lo;
      T compressed_data;
      typename LField<T,Dim>::iterator rhs(compressed_data);
      lo.getMessage(*mess);
      rhs.getMessage(*mess);
      // dbgmsg << "Received domain " << lo << " from " << any_node << endl;

      // put the received LField in our big LField
      typename LField<T,Dim>::iterator putloc = data.begin(lo);
      for (unsigned elems=lo.size(); elems > 0; ++putloc, ++rhs, --elems)
	*putloc = *rhs;

      // Free the memory in the message.
      delete mess;
    }

    // finally, we can print the big LField out
    out << data;
  }
}


//////////////////////////////////////////////////////////////////////

//
// Fill all the guard cells, including all the necessary communication.
//

template< class T, unsigned Dim >
void BareField<T,Dim>::fillGuardCells(bool reallyFill) const
{

  // This operation is logically const because the physical cells
  // of the BareField are unaffected, so cast away const here.
  BareField<T,Dim>& ncf = const_cast<BareField<T,Dim>&>(*this);

  // Only fill if we are supposed to.
  if (!reallyFill)
    return;

  // Make ourselves clean.
  ncf.clearDirtyFlag();

  // Only need to do work if we have non-zero GuardCellSizes
  if (Gc == GuardCellSizes<Dim>())
    return;

  // indicate we're doing another gc fill
  //INCIPPLSTAT(incGuardCellFills);

  // iterators for looping through LField's of this BareField
  iterator_if lf_i, lf_e = ncf.end_if();

  // ----------------------------------------
  // First the send loop.
  // Loop over all the local nodes and
  // send data to the remote ones they overlap.
  int nprocs = Ippl::getNodes();
  if (nprocs > 1) {
    
    // Build a map of the messages we expect to receive.
    typedef std::multimap< NDIndex<Dim> , LField<T,Dim>* , std::less<NDIndex<Dim> > > ac_recv_type;
    ac_recv_type recv_ac;
    bool* recvmsg = new bool[nprocs];
    

    
    // set up messages to be sent
    Message** mess = new Message*[nprocs];
    int iproc;
    for (iproc=0; iproc<nprocs; ++iproc) {
      mess[iproc] = NULL;
      recvmsg[iproc] = false;
    }
    
    // now do main loop over LFields, packing overlaps into proper messages
    for (lf_i = ncf.begin_if(); lf_i != lf_e; ++lf_i) {
      
      // Cache some information about this local array.
      LField<T,Dim> &lf = *((*lf_i).second);
      const NDIndex<Dim> &lf_domain = lf.getAllocated();
      // Find the remote ones that touch this LField's guard cells
      typename Layout_t::touch_range_dv
	rrange(ncf.getLayout().touch_range_rdv(lf_domain));
      typename Layout_t::touch_iterator_dv rv_i;
      for (rv_i = rrange.first; rv_i != rrange.second; ++rv_i) {
	// Save the intersection and the LField it is for.
	NDIndex<Dim> sub = lf_domain.intersect( (*rv_i).first );
	typedef typename ac_recv_type::value_type value_type;
	recv_ac.insert(value_type(sub,&lf));
        // note who will be sending this data
        int rnode = (*rv_i).second->getNode();
        recvmsg[rnode] = true;
      }
      

      
      const NDIndex<Dim>& lo = lf.getOwned();
      // Loop over the remote domains which have guard cells
      // that this local domain touches.
      typename Layout_t::touch_range_dv
	range( ncf.getLayout().touch_range_rdv(lo,ncf.getGC()) );
      typename Layout_t::touch_iterator_dv remote_i;
      for (remote_i = range.first; remote_i != range.second; ++remote_i) {
	// Find the intersection.
	NDIndex<Dim> intersection = lo.intersect( (*remote_i).first );
        // Find out who owns this remote domain.
        int rnode = (*remote_i).second->getNode();
	// Create an LField iterator for this intersection region,
	// and try to compress it.
        // storage for LField compression
        T compressed_value;
	LFI msgval = lf.begin(intersection, compressed_value);
	msgval.TryCompress();

	// Put intersection domain and field data into message
        if (!mess[rnode]) mess[rnode] = new Message;
        PAssert(mess[rnode]);
	mess[rnode]->put(intersection); // puts Dim items in Message
	mess[rnode]->put(msgval);       // puts 3 items in Message
      }
    
    }
    
    int remaining = 0;
    for (iproc=0; iproc<nprocs; ++iproc)
      if (recvmsg[iproc]) ++remaining;
    delete [] recvmsg;
    

    
    // Get message tag.
    int tag = Ippl::Comm->next_tag( F_GUARD_CELLS_TAG , F_TAG_CYCLE );
    
    // Send all the messages.
    for (iproc=0; iproc<nprocs; ++iproc) {
      if (mess[iproc]) {
        Ippl::Comm->send(mess[iproc], iproc, tag);
      }
    }
    
    delete [] mess;

    // ----------------------------------------
    // Handle the local fills.
    // Loop over all the local arrays.
    
    for (lf_i = ncf.begin_if(); lf_i != lf_e; ++lf_i)
    {
      // Cache some information about this LField.
      LField<T,Dim> &lf = *(*lf_i).second;
      const NDIndex<Dim>& la = lf.getAllocated();

      // Loop over all the other LFields to establish each LField's
      // cache of overlaps.
	
      // This is an N*N operation, but the expectation is that this will
      // pay off since we will reuse this cache quite often.

      if (!lf.OverlapCacheInitialized())
      {
        for (iterator_if rf_i = ncf.begin_if(); rf_i != ncf.end_if(); ++rf_i)
	  if ( rf_i != lf_i )
	  {
	    // Cache some info about this LField.
	    LField<T,Dim> &rf = *(*rf_i).second;
	    const NDIndex<Dim>& ro = rf.getOwned();

	    // If the remote has info we want, then add it to our cache.
	    if ( la.touches(ro) )
	      lf.AddToOverlapCache(&rf);
	  }
      }

      // We know at this point that the overlap cache is established.
      // Use it.

      for (typename LField<T,Dim>::OverlapIterator rf_i = lf.BeginOverlap();
	   rf_i != lf.EndOverlap(); ++rf_i)
      {
        LField<T, Dim> &rf = *(*rf_i);
        const NDIndex<Dim>& ro = rf.getOwned();

        bool c1 = lf.IsCompressed();
        bool c2 = rf.IsCompressed();
        bool c3 = *rf.begin() == *lf.begin();

        // If these are compressed we might not have to do any work.
        if ( !( c1 && c2 && c3 ) )
        {
	  

	  // Find the intersection.
	  NDIndex<Dim> intersection = la.intersect(ro);
		
	  // Build an iterator for the rhs.
	  LFI rhs = rf.begin(intersection);
		
	  // Could we compress that?
	  if ( !(c1 && rhs.CanCompress(*lf.begin())) )
	  {
	    // Make sure the left is not compressed.
	    lf.Uncompress();
		  
	    // Nope, we really have to copy.
	    LFI lhs = lf.begin(intersection);
	    // And do the assignment.
	    BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
	  }
	      
	  
        }
      }
    }
    

    // ----------------------------------------
    // Receive all the messages.
    
    while (remaining>0) {
      // Receive the next message.
      int any_node = COMM_ANY_NODE;
      
      Message* rmess = Ippl::Comm->receive_block(any_node,tag);
      PAssert(rmess);
      --remaining;
      

      // Determine the number of domains being sent
      int ndoms = rmess->size() / (Dim + 3);
      for (int idom=0; idom<ndoms; ++idom) {
        // Extract the intersection domain from the message.
        NDIndex<Dim> intersection;
        intersection.getMessage(*rmess);

        // Extract the rhs iterator from it.
        T compressed_value;
        LFI rhs(compressed_value);
        rhs.getMessage(*rmess);

        // Find the LField it is destined for.
        typename ac_recv_type::iterator hit = recv_ac.find( intersection );
        PAssert( hit != recv_ac.end() );
        // Build the lhs brick iterator.
        LField<T,Dim> &lf = *(*hit).second;
        // Check and see if we really have to do this.
        if ( !(rhs.IsCompressed() && lf.IsCompressed() &&
             (*rhs == *lf.begin())) )
	{
	  // Yep. gotta do it.
	  lf.Uncompress();
	  LFI lhs = lf.begin(intersection);
	  // Do the assignment.
	  BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
	}

        // Take that entry out of the receive list.
        recv_ac.erase( hit );
      }
      delete rmess;
    }
    
  }
  else { // single-node case
    // ----------------------------------------
    // Handle the local fills.
    // Loop over all the local arrays.
    
    for (lf_i = ncf.begin_if(); lf_i != lf_e; ++lf_i)
    {
      // Cache some information about this LField.
      LField<T,Dim> &lf = *(*lf_i).second;
      const NDIndex<Dim>& la = lf.getAllocated();

      // Loop over all the other LFields to establish each LField's
      // cache of overlaps.
	
      // This is an N*N operation, but the expectation is that this will
      // pay off since we will reuse this cache quite often.

      if (!lf.OverlapCacheInitialized())
      {
        for (iterator_if rf_i = ncf.begin_if(); rf_i != ncf.end_if(); ++rf_i)
	  if ( rf_i != lf_i )
	  {
	    // Cache some info about this LField.
	    LField<T,Dim> &rf = *(*rf_i).second;
	    const NDIndex<Dim>& ro = rf.getOwned();

	    // If the remote has info we want, then add it to our cache.
	    if ( la.touches(ro) )
	      lf.AddToOverlapCache(&rf);
	  }
      }

      // We know at this point that the overlap cache is established.
      // Use it.

      for (typename LField<T,Dim>::OverlapIterator rf_i = lf.BeginOverlap();
	   rf_i != lf.EndOverlap(); ++rf_i)
      {
        LField<T, Dim> &rf = *(*rf_i);
        const NDIndex<Dim>& ro = rf.getOwned();

        bool c1 = lf.IsCompressed();
        bool c2 = rf.IsCompressed();
        bool c3 = *rf.begin() == *lf.begin();

        // If these are compressed we might not have to do any work.
        if ( !( c1 && c2 && c3 ) )
        {
	  

	  // Find the intersection.
	  NDIndex<Dim> intersection = la.intersect(ro);
		
	  // Build an iterator for the rhs.
	  LFI rhs = rf.begin(intersection);
		
	  // Could we compress that?
	  if ( !(c1 && rhs.CanCompress(*lf.begin())) )
	  {
	    // Make sure the left is not compressed.
	    lf.Uncompress();
		  
	    // Nope, we really have to copy.
	    LFI lhs = lf.begin(intersection);
	    // And do the assignment.
	    BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
	  }
	      
	  
        }
      }
    }
    
  }
  return;
}

//////////////////////////////////////////////////////////////////////

//
// Fill guard cells with a constant value.
// This is typically used to zero out the guard cells before a scatter.
//

template <class T, unsigned Dim>
void BareField<T,Dim>::setGuardCells(const T& val) const
{
  // profiling macros
  
  

  // if there are no guard cells, we can just return
  if (Gc == GuardCellSizes<Dim>())
    return;

  // this member function is logically const, so cast away const-ness
  BareField<T,Dim>& ncf = const_cast<BareField<T,Dim>&>(*this);

  // loop over our LFields and set guard cell values
  iterator_if lf_i, lf_e = ncf.end_if();
  for (lf_i = ncf.begin_if(); lf_i != lf_e; ++lf_i) {
    // Get this LField
    LField<T,Dim>& lf = *((*lf_i).second);

    // Quick test to see if we can avoid doing any work.
    // If this LField is compressed and already contains
    // the value that is being assigned to the guard cells,
    // then we can just move on to the next LField.
    if (lf.IsCompressed() && lf.getCompressedData() == val)
      continue;

    // OK, we really have to fill the guard cells, so get 
    // the domains we will be working with.
    const NDIndex<Dim>& adom = lf.getAllocated();
    const NDIndex<Dim>& odom = lf.getOwned();
    NDIndex<Dim> dom = odom;  // initialize working domain to Owned

    // create a compressed LField with the same allocated domain
    // containing the value to be assigned to the guard cells
    LField<T,Dim> rf(odom,adom);
    rf.Compress(val);

    // Uncompress LField to be filled and get some iterators
    lf.Uncompress();
    LFI lhs, rhs;

    // now loop over dimensions and fill guard cells along each
    for (unsigned int idim = 0; idim < Dim; ++idim) {
      // set working domain to left guards in this dimension
      dom[idim] = Index(adom[idim].first(),
                        adom[idim].first() + Gc.left(idim) - 1);
      if (!dom[idim].empty()) {
	// Set iterators over working domain and do assignment
	lhs = lf.begin(dom);
	rhs = rf.begin(dom);
	BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
      }

      // Set working domain to right guards in this dimension
      dom[idim] = Index(adom[idim].last() - Gc.right(idim) + 1,
                        adom[idim].last());
      if (!dom[idim].empty()) {
	// Set iterators over working domain and do assignment
	lhs = lf.begin(dom);
	rhs = rf.begin(dom);
	BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
      }

      // Adjust working domain along this direction 
      // in preparation for working on next dimension.
      dom[idim] = adom[idim];
    }
  }

  // let's set the dirty flag, since we have modified guard cells.
  ncf.setDirtyFlag();

  return;
}

//////////////////////////////////////////////////////////////////////

//
// Accumulate guard cells into real cells, including necessary communication.
//

template <class T, unsigned Dim>
void BareField<T,Dim>::accumGuardCells()
{

  // Only need to do work if we have non-zero GuardCellSizes
  if (Gc == GuardCellSizes<Dim>())
    return;

  // iterators for looping through LField's of this BareField
  iterator_if lf_i, lf_e = end_if();

  // ----------------------------------------
  // First the send loop.
  // Loop over all the local nodes and
  // send data to the remote ones they overlap.
  int nprocs = Ippl::getNodes();
  if (nprocs > 1) {
    
    // Build a map of the messages we expect to receive.
    typedef std::multimap< NDIndex<Dim> , LField<T,Dim>* , std::less<NDIndex<Dim> > > ac_recv_type;
    ac_recv_type recv_ac;
    bool* recvmsg = new bool[nprocs];
    

    
    // set up messages to be sent
    Message** mess = new Message*[nprocs];
    int iproc;
    for (iproc=0; iproc<nprocs; ++iproc) {
      mess[iproc] = NULL;
      recvmsg[iproc] = false;
    }
    
    // now do main loop over LFields, packing overlaps into proper messages
    for (lf_i = begin_if(); lf_i != lf_e; ++lf_i) {
      
      // Cache some information about this local array.
      LField<T,Dim> &lf = *((*lf_i).second);
      const NDIndex<Dim>& lo = lf.getOwned();
      // Find the remote ones with guard cells touching this LField
      typename Layout_t::touch_range_dv
	rrange( getLayout().touch_range_rdv(lo,Gc) );
      typename Layout_t::touch_iterator_dv rv_i;
      for (rv_i = rrange.first; rv_i != rrange.second; ++rv_i) {
	// Save the intersection and the LField it is for.
	NDIndex<Dim> sub = lo.intersect( (*rv_i).first );
	recv_ac.insert(typename ac_recv_type::value_type(sub,&lf));
        // note who will be sending this data
        int rnode = (*rv_i).second->getNode();
        recvmsg[rnode] = true;
      } 
      

      
      const NDIndex<Dim> &lf_domain = lf.getAllocated();
      // Loop over the remote domains that touch this local
      // domain's guard cells
      typename Layout_t::touch_range_dv
	range(getLayout().touch_range_rdv(lf_domain));
      typename Layout_t::touch_iterator_dv remote_i;
      for (remote_i = range.first; remote_i != range.second; ++remote_i) {
	// Find the intersection.
	NDIndex<Dim> intersection = lf_domain.intersect( (*remote_i).first );
        // Find out who owns this remote domain.
        int rnode = (*remote_i).second->getNode();
	// Create an LField iterator for this intersection region,
	// and try to compress it.
        // storage for LField compression
        T compressed_value;
	LFI msgval = lf.begin(intersection, compressed_value);
	msgval.TryCompress();

	// Put intersection domain and field data into message
        if (!mess[rnode]) mess[rnode] = new Message;
        PAssert(mess[rnode]);
	mess[rnode]->put(intersection); // puts Dim items in Message
	mess[rnode]->put(msgval);       // puts 3 items in Message
      }
    
    }
    
    int remaining = 0;
    for (iproc=0; iproc<nprocs; ++iproc)
      if (recvmsg[iproc]) ++remaining;
    delete [] recvmsg;
    

    
    // Get message tag.
    int tag = Ippl::Comm->next_tag( F_GUARD_CELLS_TAG , F_TAG_CYCLE );
    
    // Send all the messages.
    for (iproc=0; iproc<nprocs; ++iproc) {
      if (mess[iproc]) {
        Ippl::Comm->send(mess[iproc], iproc, tag);
      }
    }
    
    delete [] mess;

    // ----------------------------------------
    // Handle the local fills.
    // Loop over all the local arrays.
    
    for (lf_i=begin_if(); lf_i != lf_e; ++lf_i)
    {
      // Cache some information about this LField.
      LField<T,Dim> &lf = *(*lf_i).second;
      const NDIndex<Dim>& la = lf.getAllocated();

      // Loop over all the other LFields to establish each LField's
      // cache of overlaps.
	
      // This is an N*N operation, but the expectation is that this will
      // pay off since we will reuse this cache quite often.

      if (!lf.OverlapCacheInitialized())
      {
        for (iterator_if rf_i = begin_if(); rf_i != end_if(); ++rf_i)
	  if ( rf_i != lf_i )
	  {
	    // Cache some info about this LField.
	    LField<T,Dim> &rf = *(*rf_i).second;
	    const NDIndex<Dim>& ro = rf.getOwned();

	    // If the remote has info we want, then add it to our cache.
	    if ( la.touches(ro) )
	      lf.AddToOverlapCache(&rf);
	  }
      }

      // We know at this point that the overlap cache is established.
      // Use it.

      for (typename LField<T,Dim>::OverlapIterator rf_i = lf.BeginOverlap();
	   rf_i != lf.EndOverlap(); ++rf_i)
      {
        LField<T, Dim> &rf = *(*rf_i);
        const NDIndex<Dim>& ro = rf.getOwned();

        

        // Find the intersection.
        NDIndex<Dim> intersection = la.intersect(ro);
		
        // Build iterator for lf guard cells
        LFI lhs = lf.begin(intersection);

        // check if we can skip accumulation
        if ( !lhs.CanCompress(T()) ) {

          // Make sure the right side is not compressed.
          rf.Uncompress();
		  
          // Build iterator for rf real cells
          LFI rhs = rf.begin(intersection);

          // And do the accumulation
          BrickExpression<Dim,LFI,LFI,OpAddAssign>(rhs,lhs).apply();
	}    
        
      }
    }
    

    // ----------------------------------------
    // Receive all the messages.
    
    while (remaining>0) {
      // Receive the next message.
      int any_node = COMM_ANY_NODE;
      
      Message* rmess = Ippl::Comm->receive_block(any_node,tag);
      PAssert(rmess);
      --remaining;
      

      // Determine the number of domains being sent
      int ndoms = rmess->size() / (Dim + 3);
      for (int idom=0; idom<ndoms; ++idom) {
        // Extract the intersection domain from the message.
        NDIndex<Dim> intersection;
        intersection.getMessage(*rmess);

        // Extract the rhs iterator from it.
        T compressed_value;
        LFI rhs(compressed_value);
        rhs.getMessage(*rmess);

        // Find the LField it is destined for.
        typename ac_recv_type::iterator hit = recv_ac.find( intersection );
        PAssert( hit != recv_ac.end() );

        // Build the lhs brick iterator.
        LField<T,Dim> &lf = *(*hit).second;

	// Make sure LField is uncompressed
	lf.Uncompress();
	LFI lhs = lf.begin(intersection);

	// Do the accumulation
	BrickExpression<Dim,LFI,LFI,OpAddAssign>(lhs,rhs).apply();

        // Take that entry out of the receive list.
        recv_ac.erase( hit );
      }
      delete rmess;
    }
    
  }
  else { // single-node case
    // ----------------------------------------
    // Handle the local fills.
    // Loop over all the local arrays.
    
    for (lf_i=begin_if(); lf_i != lf_e; ++lf_i)
    {
      // Cache some information about this LField.
      LField<T,Dim> &lf = *(*lf_i).second;
      const NDIndex<Dim>& la = lf.getAllocated();

      // Loop over all the other LFields to establish each LField's
      // cache of overlaps.
	
      // This is an N*N operation, but the expectation is that this will
      // pay off since we will reuse this cache quite often.

      if (!lf.OverlapCacheInitialized())
      {
        for (iterator_if rf_i = begin_if(); rf_i != end_if(); ++rf_i)
	  if ( rf_i != lf_i )
	  {
	    // Cache some info about this LField.
	    LField<T,Dim> &rf = *(*rf_i).second;
	    const NDIndex<Dim>& ro = rf.getOwned();

	    // If the remote has info we want, then add it to our cache.
	    if ( la.touches(ro) )
	      lf.AddToOverlapCache(&rf);
	  }
      }

      // We know at this point that the overlap cache is established.
      // Use it.

      for (typename LField<T,Dim>::OverlapIterator rf_i = lf.BeginOverlap();
	   rf_i != lf.EndOverlap(); ++rf_i)
      {
        LField<T, Dim> &rf = *(*rf_i);
        const NDIndex<Dim>& ro = rf.getOwned();

        

        // Find the intersection.
        NDIndex<Dim> intersection = la.intersect(ro);
		
        // Build iterator for lf guard cells
        LFI lhs = lf.begin(intersection);

        // Check if we can skip accumulation
        if ( !lhs.CanCompress(T()) ) {
          // Make sure rf is not compressed.
          rf.Uncompress();
		  
          // Build iterator for rf real cells
          LFI rhs = rf.begin(intersection);

          // And do the accumulation
          BrickExpression<Dim,LFI,LFI,OpAddAssign>(rhs,lhs).apply();
	}    
        
      }
    }
    
  }

  // since we just modified real cell values, set dirty flag if using
  // deferred guard cell fills
  setDirtyFlag();
  // fill guard cells now, unless we are deferring
  fillGuardCellsIfNotDirty();
  // try to compress the final result
  Compress();

  return;
}

//////////////////////////////////////////////////////////////////////

//
// Tell a BareField to compress itself.
// Just loop over all of the local fields and tell them to compress.
//
template<class T, unsigned Dim>
void BareField<T,Dim>::Compress() const
{
  
  

  if (!compressible_m) return;

  // This operation is logically const, so cast away const here
  BareField<T,Dim>& ncf = const_cast<BareField<T,Dim>&>(*this);
  for (iterator_if lf_i = ncf.begin_if(); lf_i != ncf.end_if(); ++lf_i)
    (*lf_i).second->TryCompress(isDirty());
}

template<class T, unsigned Dim>
void BareField<T,Dim>::Uncompress() const
{
  
  

  // This operation is logically const, so cast away const here
  BareField<T,Dim>& ncf = const_cast<BareField<T,Dim>&>(*this);
  for (iterator_if lf_i = ncf.begin_if(); lf_i != ncf.end_if(); ++lf_i)
    (*lf_i).second->Uncompress();
}

//
// Look at all the local fields and calculate the 
// fraction of all the elements that are compressed.
//
template<class T, unsigned Dim>
double BareField<T,Dim>::CompressedFraction() const
{
  
  

  // elements[0] = total elements
  // elements[1] = compressed elements
  double elements[2] = { 0.0, 0.0};
  // Loop over all of the local fields.
  for (const_iterator_if lf_i=begin_if(); lf_i != end_if(); ++lf_i)
    {
      // Get a reference to the current local field.
      LField<T,Dim> &lf = *(*lf_i).second;
      // Get the size of this one.
      double s = lf.getOwned().size();
      // Add that up.
      elements[0] += s;
      // If it is compressed...
      if ( lf.IsCompressed() )
	// Add that amount to the compressed total.
	elements[1] += s;
    }
  // Make some space to put the global sum of each of the above.
  double reduced_elements[2] = { 0.0, 0.0};
  // Do the global reduction.
  reduce(elements,elements+2,reduced_elements,OpAddAssign());
  // Return the fraction.
  return reduced_elements[1]/reduced_elements[0];
}


//////////////////////////////////////////////////////////////////////
template<class T, unsigned Dim>
void BareField<T,Dim>::Repartition(UserList* userlist)
{
  
  

  // Cast to the proper type of FieldLayout.
  Layout_t *newLayout = (Layout_t *)( userlist );

  // Build a new temporary field on the new layout.
  BareField<T,Dim> tempField( *newLayout, Gc );

  // Copy our data over to the new layout.
  tempField = *this;

  // Copy back the pointers to the new local fields.
  Locals_ac = tempField.Locals_ac;

  //INCIPPLSTAT(incRepartitions);
}


//////////////////////////////////////////////////////////////////////
// Tell the subclass that the FieldLayout is being deleted, so
// don't use it anymore
template<class T, unsigned Dim>
void BareField<T,Dim>::notifyUserOfDelete(UserList* userlist)
{
  
  

  // just set our layout pointer to NULL; if we try to do anything more
  // with this object, other than delete it, a core dump is likely
  if (Layout != 0 && Layout->get_Id() == userlist->getUserListID()) {
    // the ID refers to this layout, so get rid of it.  It is possible
    // the ID refers to some other object, in which case we do not want
    // to cast away our Layout object.
    Layout = 0;
  } else {
    // for now, print a warning, until other types of FieldLayoutUser's
    // are set up to register with a FieldLayout ... but in general,
    // this is OK and this warning should be removed
    WARNMSG("BareField: notified of unknown UserList being deleted.");
    WARNMSG(endl);
  }
}

// a simple true-false template used to select which loop to implement
// in the BareField::localElement body
template<unsigned Dim, bool exists>
class LFieldDimTag {
};

// 1, 2, 3, and N Dimensional functions
template <class T>
inline
T* PtrOffset(T* ptr, const NDIndex<1U>& pos, const NDIndex<1U>& alloc,
             LFieldDimTag<1U,true>) {
  T* newptr = ptr + pos[0].first() - alloc[0].first();
  return newptr;
}

template <class T>
inline
T* PtrOffset(T* ptr, const NDIndex<2U>& pos, const NDIndex<2U>& alloc,
               LFieldDimTag<2U,true>) {
  T* newptr = ptr + pos[0].first() - alloc[0].first() +
         alloc[0].length() * ( pos[1].first() - alloc[1].first() );
  return newptr;
}

template <class T>
inline
T* PtrOffset(T* ptr, const NDIndex<3U>& pos, const NDIndex<3U>& alloc,
               LFieldDimTag<3U,true>) {
  T* newptr = ptr + pos[0].first() - alloc[0].first() +
         alloc[0].length() * ( pos[1].first() - alloc[1].first() +
         alloc[1].length() * ( pos[2].first() - alloc[2].first() ) );
  return newptr;
}

template <class T, unsigned Dim>
inline
T* PtrOffset(T* ptr, const NDIndex<Dim>& pos, const NDIndex<Dim>& alloc,
             LFieldDimTag<Dim,false>) {
  T* newptr = ptr;
  int n=1;
  for (unsigned int d=0; d<Dim; ++d) {
    newptr += n * (pos[d].first() - alloc[d].first());
    n *= alloc[d].length();
  }
  return newptr;
}


//////////////////////////////////////////////////////////////////////
// Get a ref to a single element of the Field; if it is not local to our
// processor, print an error and exit.  This allows the user to provide
// different index values on each node, instead of using the same element
// and broadcasting to all nodes.
template<class T, unsigned Dim> 
T& BareField<T,Dim>::localElement(const NDIndex<Dim>& Indexes) const
{
  
  
  

  // Instead of checking to see if the user has asked for one element,
  // we will just use the first element specified for each dimension.

  // Is this element here? 
  // Try and find it in the local BareFields.
  const_iterator_if lf_i   = begin_if();
  const_iterator_if lf_end = end_if();
  for ( ; lf_i != lf_end ; ++lf_i ) {
    LField<T,Dim>& lf(*(*lf_i).second);
    // End-point "contains" OK since "owned" is unit stride.
    // was before CK fix: if ( lf.getOwned().contains( Indexes ) ) {
    if ( lf.getAllocated().contains( Indexes ) ) {	
      // Found it ... first uncompress, then get a pointer to the
      // requested element.
      lf.Uncompress();
      //      return *(lf.begin(Indexes));
      // instead of building an iterator, just find the value
      NDIndex<Dim> alloc = lf.getAllocated();
      T* pdata = PtrOffset(lf.getP(), Indexes, alloc,
                           LFieldDimTag<Dim,(Dim<=3)>());
      return *pdata;
    }
  }

  // if we're here, we did not find it ... it must not be local
  ERRORMSG("BareField::localElement: attempt to access non-local index ");
  ERRORMSG(Indexes << " on node " << Ippl::myNode() << endl);
  ERRORMSG("Occurred in a BareField with layout = " << getLayout() << endl);
  ERRORMSG("Calling abort ..." << endl);
  Ippl::abort();
  return *((*((*(begin_if())).second)).begin());
}


//////////////////////////////////////////////////////////////////////
//
// Get a single value and return it in the given storage.  Whichever
// node owns the value must broadcast it to the other nodes.
//
//////////////////////////////////////////////////////////////////////
template <class T, unsigned int Dim>
void BareField<T,Dim>::getsingle(const NDIndex<Dim>& Indexes, T& r) const
{
  
  

  // Instead of checking to see if the user has asked for one element,
  // we will just use the first element specified for each dimension.

  // Check to see if the point is in the boundary conditions.
  // Check for: The point is not in the owned domain, but it is in that
  //            domain augmented with the boundary condition size.
  // If so, use a different algorithm.
  if ( (!Indexes.touches(Layout->getDomain())) &&
       Indexes.touches(AddGuardCells(Layout->getDomain(),Gc)) ) {
    getsingle_bc(Indexes,r);
    return;
  }

  // create a tag for send/receives if necessary
  int tag = Ippl::Comm->next_tag(F_GETSINGLE_TAG, F_TAG_CYCLE);

  // Is it here? Try and find it in the LFields
  const_iterator_if lf_end = end_if();
  for (const_iterator_if lf_i = begin_if(); lf_i != lf_end ; ++lf_i) {
    if ( (*lf_i).second->getOwned().contains( Indexes ) ) {
      // Found it.
      // Get a pointer to the requested element.
      LField<T,Dim>& lf( *(*lf_i).second );
      typename LField<T,Dim>::iterator lp = lf.begin(Indexes);

      // Get the requested data.
      r = *lp;

      // Broadcast it if we're running in parallel
      if (Ippl::getNodes() > 1) {
	Message *mess = new Message;
	::putMessage(*mess, r);
	Ippl::Comm->broadcast_others(mess, tag);
      }

      return;
    }
  }

  // If we're here, we didn't find it: It is remote.  Wait for a message.
  if (Ippl::getNodes() > 1) {
    int any_node = COMM_ANY_NODE;
    Message *mess = Ippl::Comm->receive_block(any_node,tag);
    ::getMessage(*mess, r);
    delete mess;
  }
  else {
    // we did not find it, and we only have one node, so this must be
    // an error.
    ERRORMSG("Illegal single-value index " << Indexes);
    ERRORMSG(" in BareField::getsingle()" << endl);
  }
}

//////////////////////////////////////////////////////////////////////

//
// Get a single value from the BareField when we know that the 
// value is in the boundary condition area.
// We use a more robust but slower algorithm here because there could 
// be redundancy.
//

template<class T, unsigned D>
void
BareField<T,D>::getsingle_bc(const NDIndex<D>& Indexes, T& r) const
{
  // We will look through everything to find who is the authority
  // for the data.  We look through the locals to see if we have it,
  // and we look through the remotes to see if someone else has it.
  // The lowest numbered processor is the authority.
  int authority_proc = -1;

  // Look through all the locals to try and find it.
  // Loop over all the LFields.
  const_iterator_if lf_end = end_if();
  for (const_iterator_if lf_i = begin_if(); lf_i != lf_end ; ++lf_i) {
    // Is it in this LField?
    if ( (*lf_i).second->getAllocated().contains( Indexes ) ) {
      // Found it.  As far as we know now, this is the authority proc.
      authority_proc = Ippl::myNode();

      // Get the requested data.
      LField<T,D>& lf( *(*lf_i).second );
      typename LField<T,D>::iterator lp = lf.begin(Indexes);
      r = *lp;

      // If we're not the only guy on the block, go on to find the remotes.
      if (Ippl::getNodes() > 1) 
	break;

      // Otherwise, we're done.
      return;
    }
  }

  // Look through all the remotes to see who else has it.
  // Loop over all the remote LFields that touch Indexes.
  typename FieldLayout<D>::touch_range_dv range = 
    Layout->touch_range_rdv(Indexes,Gc);
  for (typename FieldLayout<D>::touch_iterator_dv p=range.first; 
       p!=range.second;++p) 
    // See if anybody has a lower processor number.
    if ( authority_proc > (*p).second->getNode() )
      // Someone else is the authority.
      authority_proc = (*p).second->getNode();

  // create a tag for the broadcast.
  int tag = Ippl::Comm->next_tag(F_GETSINGLE_TAG, F_TAG_CYCLE);

  if ( authority_proc == Ippl::myNode() ) {
    // If we're the authority, broadcast.
    Message *mess = new Message;
    ::putMessage(*mess, r);
    Ippl::Comm->broadcast_others(mess, tag);
  }
  else {
    // If someone else is the authority, receive the message.
    Message *mess = Ippl::Comm->receive_block(authority_proc,tag);
    ::getMessage(*mess, r);
    delete mess;
  }
  
}

/***************************************************************************
 * $RCSfile: BareField.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: BareField.cpp,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
