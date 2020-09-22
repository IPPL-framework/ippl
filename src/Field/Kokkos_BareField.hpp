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
#include "Field/Kokkos_BareField.h"
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


template< class T, unsigned Dim >
Kokkos_BareField<T,Dim>::~Kokkos_BareField() { }


template< class T, unsigned Dim >
void
Kokkos_BareField<T,Dim>::initialize(Layout_t & l) {
  
  

  // if our Layout has been previously set, we just ignore this request
  if (Layout == 0) {
    Layout = &l;
    setup();
  }
}

//////////////////////////////////////////////////////////////////////
//
// Using the data that has been initialized by the ctors,
// complete the construction by allocating the Kokkos_LFields.
//

template< class T, unsigned Dim >
void
Kokkos_BareField<T,Dim>::setup()
{
  // Reserve space for the pointers to the Kokkos_LFields.
  Locals_ac.reserve( getLayout().size_iv() );

  // Loop over all the Vnodes, creating an Kokkos_LField in each.
  for (typename Layout_t::iterator_iv v_i=getLayout().begin_iv();
       v_i != getLayout().end_iv();
       ++v_i)
    {
      // Get the owned and guarded sizes.
      const NDIndex<Dim> &owned = (*v_i).second->getDomain();
      NDIndex<Dim> guarded; // = AddGuardCells( owned , Gc );

      // Get the global vnode number (ID number, value from 0 to nvnodes-1):
      int vnode = (*v_i).second->getVnode();

      Kokkos_LField<T, Dim> *lf;
      lf = new Kokkos_LField<T,Dim>( owned, guarded, vnode );

      // Put it in the list.
      Locals_ac.insert( end_if(), 
                        typename ac_id_larray::value_type((*v_i).first,lf));
    }
}

//////////////////////////////////////////////////////////////////////

//
// Print a Kokkos_BareField out.
//

template< class T, unsigned Dim>
void 
Kokkos_BareField<T,Dim>::write(std::ostream& /*out*/)
{
 /*
  

  // Inform dbgmsg(">>>>>>>> Kokkos_BareField::write", INFORM_ALL_NODES);
  // dbgmsg << "Printing values for field at address = " << &(*this) << endl;

  // on remote nodes, we must send the subnodes Kokkos_LField's to node 0
  int tag = Ippl::Comm->next_tag(F_WRITE_TAG, F_TAG_CYCLE);
  if (Ippl::myNode() != 0) {
    for ( iterator_if local = begin_if(); local != end_if() ; ++local) {
      // Cache some information about this local field.
      Kokkos_LField<T,Dim>&  l = *((*local).second);
      NDIndex<Dim>&  lo = (NDIndex<Dim>&) l.getOwned();
      typename Kokkos_LField<T,Dim>::iterator rhs(l.begin());

      // Build and send a message containing the owned LocaKokkos_LField data
      if (Ippl::myNode() != 0) {
	Message *mess = new Message();
	lo.putMessage(*mess);	      // send the local domain of the Kokkos_LField
	rhs.putMessage(*mess);          // send the data itself
	// dbgmsg << "Sending domain " << lo << " to node 0" << endl;
	Ippl::Comm->send(mess, 0, tag);
      }
    }
  } else {    // now, on node 0, receive the remaining Kokkos_LField's ...
    // put all the Kokkos_LField's in a big, uncompressed Kokkos_LField
    Kokkos_LField<T,Dim> data(getDomain(), getDomain());
    data.Uncompress();

    // first put in our local ones
    for ( iterator_if local = begin_if(); local != end_if() ; ++local) {
      // Cache some information about this local field.
      Kokkos_LField<T,Dim>&  l = *((*local).second);
      NDIndex<Dim>&  lo = (NDIndex<Dim>&) l.getOwned();
      typename Kokkos_LField<T,Dim>::iterator rhs(l.begin());
      
      // put the local Kokkos_LField in our big Kokkos_LField
      // dbgmsg << "  Copying local domain " << lo << " from Kokkos_LField at ";
      // dbgmsg << &l << ":" << endl;
      typename Kokkos_LField<T,Dim>::iterator putloc = data.begin(lo);
      typename Kokkos_LField<T,Dim>::iterator getloc = l.begin(lo);
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

      // Extract the domain size and Kokkos_LField iterator from the message
      NDIndex<Dim> lo;
      T compressed_data;
      typename Kokkos_LField<T,Dim>::iterator rhs(compressed_data);
      lo.getMessage(*mess);
      rhs.getMessage(*mess);
      // dbgmsg << "Received domain " << lo << " from " << any_node << endl;

      // put the received Kokkos_LField in our big Kokkos_LField
      typename Kokkos_LField<T,Dim>::iterator putloc = data.begin(lo);
      for (unsigned elems=lo.size(); elems > 0; ++putloc, ++rhs, --elems)
	*putloc = *rhs;

      // Free the memory in the message.
      delete mess;
    }

    // finally, we can print the big Kokkos_LField out
    out << data;
  }*/
}


// //////////////////////////////////////////////////////////////////////
// // Get a ref to a single element of the Field; if it is not local to our
// // processor, print an error and exit.  This allows the user to provide
// // different index values on each node, instead of using the same element
// // and broadcasting to all nodes.
// template<class T, unsigned Dim>
// T/*&*/ Kokkos_BareField<T,Dim>::localElement(const NDIndex<Dim>& Indexes) /*const*/
// {
// /*
//
//
//
//   // Instead of checking to see if the user has asked for one element,
//   // we will just use the first element specified for each dimension.
//
//   // Is this element here?
//   // Try and find it in the local Kokkos_BareFields.
//   const_iterator_if lf_i   = begin_if();
//   const_iterator_if lf_end = end_if();
//   for ( ; lf_i != lf_end ; ++lf_i ) {
//     Kokkos_LField<T,Dim>& lf(*(*lf_i).second);
//     // End-point "contains" OK since "owned" is unit stride.
//     // was before CK fix: if ( lf.getOwned().contains( Indexes ) ) {
//     if ( lf.getAllocated().contains( Indexes ) ) {
//       // Found it ... first uncompress, then get a pointer to the
//       // requested element.
//       lf.Uncompress();
//       //      return *(lf.begin(Indexes));
//       // instead of building an iterator, just find the value
//       NDIndex<Dim> alloc = lf.getAllocated();
//       T* pdata = PtrOffset(lf.getP(), Indexes, alloc,
//                            Kokkos_LFieldDimTag<Dim,(Dim<=3)>());
//       return *pdata;
//     }
//   }
//
//   // if we're here, we did not find it ... it must not be local
//   ERRORMSG("Kokkos_BareField::localElement: attempt to access non-local index ");
//   ERRORMSG(Indexes << " on node " << Ippl::myNode() << endl);
//   ERRORMSG("Occurred in a Kokkos_BareField with layout = " << getLayout() << endl);
//   ERRORMSG("Calling abort ..." << endl);
//   Ippl::abort();
//   return *((*((*(begin_if())).second)).begin());*/
//
//     return double(2.0);
// }