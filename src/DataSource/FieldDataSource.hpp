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
#include "DataSource/FieldDataSource.h"
#include "Field/Field.h"
#include "Message/Communicate.h"
#include "Utility/IpplInfo.h"



////////////////////////////////////////////////////////////////////////////
// constructor: the name, the connection, the transfer method,
// the field to connect, and the parent node
template<class T, unsigned Dim, class M, class C>
FieldDataSource<T,Dim,M,C>::FieldDataSource(const char *nm, DataConnect *dc,
					    int tm, Field<T,Dim,M,C>& F)
  : DataSourceObject(nm, &F, dc, tm), MyField(F) {
  // nothing more to do here
}


////////////////////////////////////////////////////////////////////////////
// destructor
template<class T, unsigned Dim, class M, class C>
FieldDataSource<T,Dim,M,C>::~FieldDataSource() {
  // nothing to do here
}


////////////////////////////////////////////////////////////////////////////
// draw all the data together onto the Parent process
template<class T, unsigned Dim, class M, class C >
void FieldDataSource<T,Dim,M,C>::gather_data(void) {
  
  

  // the tag used to send data, my node, and total number of nodes
  int n, tag = Ippl::Comm->next_tag(DS_FIELD_TAG, DS_CYCLE);
  unsigned N = Ippl::getNodes();
  unsigned myN = Ippl::myNode();

  // First loop over all the local vnodes and send them to the connected
  // nodes (0 ... connected - 1).  Plus, on the connected nodes,
  // we put the data directly into the vtk structure.
  typename Field<T,Dim,M,C>::iterator_if local = MyField.begin_if();
  typename Field<T,Dim,M,C>::iterator_if endvn = MyField.end_if();
  for ( ; local != endvn ; ++local) {
    for (n = 0; n < getConnection()->getNodes(); ++n) {
      // Cache some information about this local field.
      LField<T,Dim>&  l           = *((*local).second);
      NDIndex<Dim>&  lo           = (NDIndex<Dim>&) l.getOwned();
      typename LField<T,Dim>::iterator rhs = l.begin();

      // We only need to send messages if there is more than one node.  
      if (n != myN) {
	// Build and send a message containing the owned LocalField data
	Message *mess = new Message();
	lo.putMessage(*mess);	      // send the local domain of the LField
	rhs.putMessage(*mess);        // send the data itself
	Ippl::Comm->send(mess, n, tag);
      } else {
	insert_data(lo, rhs);         // on the parent node, just copy in data
      }
    }
  }

  // Receive all the messages.
  if (N > 1 && getConnection()->onConnectNode()) {
    // we expect to receive one message from each remote vnode
    int remaining = MyField.getLayout().size_rdv();

    // keep receiving messages until they're all here
    for ( ; remaining > 0; --remaining) {
      // Receive the generic message.
      int any_node = COMM_ANY_NODE;
      Message *mess = Ippl::Comm->receive_block(any_node, tag);

      // Extract the domain size and LField iterator from the message
      NDIndex<Dim> localBlock;
      T lfdata;
      typename LField<T,Dim>::iterator rhs(lfdata);
      localBlock.getMessage(*mess);
      rhs.getMessage(*mess);

      // copy the data from the iterator to the vtk storage
      insert_data(localBlock, rhs);

      // Free the memory.
      delete mess;
    }
  }
}

/***************************************************************************
 * $RCSfile: FieldDataSource.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: FieldDataSource.cpp,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
