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
#include "Utility/FieldPrint.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"
#include "Field/BrickExpression.h"


#include <iomanip>


//------------------------------------------------------------------
template<class T, unsigned Dim >
void 
FieldPrint<T,Dim>::print(NDIndex<Dim>& view) {

  // check that the view is contained inside of the Field's domain
  NDIndex<Dim>& domain = (NDIndex<Dim>&) MyField.getDomain();
  if( ! domain.contains(view) ) {
    ERRORMSG("FieldPrint::print - the domain of the field: " << domain<<endl);
    ERRORMSG(" must contain the selected view: " << view << endl);
    return;
  }

  // Loop over all the local nodes and send the intersecting data to the
  // parent node
  int tag = Ippl::Comm->next_tag(FP_GATHER_TAG, FP_TAG_CYCLE);
  typedef typename LField<T,Dim>::iterator LFI;
  typename BareField<T,Dim>::iterator_if local;
  if (Ippl::Comm->myNode() != Parent) {
    // prepare a message to send to parent node
    Message *mess = new Message();
    int datahere;

    // put data for each local LField in the message
    for (local = MyField.begin_if(); local != MyField.end_if(); ++local) {
      // find the intersection of this lfield with the view, and put data in
      // message.
      LField<T,Dim> &l = *(*local).second;
      NDIndex<Dim>& lo = (NDIndex<Dim>&) l.getOwned();
      if (view.touches(lo)) {
	datahere = 1;
	NDIndex<Dim> intersection = lo.intersect(view);
	T compressed_data;
	LFI rhs = l.begin(intersection, compressed_data);
	rhs.TryCompress();
	::putMessage(*mess, datahere);
	intersection.putMessage(*mess);
	rhs.putMessage(*mess);
      }
    }

    // Send the message.
    datahere = 0;
    ::putMessage(*mess, datahere);
    Ippl::Comm->send(mess, Parent, tag);
  } else {
    // on parent, first copy all local blocks into a single LField
    LField<T,Dim> myLField(view,view);
    myLField.Uncompress();
    
    for (local = MyField.begin_if(); local != MyField.end_if(); ++local) {
      // find the intersection of this lfield with the view
      LField<T,Dim> &l = *(*local).second;
      NDIndex<Dim>& lo = (NDIndex<Dim>&) l.getOwned();
      if (view.touches(lo)) {
	NDIndex<Dim> intersection = lo.intersect(view);
	LFI lhs = myLField.begin(intersection);
	LFI rhs = l.begin(intersection);
	BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
      }
    }

    // Receive all the messages, one from each node.
    for (int remaining = Ippl::getNodes() - 1; remaining > 0; --remaining) {
      // Receive the generic message.
      int any_node = COMM_ANY_NODE;
      Message *mess = Ippl::Comm->receive_block(any_node, tag);
      PAssert(mess);

      // keep getting blocks until we're done with the message
      int datahere;
      ::getMessage(*mess, datahere);
      while (datahere != 0) {
	// Extract the intersection domain from the message.
	NDIndex<Dim> localBlock;
	localBlock.getMessage(*mess);

	// Extract the rhs iterator from it.
	T compressed_value;
	LFI rhs(compressed_value);
	rhs.getMessage(*mess);

	// copy the data into our local LField
	LFI lhs = myLField.begin(localBlock);
	BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();

	// see if there is another block
	::getMessage(*mess, datahere);
      }

      // done with the message now
      delete mess;
    }

    // now that we have populated the localfield with the view, we print
    Inform out("");

    int f0, f1, f2, n0, n1, n2, i0, i1, i2;
    LFI liter = myLField.begin();
    switch(Dim) {
    case 1:
      f0 = view[0].first();
      n0 = view[0].length();
      for (i0=0; i0<n0; ++i0) {
	out << "["<<f0+i0<<"]= "<< liter.offset(i0) << endl;
      }
      break;

    case 2:
      f0 = view[0].first();
      f1 = view[1].first();
      n0 = view[0].length();
      n1 = view[1].length();
      for (i0=0; i0<n0; ++i0) {
	for (i1=0; i1<n1; ++i1) {
	  out << "["<<f0+i0<<"]["<<f1+i1<<"]= "<< liter.offset(i0,i1)<< endl;
	}
      }
      break;

    case 3:
      f0 = view[0].first();
      f1 = view[1].first();
      f2 = view[2].first();
      n0 = view[0].length();
      n1 = view[1].length();
      n2 = view[2].length();
      if(Scientific) {
	out.setf(std::ios::scientific);
      }
      for (i0=0; i0<n0; ++i0) {
	for (i1=0; i1<n1; ++i1) {
	  out << "[" << std::setw(IndexWidth) << f0+i0 << "]"
              << "[" << std::setw(IndexWidth) << f1+i1 << "]"
              << "[" << std::setw(IndexWidth) << f2    << ":" << std::setw(IndexWidth) << f2+n2-1 << "] ";
	  for (i2=0; i2<n2; ++i2) {
	    out << std::setprecision(DataPrecision) << std::setw(DataWidth)
		<< liter.offset(i0,i1,i2)<<" ";
	    if( CarReturn > 0 ) {
	      if( i2 != 0 && i2 != ( n2-1) && !( (i2+1) % CarReturn) ) {
		out << endl << "                    ";
	      }
	    }
	  }
	  out << endl;
	}
      }
      break;

    default:
      ERRORMSG("bad Dimension \""<<Dim<<"\" in FieldPrint::print()"<< endl);
      return;
    }
  }
}


/***************************************************************************
 * $RCSfile: FieldPrint.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: FieldPrint.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
