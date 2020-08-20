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
#include "Utility/FieldDebugPrint.h"

#include "Field/BrickExpression.h"
#include "Field/GuardCellSizes.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"
#include "Message/Communicate.h"
#include "Message/Message.h"
#include <iomanip>

// Print out the data for a given field, using it's whole domain
template<class T, unsigned Dim>
void FieldDebugPrint<T,Dim>::print(BareField<T,Dim>& F,
				   Inform& out, bool allnodes) {
  NDIndex<Dim> domain;
  if (PrintBC)
    domain = AddGuardCells(F.getDomain(), F.getGuardCellSizes());
  else
    domain = F.getDomain();
  print(F, domain, out, allnodes);
}


// print out the data for a given field, using it's whole domain
template<class T, unsigned Dim>
void FieldDebugPrint<T,Dim>::print(BareField<T,Dim>& F, bool allnodes) {
  NDIndex<Dim> domain;
  if (PrintBC)
    domain = AddGuardCells(F.getDomain(), F.getGuardCellSizes());
  else
    domain = F.getDomain();
  Inform out("");
  print(F, domain, out, allnodes);
}


// print out the data for a given field and domain
template<class T, unsigned Dim>
void FieldDebugPrint<T,Dim>::print(BareField<T,Dim>& F,
				   const NDIndex<Dim>& view, bool allnodes) {
  Inform out("");
  print(F, view, out, allnodes);
}


// print out the data for a given field and domain
template<class T, unsigned Dim>
void FieldDebugPrint<T,Dim>::print(BareField<T,Dim>& F,
				   const NDIndex<Dim>& view,
				   Inform& out,
                                   bool allnodes) {

  // generate a new communication tag, if necessary
  int tag;
  if (allnodes) {
    tag = Ippl::Comm->next_tag(FP_GATHER_TAG, FP_TAG_CYCLE);
  }

  // determine the maximum domain of the field
  NDIndex<Dim> domain;
  if (PrintBC)
    domain = AddGuardCells(F.getDomain(), F.getGuardCellSizes());
  else
    domain = F.getDomain();

  // check that the view is contained inside of the Field's domain
  if(!domain.contains(view)) {
    ERRORMSG("FieldDebugPrint::print - the domain of the field: " << domain<<endl);
    ERRORMSG(" must contain the selected view: " << view << endl);
    return;
  }

  // In order to print this stuff right, we need to make sure the
  // guard cells are filled.  But we can only do this if this is
  // being called by all the nodes, since it requires communication.
  if (allnodes && F.isDirty()) {
    F.fillGuardCells();
  }

  // create an LField to store the information for the view, and
  // an LField to indicate which elements we actually have (they might
  // not be available if we're printing out from the debugger without
  // any communication).  Start off initially saying we do not have
  // any data.
  typedef typename LField<T,Dim>::iterator LFI;
  typedef typename LField<bool,Dim>::iterator BLFI;
  typedef PETE_Scalar<bool> BPS;
  LField<T,Dim> myLField(view,view);
  LField<bool,Dim> availLField(view,view);
  BPS trueitem(true);
  BPS falseitem(false);
  BLFI blhs = availLField.begin();
  BrickExpression<Dim,BLFI,BPS,OpAssign>(blhs,falseitem).apply();

  // create an iterator for looping over local vnodes
  typename BareField<T,Dim>::iterator_if local;

  // If we're printing involving all the nodes, send data from local vnodes to
  // node zero and have that node print.  If we're doing this on individual
  // nodes, just have that node fill in the data it has, and print that
  // (which does not require any communication).
  if (allnodes) {
    // Loop over all the local nodes and send the intersecting data to the
    // parent node
    if (Ippl::myNode() != 0) {
      // prepare a message to send to parent node
      Message *mess = new Message();
      int datahere;
      // put data for each local LField in the message
      for (local = F.begin_if(); local != F.end_if(); ++local) {
        // find the intersection of this lfield with the view, and put data in
        // message.
        LField<T,Dim> &l = *(*local).second;
        NDIndex<Dim>& lo = (NDIndex<Dim>&) l.getAllocated();
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
      Ippl::Comm->send(mess, 0, tag);
    } else {
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
          myLField.Uncompress();
          availLField.Uncompress();
	  LFI lhs = myLField.begin(localBlock);
	  BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
          BLFI blhs = availLField.begin(localBlock);
          BrickExpression<Dim,BLFI,BPS,OpAssign>(blhs,trueitem).apply();

	  // see if there is another block
	  ::getMessage(*mess, datahere);
        }

        // done with the message now
        delete mess;
      }
    }
  }

  // Now that we have populated the localfield with the view, we print.
  // Only print on node 0, or on all the nodes if allnodes = false.
  if (!allnodes || Ippl::myNode() == 0) {

    // TJW 1/22/1999

    // To make this correct with defergcfill=true and allnodes=false (to avoid
    // inconsistent/unset guard element values being output as physical
    // elements, do this in two stages. The first fills the single LField
    // based on intersections with Allocated domains, which can put some
    // inconsistent/unset guard-element values into physical element
    // locations. The second fills the single LField based on intersections
    // with Owned domains, which will overwrite those bogus values with
    // correct values from owned physical elements in the original BareField's
    // LFields. There is undoubtably a more efficient way to achieve this end,
    // but do it this way for now, to fix the bug:
    
    // TJW: First stage: intersect with Allocated:

    // first need to copy all local blocks into the single LField
    for (local = F.begin_if(); local != F.end_if(); ++local) {
      // find the intersection of this lfield with the view
      LField<T,Dim> &l = *(*local).second;
      NDIndex<Dim>& lo = (NDIndex<Dim>&) l.getAllocated();
      if (view.touches(lo)) {
        myLField.Uncompress();
        availLField.Uncompress();
        NDIndex<Dim> intersection = lo.intersect(view);
        LFI lhs = myLField.begin(intersection);
        LFI rhs = l.begin(intersection);
        BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
        BLFI blhs = availLField.begin(intersection);
        BrickExpression<Dim,BLFI,BPS,OpAssign>(blhs,trueitem).apply();
      }
    }

    // TJW: Second stage: intersect with Owned:

    // first need to copy all local blocks into the single LField
    for (local = F.begin_if(); local != F.end_if(); ++local) {
      // find the intersection of this lfield with the view
      LField<T,Dim> &l = *(*local).second;
      NDIndex<Dim>& lo = (NDIndex<Dim>&) l.getOwned();
      if (view.touches(lo)) {
        myLField.Uncompress();
        availLField.Uncompress();
        NDIndex<Dim> intersection = lo.intersect(view);
        LFI lhs = myLField.begin(intersection);
        LFI rhs = l.begin(intersection);
        BrickExpression<Dim,LFI,LFI,OpAssign>(lhs,rhs).apply();
        BLFI blhs = availLField.begin(intersection);
        BrickExpression<Dim,BLFI,BPS,OpAssign>(blhs,trueitem).apply();
      }
    }

    // finally, we can print
    out << "~~~~~~~~ field slice ";
    for (unsigned int pd=0; pd < Dim; ++pd) {
      out << (pd == 0 ? "(" : ", ");
      out << view[pd].first() << ":" << view[pd].last();
      out << ":" << view[pd].stride();
    }
    out << ") ~~~~~~~~" << endl;
    if(Scientific)
      out.setf(std::ios::scientific);

    unsigned int i0, i1, i2;
    LFI liter = myLField.begin();
    BLFI bliter = availLField.begin();
    switch(Dim) {
    case 1:
      for (i0=0; i0 < view[0].length(); ++i0, ++liter, ++bliter)
	printelem(*bliter, *liter, i0, out);
      out << endl;
      break;

    case 2:
      for (i1=0; i1 < view[1].length(); ++i1) {
	out << "--------------------------------------------------J = ";
	out << view[1].first() + i1 << endl;
	for (i0=0; i0 < view[0].length(); ++i0, ++liter, ++bliter)
	  printelem(*bliter, *liter, i0, out);
	out << endl;
	out << endl;
      }
      break;

    case 3:
      for (i2=0; i2 < view[2].length(); ++i2) {
	out << "==================================================K = ";
	out << view[2].first() + i2 << endl;
	for (i1=0; i1 < view[1].length(); ++i1) {
	  out << "--------------------------------------------------J = ";
	  out << view[1].first() + i1 << endl;
	  for (i0=0; i0 < view[0].length(); ++i0, ++liter, ++bliter)
	    printelem(*bliter, *liter, i0, out);
	  out << endl;
	  out << endl;
	}
      }
      break;

    default:
      ERRORMSG("bad Dimension \"" << Dim << "\" in FieldDebugPrint::print()");
      ERRORMSG(endl);
      Ippl::abort();
    }
  }
}


// print a single value to the screen
template<class T, unsigned Dim>
void FieldDebugPrint<T,Dim>::printelem(bool isavail, T &val,
                                       unsigned int i0, Inform& out) {
  if (DataPrecision > 0)
    out << std::setprecision(DataPrecision);
  if (DataWidth > 0)
    out << std::setw(DataWidth);
  if(Scientific)
    out.setf(std::ios::scientific);
  if (isavail)
    out << val;
  else
    out << '-';
  if (CarReturn > 0 && ((i0+1) % CarReturn) == 0)
    out << endl;
  else
    out << " ";
}

  
/***************************************************************************
 * $RCSfile: FieldDebugPrint.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: FieldDebugPrint.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
