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
#include "Particle/ParticleUniformLayout.h"
#include "Utility/IpplInfo.h"
#include "Message/Communicate.h"
#include "Message/Message.h"

#include <cstddef>


/////////////////////////////////////////////////////////////////////
// constructor
// create storage for per-node data
template<class T, unsigned Dim>
ParticleUniformLayout<T, Dim>::ParticleUniformLayout() {
  
  int N = Ippl::getNodes();
  LocalSize = new int[N];
  Change = new int[N];
  MsgCount = new int[N];
}


/////////////////////////////////////////////////////////////////////
// destructor
template<class T, unsigned Dim>
ParticleUniformLayout<T, Dim>::~ParticleUniformLayout() {
  
  delete [] LocalSize;
  delete [] Change;
  delete [] MsgCount;
}


/////////////////////////////////////////////////////////////////////
// Update the location and indices of all atoms in the given IpplParticleBase
// object.  This handles swapping particles among processors if
// needed, and handles create and destroy requests.  When complete,
// all nodes have correct layout information.
template<class T, unsigned Dim>
void ParticleUniformLayout<T, Dim>::update(
  IpplParticleBase< ParticleUniformLayout<T, Dim> >& PData,
  const ParticleAttrib<char>* canSwap) {

  int i, j;			// loop variables
  int N = Ippl::getNodes();
  int myN = Ippl::myNode();
  size_t TotalNum   = PData.getTotalNum();
  size_t LocalNum   = PData.getLocalNum();
  size_t DestroyNum = PData.getDestroyNum();
  float Weight = 1.0 / (float)N;  // the fraction of the total particles
				  // to go on each node
  // create Inform object for printing
  //Inform dbgmsg("UniformLayout", INFORM_ALL_NODES);
  //dbgmsg << "At start on node " << myN << ": local=" << LocalNum;
  //dbgmsg << ", total=" << TotalNum << ", destroy=" << DestroyNum << endl;

  // if we just have one node, this is simple: just create and destroy
  // particles we need to.
  if (N == 1) {
    // delete unsightly particles
    PData.performDestroy();

    // adjust local num
    LocalNum -= DestroyNum;

    // update particle counts
    PData.setTotalNum(LocalNum);
    PData.setLocalNum(LocalNum);

    return;
  }

  // data and tags for send/receive's
  int tag1 = Ippl::Comm->next_tag(P_WEIGHTED_LAYOUT_TAG, P_LAYOUT_CYCLE);
  int tag2 = Ippl::Comm->next_tag(P_WEIGHTED_RETURN_TAG, P_LAYOUT_CYCLE);
  int tag3 = Ippl::Comm->next_tag(P_WEIGHTED_TRANSFER_TAG, P_LAYOUT_CYCLE);
  int node, sendnum, recnum;
  int nodedata[3];
  Message *msg = 0;

  // perform tasks on client nodes
  if (myN != 0) {
    // step 1: forward our current size and delete requests
    // to master node
    nodedata[0] = LocalNum;
    nodedata[1] = DestroyNum;
    msg = new Message;
    msg->put(nodedata, nodedata + 2);
    //dbgmsg << "Sending to parent: LocalNum=" << nodedata[0];
    //dbgmsg << ", DestroyNum=" << nodedata[1] << endl;
    Ippl::Comm->send(msg, 0, tag1);

  } else {			// do update tasks particular to node 0

    //
    // step 1: get info on requests from client nodes
    //

    // fill in data on the size, etc. of node 0
    LocalSize[0] = LocalNum - DestroyNum;
    TotalNum = LocalSize[0];
    //dbgmsg << "Master initially has TotalNum = " << TotalNum << endl;

    // receive messages from other nodes describing what they have
    int notrecvd = N - 1;	// do not need to receive from node 0
    while (notrecvd > 0) {
      // receive a message from another node.  After recv, node == sender.
      node = Communicate::COMM_ANY_NODE;
      msg = Ippl::Comm->receive_block(node, tag1);
      msg->get(nodedata);
      delete msg;

      // fill in data on the size, etc. of other nodes
      LocalSize[node] = nodedata[0] - nodedata[1];
      TotalNum += LocalSize[node];
      //dbgmsg << "Master received local[" << node << "] = " << LocalSize[node];
      //dbgmsg << ", new TotalNum = " << TotalNum << endl;
      notrecvd--;
    }

    // now calculate how many particles go on each node
    if (getUpdateFlag(ParticleLayout<T,Dim>::SWAP)) {
      int accounted = 0;
      for (i = 0; i < N; i++) {
	Change[i] = (int)((float)TotalNum * Weight);
	accounted += Change[i];
      }
      accounted -= TotalNum;
      if (accounted < 0) {
	while (accounted != 0) {
	  Change[(-accounted) % N]++;
	  accounted++;
	}
      } else {
	int whichnode = 0;
	while (accounted != 0) {
	  if (Change[whichnode] > 0) {
	    Change[whichnode]--;
	    accounted--;
	  }
	  whichnode = (whichnode + 1) % N;
	}
      }
      for (i = 0; i < N; i++) {
	Change[i] = Change[i] - LocalSize[i];
	MsgCount[i] = Change[i];
      }
    } else {
      for (i = 0; i < N; i++) {
        Change[i] = MsgCount[i] = 0;
      }
    }

    // send out instructions to all nodes, while calculating what goes where
    for (i = 0; i < N; i++) {
      // put header info into the message
      nodedata[0] = TotalNum;	  // new total number of particles
      if (Change[i] <= 0) {	  // if the change is < 0, send out particles
	nodedata[1] = -Change[i]; // number of particles to send
	nodedata[2] = 0;	  // number of particles to receive
      }
      else {			  // if the change is > 0, receive particles
	nodedata[1] = 0;	  // number of particles to send
	nodedata[2] = Change[i];  // number of particles to receive
      }
      msg = new Message;
      msg->put(nodedata, nodedata + 3);

      // if we must send out particles to other nodes
      // put info on where to send into the message
      if (Change[i] <= 0) {
	for (j = 0; j < N && MsgCount[i] < 0; j++) {
	  if (j != i && MsgCount[j] > 0) {
	    nodedata[0] = j;
	    if ((-MsgCount[i]) > MsgCount[j])
	      nodedata[1] = MsgCount[j];
	    else
	      nodedata[1] = -MsgCount[i];
	    MsgCount[i] += nodedata[1];
	    MsgCount[j] -= nodedata[1];
	    msg->put(nodedata, nodedata + 2);
	  }
	}
      }

      Ippl::Comm->send(msg, i, tag2);
    }
  }

  // step 2: retrieve instructions on what to create, and what to send/rec
  node = Communicate::COMM_ANY_NODE;
  // dbgmsg << "Receiving instructions ..." << endl;
  msg = Ippl::Comm->receive_block(node, tag2);
  msg->get(nodedata);
  TotalNum = nodedata[0];	// new total number of particles
  sendnum = nodedata[1];	// how many particles to send out
  recnum = nodedata[2];		// how many particles to receive
  //dbgmsg << "Received new TotalNum=" << TotalNum << ", sendnum=" << sendnum;
  //dbgmsg << ", recnum=" << recnum << endl;

  // step 3: delete unwanted particles, update local num
  PData.performDestroy();
  LocalNum -= DestroyNum;

  // step 4: send out particles which need to be sent.  In this case,
  // we just respond to the instructions from the master node, and
  // take particles from the end of our list
  if (canSwap==0) {
    while (sendnum > 0) {
      // get number of particles to send, and where
      msg->get(nodedata);	// node, number of particles
      LocalNum -= nodedata[1];
      sendnum -= nodedata[1];

      //dbgmsg << "Sending " << nodedata[1] << " particles to node ";
      //dbgmsg << nodedata[0] << endl;

      // put the particles in a new message
      Message *sendmsg = new Message;
      PData.putMessage(*sendmsg, nodedata[1], LocalNum);
      Ippl::Comm->send(sendmsg, nodedata[0], tag3);

      // After putting particles in the message, we can delete them.
      PData.destroy(nodedata[1], LocalNum, true);
    }
  }
  else {
    while (sendnum > 0) {
      // get number of particles to send, and where
      msg->get(nodedata);	// node, number of particles

      // put the particles in a new message
      Message *sendmsg = new Message;
      int delpart = LocalNum-1;
      for (int ip=0; ip<nodedata[1]; ip++) {
        while ( !(bool((*canSwap)[delpart])) ) { --delpart; }
        PData.putMessage(*sendmsg, 1, delpart);
        // After putting particles in the message, we can delete them.
        PData.destroy(1, delpart, true);
      }
      LocalNum -= nodedata[1];
      sendnum -= nodedata[1];

      //dbgmsg << "Sending " << nodedata[1] << " particles to node ";
      //dbgmsg << nodedata[0] << endl;

      Ippl::Comm->send(sendmsg, nodedata[0], tag3);
    }
  }

  // we no longer need the message from node 0 telling us what to do
  delete msg;

  // step 5: receive particles, add them to our list
  while (recnum > 0) {
    // receive next message with particle data for us
    node = Communicate::COMM_ANY_NODE;
    // dbgmsg<< "Receiving particles (" << recnum << " yet to arrive)" << endl;
    msg = Ippl::Comm->receive_block(node, tag3);
    int recvamt = PData.getMessage(*msg);
    delete msg;

    //dbgmsg << "Received " << recvamt << " particles" << endl;
    LocalNum += recvamt;
    recnum -= recvamt;
  }

  // finally, update our particle number counts
  PData.setTotalNum(TotalNum);	// set the total atom count
  PData.setLocalNum(LocalNum);	// set the number of local atoms
}


/////////////////////////////////////////////////////////////////////
// print it out
template<class T, unsigned Dim>
inline
std::ostream& operator<<(std::ostream& out, const ParticleUniformLayout<T,Dim>& /*L*/) {

  out << "ParticleUniformLayout" << std::endl;
  return out;
}


/////////////////////////////////////////////////////////////////////
// print out debugging information
template<class T, unsigned Dim>
void ParticleUniformLayout<T, Dim>::printDebug(Inform& o) {

  o << "ParticleUniformLayout";
}


/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $
 ***************************************************************************/

