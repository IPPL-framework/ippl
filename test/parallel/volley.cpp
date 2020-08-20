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
 ***************************************************************************/

// test program to demonstrate use of parallel master->client->master volley
#include "Ippl.h"


void report(const char *str, bool result) {
   
  Inform msg("Results", INFORM_ALL_NODES);
  msg << "Test " << str << ": ";
  msg << (result ? "PASSED" : "FAILED") << endl;
}


int main(int argc, char *argv[]) {
  
  Ippl ippl(argc, argv);

  Message *msg;
  int mynode = ippl.myNode();
  char recbuf[128];
  int recint = 0;
  int sendtag = 80;

  if (mynode == 0) {
    // first have the master node send a message to the other nodes
    msg = new Message();
    msg->put("This is a message from the master node: ");
    msg->put(mynode);
    ippl.Comm->broadcast_all(msg, sendtag);

    // now have the master receive back all the replies and print them
    int notReceived = ippl.getNodes();
    while (notReceived > 0) {
      int node = COMM_ANY_NODE;
      int tag = sendtag;
      msg = ippl.Comm->receive_block(node, tag);
      if (msg == 0) {
	report("Master receive", false);
	ERRORMSG("Could not receive from client nodes in main." << endl);
	return 1;
      }
      msg->get(recbuf).get(recint);
      delete msg;
      if (recint != node) {
	report("Master receive", false);
	ERRORMSG("Incorrect message from node " << node << endl);
	return 1;
      }
      notReceived--;
    }
    report("Master receive", true);
  } else {
    // the client nodes just receive the message and send one back
    int node = 0;
    int tag = sendtag;
    msg = ippl.Comm->receive_block(node, tag);
    if (msg == 0) {
      report("Client receive", false);
      ERRORMSG("Could not receive from master node in main." << endl);
      return 1;
    }
    msg->get(recbuf).get(recint);
    delete msg;
    if (recint != node) {
      report("Client receive", false);
      ERRORMSG("Incorrect message from node " << node << endl);
      return 1;
    }
    report("Client receive", true);
    msg = new Message;
    msg->put("This is a reply message from the client node: ");
    msg->put(mynode);
    ippl.Comm->send(msg, 0, sendtag);
  }

  return 0;
}

/***************************************************************************
 * $RCSfile: volley.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:38 $
 * IPPL_VERSION_ID: $Id: volley.cpp,v 1.1.1.1 2003/01/23 07:40:38 adelmann Exp $ 
 ***************************************************************************/
