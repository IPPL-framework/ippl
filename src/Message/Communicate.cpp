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

//////////////////////////////////////////////////////////////////////////////
// Communicate - common member functions for Communicate object.
// On-node traffic
// is handle here and architecture specific routines are called for off-node
// traffic.  This is the base class for all comm-lib-specific subclasses.
//////////////////////////////////////////////////////////////////////////////

// include files
#include "Message/Communicate.h"
#include "Message/Message.h"

#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"
#include "Utility/PAssert.h"
#include <cstdio>


////////////////////////////////////////////////////////////////////////////
// print summary of this class to the given output stream
std::ostream& operator<<(std::ostream& o, const Communicate& c)
{

    o << "Parallel communication method: " << c.name() << "\n";
    o << "  Total nodes: " << c.getNodes() << ", Current node: ";
    o << c.myNode() << "\n";
    o << "  Queued received messages: ";
    o << c.getReceived() <<"\n";

    return o;
}


////////////////////////////////////////////////////////////////////////////
// Constructor.
// 	arguments: command-line args, and number of processes
// to start (if < 0, start the 'default' number, i.e. the number of
// hosts 
// Note: The base-class constructor does not need the argument info or
// the number of nodes, it just by default sets the number of nodes=1
Communicate::Communicate(int, char **, int)
        : nextMsgNum(1)
{

    // initialize data for Communicate
    TotalNodes = 1;
    myHost = 0;
    ErrorStatus = COMM_NOERROR;
}


////////////////////////////////////////////////////////////////////////////
// Destructor.  Nothing to do at present.
Communicate::~Communicate()
{ }


////////////////////////////////////////////////////////////////////////////
// Add a new on-node message to the linked list.  Return success.
bool Communicate::add_msg(Message *msg, int node, int tag)
{
    recMsgList.push_back(MessageData(node, tag, msg));
    return true;
}


////////////////////////////////////////////////////////////////////////////
// Looks for a message in the message queue from the specified node
// and tag.  This understands wildcards for node and tag.
// Returns a pointer to the Message object found, and sets node and
// tag equal to the proper values.  Also, this will remove the item from
// the queue.
Message* Communicate::find_msg(int& node, int& tag)
{

    // just find the first message that meets the criteria
    std::vector<MessageData>::iterator qi   = recMsgList.begin();
    std::vector<MessageData>::iterator qend = recMsgList.end();
    for ( ; qi != qend ; ++qi)
    {
        if ((node == COMM_ANY_NODE || (*qi).node == node) &&
                (tag  == COMM_ANY_TAG  || (*qi).tag  == tag))
        {
            node = (*qi).node;
            tag = (*qi).tag;
            Message *retval = (*qi).msg;
            recMsgList.erase(qi);
            //INCIPPLSTAT(incMessageReceived);
            //INCIPPLSTAT(incMessageReceivedFromQueue);
            return retval;
        }
    }

    // if we're here, no message was found
    return 0;
}


////////////////////////////////////////////////////////////////////////////
// Default version of virtual send function ... here, does nothing.
bool Communicate::mysend(Message *, int, int, int)
{

    // just return false, since we cannot send a message with this function
    return false;
}


////////////////////////////////////////////////////////////////////////////
// Default version of virtual receive function ... here, does nothing.
Message* Communicate::myreceive(int&, int&, int)
{

    // just return NULL, since we cannot find a message with this function
    return 0;
}


////////////////////////////////////////////////////////////////////////////
// Default version of virtual barrier function ... here, does nothing.
void Communicate::mybarrier(void)
{
    

    // just return NULL, since we cannot find a message with this function
    return;
}


////////////////////////////////////////////////////////////////////////////
// resent a message buffer that has been previously packed and copied
// into the provided buffer.  Return success.
bool Communicate::resend(void *, int, int, int)
{

    // just return false, since we cannot resend a message with this function
    return false;
}


////////////////////////////////////////////////////////////////////////////
// Send data to the given node, with given tag.  If delmsg==true, the
// message will be deleted after it is sent, otherwise it will be left alone.
bool Communicate::send(Message *msg, int node, int tag, bool delmsg)
{
    bool retval;

    // check for problems ...
    if ( node < 0 || node >= getNodes() || tag < 0 )
    {
        ERRORMSG("Communicate: illegal send node " << node << endl);
        ErrorStatus = COMM_ERROR;
        return false;
    }

    // if the message is addressed to this node, put it in the local receive
    // queue immediately
    if ( node == myNode() )
    {
        retval = add_msg(msg, node, tag);
        //INCIPPLSTAT(incMessageSent);
        //INCIPPLSTAT(incMessageSentToSelf);
    }
    else
    {
        // the message must be sent elsewhere ... call the proper function
        retval = mysend(msg, node, tag, COMM_SEND_TAG);

        // if the send was successful, delete the message if requested
        if (retval)
        {
            //INCIPPLSTAT(incMessageSent);
            //INCIPPLSTAT(incMessageSentToOthers);
            if (delmsg)
                delete msg;
        }
    }

    // set error code
    ErrorStatus = (retval != 0 ? COMM_NOERROR : COMM_NOSEND);

    // return the success of the operation
    return retval;
}


////////////////////////////////////////////////////////////////////////////
// Receive data from another node.  Returns newly created Message object
// with received message, or NULL if no message is available.
// If node == COMM_ANY_NODE, this will receive the next message with the given
// tag from any node.
// If tag == COMM_ANY_TAG, this will receive the next message with
// any tag from the given node.  If both are wildcards, this will receive the
// next message, period.  node and tag are passed by reference; if either
// is a wildcard, and a message is received, they are changed to their actual
// values.
// Messages are searched for in this order (if node == COMM_ANY_NODE) :
//      1. Pending in network
//      2. In receive queue
Message* Communicate::receive(int& node, int& tag)
{
    // do a check for a message from another node
    //dbgmsg << "Checking for queued message ..." << endl;
    Message *msg = find_msg(node, tag);
    //dbgmsg << "Found one? " << (msg != 0 ? "yes" : "no") << endl;

    if (msg == 0 && myNode() != node)
    {
        int checknode = node;
        int checktag = tag;
        //INCIPPLSTAT(incMessageReceiveChecks);
        //dbgmsg << "Checking for remote message ..." << endl;
        if ((msg = myreceive(checknode, checktag, COMM_SEND_TAG)) != 0)
        {
            // see if the message matches our criteria for searching
            //dbgmsg << "Message found from node " << checknode << " with tag ";
            //dbgmsg << checktag << endl;
            if ((node != COMM_ANY_NODE && node != checknode) ||
                    (tag  != COMM_ANY_TAG  && tag  != checktag ))
            {
                // the message does not match; queue it and report no msg found
                //dbgmsg << "But it's not what we want." << endl;
                add_msg(msg, checknode, checktag);
                msg = 0;
            }
            else
            {
                // the message matches; save the node and tag and return the msg
                //dbgmsg << "And it is what we want!" << endl;
                node = checknode;
                tag = checktag;
                //INCIPPLSTAT(incMessageReceived);
                //INCIPPLSTAT(incMessageReceivedFromNetwork);
            }
        }
        else
        {
            //INCIPPLSTAT(incMessageReceiveChecksFailed);
        }
    }

    // set error code
    ErrorStatus = (msg != 0 ? COMM_NOERROR : COMM_NORECEIVE);

    // return the message, or NULL if none was found
    return msg;
}


////////////////////////////////////////////////////////////////////////////
// A blocking version of receive.
Message *Communicate::receive_block(int& node, int &tag)
{
    
    
    
    

    //Inform dbgmsg("Comm::receive_block", INFORM_ALL_NODES);
    //dbgmsg << "Doing blocking receive from node " << node;
    //dbgmsg << ", tag " << tag << endl;

    // If we haven't already found a message, check the local messages
    //dbgmsg << "Checking for queued message ..." << endl;
    
    Message *msg = find_msg(node, tag);
    
    //dbgmsg << "Found one? " << (msg != 0 ? "yes" : "no") << endl;

    // keep checking for remote msgs until we get one
    
    if (myNode() != node)
    {
        while (msg == 0)
        {
            // process list of resend requests
            int checknode = node;
            int checktag = tag;
            //INCIPPLSTAT(incMessageReceiveChecks);
            //dbgmsg << "Checking for remote message ..." << endl;
            if ((msg = myreceive(checknode, checktag, COMM_SEND_TAG)) != 0)
            {
                // see if the message matches our criteria for searching
                //dbgmsg << "Message found from node " << checknode << " with tag ";
                //dbgmsg << checktag << endl;
                if ((node != COMM_ANY_NODE && node != checknode) ||
                        (tag  != COMM_ANY_TAG  && tag  != checktag ))
                {
                    // the message does not match; queue it and report no msg found
                    //dbgmsg << "But it's not what we want." << endl;
                    add_msg(msg, checknode, checktag);
                    msg = 0;
                }
                else
                {
                    // the message matches; save the node and tag and return the msg
                    //dbgmsg << "And it is what we want!" << endl;
                    node = checknode;
                    tag = checktag;
                    //INCIPPLSTAT(incMessageReceived);
                    //INCIPPLSTAT(incMessageReceivedFromNetwork);
                }
            }
        }
    }
    

    // If we're on just one node, and we did not find a message, this is
    // a big problem.
    PInsist(!(myNode() == node && msg == 0),
            "Local message not found in Communicate::receive_block!!");

    // set error code
    ErrorStatus = (msg != 0 ? COMM_NOERROR : COMM_NORECEIVE);

    // return the message, or NULL if none was found
    return msg;
}


////////////////////////////////////////////////////////////////////////////
// Broadcast the given message to ALL nodes, including this node.
// Return number of nodes sent to.
// Arguments are the Message, and the tag for the message.
int Communicate::broadcast_all(Message *msg, int tag)
{
    int i;			// loop variable

    // send message to all other nodes
    for (i=(getNodes() - 1); i >= 0; i--)
    {
        if (i != myNode())
        {
            mysend(msg, i, tag, COMM_SEND_TAG);
            //INCIPPLSTAT(incMessageSent);
            //INCIPPLSTAT(incMessageSentToOthers);
        }
    }

    // send message to this node; since we do this, don't need to delete msg
    add_msg(msg, myNode(), tag);
    //INCIPPLSTAT(incMessageSent);
    //INCIPPLSTAT(incMessageSentToSelf);

    return getNodes();
}


////////////////////////////////////////////////////////////////////////////
// Broadcast the given message to all OTHER nodes, but not this node.
// Return number of nodes sent to.
// Arguments are the Message, and the tag for the message, and whether
// we should delete the given message object.
int Communicate::broadcast_others(Message *msg, int tag, bool delmsg)
{
    int i;			// loop variable

    // send message to all other nodes
    for (i=(getNodes() - 1); i >= 0; i--)
    {
        if (i != myNode())
        {
            mysend(msg, i, tag, COMM_SEND_TAG);
            //INCIPPLSTAT(incMessageSent);
            //INCIPPLSTAT(incMessageSentToOthers);
        }
    }

    // delete message
    if (delmsg)
        delete msg;

    return getNodes() - 1;
}


////////////////////////////////////////////////////////////////////////////
// Synchronize all processors (everybody waits for everybody
// else to get here before returning to calling function).
void Communicate::barrier()
{
    

    mybarrier();
    //INCIPPLSTAT(incBarriers);
}


////////////////////////////////////////////////////////////////////////////
// clean up after a Message has been used (called by Message).  By
// default, does nothing.
void Communicate::cleanupMessage(void *) { }


////////////////////////////////////////////////////////////////////////////
// calculate how big the buffer must be to send the given message
int Communicate::find_msg_length(Message &msg)
{

    static const unsigned int longsize = wordround(sizeof(MsgNum_t));
    static const unsigned int intsize4 = wordround(4 * sizeof(int));
    static const unsigned int intsize2 = wordround(2 * sizeof(int));

    // the message contains a long and three integers at the start with the
    // msg num, node, tag, and number of items
    unsigned int buffsize = longsize + intsize4;

    // now include the sizes of the elements themselves.  For each item,
    // we also include two integers with size information.
    int nitems = msg.size();
    for (int i=0; i < nitems; ++i)
        buffsize += (intsize2 + wordround(msg.item(i).numBytes()));

    return buffsize;
}


////////////////////////////////////////////////////////////////////////////
// put data from the given Message into the given buffer
void Communicate::fill_msg_buffer(void *buffer, Message &msg, int tag,
                                  int bufsize, int /*node*/)
{

    void *pos = buffer;		  // location in buffer to pack data
    int nitems = msg.size();	  // Number of items in Message
    int mdata[4];			  // Array to store msg header info
    MsgNum_t mnum = (nextMsgNum++); // Message ID

    //Inform dbgmsg("***Communicate::fill_msg_buffer", INFORM_ALL_NODES);
    //dbgmsg << "Preparing to send out message " << mnum;
    //dbgmsg << " with tag " << tag << " of size " << bufsize << endl;

    // we must make sure to zero out the buffer if we're using checksums,
    // so that random data values do not occur in the spaces where word
    // alignment padding is used

    // put message ID info into the buffer
    pack(&mnum, pos, sizeof(MsgNum_t));

    // put message header info into the buffer
    mdata[0] = tag;
    mdata[1] = myNode();
    mdata[2] = nitems;
    mdata[3] = bufsize;
    pack(mdata, pos, 4*sizeof(int));

    // finally pack in the data
    for (int i=0; i < nitems; ++i)
    {
        Message::MsgItem &msgitem = msg.item(i);
        mdata[0] = msgitem.numElems();
        mdata[1] = msgitem.numBytes();
        pack(mdata, pos, 2*sizeof(int));
        if (mdata[1] > 0)
            pack(msgitem.data(), pos, mdata[1]);
    }

    ADDIPPLSTAT(incMessageBytesSent,bufsize);
}


////////////////////////////////////////////////////////////////////////////
// get data out of a buffer and create a Message
Message* Communicate::unpack_message(int &node, int &tag, void *buffer)
{

    Message *newmsg = 0;

    // pos will always point to the next location in the buffer to get data
    void *pos = buffer;

    addwordround(pos, sizeof(MsgNum_t));

    // get the tag, sender, and number of messages
    int *mdata = static_cast<int *>(pos);
    tag = mdata[0];
    node = mdata[1];
    int nitems = mdata[2];
    int bufsize = mdata[3];
    addwordround(pos, 4*sizeof(int));

    //WARNMSG("Received message " << mnum << " from node " << node);
    //WARNMSG(" with tag " << tag << " of size " << bufsize << endl);

    // check for special tags, to abort, retransmit, or just receive
    if (tag == IPPL_ABORT_TAG)
    {
        ERRORMSG("Stopping due to abort request sent from node " << node << endl);
        ::abort();

    }
    else if (tag == IPPL_EXIT_TAG)
    {
        ERRORMSG("Exiting due to exit request sent from node " << node << endl);
        ::exit(1);

    }
    else
    {
        // this is just a regular message

        // if we're here, the checksums (if enabled) were OK, so receive the
        // message

        // create data structure for this message
        newmsg = new Message(nitems);

        // get all the items and add to the message
        for (int j = 0; j < nitems; j++)
        {
            int *hdr = static_cast<int *>(pos);
            int elements = hdr[0];
            int bytesize = hdr[1];
            addwordround(pos, 2*sizeof(int));

            // for each item, find the pointer to the actual data and give
            // that pointer to the Message object.  The Message object then
            // does not delete the data until the very end, when the Message
            // is deleted.
            if (bytesize > 0 && elements > 0)
            {
                newmsg->setCopy(false);
                newmsg->setDelete(false);
                newmsg->putmsg(pos, bytesize/elements, elements);
                addwordround(pos, bytesize);
            }
        }

        // indicate we've received a normal message
        ADDIPPLSTAT(incMessageBytesReceived,bufsize);

    }

    // return the new message, or zero to indicate the buffer contained
    // something else than an actual message
    return newmsg;
}