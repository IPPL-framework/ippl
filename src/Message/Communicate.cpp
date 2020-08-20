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
#include "Message/CRC.h"
#include "PETE/IpplExpressions.h"

#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"
#include "Utility/RandomNumberGen.h"
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
// Also note: the derived classes should erase Contexts and Processes, and
// put in the proper values.
Communicate::Communicate(int, char **, int)
        : nextMsgNum(1)
{

    // initialize data for Communicate
    TotalNodes = 1;
    myHost = 0;
    ErrorStatus = COMM_NOERROR;
    Contexts.push_back(1);
    Processes.push_back(Contexts); // using Contexts is just convenient here
}


////////////////////////////////////////////////////////////////////////////
// Destructor.  Nothing to do at present.
Communicate::~Communicate(void)
{
    

    // delete the cached messages
    SentCache_t::iterator cachei = sentMsgCache.begin();
    for ( ; cachei != sentMsgCache.end(); ++cachei)
        (*cachei).second.freebuf();
}


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

    // process list of resend requests
    //process_resend_requests();

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
    

    //Inform dbgmsg("Comm::receive", INFORM_ALL_NODES);
    //dbgmsg << "Doing receive from node " << node << ", tag " << tag << endl;

    // process list of resend requests
    process_resend_requests();

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
    
    
    
    

    // process list of resend requests
    process_resend_requests();

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
            process_resend_requests();

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
                if (Ippl::retransmit())
                    msg = find_msg(node, tag);
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

    // if checksums are to be performed, add in space for the 32-bit checksum
    if (Ippl::useChecksums())
        buffsize += sizeof(CRCTYPE);

    return buffsize;
}


////////////////////////////////////////////////////////////////////////////
// put data from the given Message into the given buffer
void Communicate::fill_msg_buffer(void *buffer, Message &msg, int tag,
                                  int bufsize, int node)
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
    if (Ippl::useChecksums())
        memset(pos, 0, bufsize);

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

    // if checksums are on, find the checksum and append it to the buffer
    if (Ippl::useChecksums())
    {
        // calculate the crc
        int crcsize = bufsize - sizeof(CRCTYPE);
        CRCTYPE crcval = crc(buffer, crcsize);

        // append it to the end of the buffer
        *(static_cast<CRCTYPE *>(pos)) = crcval;

        // if we're trying to retransmit, cache the message
        if (Ippl::retransmit())
        {
            if (tag != IPPL_RETRANSMIT_TAG && tag != IPPL_MSG_OK_TAG)
            {
                //dbgmsg << "Adding message " << mnum << " of size " << bufsize;
                //dbgmsg << " with tag " << tag << " to sent cache." << endl;
                add_to_send_cache(buffer, mnum, bufsize, node);
            }
            else
            {
                //dbgmsg << "NOT adding msg with tag " << tag << " to cache" << endl;
            }
        }
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

    // get the message ID number
    MsgNum_t mnum = *(static_cast<MsgNum_t *>(pos));
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
    else if (tag == IPPL_RETRANSMIT_TAG)
    {
        // get the retransmit message number and tag out of the current buffer
        unpack_retransmission_request(nitems, pos);

    }
    else if (tag == IPPL_MSG_OK_TAG)
    {
        // clear out the messages that this message lists are OK to be deleted
        clear_ok_messages(nitems, pos);

    }
    else
    {
        // this is just a regular message

        // do checksum comparison, if checksums are on
        if (Ippl::useChecksums())
        {
            // calculate the crc
            int crcsize = bufsize - sizeof(CRCTYPE);
            CRCTYPE crcval = crc(buffer, crcsize);

            // as a test, randomly change crcval
            //if (IpplRandom() < 0.1)
            //  crcval += 1;

            // compare this crc to the original one
            void *origloc = static_cast<void *>(static_cast<char *>(buffer)+crcsize);
            CRCTYPE origcrc = *(static_cast<CRCTYPE *>(origloc));
            if (crcval != origcrc)
            {
                ERRORMSG("Failed CRC check (" << crcval << " != " << origcrc);
                ERRORMSG(") on node " << Ippl::myNode());
                ERRORMSG(" for message " << mnum << " of size " << bufsize);
                ERRORMSG(" bytes sent from node ");
                ERRORMSG(node << " with tag " << tag << endl);
                if (Ippl::retransmit())
                {
                    // send off a request to have message 'mnum' resent to us by 'node'
                    requestList.push_back(std::pair<int,MsgNum_t>(node, mnum));
                }
                else
                {
                    // since we're not trying to retransmit, we just quit.
                    PInsist(crcval == origcrc, "Exiting due to CRC check failure.");
                }

                // and then return 0 so that the caller knows there was a problem
                return 0;
            }
        }

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

        // tell the sender that we received this message OK
        if (Ippl::retransmit())
            informOKList.push_back(std::pair<int,MsgNum_t>(node, mnum));
    }

    // return the new message, or zero to indicate the buffer contained
    // something else than an actual message
    return newmsg;
}


////////////////////////////////////////////////////////////////////////////
// put the given message buffer in the sent-message cache, as a new
// CommSendInfo object storing the buffer and other information.
void Communicate::add_to_send_cache(void *msgbuf, MsgNum_t mnum, int msgsize,
                                    int node)
{

    // make sure we do not already have this message
    SentCache_t::iterator senti = sentMsgCache.find(mnum);
    if (senti != sentMsgCache.end())
    {
        ERRORMSG("ERROR: Trying to cache an already-cached message with num = ");
        ERRORMSG(mnum << endl);
        return;
    }

    //Inform dbgmsg("***Communicate::add_to_send_cache", INFORM_ALL_NODES);
    //dbgmsg << "Adding message " << mnum << " to cache with size = " << msgsize;
    //dbgmsg << endl;

    // make a copy of the message
    char *copybuf = new char[msgsize];
    memcpy(copybuf, msgbuf, msgsize);

    // add the message to the cache list
    CommSendInfo csi(msgsize, copybuf, node);
    sentMsgCache.insert(SentCache_t::value_type(mnum, csi));

    //dbgmsg<<"Sent message cached; now " << sentMsgCache.size() << " buffers ";
    //dbgmsg << "in the cache." << endl;
}


////////////////////////////////////////////////////////////////////////////
// send off a request to have this message retransmitted to us
void Communicate::request_retransmission(int node, MsgNum_t mnum)
{
    Inform dbgmsg("***Communicate::request_retransmission", INFORM_ALL_NODES);
    dbgmsg << "Requesting retransmission of message " << mnum << " from node ";
    dbgmsg << node << endl;

    // create a regular message, but with the
    // special retransmit tag and the message number as the one item
    // in the Message
    Message msg(1);
    msg.put(mnum);
    send(&msg, node, IPPL_RETRANSMIT_TAG, false); // does not delete message
}


////////////////////////////////////////////////////////////////////////////
// get the resend information from a buffer sent in a message requesting
// retransmission
void Communicate::unpack_retransmission_request(int nitems, void *pos)
{
    Inform dbgmsg("***Communicate::unpack_retrans_req", INFORM_ALL_NODES);
    //dbgmsg << "Unpacking retransmission request ..." << endl;

    // retransmission messages have the following information as separate
    // items:
    //   message number to retransmit (type == MsgNum_t)
    // so, nitems should be one, and the bytesize should match
    PInsist(nitems == 1, "Wrong number of items in retransmit request.");

    // get the retransmit message number item header info
    int *hdr = static_cast<int *>(pos);
    PInsist(hdr[0] == 1 && hdr[1] == sizeof(MsgNum_t),
            "Wrong message info for retransmit message number.");
    addwordround(pos, 2*sizeof(int));

    // get the actual retransmit message number
    MsgNum_t mnum = *(static_cast<MsgNum_t *>(pos));
    dbgmsg << "Received request to resend message " << mnum << endl;
    resendList.push_back(mnum);
}


////////////////////////////////////////////////////////////////////////////
// for message mnum, resend the data
void Communicate::perform_resend(MsgNum_t mnum)
{
    // get the message info from our cache.
    SentCache_t::iterator senti = sentMsgCache.find(mnum);
    PInsist(senti != sentMsgCache.end(),
            "Could not find message in local sent cache to retransmit.");

    // get the node and size
    int size = (*senti).second.size();
    int node = (*senti).second.node();

    // resend the data
    ERRORMSG("WARNING: Resending message number " << mnum);
    ERRORMSG(" of size " << size << " from node ");
    ERRORMSG(myNode() << " to node " << node << " ..." << endl);
    resend((*senti).second.buf(), size, node, COMM_SEND_TAG);
}


////////////////////////////////////////////////////////////////////////////
// tell the sender that we received this message OK
void Communicate::send_ok_message(int node, MsgNum_t mnum)
{
    Inform dbgmsg("***Communicate::send_ok_message", INFORM_ALL_NODES);
    dbgmsg << "Informing node " << node << " that message " << mnum;
    dbgmsg << " was received ok." << endl;

    Message msg(1);
    msg.put(mnum);		// the list of message numbers, one at a time

    send(&msg, node, IPPL_MSG_OK_TAG, false); // does not delete message
}


////////////////////////////////////////////////////////////////////////////
// unpack message with a list of OK message numbers, and delete them
// from our cache
void Communicate::clear_ok_messages(int nitems, void *pos)
{
    Inform dbgmsg("***Communicate::clear_ok_messages", INFORM_ALL_NODES);
    //dbgmsg << "Unpacking messages-ok information for " << nitems;
    //dbgmsg << " messages ..." << endl;

    // message-ok messages have the following information as separate
    // items:
    //   the number of OK messages (type == int)
    //   the first OK message number (type == MsgNum_t)
    //   the second OK message number (type == MsgNum_t)
    //   etc
    PInsist(nitems >= 1, "Wrong number of items in retransmit request.");

    // loop through the list of items, get the message number from each,
    // and remove that message from our queue
    for (int i=0; i < nitems; ++i)
    {
        // get the message-ok header
        int *hdr = static_cast<int *>(pos);
        PInsist(hdr[0] == 1 && hdr[1] == sizeof(MsgNum_t),
                "Wrong message info for message-ok number.");
        addwordround(pos, 2*sizeof(int));

        // get the message-ok number
        MsgNum_t mnum = *(static_cast<MsgNum_t *>(pos));
        addwordround(pos, sizeof(MsgNum_t));

        // add this number to our list of messages to say are OK
        dbgmsg << "Will clear message " << mnum << " as OK." << endl;
        sentOKList.push_back(mnum);
    }
}


////////////////////////////////////////////////////////////////////////////
// unpack message with a list of OK message numbers, and delete them
// from our cache
void Communicate::remove_single_ok_message(MsgNum_t mnum)
{
    Inform dbgmsg("***Communicate::remove_single_ok_message", INFORM_ALL_NODES);

    // check if we have that message
    SentCache_t::iterator senti = sentMsgCache.find(mnum);
    if (senti == sentMsgCache.end())
    {
        // we do not have it; print an error message
        ERRORMSG("ERROR: Received 'message ok' for message " << mnum);
        ERRORMSG(", but this node does not have that message in cache." << endl);

    }
    else
    {
        // we have it, so remove it after freeing the buffer
        (*senti).second.freebuf();
        sentMsgCache.erase(senti);
        dbgmsg << "Removed message " << mnum << " from send cache; now ";
        dbgmsg << sentMsgCache.size() << " messages in list." << endl;
    }
}


////////////////////////////////////////////////////////////////////////////
// process list of resend requests
void Communicate::process_resend_requests()
{
    if (resendList.size() > 0)
    {
        Inform dbgmsg("***Communicate::process_resend_reqs", INFORM_ALL_NODES);
        dbgmsg << "Clearing " << sentOKList.size() << " and resending ";
        dbgmsg << resendList.size() << " messages ..." << endl;
    }

    // clear out OK messages
    while (sentOKList.size() > 0)
    {
        MsgNum_t mnum = *(sentOKList.begin());
        sentOKList.erase(sentOKList.begin());
        remove_single_ok_message(mnum);
    }

    // resend a message, if necessary
    while (resendList.size() > 0)
    {
        MsgNum_t mnum = *(resendList.begin());
        resendList.erase(resendList.begin());
        perform_resend(mnum);
    }

    // inform other nodes that we've received their messages ok
    while (informOKList.size() > 0)
    {
        int node = (*(informOKList.begin())).first;
        MsgNum_t mnum = (*(informOKList.begin())).second;
        informOKList.erase(informOKList.begin());
        send_ok_message(node, mnum);
    }

    // request resends from other nodes
    while (requestList.size() > 0)
    {
        int node = (*(requestList.begin())).first;
        MsgNum_t mnum = (*(requestList.begin())).second;
        requestList.erase(requestList.begin());
        request_retransmission(node, mnum);
    }
}
