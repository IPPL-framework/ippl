// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef COMMUNICATE_H
#define COMMUNICATE_H

/***************************************************************************
 * Communicate.h - communications object for use with Ippl framework.  Allows
 * user to establish id's for available nodes, establish connections, and
 * send/receive data.
 ***************************************************************************/

// include files
#include "Message/TagMaker.h"
#include "Message/Tags.h"
#include <cstdlib>
#include <memory>
#include <cstring>

#include <vector>
#include <utility>

#include <iostream>

#include <mpi.h>

// forward declarations
class Message;
class Communicate;
std::ostream& operator<<(std::ostream&, const Communicate&);

// special codes used as 'wildcards' to match any node or tag
const int COMM_ANY_NODE = (-1);
const int COMM_ANY_TAG  = (-1);


// A simple class used to store information for caching sent messages.  This
// is only used if the 'retransmit' option is active.
class CommSendInfo
{
public:
    CommSendInfo()
            : size_m(0), buf_m(0)
    {
    }

    CommSendInfo(int size, char *buf, int node)
            : size_m(size), node_m(node), buf_m(buf)
    {
    }

    CommSendInfo(const CommSendInfo &c)
            : size_m(c.size_m), node_m(c.node_m), buf_m(c.buf_m)
    {
    }

    ~CommSendInfo()
    {
        // the user is actually responsible for freeing the buffer.  We
        // do not do this automatically here
    }

    CommSendInfo &operator=(const CommSendInfo &c)
    {
        size_m = c.size_m;
        buf_m = c.buf_m;
        node_m = c.node_m;
        return *this;
    }

    int size() const
    {
        return size_m;
    }

    int node() const
    {
        return node_m;
    }

    char *buf()
    {
        return buf_m;
    }
    const char *buf() const
    {
        return buf_m;
    }

    void freebuf()
    {
        if (buf_m != 0)
            delete [] buf_m;
        buf_m = 0;
    }

private:
    int size_m;
    int node_m;
    char *buf_m;
};


// The base class for all specific Communicate objects
class Communicate : public TagMaker
{

public:
    // default error codes, may be overridden by inherited classes.
    enum CommErrors { COMM_NOERROR, COMM_ERROR, COMM_NOSEND, COMM_NORECEIVE };

    // special tags used by this class ... 32000 is arbitrary
    enum CommTags { COMM_HOSTS_TAG = 32000, COMM_DIE_TAG, COMM_SEND_TAG };

    // special codes used as 'wildcards' to match any node or tag
    // These are listed again because they should be here, but the global
    // values are kept for compatibility.
    enum CommCodes { COMM_ANY_NODE = (-1), COMM_ANY_TAG = (-1) };

public:

    // constructor and destructor
    // constructor arguments: command-line args, and number of processes
    // to start (if < 0, start the 'default' number, i.e. the number of
    // hosts in a PVM virtual machine, the number of nodes in an O2K, etc)
    Communicate(int argc = 0, char** argv = NULL, int procs = (-1));
    virtual ~Communicate(void);

    // return the name of this item
    virtual const char *name() const
    {
        return "Serial";
    }

    // return info about connections in general
    int getNodes() const
    {
        return TotalNodes;
    }
    int getContexts(const int n) const
    {
        return Contexts[n];
    }
    int getProcesses(const int n, const int c) const
    {
        return Processes[n][c];
    }
    int myNode() const
    {
        return myHost;
    }
    int getError() const
    {
        return ErrorStatus;
    }
    int getReceived() const
    {
        return recMsgList.size();
    }

    //
    //    nonvirtual routines to send/receive data
    //

    // send data to another node.  Returns success (T or F).
    // last argument specifies whether to delete the Message after sending
    // (if message is for another node).  Note that if the send is not
    // successful, the message will NOT be deleted, regardless of delmsg.
    bool send(Message *, int node, int tag, bool delmsg = true);

    // receive data from another node.  Returns newly created Message object
    // with received message, or NULL if no message is available.
    // If node is < 0, this will receive the next message with the given tag
    // from any node.  If tag < 0, this will receive the next message with
    // any tag from the given node.  If both are < 0, this will receive the
    // next message, period.  node and tag are passed by reference; if either
    // is < 0, and a message is received, they are changed to their actual
    // values.
    Message *receive(int& node, int& tag);

    // a blocking version of receive;
    Message *receive_block(int& node, int& tag);

    //send and receive for raw data
    virtual bool raw_send(void *, int , int , int )
    {
        return false;
    }
    virtual MPI_Request raw_isend(void *, int , int , int )
    {
        return MPI_Request();
    }
    virtual int raw_receive(char *, int , int &, int &)
    {
        return 0;
    }
    virtual MPI_Request raw_ireceive(char *, int , int , int )
    {
        return MPI_Request();
    }
    virtual int raw_probe_receive(char *&, int &, int &)
    {
        return 0;
    }
     

    //
    //    virtual routines to broadcast data
    //

    // broadcast the current message to other nodes.
    // Return number of nodes actually sent to.
    // The first version sends to all nodes including this node.
    // The second version sends to all nodes except this node.
    // The first argument is the Message; the last argument is the tag.
    virtual int broadcast_all(Message *, int);
    virtual int broadcast_others(Message *, int, bool delmsg=true);


    //
    //    routines to synchronize processors at a barrier
    //

    // Synchronize all processors (everybody waits for everybody
    // else to get here before returning to calling function).
    void barrier(void);

    //
    //    virtual routines to deal with memory management
    //

    // clean up after a Message has been used (called by Message).  By
    // default, does nothing.
    virtual void cleanupMessage(void *);

protected:
    // struct used to store messages, tags, and nodes
    struct MessageData
    {
        int node;			// sending/receiving node
        int tag;			// tag of the message
        Message *msg;		// pointer to the message itself
        MessageData(int n, int t, Message *m) : node(n),tag(t),msg(m) {}
        MessageData() : node(-1), tag(-1), msg(0) { }
        MessageData(const MessageData& m) : node(m.node),tag(m.tag),msg(m.msg) {}
        ~MessageData() {}
    };

    // a list of messages which have already been received, but not yet
    // delivered
    std::vector<MessageData> recMsgList;

    // the following items should be filled in by the derived classes
    int TotalNodes;		// number of nodes available (0 ... # nodes-1)
    int myHost;			// which node am I?
    int ErrorStatus;		// error code, from above enumeration
    std::vector<int> Contexts;		// the number of contexts per node
    std::vector< std::vector<int> > Processes;   // number of running processes per context

    // An integer message number identifier; this is included in each
    // message, and continually increases as more messages are sent.
    typedef long MsgNum_t;
    MsgNum_t nextMsgNum;

    // An optional sent-message cache, used to attempt to retransmit
    // messages if they are corrupted in-transit.  Messages are keyed on
    // a message number, which is is unique for each message.
    typedef std::map<MsgNum_t, CommSendInfo> SentCache_t;
    SentCache_t sentMsgCache;

    // a list of things to resend at the next opportunity
    std::vector<MsgNum_t> resendList;

    // a list of messages which have been received OK
    std::vector<MsgNum_t> sentOKList;

    // a list of messages which should be cleared out on other nodes
    std::vector<std::pair<int,MsgNum_t> > informOKList;

    // a list of requests we must make to other nodes to resend messages
    std::vector<std::pair<int,MsgNum_t> > requestList;

    // add a new message to the received message queues.  Return success.
    // arguments: message, sending node, tag
    bool add_msg(Message *, int, int);

    // Looks for a message in the message queue from the specified node
    // and tag.  This understands wildcards for node and tag.
    // Returns a pointer to the Message object found, and sets node and
    // tag equal to the proper values.  Also, this will remove the item from
    // the queue.
    Message* find_msg(int&, int&);

    //
    // implementation-specific routines (which begin with 'my')
    //	these should be provided in a derived class, and contain the
    //	comm-library-specific code
    //

    // send a message ... arguments are the Message itself, the
    // destination node, the 'user' tag, and the 'encoding' tag.
    // Messages should be sent via the underlying mechanism by using the
    // encoding tag (one of the COMM_ tags),
    // and should embed the information about what the user
    // tag is in the data sent between nodes.  Return success.
    virtual bool mysend(Message *, int node, int utag, int etag);

    // receive a message from the given node and user tag.  Return a NEW
    // Message object if a message arrives, or NULL if no message available.
    // node will be set to the node from which the message was sent.
    // tag will be set to the 'user tag' for that message.
    // etag is the 'encoding' tag, and must be one of the COMM_ tags.
    // Only message sent via the underlying mechanism with the
    // given etag are checked.  When one is found, the user tag and sending
    // node are extracted from the sent data.
    // If node = COMM_ANY_NODE, checks for messages from any node.
    // If tag = COMM_ANY_TAG, checks for messages with any user tag.
    virtual Message *myreceive(int& node, int& tag, int etag);

    // Synchronize all processors (everybody waits for everybody
    // else to get here before returning to calling function).
    virtual void mybarrier(void);

    // resent a message buffer that has been previously packed and copied
    // into the provided buffer.  Return success.
    virtual bool resend(void *buf, int size, int node, int etag);

    //
    // utility functions used to serialize data into and out of byte buffers
    //
    // standard way to create and free buffer storage
    static inline void *makebuffer(int size)
    {
        return malloc(size);
    }
    static inline void freebuffer(void *buf)
    {
        free(buf);
    }

    // compute the size of storage needed to add 'size' bytes to a buffer,
    // in order to keep everything word-aligned
    static inline unsigned int wordround(int size)
    {
        return sizeof(long long) *
               ((size + sizeof(long long) - 1)/sizeof(long long));
    }

    // compute a wordround value for 'size' bytes, then add that to the
    // given 'pos' pointer
    static inline void addwordround(void * &pos, int size)
    {
        pos = static_cast<void *>(wordround(size) + static_cast<char *>(pos));
    }

    // memcpy data into the given location, and then increment the pointer
    static inline void pack(void *packdata, void * &pos, int size)
    {
        memcpy(pos, packdata, size);
        addwordround(pos, size);
    }

    // memcpy data out of a given location to another, updating 'pos'
    static inline void unpack(void * &pos, void *packdata, int size)
    {
        memcpy(packdata, pos, size);
        addwordround(pos, size);
    }

    //
    // utility functions used in packing and unpacking Message data
    //

    // calculate how big the buffer must be to send the given message
    int find_msg_length(Message &);

    // put data from the given Message into the given buffer, with tag value.
    // the final arguments are the buffer size, in bytes, and the dest node.
    void fill_msg_buffer(void *, Message &, int, int, int);

    // take data out of the current receive buf and create a new Message
    Message *unpack_message(int &node, int &tag, void *pos);

    //
    // utility functions used for message caching/retransmit
    //

    // put the given message buffer in the sent-message cache, as a new
    // CommSendInfo object storing the buffer and other information.
    void add_to_send_cache(void *pos, MsgNum_t mnum, int size, int node);

    // send off a request to have this message retransmitted to us
    void request_retransmission(int node, MsgNum_t mnum);

    // resend the data for message mnum ... calls the virtual 'resend'
    void perform_resend(MsgNum_t mnum);

    // get the resend information from a buffer sent in a message requesting
    // retransmission
    void unpack_retransmission_request(int nitems, void *pos);

    // tell the sender that we received this message OK
    void send_ok_message(int node, MsgNum_t mnum);

    // unpack message with a list of OK message numbers, and delete them
    // from our cache
    void clear_ok_messages(int nitems, void *pos);

    // remove a single OK message
    void remove_single_ok_message(MsgNum_t mnum);

    // process list of resend requests
    void process_resend_requests();
};

#endif // COMMUNICATE_H
