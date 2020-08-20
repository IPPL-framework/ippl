// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef COMM_MPI_H
#define COMM_MPI_H

/***************************************************************************
 * CommMPI.h - MPI-specific communications object for use with the
 * Ippl framework.
 * Allows user to establish id's for available nodes, establish connections,
 * and send/receive data.
 ***************************************************************************/

// include files
#include "Message/Communicate.h"
#include <mpi.h>


class CommMPI : public Communicate
{

public:
    // constructor and destructor
    // constructor arguments: command-line args, and number of processes
    // to start (if < 0, start the 'default' number, i.e. the number of
    // hosts in a MPI virtual machine, the number of nodes in an O2K, etc)
    CommMPI(int& argc, char**& argv, int procs = (-1), bool mpiinit = true, 
            MPI_Comm mpicomm = MPI_COMM_WORLD);
    virtual ~CommMPI(void);

    // return the name of this item
    virtual const char *name() const
    {
        return "MPI";
    }

    //
    //    virtual routines to deal with memory management
    //

    // clean up after a Message has been used (called by Message).
    virtual void cleanupMessage(void *);

    virtual bool raw_send(void*, int size, int node, int tag);
    virtual MPI_Request raw_isend(void *, int size, int node, int tag);
	virtual MPI_Request raw_ireceive(char *buf, int size, int node, int tag);
    virtual int raw_receive(char*, int size, int &node, int &tag);
    virtual int raw_probe_receive(char *&, int &node, int &tag);

protected:

    // implementation-specific routines (which begin with 'my')
    // these should be provided in a derived class, and contain the
    // comm-library-specific code

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

private:
    // an MPI communicator for this object to use, to avoid interfering
    // with other MPI usage.
    MPI_Comm communicator;

    // a flag indicating whether we initialized the communication or not.
    bool weInitialized;

    // take data from the given Message, and pack it into the current send buf
    void *pack_message(Message *msg, int tag, int &buffsize, int node);
};


#endif // COMM_MPI_H
