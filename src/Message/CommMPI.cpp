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
#include "Message/CommMPI.h"
#include "Message/Message.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"


#include "Utility/IpplMessageCounter.h"

// include mpi header file
#include <mpi.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <vector>

// if an error occurs during myreceive more times than this, CommMPI
// will just exit.  Make it negative to totally disable checking for a
// maximum number of errors
#define MAX_MPI_ERRS	500


// static data to keep track of errors
static int numErrors = 0;
static int size_of_MPI_INT; /* needed for tracing */

// temporary buffer used for speed
#define PSIZE 1024*16
#define PACKSIZE ((PSIZE)*sizeof(long))
static long mpipackbuf[PSIZE];



////////////////////////////////////////////////////////////////////////////
// constructor.  arguments: command-line args, and number of processes
// to start (if < 0, start the 'default' number, i.e. the number of
// hosts in a MPI virtual machine, the number of nodes in an O2K, etc)
// Note: The base-class constructor does not need the argument info or
// the number of nodes, it just by default sets the number of nodes=1
// The final argument indicates whether to run MPI_Init or not; IPPL
// may be run as another user of MPI, in a context where MPI_Init has
// already been called, in which case it can skip that step.
CommMPI::CommMPI(int& argc , char**& argv, int procs, bool mpiinit, MPI_Comm mpicomm)
        : Communicate(argc, argv, procs), weInitialized(mpiinit)
{

    int i, reported, rep_host, ierror, result_len;
    MPI_Status stat;
    char *currtok, *nexttok, *execname;

    // a little "string magic" to strip the absolute pathname off the executable
    currtok = strstr(argv[0],"/");
    if (!currtok)
    {
        execname = strdup(argv[0]);
    }
    else
    {
        currtok++;
        nexttok = strstr(currtok,"/");
        while (nexttok)
        {
            currtok = nexttok+1;
            nexttok = strstr(currtok,"/");
        }
        execname = strdup(currtok);
    }

    // initialize mpi
    if (weInitialized) {
#ifdef _OPENMP
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        INFOMSG("Ippl will be initialized with " <<
                omp_get_max_threads() << " OMP threads\n");
        
        if ( provided != MPI_THREAD_FUNNELED )
            ERRORMSG("CommMPI: Didn't get requested MPI-OpenMP setting.\n");
#else
        MPI_Init(&argc, &argv);
#endif
    }
    //else
    //    INFOMSG("NOT initializing MPI = " << endl);

    // restore original executable name without absolute path
    strcpy(argv[0],execname);

    free(execname);

    // duplicate the MPI_COMM_WORLD communicator, so that we can use
    // a communicator that will not conflict with other users of MPI_COMM_WORLD
    MPI_Comm_dup(mpicomm, &communicator);

    // determine the number of nodes running and my node number
    MPI_Comm_size(communicator,&TotalNodes);
    MPI_Comm_rank(communicator,&myHost);

    // make sure we do not have too many processes running
    if (procs > 0 && procs < TotalNodes)
    {
        // if this is a process that is beyond what we had requested, just exit
        if (myHost >= procs)
            Ippl::abort();
        TotalNodes = procs;
    }

    MPI_Type_size ( MPI_INT, &size_of_MPI_INT );
    if (myHost == 0)      // this code is run by the master process
    {
        // send a messages to each child node
        for (i = 1; i < TotalNodes; i++)
        {
            MPI_Send(&myHost, 1, MPI_INT, i, COMM_HOSTS_TAG, communicator);
        }

        // wait for the spawned processes to report back that they're ready
        //~ int *child_ready = new int[TotalNodes];
        std::vector<int> child_ready(TotalNodes);
        for (i = 0; i < TotalNodes; child_ready[i++] = 0)
            ;
        INFOMSG("CommMPI: Parent process waiting for children ..." << endl);
        reported = 1;		// since the parent is already ready
        while (reported < TotalNodes)
        {
            ierror = MPI_Recv(&rep_host, 1, MPI_INT, MPI_ANY_SOURCE,
                              COMM_HOSTS_TAG, communicator, &stat);

            if (rep_host >= 0 && rep_host < TotalNodes && !(child_ready[rep_host]))
            {
                child_ready[rep_host] = 1;
                reported++;
                INFOMSG("CommMPI: Child " << rep_host << " ready." << endl);
            }
            else
            {
                ERRORMSG("CommMPI: Error with child reporting to parent.  ");
                ERRORMSG("rep_host = " << rep_host);
                ERRORMSG(", child_ready[] = " << child_ready[rep_host] << endl);
            }
        }

        //~ delete [] child_ready;
        INFOMSG("CommMPI: Initialization complete." << endl);

    }
    else  			// this is a child process; get data from pops
    {
        char host_name[MPI_MAX_PROCESSOR_NAME];
        ierror = MPI_Get_processor_name(host_name, &result_len);
        if (ierror >= 0)
        {
            INFOMSG("CommMPI: Started job " << myHost << " on host `");
            INFOMSG(host_name <<  "'." << endl);
        }
        else
        {
            ERRORMSG("CommMPI: failed" << endl);
        }

        // receive message from the master node
        int checknode;
        MPI_Recv(&checknode, 1, MPI_INT, 0, COMM_HOSTS_TAG, communicator,
                 &stat);

        if (checknode != 0)
            WARNMSG("CommMPI: Child received bad message during startup." << endl);

        // send back an acknowledgement
        MPI_Send(&myHost, 1, MPI_INT, 0, COMM_HOSTS_TAG, communicator);

    }

    // set up the contexts and processes arrays properly
    if (TotalNodes > 1)
    {
        std::vector<int> proccount;
        proccount.push_back(1);
        for (i = 1; i < TotalNodes; i++)
        {
            Contexts.push_back(1);
            Processes.push_back(proccount);
        }
    }

}


////////////////////////////////////////////////////////////////////////////
// class destructor
CommMPI::~CommMPI(void)
{

    int i, dieCode = 0;
    MPI_Status stat;

    // on all nodes, when running in parallel, get any extra messages not
    // yet received
    if (TotalNodes > 1)
    {
        int trial, node, tag;
        Message *msg;
        for (trial = 0; trial < 50000; ++trial)
        {
            do
            {
                node = COMM_ANY_NODE;
                tag = COMM_ANY_TAG;
                msg = myreceive(node, tag, COMM_SEND_TAG);
                if (msg != 0 && tag != IPPL_ABORT_TAG && tag != IPPL_EXIT_TAG)
                {
                    WARNMSG("CommMPI: Found extra message from node " << node);
                    WARNMSG(", tag " << tag << ": msg = " << *msg << endl);
                }
            }
            while (msg != 0);
        }
    }

    // broadcast a message to all other nodes to tell them to quit
    if (myNode() == 0)
    {
        // on master node, send out messages
        for (i = 1; i < TotalNodes; i++)
        {
            MPI_Send(&dieCode, 1, MPI_INT, i, COMM_DIE_TAG, communicator);

        }
    }
    else
    {
        // on client nodes, receive message
        MPI_Recv(&dieCode, 1, MPI_INT, 0, COMM_DIE_TAG, communicator, &stat);

    }

    MPI_Barrier(communicator);
    // delete the communicator we used
    MPI_Comm_free(&communicator);

    // if we did our own initialization, also do finalize operation.  But
    // if we did not initialize, skip the finalize as well

    /*
    ada: it make no sense to call finalize from the application,
    because the IPPL destructor is called AFTER that, which causes:
    "0032-151 MPI is already finalized in string, task number"
    */
    if (weInitialized)
        MPI_Finalize();
}


////////////////////////////////////////////////////////////////////////////
// take the data from a Message object and pack it into the current send buf.
// each message is packed in this order:
//      tag, sending node, number of items             (3-int array)
//              type of item 1  (short)
//              size of item 1, in number of elements   (int)
//              item 1 data     (various)
//              ...
//              type of item N  (short)
//              size of item N, in number of elements   (int)
//              item N data     (various)
void *CommMPI::pack_message(Message *msg, int tag, int &buffsize, int node)
{

    // calculate size of buffer
    buffsize = find_msg_length(*msg);

    // allocate storage for buffer
    void *pos = ((unsigned int) buffsize > PACKSIZE) ? makebuffer(buffsize) : mpipackbuf;

    // pack message data and return the necessary pointer
    fill_msg_buffer(pos, *msg, tag, buffsize, node);
    return pos;
}


////////////////////////////////////////////////////////////////////////////
// send a message ... arguments are the Message itself, the
// destination node, the 'user' tag, and the 'encoding' tag.
// Messages should be sent via the underlying mechanism by using the
// encoding tag (one of the COMM_ tags),
// and should embed the information about what the user
// tag is in the data sent between nodes.  Return success.
bool CommMPI::mysend(Message *msg, int node, int tag, int etag)
{

    int errstat = (-1);
    int flag = false;
    MPI_Request request;
    MPI_Status status;

    MPI_Status rec_status;
    int src_node, rec_node, rec_tag, rec_size, rec_utag, bufid, rec_flag = 0;
    Message* newmsg = NULL;

    // pack the message data into the buffer
    int size;
    void *outbuffer = pack_message(msg, tag, size, node);

    // send the message (non-blocking)
    // Inform dbgmsg("CommMPI", INFORM_ALL_NODES);
    // dbgmsg << "Sending MPI message of size " << size << " to node " << node;
    // dbgmsg << " with tag " << tag << "." << endl;

    //messaging "profiler"
    IpplMessageCounter::getInstance().registerMessage(size);

    errstat = MPI_Isend(outbuffer, size, MPI_BYTE, node, etag,
                        communicator, &request);


    while (!flag)
    {
        if (!Ippl::retransmit())
        {
            // get info about messages to be received
            bufid = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, communicator,
                               &rec_flag, &rec_status);
            if ((bufid >= 0) && (rec_flag != 0) )
            {
                // a message is available to be received
                src_node = rec_status.MPI_SOURCE;
                rec_tag = rec_status.MPI_TAG;
                MPI_Get_count(&rec_status, MPI_BYTE, &rec_size);
                // dbgmsg<<"Receiving MPI message of size " << rec_size << " from node ";
                // dbgmsg << src_node << "." << endl;
                if ( (rec_size >= 0) && (rec_tag >= 0) && (src_node >= 0) )
                {
                    // message is a valid one, so malloc the output buffer
                    void *rec_buff = makebuffer(rec_size);

                    // blocking receive, unpack message
                    MPI_Recv(rec_buff, rec_size, MPI_BYTE, src_node, rec_tag,
                             communicator, &rec_status);

                    newmsg = unpack_message(rec_node, rec_utag, rec_buff);

                    // if there was an error unpacking, then the message had a problem
                    // and is invalid, so throw this one away
                    if (newmsg == 0)
                    {
                        // free up the buffer
                        cleanupMessage(rec_buff);

                    }
                    else
                    {
                        // tell the message to inform us when the buffer is finished
                        newmsg->useCommunicate(this, rec_buff);

                        // put message in my message queue
                        add_msg(newmsg, rec_node, rec_utag);
                    }

                    // reset other receive information
                    newmsg = NULL; // reset message pointer
                    rec_flag = 0; // reset receive flag
                }
            }
        }

        // check for completion of send
        MPI_Test(&request, &flag, &status);
    }

    //  free up the send buffer
    if ((unsigned int) size > PACKSIZE)
        freebuffer(outbuffer);

    // return the success of the operation
    return (errstat == 0);
}


////////////////////////////////////////////////////////////////////////////
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
Message *CommMPI::myreceive(int& node, int& tag, int etag)
{

    int bufid, size, checknode, checktag, flag = false;
    Message *newmsg = 0;
    MPI_Status stat;

    checknode = (node < 0 || node >= TotalNodes ? MPI_ANY_SOURCE : node);
    checktag = etag;

    // get info about message
    bufid = MPI_Iprobe(checknode, checktag, communicator, &flag, &stat);
    if (bufid < 0)
    {
        // an error has occurred
        ERRORMSG("CommMPI: cannot receive msg from node " << checknode);
        ERRORMSG(", tag " << checktag << endl);

        if (MAX_MPI_ERRS > 0 && ++numErrors > MAX_MPI_ERRS)
        {
            ERRORMSG("Maximum number of MPI receive errors (" << numErrors);
            ERRORMSG(") exceeded. MPI is hosed!!" << endl);
            Ippl::abort();
        }
    }

    // if the message is actually available, see if we can get it now
    if (flag == true)
    {
        MPI_Get_count(&stat,MPI_BYTE,&size);
        if (size < 0)
        {
            ERRORMSG("CommMPI: received message has size " << size << endl);
        }
        else if ((stat.MPI_TAG != checktag) || (stat.MPI_TAG < 0))
        {
            ERRORMSG("CommMPI: received message with invalid tag ");
            ERRORMSG(stat.MPI_TAG << endl);
        }
        else if (stat.MPI_SOURCE < 0)
        {
            ERRORMSG("CommMPI: received message from invalid source ");
            ERRORMSG(stat.MPI_SOURCE << endl);
        }
        else
        {
            checknode = stat.MPI_SOURCE;
            checktag = stat.MPI_TAG;

            // malloc the receive buffer
            void *outbuff = makebuffer(size);

            // blocking receive
            // Inform dbgmsg("CommMPI", INFORM_ALL_NODES);
            // dbgmsg << "Receiving MPI message of size " << size << " from node ";
            // dbgmsg << checknode << "." << endl;
            MPI_Recv(outbuff, size, MPI_BYTE, checknode, checktag,
                     communicator, &stat);

            newmsg = unpack_message(node, tag, outbuff);

            // if there was an error unpacking, then the message had a problem
            // and is invalid, so throw this one away
            if (newmsg == 0)
            {
                // free up the buffer
                cleanupMessage(outbuff);
            }
            else
            {
                // tell the message to notify us when its done with the buffer
                newmsg->useCommunicate(this, outbuff);
            }

            // zero out the count of MPI-specific errors
            numErrors = 0;
        }

    }

    // return the new Message, or NULL if no message available
    return newmsg;
}


////////////////////////////////////////////////////////////////////////////
// Synchronize all processors (everybody waits for everybody
// else to get here before returning to calling function).
// Uses MPI barrier for all procs
void CommMPI::mybarrier(void)
{
    MPI_Barrier(communicator);
}


////////////////////////////////////////////////////////////////////////////
// resent a message buffer that has been previously packed and copied
// into the provided buffer.  Return success.
bool CommMPI::resend(void *buf, int buffsize, int node, int etag)
{

    //Inform dbgmsg("CommMPI::resend", INFORM_ALL_NODES);
    //dbgmsg << "About to resend buffer of size " << buffsize << " to node ";
    //dbgmsg << node << " with etag = " << etag << endl;

    // this will only work if we're sending to another node
    PInsist(node != myNode(), "Can only retransmit to other nodes");

    IpplMessageCounter::getInstance().registerMessage(buffsize);

    // send the buffer out
    MPI_Request request;
    int errstat = MPI_Isend(buf, buffsize, MPI_BYTE, node, etag,
                            communicator, &request);


    int flag = false;
    MPI_Status status;
    while (!flag)
    {
        // check for completion of send
        MPI_Test(&request, &flag, &status);
    }

    // return the success of the operation
    return (errstat == 0);
}


////////////////////////////////////////////////////////////////////////////
// clean up after a Message has been used (called by Message).
void CommMPI::cleanupMessage(void *d)
{
    // need to free the allocated storage
    freebuffer(d);
}


bool CommMPI::raw_send(void *data, int size, int node, int tag)
{
    IpplMessageCounter::getInstance().registerMessage(size);

    return MPI_Send(data, size, MPI_BYTE, node, tag, communicator)
           == MPI_SUCCESS;
}

MPI_Request CommMPI::raw_isend(void *data, int size, int node, int tag)
{
    MPI_Request request;

    IpplMessageCounter::getInstance().registerMessage(size);

    MPI_Isend(data, size, MPI_BYTE, node, tag, communicator, &request);
    return request;
}

int CommMPI::raw_receive(char *data, int size, int &node, int &tag)
{
    if (node == COMM_ANY_NODE)
        node = MPI_ANY_SOURCE;
    if (tag == COMM_ANY_TAG)
        tag = MPI_ANY_TAG;

    MPI_Status stat;
    MPI_Recv(data, size, MPI_BYTE, node, tag, communicator, &stat);

    node = stat.MPI_SOURCE;
    tag = stat.MPI_TAG;
    int count;
    MPI_Get_count(&stat, MPI_BYTE, &count);
    return count;
}

MPI_Request CommMPI::raw_ireceive(char *buf, int size, int node, int tag)
{
    if (node == COMM_ANY_NODE)
        node = MPI_ANY_SOURCE;
    if (tag == COMM_ANY_TAG)
        tag = MPI_ANY_TAG;

    MPI_Request request;
    MPI_Irecv(buf, size, MPI_BYTE, node, tag, communicator, &request);

    return request;
}

int CommMPI::raw_probe_receive(char *&data, int &node, int &tag)
{
    if (node == COMM_ANY_NODE)
        node = MPI_ANY_SOURCE;
    if (tag == COMM_ANY_TAG)
        tag = MPI_ANY_TAG;
    MPI_Status stat;

    MPI_Probe(node, tag, communicator, &stat);
    int count;
    MPI_Get_count(&stat, MPI_BYTE, &count);
    if(count>0)
		data = new char[count];
	else
		data = 0;
    node = stat.MPI_SOURCE;
    tag = stat.MPI_TAG;

    MPI_Recv(data, count, MPI_BYTE, node, tag, communicator, &stat);

    return count;
}
