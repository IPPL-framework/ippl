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
#include "Message/GlobalComm.h"
#include "Message/Communicate.h"
#include "Message/Message.h"
#include "Message/Tags.h"
#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"
#include "Message/DataTypes.h"
#include "Message/Operations.h"

#include <algorithm>

////////////////////////////////////////////////////////////////////////////
// Reduce equally-sized arrays across the machine, by sending to node
// 0 and broadcasting back the result.  The arguments are two begin,end
// iterators for the source of the data, an iterator pointing to
// where the summed data should go, and an operation to perform in
// the reduction ... this last should be from PETE, e.g., OpAdd, etc.
// template classes found in either ../Expression/Applicative.h or
// ../Expression/TypeComputations.h. The are simple classes  such as
// OpAnd(), OpMax(), OpAdd(), OpMultipply(), etc....
// Return success.
// The final argument indicates whether the LOCAL NODE should have it's
// values included in the reduction (by default, this is true).  If this
// pointer to the boolean array is null, all the values will be included.
// NOTE: The input iterators must iterate over simple data objects,
// which do not require their own special getMessage/putMessage.  If you
// need to reduce a complex quantity, use the scalar version of reduce.
template <class InputIterator, class OutputIterator, class ReduceOp>
bool reduce(Communicate& comm, InputIterator s1, InputIterator s2,
            OutputIterator t1, const ReduceOp& op, bool *IncludeVal)
{


    // Inform dbgmsg("reduce-vector", INFORM_ALL_NODES);

    // determine destination node and tags
    int parent = 0;
    int sendtag = comm.next_tag(COMM_REDUCE_SEND_TAG, COMM_REDUCE_CYCLE);
    int rectag  = comm.next_tag(COMM_REDUCE_RECV_TAG, COMM_REDUCE_CYCLE);

    // determine how many elements we have to reduce
    unsigned int elements = 0;
    for (InputIterator tmps1 = s1; tmps1 != s2; ++tmps1, ++elements);
    if (elements == 0)
    {
        ERRORMSG("reduce: nothing to reduce." << endl);
    }

    // create flags, if they are not provided
    bool *useFlags = IncludeVal;
    if (useFlags == 0 && elements > 0)
    {
        useFlags = new bool[elements];
        for (unsigned int u=0; u < elements; useFlags[u++] = true);
    }

    if ( comm.myNode() != parent )
    {
        // send the source data to node 0 if we are not node 0
        Message *msg = new Message;
        // dbgmsg << "sending message with " << elements << " elements to node ";
        // dbgmsg << parent << " with tag " << sendtag << endl;
        ::putMessage(*msg, elements);
        if (elements > 0)
        {
            ::putMessage(*msg, s1, s2);
            ::putMessage(*msg, useFlags, useFlags + elements);
        }
        if ( ! comm.send(msg, parent, sendtag) )
        {
            Ippl::abort("reduce: cannot send reduce buffers.");
        }

        // then we get the results back
        msg = comm.receive_block(parent, rectag);
        // dbgmsg << "received message with size = " << msg->size();
        // dbgmsg << " from node " << parent << " with tag " << rectag << endl;
        if ( ! msg || msg->size() < 1 )
            Ippl::abort("reduce: cannot receive reduce results.");
        getMessage(*msg, *t1);
        delete msg;

    }
    else
    {
        // first copy the source into the target; this is like receiving
        // from ourselves
        InputIterator  tmp1  = s1;
        OutputIterator t2    = t1;
        bool*          copyf = useFlags;
        for ( ; tmp1 != s2; ++tmp1, ++t2, ++copyf)
            if (*copyf)
                *t2 = *tmp1;

        // the parent receives all the messages and then broadcasts the
        // reduced result
        int notReceived = comm.getNodes() - 1;
        while (notReceived > 0)
        {
            // receive message
            int fromnode = COMM_ANY_NODE;
            Message *recmsg = comm.receive_block(fromnode, sendtag);
            // dbgmsg << "received message with size = " << recmsg->size();
            // dbgmsg << " from node " << fromnode << " with tag "<<sendtag<<endl;
            if ( ! recmsg || recmsg->size() < 1 )
                Ippl::abort("reduce: cannot receive reduce buffers.");

            // get data from message
            int recelems;
            ::getMessage(*recmsg, recelems);
            if ((unsigned int) recelems != elements)
                Ippl::abort("reduce: mismatched element count in vector reduction.");
            if (elements > 0)
            {
                InputIterator reci = (InputIterator)(recmsg->item(0).data());
                bool *recflag = (bool *)(recmsg->item(1).data());

                // the target buffer must have size >= size of the source, so
                // we can iterate over the source iterator
                unsigned int u;
                for (u=0, t2=t1; u < elements; ++t2, ++reci, ++u)
                {
                    if (recflag[u])
                    {
                        if (useFlags[u])
                        {
                            PETE_apply(op, *t2, *reci);
                        }
                        else
                        {
                            *t2 = *reci;
                            useFlags[u] = true;
                        }
                    }
                }
            }

            // finished with this node's data
            delete recmsg;
            notReceived--;
        }

        // Finally, broadcast the results out.  t2 should now point to the
        // end of the target buffer.
        if (comm.getNodes() > 1)
        {
            Message *sendmsg = new Message();
            putMessage(*sendmsg, t1, t2);
            // dbgmsg << "sending message with size " << sendmsg->size();
            // dbgmsg << " to all nodes with tag " << rectag << endl;
            if (comm.broadcast_others(sendmsg, rectag) != (comm.getNodes() - 1))
                Ippl::abort("reduce: cannot send reduce results.");
        }
    }

    // we're done
    if (useFlags != 0 && useFlags != IncludeVal)
        delete [] useFlags;

    //INCIPPLSTAT(incReductions);
    return true;
}


////////////////////////////////////////////////////////////////////////////
// same as above, but this uses the default Communicate object
template <class InputIterator, class OutputIterator, class ReduceOp>
bool reduce(InputIterator s1, InputIterator s2,
            OutputIterator t1, const ReduceOp& op, bool *IncludeVal)
{
    return reduce(*Ippl::Comm, s1, s2, t1, op, IncludeVal);
}


////////////////////////////////////////////////////////////////////////////
// masked scalar versions of reduce ... instead of iterators, these versions
// expect a single quantity to reduce and a location to place the result.
// The final argument indicates whether the LOCAL NODE should have it's
// value included in the reduction (by default, this is true).
// Return success of operation.
template <class T, class ReduceOp>
bool reduce_masked(Communicate& comm, T& input, T& output,
                   const ReduceOp& op, bool IncludeVal)
{

    // Inform dbgmsg("reduce_masked", INFORM_ALL_NODES);

    // determine destination node and tags
    int parent = 0;
    int sendtag = comm.next_tag(COMM_REDUCE_SEND_TAG, COMM_REDUCE_CYCLE);
    int rectag  = comm.next_tag(COMM_REDUCE_RECV_TAG, COMM_REDUCE_CYCLE);

    if (comm.myNode() != parent)
    {
        // send the source data to node 0 if we are not node 0
        Message *msg = new Message;
        // dbgmsg << "sending message, includeflag=" << IncludeVal << ", to node ";
        // dbgmsg << parent << " with tag " << sendtag << endl;
        ::putMessage(*msg, IncludeVal);
        if (IncludeVal)
            ::putMessage(*msg, input);
        if ( ! comm.send(msg, parent, sendtag) )
        {
            Ippl::abort("reduce: cannot send reduce scalar.");
        }

        // then we get the results back
        msg = comm.receive_block(parent, rectag);
        // dbgmsg << "received message with size = " << msg->size();
        // dbgmsg << " from node " << parent << " with tag " << rectag << endl;
        if ( ! msg || msg->size() < 1 )
            Ippl::abort("reduce: cannot receive reduce results.");
        getMessage(*msg, output);
        delete msg;

    }
    else
    {
        // first copy the source into the target; this is like receiving
        // from ourselves
        if (IncludeVal)
            output = input;

        // if there are several nodes, we must get the other results
        if (comm.getNodes() > 1)
        {

            // the parent receives all the messages and then broadcasts the
            // reduced result
            int notReceived = comm.getNodes() - 1;

            // create a temporary array to store values from other nodes
            T *recval = new T[notReceived];
            bool *recflag = new bool[notReceived];

            // get all messages
            while (notReceived > 0)
            {
                // receive message
                int fromnode = COMM_ANY_NODE;
                Message *recmsg = comm.receive_block(fromnode, sendtag);
                if ( ! recmsg || recmsg->size() < 1 )
                    Ippl::abort("reduce: cannot receive reduce buffers.");

                // get flag indicating if the message has any data; if it does,
                // get it and store it
                ::getMessage(*recmsg, recflag[fromnode - 1]);
                if (recflag[fromnode - 1])
                    ::getMessage(*recmsg, recval[fromnode - 1]);

                // finished with this node's data
                delete recmsg;
                notReceived--;
            }

            // now loop through the received values and do the reduction
            for (int n=1; n < comm.getNodes(); ++n)
            {
                if (recflag[n-1])
                {
                    if (IncludeVal)
                    {
                        PETE_apply(op, output, recval[n-1]);
                    }
                    else
                    {
                        output = recval[n-1];
                        IncludeVal = true;
                    }
                }
            }

            // done with the temporary storage
            delete [] recflag;
            delete [] recval;
        }

        // Finally, broadcast the results out.  t2 should now point to the
        // end of the target buffer.
        if (comm.getNodes() > 1)
        {
            Message *sendmsg = new Message();
            ::putMessage(*sendmsg, output);
            // dbgmsg << "sending message with size " << sendmsg->size();
            // dbgmsg << " to all nodes with tag " << rectag << endl;
            if (comm.broadcast_others(sendmsg, rectag) != (comm.getNodes() - 1))
                Ippl::abort("reduce: cannot send reduce results.");
        }

        // we're done ... but do a check to see that we reduced SOMETHING
	/* ADA: can be "savely" ignored ...
        if (!IncludeVal)
        {
            WARNMSG("reduce: there was nothing to reduce, since the masks ");
            WARNMSG("were all false." << endl);
        }
	*/
    }

    //INCIPPLSTAT(incReductions);
    return true;
}


////////////////////////////////////////////////////////////////////////////
// same as above, but this uses the default Communicate object
template <class T, class ReduceOp>
bool reduce_masked(T& input, T& output, const ReduceOp& op,
                   bool IncludeVal)
{

    return reduce_masked(*Ippl::Comm, input, output, op, IncludeVal);
}


////////////////////////////////////////////////////////////////////////////
// Scatter the data in the given source container to all other nodes.
// The data is read using the first two begin,end iterators, and written
// to the location indicated by the third iterator.  The next two
// arrays are for target nodes and target indices for the data in the
// source array; they should be of the same length as the source array.
// the final argument is an STL predicate which is used to combine data
// when two or more items are scattered into the same location on the
// same node.
// Return success of operation.
template <class InputIterator, class RandomIterator, class ScatterOp>
bool scatter(Communicate& comm, InputIterator s1, InputIterator s2,
             RandomIterator t1, int *target_node,
             int *target_position, const ScatterOp& op)
{

    int i;			// loop variables
    int tag = comm.next_tag(COMM_REDUCE_SCATTER_TAG, COMM_REDUCE_CYCLE);

    // Create a number of send messages equal to TotalNodes
    // these messages will be packed with the data from the source
    // data and sent to the node indicated by target node array
    // some empty messages will be sent so the recieving node knows when
    // it has recieved all the messages
    Message* msg = new Message[comm.getNodes()];

    // Loop over each item of the source array and pack the send messages.
    // The message is packed in pairs, the first element of each pair is
    // an integer representing the array offset in the target. The second
    // element is the data to be placed in that offset.
    int *tn = target_node;
    int *tp = target_position;
    InputIterator si;
    for ( si = s1; si != s2 ; si++, tn++, tp++ )
    {
        if ( *tn < 0 || *tn >= comm.getNodes() )
        {
            ERRORMSG("scatter: bad scatter target " << *tn << endl);
            return false;
        }
        //    msg[*tn].put(*tp).put(*si);
        putMessage(msg[*tn], *tp);
        putMessage(msg[*tn], *si);
    }

    // Send out the messages.  We do not delete the messages here after the
    // send, however.
    for ( i = comm.getNodes() - 1; i >= 0; i-- )
    {
        if ( ! comm.send(msg + i, i, tag, false) )
        {
            ERRORMSG("scatter: cannot send scatter buffer " << i << endl);
            return false;
        }
    }

    // Receive the scatter messages back now.
    int notReceived = comm.getNodes();
    while (notReceived > 0)
    {
        int fromnode = COMM_ANY_NODE;
        Message *recmsg = comm.receive_block(fromnode, tag);
        if ( ! recmsg )
        {
            ERRORMSG("scatter: cannot receive scatter message." << endl);
            return false;
        }

        // for each (pos, val) pair, get it and put results in target storage
        int pairs = recmsg->size() / 2;
        int datapos;
        InputIterator reci;
        for ( i = 0 ; i < pairs ; i++ )
        {
            //      recmsg->get(datapos);
            getMessage(*recmsg, datapos);
            reci = (InputIterator)(recmsg->item(0).data());
            PETE_apply(op, t1[datapos], *reci);
            recmsg->get();	// cleans out the item without another copy
        }

        // Finished with this message.  Delete it if it is from another node; if
        // it is not, we sent it to ourselves and will delete it later.
        if ( fromnode != comm.myNode() )
            delete recmsg;
        notReceived--;
    }

    // at the end, delete the scatter messages, and return success
    delete [] msg;

    //INCIPPLSTAT(incScatters);
    return true;
}


// same as above, but this uses the default Communicate object
template <class InputIterator, class RandomIterator, class ScatterOp>
bool scatter(InputIterator s1, InputIterator s2,
             RandomIterator t1, int *target_node,
             int *target_position, const ScatterOp& op)
{

    return scatter(*Ippl::Comm, s1, s2, t1, target_node, target_position, op);
}

template <typename T>
void gather(const T* input, T* output, int count, int root) {
    MPI_Datatype type = get_mpi_datatype<T>(*input);

    MPI_Gather(const_cast<T*>(input), count, type,
               output, count, type, root, Ippl::getComm());
}


template <typename T>
void scatter(const T* input, T* output, int count, int root) {
    MPI_Datatype type = get_mpi_datatype<T>(*input);

    MPI_Scatter(const_cast<T*>(input), count, type,
                output, count, type, root, Ippl::getComm());
}


template <typename T, class Op>
void reduce(const T* input, T* output, int count, Op op, int root) {
    MPI_Datatype type = get_mpi_datatype<T>(*input);

    MPI_Op mpiOp = get_mpi_op<Op>(op);

    MPI_Reduce(const_cast<T*>(input), output, count, type,
               mpiOp, root, Ippl::getComm());
}

template <typename T, class Op>
void new_reduce(const T* input, T* output, int count, Op op, int root) {
    MPI_Datatype type = get_mpi_datatype<T>(*input);

    MPI_Op mpiOp = get_mpi_op<Op>(op);

    MPI_Reduce(const_cast<T*>(input), output, count, type,
               mpiOp, root, Ippl::getComm());
}


template <typename T, class Op>
void new_reduce(T* inout, int count, Op op, int root) {
    MPI_Datatype type = get_mpi_datatype<T>(*inout);

    MPI_Op mpiOp = get_mpi_op<Op>(op);

    if (Ippl::myNode() == root) {
        MPI_Reduce(MPI_IN_PLACE, inout, count, type,
                   mpiOp, root, Ippl::getComm());
    } else {
        MPI_Reduce(inout, inout, count, type,
                   mpiOp, root, Ippl::getComm());
    }
}


template <typename T, class Op>
void reduce(const T& input, T& output, int count, Op op, int root) {
    reduce(&input, &output, count, op, root);
}


template <typename T, class Op>
void allreduce(const T* input, T* output, int count, Op op) {
    MPI_Datatype type = get_mpi_datatype<T>(*input);

    MPI_Op mpiOp = get_mpi_op<Op>(op);

    MPI_Allreduce(const_cast<T*>(input), output, count, type,
                  mpiOp, Ippl::getComm());
}

template <typename T, class Op>
void allreduce(const T& input, T& output, int count, Op op) {
    allreduce(&input, &output, count, op);
}


template <typename T, class Op>
void allreduce(T* inout, int count, Op op) {
    MPI_Datatype type = get_mpi_datatype<T>(*inout);

    MPI_Op mpiOp = get_mpi_op<Op>(op);

    MPI_Allreduce(MPI_IN_PLACE, inout, count, type,
                  mpiOp, Ippl::getComm());
}


template <typename T, class Op>
void allreduce(T& inout, int count, Op op) {
    allreduce(&inout, count, op);
}