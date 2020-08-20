// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef GLOBAL_COMM_H
#define GLOBAL_COMM_H

/*
 * GlobalComm.h - Global communication functions, such as reduce and scatter.
 */


// forward declarations
class Communicate;


// Reduce equally-sized arrays across the machine, by sending to node
// 0 and broadcasting back the result.  The arguments are two begin,end
// iterators for the source of the data, an iterator pointing to
// where the summed data should go, and an operation to perform in
// the reduction.  Return success of operation.
// The final argument indicates whether the LOCAL NODE should have it's
// values included in the reduction (by default, this is true).  If this
// pointer to the boolean array is null, all the values will be included.
// NOTE: The input iterators must iterate over simple data objects,
// which do not require their own special getMessage/putMessage.  If you
// need to reduce a complex quantity, use the scalar version of reduce.
template <class InputIterator, class OutputIterator, class ReduceOp>
bool reduce(Communicate&, InputIterator, InputIterator, OutputIterator,
            const ReduceOp&, bool *IncludeVal = 0);

// same as above, but this uses the default Communicate object
template <class InputIterator, class OutputIterator, class ReduceOp>
bool reduce(InputIterator, InputIterator, OutputIterator,
            const ReduceOp&, bool *IncludeVal = 0);

// scalar versions of reduce ... instead of iterators, these versions
// expect a single quantity to reduce and a location to place the result.
template <class T, class ReduceOp>
bool reduce(Communicate& comm, T& input, T& output, const ReduceOp& op);

// same as above, but this uses the default Communicate object
template <class T, class ReduceOp>
bool reduce(T& input, T& output, const ReduceOp& op);


// masked scalar versions of reduce ... instead of iterators, these versions
// expect a single quantity to reduce and a location to place the result.
// The final argument indicates whether the LOCAL NODE should have it's
// value included in the reduction (by default, this is true).
// Return success of operation.
template <class T, class ReduceOp>
bool reduce_masked(Communicate& comm, T& input, T& output, const ReduceOp& op,
                   bool IncludeVal);

// same as above, but this uses the default Communicate object
template <class T, class ReduceOp>
bool reduce_masked(T& input, T& output, const ReduceOp& op,
                   bool IncludeVal);


// scalar versions of reduce ... instead of iterators, these versions
// expect a single quantity to reduce and a location to place the result.
template <class T, class ReduceOp>
bool reduce(Communicate& comm, T& input, T& output, const ReduceOp& op)
{
    return reduce_masked(comm, input, output, op, true);
}

// same as above, but this uses the default Communicate object
template <class T, class ReduceOp>
bool reduce(T& input, T& output, const ReduceOp& op)
{
    return reduce_masked(input, output, op, true);
}


// Scatter the data in the given source container to all other nodes.
// The data is read using the first two begin,end iterators, and written
// to the location indicated by the third iterator.  The next two
// arrays are for target nodes and target indices for the data in the
// source array; they should be of the same length as the source array.
// The final argument is an STL predicate which is used to combine data
// when two or more items are scattered into the same location on the
// same node.
// Return success of operation.
template <class InputIterator, class RandomIterator, class ScatterOp>
bool scatter(Communicate&, InputIterator, InputIterator, RandomIterator,
             int *, int *, const ScatterOp&);

// same as above, but this uses the default Communicate object
template <class InputIterator, class RandomIterator, class ScatterOp>
bool scatter(InputIterator, InputIterator, RandomIterator,
             int *, int *, const ScatterOp&);



/* Gather the data in the given source container from all other nodes to a
 * specific node (default: 0).
 */
template <typename T>
void gather(const T* input, T* output, int count, int root = 0);


/* Scatter the data from all other nodes to a
 * specific node (default: 0).
 */
template <typename T>
void scatter(const T* input, T* output, int count, int root = 0);

/* Reduce data coming from all nodes to a specific node
 * (default: 0). Apply certain operation
 *
 */
template <typename T, class Op>
void reduce(const T* input, T* output, int count, Op op, int root = 0);

template <typename T, class Op>
void new_reduce(const T* input, T* output, int count, Op op, int root = 0);

template <typename T, class Op>
void new_reduce(T* inout, int count, Op op, int root = 0);

template <typename T, class Op>
void reduce(const T& input, T& output, int count, Op op, int root = 0);

template <typename T, class Op>
void allreduce(const T* input, T* output, int count, Op op);

template <typename T, class Op>
void allreduce(const T& input, T& output, int count, Op op);

template <typename T, class Op>
void allreduce(T* inout, int count, Op op);

template <typename T, class Op>
void allreduce(T& inout, int count, Op op);


#include "Message/GlobalComm.hpp"

#endif // GLOBAL_COMM_H