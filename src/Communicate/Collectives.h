// Global communication functions, such as reduce and scatter.

#ifndef IPPL_MPI_COLLECTIVES_H
#define IPPL_MPI_COLLECTIVES_H

namespace ippl {
    namespace mpi {
        /* Gather the data in the given source container from all other nodes to a
        * specific node (default: 0).
        */
        template <typename T>
        void gather(const T* input, T* output, int count,
                    const MPI_Comm& comm = Comm->getCommunicator(), int root = 0);

        /* Scatter the data from all other nodes to a
        * specific node (default: 0).
        */
        template <typename T>
        void scatter(const T* input, T* output, int count,
                     const MPI_Comm& comm = Comm->getCommunicator(), int root = 0);

        /* Reduce data coming from all nodes to a specific node
        * (default: 0). Apply certain operation
        *
        */
        template <typename T, class Op>
        void reduce(const T* input, T* output, int count, Op op,
                    const MPI_Comm& comm = Comm->getCommunicator(), int root = 0);

        template <typename T, class Op>
        void reduce(const T& input, T& output, int count, Op op,
                    const MPI_Comm& comm = Comm->getCommunicator(), int root = 0);

        template <typename T, class Op>
        void allreduce(const T* input, T* output, int count, Op op,
                       const MPI_Comm& comm = Comm->getCommunicator());

        template <typename T, class Op>
        void allreduce(const T& input, T& output, int count, Op op,
                       const MPI_Comm& comm = Comm->getCommunicator());

        template <typename T, class Op>
        void allreduce(T* inout, int count, Op op, const MPI_Comm& comm = Comm->getCommunicator());

        template <typename T, class Op>
        void allreduce(T& inout, int count, Op op, const MPI_Comm& comm = Comm->getCommunicator());
    }
}

#include "Communicate/Collectives.hpp"

#endif
