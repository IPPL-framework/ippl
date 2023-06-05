#include <algorithm>

#include "Communicate/DataTypes.h"
#include "Communicate/Operations.h"
#include "Communicate/Collectives.h"

namespace ippl {
    namespace mpi {
        template <typename T>
        void gather(const T* input, T* output, int count, int root) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Gather(const_cast<T*>(input), count, type, output, count, type, root, Comm->getCommunicator());
        }

        template <typename T>
        void scatter(const T* input, T* output, int count, int root) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Scatter(const_cast<T*>(input), count, type, output, count, type, root, Comm->getCommunicator());
        }

        template <typename T, class Op>
        void reduce(const T* input, T* output, int count, Op op, int root) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Op mpiOp = get_mpi_op<Op>(op);

            MPI_Reduce(const_cast<T*>(input), output, count, type, mpiOp, root, Comm->getCommunicator());
        }

        template <typename T, class Op>
        void reduce(const T& input, T& output, int count, Op op, int root) {
            reduce(&input, &output, count, op, root);
        }

        template <typename T, class Op>
        void allreduce(const T* input, T* output, int count, Op op) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Op mpiOp = get_mpi_op<Op>(op);

            MPI_Allreduce(const_cast<T*>(input), output, count, type, mpiOp, Comm->getCommunicator());
        }

        template <typename T, class Op>
        void allreduce(const T& input, T& output, int count, Op op) {
            allreduce(&input, &output, count, op);
        }

        template <typename T, class Op>
        void allreduce(T* inout, int count, Op op) {
            MPI_Datatype type = get_mpi_datatype<T>(*inout);

            MPI_Op mpiOp = get_mpi_op<Op>(op);

            MPI_Allreduce(MPI_IN_PLACE, inout, count, type, mpiOp, Comm->getCommunicator());
        }

        template <typename T, class Op>
        void allreduce(T& inout, int count, Op op) {
            allreduce(&inout, count, op);
        }
    }
}
