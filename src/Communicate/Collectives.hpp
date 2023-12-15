#include "Communicate/DataTypes.h"

#include "Communicate/Operations.h"

namespace ippl {
    namespace mpi {
        template <typename T>
        void Communicator::gather(const T* input, T* output, int count, int root) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Gather(const_cast<T*>(input), count, type, output, count, type, root, *comm_m);
        }

        template <typename T>
        void Communicator::scatter(const T* input, T* output, int count, int root) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Scatter(const_cast<T*>(input), count, type, output, count, type, root, *comm_m);
        }

        template <typename T, class Op>
        void Communicator::reduce(const T* input, T* output, int count, Op op, int root) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Op mpiOp = get_mpi_op<Op, T>(op);

            MPI_Reduce(const_cast<T*>(input), output, count, type, mpiOp, root, *comm_m);
        }

        template <typename T, class Op>
        void Communicator::reduce(const T& input, T& output, int count, Op op, int root) {
            reduce(&input, &output, count, op, root);
        }

        template <typename T, class Op>
        void Communicator::allreduce(const T* input, T* output, int count, Op op) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Op mpiOp = get_mpi_op<Op, T>(op);

            MPI_Allreduce(const_cast<T*>(input), output, count, type, mpiOp, *comm_m);
        }

        template <typename T, class Op>
        void Communicator::allreduce(const T& input, T& output, int count, Op op) {
            allreduce(&input, &output, count, op);
        }

        template <typename T, class Op>
        void Communicator::allreduce(T* inout, int count, Op op) {
            MPI_Datatype type = get_mpi_datatype<T>(*inout);

            MPI_Op mpiOp = get_mpi_op<Op, T>(op);

            MPI_Allreduce(MPI_IN_PLACE, inout, count, type, mpiOp, *comm_m);
        }

        template <typename T, class Op>
        void Communicator::allreduce(T& inout, int count, Op op) {
            allreduce(&inout, count, op);
        }
    }  // namespace mpi
}  // namespace ippl
