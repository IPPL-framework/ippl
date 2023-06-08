namespace ippl {
    namespace mpi {

        template <typename T>
        void Communicator::send(const T& buf, int count, int dest, int tag) {
            this->send(&buf, count, dest, tag);
        }

        template <typename T>
        void Communicator::send(const T* buf, int count, int dest, int tag) {
            MPI_Datatype type = get_mpi_datatype<T>(buf);

            MPI_Send(&buf, count, type, dest, tag, comm_m);
        }

        template <typename T>
        void Communicator::recv(T& output, int count, int source, int tag, Status& status) {
            this->recv(&output, count, source, tag, status);
        }

        template <typename T>
        void Communicator::recv(T* output, int count, int source, int tag, Status& status) {
            MPI_Datatype type = get_mpi_datatype<T>(*output);

            MPI_Recv(output, count, type, source, tag, comm_m, &status);
        }
    }  // namespace mpi
}  // namespace ippl
