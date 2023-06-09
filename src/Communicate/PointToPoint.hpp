namespace ippl {
    namespace mpi {

        /*
         * Blocking point-to-point communication
         */

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

        /*
         * Non-blocking point-to-point communication
         */

        template <typename T>
        void Communicator::isend(const T& buffer, int count, int dest, int tag, Request& request) {
            this->isend(&buffer, count, dest, tag, request);
        }

        template <typename T>
        void Communicator::isend(const T* buffer, int count, int dest, int tag, Request& request) {
            MPI_Datatype type = get_mpi_datatype<T>(*buffer);

            MPI_Isend(buffer, count, type, dest, tag, *comm_m, request);
        }

        template <typename T>
        void Communicator::irecv(T& buffer, int count, int source, int tag, Request& request) {
            this->irecv(&buffer, count, source, tag, request);
        }

        template <typename T>
        void Communicator::irecv(T* buffer, int count, int source, int tag, Request& request) {
            MPI_Datatype type = get_mpi_datatype<T>(*buffer);

            MPI_Irecv(buffer, count, type, source, tag, *comm_m, request);
        }

    }  // namespace mpi
}  // namespace ippl
