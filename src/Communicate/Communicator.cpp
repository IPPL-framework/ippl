
#include "Communicate/Communicator.h"

namespace ippl {
    namespace mpi {

        Communicator::Communicator()
            : comm_m(new MPI_Comm(MPI_COMM_WORLD)) {
            MPI_Comm_rank(*comm_m, &rank_m);
            MPI_Comm_size(*comm_m, &size_m);
        }

        Communicator::Communicator(MPI_Comm& comm) {
            comm_m = std::make_shared<MPI_Comm>(comm);
            MPI_Comm_rank(*comm_m, &rank_m);
            MPI_Comm_size(*comm_m, &size_m);
        }

        Status Communicator::probe(int source, int tag) {
            Status status;
            MPI_Probe(source, tag, *comm_m, status);
            return status;
        }
    }  // namespace mpi
}  // namespace ippl
