
#include "Communicate/Communicator.h"

namespace ippl {
    namespace mpi {

        Communicator::Communicator()
            : comm_m(new MPI_Comm(MPI_COMM_WORLD)) {
            MPI_Comm_rank(*comm_m, &rank_m);
            MPI_Comm_size(*comm_m, &size_m);
        }

        Communicator::Communicator(MPI_Comm comm) {
            comm_m = std::make_shared<MPI_Comm>(comm);
            MPI_Comm_rank(*comm_m, &rank_m);
            MPI_Comm_size(*comm_m, &size_m);
        }

        Communicator& Communicator::operator=(MPI_Comm comm) {
            comm_m = std::make_shared<MPI_Comm>(comm);
            MPI_Comm_rank(*comm_m, &rank_m);
            MPI_Comm_size(*comm_m, &size_m);
            return *this;
        }

        Communicator Communicator::Communicator::split(int color, int key) const {
            MPI_Comm newcomm;
            MPI_Comm_split(*comm_m, color, key, &newcomm);
            return Communicator(newcomm);
        }

        void Communicator::probe(int source, int tag, Status& status) {
            MPI_Probe(source, tag, *comm_m, status);
        }

        bool Communicator::iprobe(int source, int tag, Status& status) {
            int flag = 0;
            MPI_Iprobe(source, tag, *comm_m, &flag, status);
            return (flag != 0);
        }
    }  // namespace mpi
}  // namespace ippl
