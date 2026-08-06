
#include "Communicate/Communicator.h"

namespace ippl::mpi {

    namespace {
        // Populate rank_m / size_m from the live communicator.
        void cacheRankAndSize(const MPI_Comm& comm, int& rank, int& size) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        }
    }  // namespace

    Communicator::Communicator()
        : comm_m(new MPI_Comm(MPI_COMM_WORLD)) {
        cacheRankAndSize(*comm_m, rank_m, size_m);
    }

    Communicator::Communicator(MPI_Comm comm) {
        comm_m = std::make_shared<MPI_Comm>(comm);
        cacheRankAndSize(*comm_m, rank_m, size_m);
    }

    Communicator& Communicator::operator=(MPI_Comm comm) {
        comm_m = std::make_shared<MPI_Comm>(comm);
        cacheRankAndSize(*comm_m, rank_m, size_m);
        return *this;
    }

    Communicator Communicator::split(int color, int key) const {
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

}  // namespace ippl::mpi
