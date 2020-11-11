#include "CommBoostMpi.h"

namespace ippl {
    Communicate::Communicate(const MPI_Comm& comm)
    : boost::mpi::communicator(comm, kind_type::comm_duplicate)
    {}
}