#include "CommBoostMpi.h"

#include <Kokkos_Core.hpp>

namespace ippl {
    Communicate::Communicate(int argc, char* argv[],
                             const MPI_Comm& comm)
    : env_m(argc, argv)
    , world_m(comm, kind_type::comm_duplicate)
    , rank_m(world_m.rank())
    , size_m(world_m.size())
    {
        Kokkos::initialize(argc, argv);
    }




    Communicate::~Communicate() {
        Kokkos::finalize();
    }
}