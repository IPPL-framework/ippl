//
// Class Communicate
//
#include "Ippl.h"

#include "Communicate.h"

namespace ippl {
    Communicate::Communicate(int& argc, char**& argv)
        : Communicate(argc, argv, MPI_COMM_WORLD) {}

    Communicate::Communicate(int& argc, char**& argv, const MPI_Comm& comm)
        : comm_m(comm) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(comm_m, &rank_m);
        MPI_Comm_size(comm_m, &size_m);
    }

    Communicate::~Communicate() {
        MPI_Finalize();
    }

    void Communicate::irecv(int src, int tag, archive_type<>& ar, MPI_Request& request,
                            size_type msize) {
        if (msize > INT_MAX) {
            std::cerr << "Message size exceeds range of int" << std::endl;
            this->abort();
        }
        MPI_Irecv(ar.getBuffer(), msize, MPI_BYTE, src, tag, comm_m, &request);
    }
}  // namespace ippl
