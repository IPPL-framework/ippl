//
// Class Environment
//
#include "Ippl.h"

#include "Environment.h"

namespace ippl {
    namespace mpi {

        Environment::Environment(int& argc, char**& argv, const MPI_Comm& comm)
            : comm_m(comm) {
            if (!initialized()) {
                MPI_Init(&argc, &argv);
            }
        }

        Environment::~Environment() {
            if (!finalized()) {
                MPI_Finalize();
            }
        }

        bool Environment::initialized() {
            int flag = 0;
            MPI_Initialized(&flag);
            return (flag != 0);
        }

        bool Environment::finalized() {
            int flag = 0;
            MPI_Finalized(&flag);
            return (flag != 0);
        }
    }  // namespace mpi
}  // namespace ippl
