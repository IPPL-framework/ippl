//
// Class Environment
//
#ifndef IPPL_MPI_ENVIRONMENT_H
#define IPPL_MPI_ENVIRONMENT_H

#include <mpi.h>

namespace ippl {
    namespace mpi {
        /*!
         * @file Environment.h
         */
        class Environment {
        public:
            Environment() = delete;

            Environment(int& argc, char**& argv, const MPI_Comm& comm = MPI_COMM_WORLD);

            ~Environment();

            static bool initialized();

            static bool finalized();

            bool threadMultiple() noexcept { return threadMultiple_m; }

            void abort(int errorcode = -1) noexcept { MPI_Abort(comm_m, errorcode); }

        private:
            MPI_Comm comm_m;
            bool threadMultiple_m;
        };
    }  // namespace mpi
}  // namespace ippl

#endif
