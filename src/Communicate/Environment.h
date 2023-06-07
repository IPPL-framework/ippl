//
// Class Environment
//
// Copyright (c) 2023, Matthias Frey, University of St Andrews, UK
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
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

            void abort(int errorcode = -1) noexcept { MPI_Abort(comm_m, errorcode); }

        private:
            MPI_Comm comm_m;
        };
    } // namespace mpi
}  // namespace ippl

#endif
