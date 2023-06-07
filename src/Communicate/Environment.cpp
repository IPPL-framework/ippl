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
#include "Ippl.h"

#include "Environment.h"

namespace ippl {
    namespace mpi {

        Environment::Environment(int& argc, char**& argv, const MPI_Comm& comm)
            : comm_m(comm)
        {
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
    } // namespace mpi
}  // namespace ippl
