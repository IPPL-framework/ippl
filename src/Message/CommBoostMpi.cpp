//
// Class Communicate
//   Communicator class using Boost.MPI
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#include "CommBoostMpi.h"

namespace ippl {
    Communicate::Communicate()
    : Communicate(MPI_COMM_WORLD)
    {}


    Communicate::Communicate(const MPI_Comm& comm)
    : boost::mpi::communicator(comm, kind_type::comm_duplicate)
    {}
}