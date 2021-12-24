//
// Class Communicate
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
#include "Communicate.h"

namespace ippl {
    Communicate::Communicate(int& argc, char**& argv)
    : Communicate(argc, argv, MPI_COMM_WORLD)
    {}


    Communicate::Communicate(int& argc, char**& argv, const MPI_Comm& comm)
    : comm_m(comm)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(comm_m, &rank_m);
        MPI_Comm_size(comm_m, &size_m);
    }

    Communicate::~Communicate() {
        MPI_Finalize();
    }

    void Communicate::irecv(int src, int tag,
                            archive_type& ar, MPI_Request& request, size_type msize)
    {
        if (msize > INT_MAX) {
            std::cerr << "Message size exceeds range of int" << std::endl;
            std::abort();
        }
        MPI_Irecv(ar.getBuffer(), msize,
                MPI_BYTE, src, tag, comm_m, &request);
    }
}
