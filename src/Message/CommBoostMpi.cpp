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

#include "Archive.h"

namespace ippl {
    Communicate::Communicate()
    : Communicate(MPI_COMM_WORLD)
    {}


    Communicate::Communicate(const MPI_Comm& comm)
    : boost::mpi::communicator(comm, kind_type::comm_duplicate)
    {}


    template <class Buffer>
    void Communicate::send(int dest, int tag, Buffer& buffer)
    {
        detail::Archive ar;
        buffer.serialize(ar);
        this->send(dest, tag, ar.getBuffer(), ar.getSize());
    }


    template <class Buffer>
    void Communicate::recv(int src, int tag, Buffer& buffer)
    {
        boost::mpi::status msg = this->probe();

        detail::Archive ar;

//         if (msg.source() != src) {

//         }

        this->recv(src, tag, ar.getBuffer(), msg.m_count);

        buffer.deserialize(ar);
    }
}
