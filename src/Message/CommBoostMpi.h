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
#ifndef IPPL_COMMUNICATE_H
#define IPPL_COMMUNICATE_H

#include <boost/mpi/communicator.hpp>

// To be removed
#include "Message/Tags.h"
#include "Message/TagMaker.h"
class Message;

namespace ippl {
    class Communicate : public boost::mpi::communicator
                      , public TagMaker
    {

    public:
        using kind_type = boost::mpi::comm_create_kind;

        Communicate();

        Communicate(const MPI_Comm& comm = MPI_COMM_WORLD);


        ~Communicate() = default;


        [[deprecated]]
        int myNode() const noexcept {
            return this->rank();
        }

        [[deprecated]]
        int getNodes() const noexcept {
            return this->size();
        }


        [[deprecated]]
        const char *name() const noexcept {
            return "MPI";
        }


        using boost::mpi::communicator::send;
        using boost::mpi::communicator::recv;

        template <class Buffer>
        void send(int dest, int tag, Buffer& buffer);

        template <class Buffer>
        void recv(int dest, int tag, Buffer& buffer);


        [[deprecated]]
        int broadcast_others(Message *, int, bool = true) {
            return 0;
        }

        [[deprecated]]
        Message *receive_block(int& /*node*/, int& /*tag*/) {
            return nullptr;
        }
    };
}


#endif