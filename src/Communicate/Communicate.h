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

#include "Communicate/Archive.h"
#include "Communicate/Tags.h"
#include "Communicate/TagMaker.h"

namespace ippl {
    /*!
     * @file Communicate.h
     *
     * \remark Calling the plain *this pointer returns the MPI communicator, e.g. MPI_COMM_WORLD.
     */
    class Communicate : public boost::mpi::communicator
                      , public TagMaker
    {

    public:
        using kind_type = boost::mpi::comm_create_kind;

        // Attention: only works with default spaces
        using archive_type = detail::Archive<>;

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

        /*!
         * \warning Only works with default spaces!
         */
        template <class Buffer>
        void send(int dest, int tag, Buffer& buffer);


        /*!
         * \warning Only works with default spaces!
         */
        template <class Buffer>
        void recv(int src, int tag, Buffer& buffer);


        /*!
         * \warning Only works with default spaces!
         */
        template <class Buffer>
        void isend(int dest, int tag, Buffer& buffer, archive_type&, MPI_Request&);


        /*!
         * \warning Only works with default spaces!
         */
        template <class Buffer>
        void irecv(int src, int tag, Buffer& buffer);

    };


    template <class Buffer>
    void Communicate::send(int dest, int tag, Buffer& buffer)
    {
        // Attention: only works with default spaces
        archive_type ar;

        buffer.serialize(ar);

        MPI_Send(ar.getBuffer(), ar.getSize(),
                 MPI_BYTE, dest, tag, *this);
    }


    template <class Buffer>
    void Communicate::recv(int src, int tag, Buffer& buffer)
    {
        MPI_Status status;

        MPI_Probe(src, tag, *this, &status);

        int msize = 0;
        MPI_Get_count(&status, MPI_BYTE, &msize);

        // Attention: only works with default spaces
        archive_type ar(msize);

        MPI_Recv(ar.getBuffer(), ar.getSize(),
                MPI_BYTE, src, tag, *this, &status);

        buffer.deserialize(ar);
    }


    template <class Buffer>
    void Communicate::isend(int dest, int tag, Buffer& buffer,
                            archive_type& ar, MPI_Request& request)
    {
        buffer.serialize(ar);

        MPI_Isend(ar.getBuffer(), ar.getSize(),
                  MPI_BYTE, dest, tag, *this, &request);
    }
}


#endif
