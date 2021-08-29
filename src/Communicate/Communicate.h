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
#include <map>

#include "Communicate/Archive.h"
#include "Communicate/Tags.h"
#include "Communicate/TagMaker.h"
#include "Communicate/BufferIDs.h"

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
        using buffer_type = std::shared_ptr<archive_type>;
        //using buffer_type = archive_type;

        using size_type = detail::size_type;
        using count_type = detail::count_type;

        Communicate();

        Communicate(const MPI_Comm& comm = MPI_COMM_WORLD);

        /**
         * Query the current default overallocation factor
         * @return Factor by which new buffers are overallocated by default
         */
        int getDefaultOverallocation() const { return defaultOveralloc; }

        /**
         * Set the default overallocation factor
         * @param factor New overallocation factor for new buffers
         */
        void setDefaultOverallocation(int factor);

        /**
         * Obtain a buffer of at least the requested size that is associated
         * with the given ID, overallocating memory for the buffer if it's new
         * @param id The numerical ID with which the buffer is associated (allows buffer reuse)
         * @param size The minimum size of the buffer
         * @param overallocation The factor by which memory for the buffer should be overallocated
         *                       (default 1); only used if the buffer with the given ID has not been
         *                       allocated before
         * @return A shared pointer to the buffer with the requested properties
         */
        buffer_type getBuffer(int id, size_type size, int overallocation = 1);

        /**
         * Deletes a buffer
         * @param id Buffer ID
         */
        void deleteBuffer(int id);

        /**
         * Deletes all buffers created by the buffer factory
         */
        void deleteAllBuffers();

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

        template <class Buffer>
        void recv(int src, int tag, Buffer& buffer, archive_type& ar, size_type msize, count_type nrecvs);

        template <class Buffer>
        void recv(int src, int tag, Buffer& buffer, archive_type& ar, count_type nrecvs);


        /*!
         * \warning Only works with default spaces!
         */
        template <class Buffer>
        void isend(int dest, int tag, Buffer& buffer, archive_type&, MPI_Request&, count_type nsends);


        /*!
         * \warning Only works with default spaces!
         */

        void irecv(int src, int tag, archive_type&, MPI_Request&, size_type msize);

    private:
        std::map<int, buffer_type> buffers;
        int defaultOveralloc = 1;
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
    void Communicate::recv(int src, int tag, Buffer& buffer, archive_type& ar, size_type msize, count_type nrecvs)
    {
        MPI_Status status;
        MPI_Recv(ar.getBuffer(), msize,
                MPI_BYTE, src, tag, *this, &status);

        buffer.deserialize(ar, nrecvs);
    }

    template <class Buffer>
    void Communicate::recv(int src, int tag, Buffer& buffer, archive_type& ar, count_type nrecvs)
    {
        MPI_Status status;
        MPI_Probe(src, tag, *this, &status);

        int msize = 0;
        MPI_Get_count(&status, MPI_BYTE, &msize);

        MPI_Recv(ar.getBuffer(), msize,
                MPI_BYTE, src, tag, *this, &status);

        buffer.deserialize(ar, nrecvs);
    }


    template <class Buffer>
    void Communicate::isend(int dest, int tag, Buffer& buffer,
                            archive_type& ar, MPI_Request& request, count_type nsends)
    {

        buffer.serialize(ar, nsends);
        MPI_Isend(ar.getBuffer(), ar.getSize(),
                  MPI_BYTE, dest, tag, *this, &request);
    }
}

#endif
