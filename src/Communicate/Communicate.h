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
#ifndef IPPL_COMMUNICATE_H
#define IPPL_COMMUNICATE_H

#include <mpi.h>
#include <map>

// For message size check; see below
#include <climits>
#include <cstdlib>

#include "Communicate/Archive.h"
#include "Communicate/TagMaker.h"
#include "Communicate/Tags.h"

namespace ippl {
    /*!
     * @file Communicate.h
     *
     * \remark Calling the plain *this pointer returns the MPI communicator, e.g. MPI_COMM_WORLD.
     */
    class Communicate : public TagMaker {
    public:
        // Attention: only works with default spaces
        using archive_type = detail::Archive<>;
        using buffer_type  = std::shared_ptr<archive_type>;

        using size_type = detail::size_type;

        Communicate(int& argc, char**& argv);

        Communicate(int& argc, char**& argv, const MPI_Comm& comm = MPI_COMM_WORLD);

        ~Communicate();

        /**
         * Query the current default overallocation factor
         * @return Factor by which new buffers are overallocated by default
         */
        double getDefaultOverallocation() const {
            return defaultOveralloc_m;
        }

        /**
         * Set the default overallocation factor
         * @param factor New overallocation factor for new buffers
         */
        void setDefaultOverallocation(double factor);

        /**
         * Obtain a buffer of at least the requested size that is associated
         * with the given ID, overallocating memory for the buffer if it's new
         * @tparam T The datatype to be stored in the buffer; in particular, the size
         *           is scaled by the size of the data type (default char for when
         *           the size is already in bytes)
         * @param id The numerical ID with which the buffer is associated (allows buffer reuse)
         * @param size The minimum size of the buffer, measured in number of elements
         *             of the provided datatype (if the size is in bytes, the default
         *             type char should be used)
         * @param overallocation The factor by which memory for the buffer should be
         *                       overallocated; only used if the buffer with the given
         *                       ID has not been allocated before; by default, the larger
         *                       value between 1 and the defaultOveralloc_m member
         *                       is used
         * @return A shared pointer to the buffer with the requested properties
         */
        template <typename T = char>
        buffer_type getBuffer(int id, size_type size, double overallocation = 1.0);

        /**
         * Deletes a buffer
         * @param id Buffer ID
         */
        void deleteBuffer(int id);

        /**
         * Deletes all buffers created by the buffer factory
         */
        void deleteAllBuffers();

        [[deprecated]] int myNode() const noexcept {
            return rank_m;
        }

        [[deprecated]] int getNodes() const noexcept {
            return size_m;
        }

        [[deprecated]] const char* name() const noexcept {
            return "MPI";
        }

        int size() const noexcept {
            return size_m;
        }

        int rank() const noexcept {
            return rank_m;
        }

        /*!
         * \warning Only works with default spaces!
         */
        template <class Buffer>
        void recv(
            int src, int tag, Buffer& buffer, archive_type& ar, size_type msize, size_type nrecvs);

        /*!
         * \warning Only works with default spaces!
         */
        template <class Buffer>
        void isend(
            int dest, int tag, Buffer& buffer, archive_type&, MPI_Request&, size_type nsends);

        /*!
         * \warning Only works with default spaces!
         */
        void irecv(int src, int tag, archive_type&, MPI_Request&, size_type msize);

        MPI_Comm* getCommunicator() noexcept {
            return &comm_m;
        }

        void barrier() noexcept {
            MPI_Barrier(comm_m);
        }

    private:
        std::map<int, buffer_type> buffers_m;
        double defaultOveralloc_m = 1.0;

        MPI_Comm comm_m;
        int size_m;
        int rank_m;
    };

    template <class Buffer>
    void Communicate::recv(
        int src, int tag, Buffer& buffer, archive_type& ar, size_type msize, size_type nrecvs) {
        // Temporary fix. MPI communication seems to have problems when the
        // count argument exceeds the range of int, so large messages should
        // be split into smaller messages
        if (msize > INT_MAX) {
            std::cerr << "Message size exceeds range of int" << std::endl;
            std::abort();
        }
        MPI_Status status;
        MPI_Recv(ar.getBuffer(), msize, MPI_BYTE, src, tag, comm_m, &status);

        buffer.deserialize(ar, nrecvs);
    }

    template <class Buffer>
    void Communicate::isend(
        int dest, int tag, Buffer& buffer, archive_type& ar, MPI_Request& request,
        size_type nsends) {
        if (ar.getSize() > INT_MAX) {
            std::cerr << "Message size exceeds range of int" << std::endl;
            std::abort();
        }
        buffer.serialize(ar, nsends);
        MPI_Isend(ar.getBuffer(), ar.getSize(), MPI_BYTE, dest, tag, comm_m, &request);
    }
}  // namespace ippl

#include "Communicate/Buffers.hpp"

#endif
