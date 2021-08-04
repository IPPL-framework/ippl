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

        int getDefaultOverallocation() const { return defaultOveralloc; }
        void setDefaultOverallocation(int factor);

        buffer_type getBuffer(int id, size_type size, int overallocation = 1);
        void deleteBuffer(int id);
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
        //template <class Buffer>
        //void irecv(int src, int tag, Buffer& buffer);

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
        //if(this->rank() == 0) {
        //    std::cout << "Rank " << this->rank() << " before receive details " 
        //              << " msize: " << msize
        //              << " src: " << src
        //              << " tag: " << tag
        //              << " buffer size: " << ar.getBufferSize() << std::endl;
        //}
        MPI_Recv(ar.getBuffer(), msize,
                MPI_BYTE, src, tag, *this, &status);

        //if(this->rank() == 0) {
        //    std::cout << "Rank " << this->rank() << " MPI receive from rank " << src << "completed " << std::endl;
        //}
        buffer.deserialize(ar, nrecvs);
        //if(this->rank() == 0) {
        //    std::cout << "Rank " << this->rank() << " deserialize completed " << std::endl;
        //}
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
        //buffer.serialize(ar, nsends);
        ////ar.resetWritePos();
        //if(dest == 0) {
        //    buffer.deserialize(ar, nsends);
        //    ar.resetReadPos();
        //    typename Buffer::particle_position_type::HostMirror R_host = buffer.R.getHostMirror();
        //    typename Buffer::particle_position_type::HostMirror P_host = buffer.P.getHostMirror();
        //    typename Buffer::particle_position_type::HostMirror E_host = buffer.E.getHostMirror();
        //    typename Buffer::particle_index_type::HostMirror ID_host = buffer.ID.getHostMirror();
        //    typename Buffer::particle_charge_type::HostMirror q_host = buffer.q.getHostMirror();
        //    Kokkos::deep_copy(R_host, buffer.R.getView());
        //    Kokkos::deep_copy(P_host, buffer.P.getView());
        //    Kokkos::deep_copy(E_host, buffer.E.getView());
        //    Kokkos::deep_copy(ID_host, buffer.ID.getView());
        //    Kokkos::deep_copy(q_host, buffer.q.getView());
        //    std::cout << "Rank " << this->rank() << " send details " << std::endl; 
        //    std::cout << " nsends: " << nsends << std::endl;
        //    std::cout << "particle ID: " << ID_host(0) << std::endl;
        //    std::cout << "particle charge: " << q_host(0) << std::endl;
        //    std::cout << "particle R: " << R_host(0) << std::endl;
        //    std::cout << "particle momentum: " << P_host(0) << std::endl;
        //    std::cout << "particle E field: " << E_host(0) << std::endl;
        //}



        buffer.serialize(ar, nsends);
        if(this->rank() == 6) {
            std::cout << "Rank " << this->rank() << " send details " 
                      << " msize: " << ar.getSize()
                      //<< " buffer size: " << ar.getBuffer()->getBufferSize()
                      << " dest: " << dest
                      << " tag: " << tag << std::endl;
        }
        MPI_Isend(ar.getBuffer(), ar.getSize(),
                  MPI_BYTE, dest, tag, *this, &request);
    }
}

#endif
