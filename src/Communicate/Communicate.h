//
// Class Communicate
//
#ifndef IPPL_COMMUNICATE_H
#define IPPL_COMMUNICATE_H

#include <map>
#include <mpi.h>

// For message size check; see below
#include <climits>
#include <cstdlib>
#include <variant>

#include "Utility/TypeUtils.h"

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
        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
        using archive_type = detail::Archive<MemorySpace>;

        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
        using buffer_type = std::shared_ptr<archive_type<MemorySpace>>;

    private:
        template <typename MemorySpace>
        using map_type = std::map<int, buffer_type<MemorySpace>>;

        using buffer_map_type = typename detail::ContainerForAllSpaces<map_type>::type;

    public:
        using size_type = detail::size_type;

        Communicate(int& argc, char**& argv);

        Communicate(int& argc, char**& argv, const MPI_Comm& comm = MPI_COMM_WORLD);

        ~Communicate();

        /**
         * Query the current default overallocation factor
         * @return Factor by which new buffers are overallocated by default
         */
        double getDefaultOverallocation() const { return defaultOveralloc_m; }

        /**
         * Set the default overallocation factor
         * @param factor New overallocation factor for new buffers
         */
        void setDefaultOverallocation(double factor);

        /**
         * Obtain a buffer of at least the requested size that is associated
         * with the given ID, overallocating memory for the buffer if it's new
         * @tparam MemorySpace The Kokkos memory space for which to allocate the buffer
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
        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
                  typename T           = char>
        buffer_type<MemorySpace> getBuffer(int id, size_type size, double overallocation = 1.0);

        /**
         * Deletes a buffer
         * @param id Buffer ID
         */
        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
        void deleteBuffer(int id) {
            buffers_m.get<MemorySpace>().erase(id);
        }

        /**
         * Deletes all buffers created by the buffer factory
         */
        void deleteAllBuffers();

        [[deprecated]] int myNode() const noexcept { return rank_m; }

        [[deprecated]] int getNodes() const noexcept { return size_m; }

        [[deprecated]] const char* name() const noexcept { return "MPI"; }

        int size() const noexcept { return size_m; }

        int rank() const noexcept { return rank_m; }

        template <class Buffer, typename Archive>
        void recv(int src, int tag, Buffer& buffer, Archive& ar, size_type msize, size_type nrecvs);

        template <class Buffer, typename Archive>
        void isend(int dest, int tag, Buffer& buffer, Archive&, MPI_Request&, size_type nsends);

        /*!
         * \warning Only works with default spaces!
         */
        void irecv(int src, int tag, archive_type<>&, MPI_Request&, size_type msize);

        const MPI_Comm& getCommunicator() const noexcept { return comm_m; }

        void setCommunicator(const MPI_Comm& comm) noexcept { comm_m = comm; }

        void barrier() noexcept { MPI_Barrier(comm_m); }

        void abort(int errorcode = -1) noexcept { MPI_Abort(comm_m, errorcode); }

    private:
        /*!
         * For each memory space, store a map of all buffers
         * created for that space
         */
        buffer_map_type buffers_m;
        double defaultOveralloc_m = 1.0;

        MPI_Comm comm_m;
        int size_m;
        int rank_m;
    };

    template <class Buffer, typename Archive>
    void Communicate::recv(int src, int tag, Buffer& buffer, Archive& ar, size_type msize,
                           size_type nrecvs) {
        // Temporary fix. MPI communication seems to have problems when the
        // count argument exceeds the range of int, so large messages should
        // be split into smaller messages
        if (msize > INT_MAX) {
            std::cerr << "Message size exceeds range of int" << std::endl;
            this->abort();
        }
        MPI_Status status;
        MPI_Recv(ar.getBuffer(), msize, MPI_BYTE, src, tag, comm_m, &status);

        buffer.deserialize(ar, nrecvs);
    }

    template <class Buffer, typename Archive>
    void Communicate::isend(int dest, int tag, Buffer& buffer, Archive& ar, MPI_Request& request,
                            size_type nsends) {
        if (ar.getSize() > INT_MAX) {
            std::cerr << "Message size exceeds range of int" << std::endl;
            this->abort();
        }
        buffer.serialize(ar, nsends);
        MPI_Isend(ar.getBuffer(), ar.getSize(), MPI_BYTE, dest, tag, comm_m, &request);
    }
}  // namespace ippl

#include "Communicate/Buffers.hpp"

#endif
