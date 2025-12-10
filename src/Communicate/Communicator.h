//
// Class Communicator
//   Defines a class to do MPI communication.
//
#ifndef IPPL_MPI_COMMUNICATOR_H
#define IPPL_MPI_COMMUNICATOR_H

#include <climits>
#include <cstdlib>
#include <memory>
#include <mpi.h>

#include "Utility/TypeUtils.h"

#include "Communicate/Archive.h"
#include "Communicate/BufferHandler.h"
#include "Communicate/LogEntry.h"
#include "Communicate/Request.h"
#include "Communicate/Status.h"
#include "Communicate/TagMaker.h"
#include "Communicate/Tags.h"

////////////////////////////////////////////////////

namespace ippl::mpi {

    class Communicator : public TagMaker {
    public:
        Communicator();

        Communicator(MPI_Comm comm);

        Communicator& operator=(MPI_Comm comm);

        ~Communicator() = default;

        Communicator split(int color, int key) const;

        operator const MPI_Comm&() const noexcept { return *comm_m; }

        int size() const noexcept { return size_m; }

        int rank() const noexcept { return rank_m; }

        void barrier() { MPI_Barrier(*comm_m); }

        void abort(int errorcode = -1) { MPI_Abort(*comm_m, errorcode); }

        /*
         * Blocking point-to-point communication
         *
         */

        template <typename T>
        void send(const T& buffer, int count, int dest, int tag);

        template <typename T>
        void send(const T* buffer, int count, int dest, int tag);

        template <typename T>
        void recv(T& output, int count, int source, int tag, Status& status);

        template <typename T>
        void recv(T* output, int count, int source, int tag, Status& status);

        void probe(int source, int tag, Status& status);

        /*
         * Non-blocking point-to-point communication
         *
         */

        template <typename T>
        void isend(const T& buffer, int count, int dest, int tag, Request& request);

        template <typename T>
        void isend(const T* buffer, int count, int dest, int tag, Request& request);

        template <typename T>
        void irecv(T& buffer, int count, int source, int tag, Request& request);

        template <typename T>
        void irecv(T* buffer, int count, int source, int tag, Request& request);

        bool iprobe(int source, int tag, Status& status);

        /*
         * Collective communication
         */

        /* Gather the data in the given source container from all other nodes to a
         * specific node (default: 0).
         */
        template <typename T>
        void gather(const T* input, T* output, int count, int root = 0);

        /* Scatter the data from all other nodes to a
         * specific node (default: 0).
         */
        template <typename T>
        void scatter(const T* input, T* output, int count, int root = 0);

        /* Reduce data coming from all nodes to a specific node
         * (default: 0). Apply certain operation
         *
         */
        template <typename T, class Op>
        void reduce(const T* input, T* output, int count, Op op, int root = 0);

        template <typename T, class Op>
        void reduce(const T& input, T& output, int count, Op op, int root = 0);

        template <typename T, class Op>
        void allreduce(const T* input, T* output, int count, Op op);

        template <typename T, class Op>
        void allreduce(const T& input, T& output, int count, Op op);

        template <typename T, class Op>
        void allreduce(T* inout, int count, Op op);

        template <typename T, class Op>
        void allreduce(T& inout, int count, Op op);

    private:
        template <typename MemorySpace>
        using buffer_container_type = comms::DefaultBufferHandler<MemorySpace>;

        using buffer_handler_type =
            typename detail::ContainerForAllSpaces<buffer_container_type>::type;

    public:
        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
        using buffer_type = buffer_container_type<MemorySpace>::buffer_type;

    public:
        template <typename MemorySpace>
        struct async_send_data {
            buffer_type<MemorySpace> async_buffer;
            int tag;
            MPI_Request request;
        };

    public:
        using size_type = detail::size_type;
        double getDefaultOverallocation() const { return defaultOveralloc_m; }
        void setDefaultOverallocation(double factor);

        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
                  typename T           = char>
        buffer_type<MemorySpace> getBuffer(size_type size, double overallocation = 1.0);

        void deleteAllBuffers();
        void freeAllBuffers();

        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
        void freeBuffer(buffer_type<MemorySpace> buffer);

        const MPI_Comm& getCommunicator() const noexcept { return *comm_m; }

        template <class Buffer, typename Archive>
        void recv(int src, int tag, Buffer& buffer, Archive& ar, size_type msize,
                  size_type nrecvs) {
            // Temporary fix. MPI communication seems to have problems when the
            // count argument exceeds the range of int, so large messages should
            // be split into smaller messages
            if (msize > INT_MAX) {
                std::cerr << "Message size exceeds range of int" << std::endl;
                this->abort();
            }
            MPI_Status status;
            MPI_Recv(ar.getData(), msize, MPI_BYTE, src, tag, *comm_m, &status);
            SPDLOG_DEBUG("Recv buf {}, size {:04}, src {:02}, tag {:04}", (void*)(ar.getData()),
                         msize, src, tag);
            buffer.deserialize(ar, nrecvs);
        }

        template <class Buffer, typename Archive>
        void isend(int dest, int tag, Buffer& buffer, Archive& ar, MPI_Request& request,
                   size_type nsends)  //
        {
            if (ar.getSize() > INT_MAX) {
                std::cerr << "Message size exceeds range of int" << std::endl;
                this->abort();
            }
            buffer.serialize(ar, nsends);
            MPI_Isend(ar.getData(), ar.getSize(), MPI_BYTE, dest, tag, *comm_m, &request);
            SPDLOG_DEBUG("Isend buf {}, size {:04}, dst {:02}, tag {:04}, req {}",
                         (void*)(ar.getData()), ar.getSize(), dest, tag,
                         static_cast<uintptr_t>(request));
        }

        template <typename Archive>
        void irecv(int src, int tag, Archive& ar, MPI_Request& request, size_type msize) {
            if (msize > INT_MAX) {
                std::cerr << "Message size exceeds range of int" << std::endl;
                this->abort();
            }

            MPI_Irecv(ar.getData(), msize, MPI_BYTE, src, tag, *comm_m, &request);
            SPDLOG_DEBUG("Irecv buf {}, size {:04}, src {:02}, tag {:04}, req {}",
                         (void*)(ar.getData()), msize, src, tag, static_cast<uintptr_t>(request));
        }

        void printLogs(const std::string& filename);

    private:
        std::vector<LogEntry> gatherLocalLogs();
        void sendLogsToRank0(const std::vector<LogEntry>& localLogs);
        std::vector<LogEntry> gatherLogsFromAllRanks(const std::vector<LogEntry>& localLogs);
        void writeLogsToFile(const std::vector<LogEntry>& allLogs, const std::string& filename);

        std::shared_ptr<buffer_handler_type> buffer_handlers_m;
        double defaultOveralloc_m = 1.0;

        /////////////////////////////////////////////////////////////////////////////////////

    protected:
        std::shared_ptr<MPI_Comm> comm_m;
        int size_m;
        int rank_m;

    public:
        std::shared_ptr<buffer_handler_type> get_buffer_handler_instance();
    };

}  // namespace ippl::mpi

#include "Communicate/Collectives.hpp"
#include "Communicate/PointToPoint.hpp"

////////////////////////////////////

#include "Communicate/Buffers.hpp"

////////////////////////////////////

#endif
