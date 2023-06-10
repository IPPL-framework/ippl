#ifndef IPPL_MPI_COMMUNICATOR_H
#define IPPL_MPI_COMMUNICATOR_H

#include <memory>
#include <mpi.h>

#include "Communicate/Request.h"
#include "Communicate/Status.h"

////////////////////////////////////////////////
// For message size check; see below
#include <climits>
#include <cstdlib>

#include "Communicate/Archive.h"
#include "Communicate/TagMaker.h"
#include "Communicate/Tags.h"
////////////////////////////////////////////////////

namespace ippl {
    namespace mpi {

        class Communicator : public TagMaker {
        public:
            Communicator();

            Communicator(MPI_Comm& comm);

            ~Communicator() = default;

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

            /////////////////////////////////////////////////////////////////////////////////////
            using archive_type = detail::Archive<>;
            using buffer_type  = std::shared_ptr<archive_type>;
            using size_type    = detail::size_type;
            double getDefaultOverallocation() const { return defaultOveralloc_m; }
            void setDefaultOverallocation(double factor);

            template <typename T = char>
            buffer_type getBuffer(int id, size_type size, double overallocation = 1.0);

            void deleteBuffer(int id);
            void deleteAllBuffers();

            const MPI_Comm& getCommunicator() const noexcept { return *comm_m; }

            //         void setCommunicator(const MPI_Comm& comm) noexcept { comm_m.reset(comm); }

            template <class Buffer>
            void recv(int src, int tag, Buffer& buffer, archive_type& ar, size_type msize,
                      size_type nrecvs) {
                // Temporary fix. MPI communication seems to have problems when the
                // count argument exceeds the range of int, so large messages should
                // be split into smaller messages
                if (msize > INT_MAX) {
                    std::cerr << "Message size exceeds range of int" << std::endl;
                    this->abort();
                }
                MPI_Status status;
                MPI_Recv(ar.getBuffer(), msize, MPI_BYTE, src, tag, *comm_m, &status);

                buffer.deserialize(ar, nrecvs);
            }

            template <class Buffer>
            void isend(int dest, int tag, Buffer& buffer, archive_type& ar, MPI_Request& request,
                       size_type nsends) {
                if (ar.getSize() > INT_MAX) {
                    std::cerr << "Message size exceeds range of int" << std::endl;
                    this->abort();
                }
                buffer.serialize(ar, nsends);
                MPI_Isend(ar.getBuffer(), ar.getSize(), MPI_BYTE, dest, tag, *comm_m, &request);
            }

            void irecv(int src, int tag, archive_type& ar, MPI_Request& request, size_type msize) {
                if (msize > INT_MAX) {
                    std::cerr << "Message size exceeds range of int" << std::endl;
                    this->abort();
                }
                MPI_Irecv(ar.getBuffer(), msize, MPI_BYTE, src, tag, *comm_m, &request);
            }

        private:
            std::map<int, buffer_type> buffers_m;
            double defaultOveralloc_m = 1.0;

            /////////////////////////////////////////////////////////////////////////////////////

        protected:
            std::shared_ptr<MPI_Comm> comm_m;
            int size_m;
            int rank_m;
        };
    }  // namespace mpi
}  // namespace ippl

#include "Communicate/Collectives.hpp"
#include "Communicate/PointToPoint.hpp"

////////////////////////////////////

#include "Communicate/Buffers.hpp"

////////////////////////////////////

#endif
