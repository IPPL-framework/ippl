//
// Class Request
//   A communication request handle for non-blocking communication.
//
#ifndef IPPL_MPI_REQUEST_H
#define IPPL_MPI_REQUEST_H

#include "Communicate/Status.h"

namespace ippl {
    namespace mpi {

        class Request {
        public:
            Request()
                : request_m(MPI_REQUEST_NULL) {}

            ~Request();

            //             operator MPI_Request&() noexcept { return request_m; }

            //             operator const MPI_Request&() const noexcept { return request_m; }

            operator MPI_Request*() noexcept { return &request_m; }

            operator const MPI_Request*() const noexcept { return &request_m; }

            bool completed();

            void free() { MPI_Request_free(&request_m); }

            void wait() { MPI_Wait(&request_m, MPI_STATUS_IGNORE); }

        private:
            MPI_Request request_m;
        };
    }  // namespace mpi
}  // namespace ippl

#endif
