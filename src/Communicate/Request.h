#ifndef IPPL_MPI_REQUEST_H
#define IPPL_MPI_REQUEST_H

#include "Communicate/Status.h"

namespace ippl {
    namespace mpi {

        class Request {
            Request()
                : request_m(MPI_REQUEST_NULL) {}

            ~Request();

            operator MPI_Request&() noexcept { return request_m; }

            operator const MPI_Request&() const noexcept { return request_m; }

            bool completed();

            void wait();

            void free() { MPI_Request_free(&request_m); }

        private:
            MPI_Request request_m;
        };

        Request::~Request() {
            if (request_m != MPI_REQUEST_NULL) {
                this->free();
            }
        }

        bool Request::completed() {
            int flag = 0;

            Status status;

            // MPI_STATUS_IGNORE
            MPI_Request_get_status(request_m, &flag, &status);

            if (flag != 0) {
                // valid Status instance
                MPI_Test(&request_m, &flag, &status);
            } else {
                // Although we free the request, any ongoing communication
                // associated with this request is allowed to complete.
                this->free();
            }

            return (flag != 0);
        }

        void Request::wait() {
            MPI_Wait(&request_m, MPI_STATUS_IGNORE);
        }
    }  // namespace mpi
}  // namespace ippl
