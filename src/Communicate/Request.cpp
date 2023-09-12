//
// Class Request
//   A communication request handle for non-blocking communication.
//
#include "Communicate/Request.h"

namespace ippl {
    namespace mpi {

        Request::~Request() {
            if (request_m != MPI_REQUEST_NULL) {
                this->free();
            }
        }

        bool Request::completed() {
            int flag = 0;

            Status status;

            // MPI_STATUS_IGNORE
            MPI_Request_get_status(request_m, &flag, status);

            if (flag != 0) {
                // valid Status instance
                MPI_Test(&request_m, &flag, status);
            } else {
                // Although we free the request, any ongoing communication
                // associated with this request is allowed to complete.
                this->free();
            }

            return (flag != 0);
        }
    }  // namespace mpi
}  // namespace ippl
