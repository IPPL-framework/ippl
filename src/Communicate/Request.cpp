//
// Class Request
//   A communication request handle for non-blocking communication.
//
//
// Copyright (c) 2023, Matthias Frey, University of St Andrews, UK
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
