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
