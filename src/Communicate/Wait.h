//
// Global MPI Wait functions
//   Defines wait functions for non-blocking send/receive communication.
//
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
#ifndef IPPL_MPI_WAIT_H
#define IPPL_MPI_WAIT_H

#include <iterator>

#include "Communicate/Request.h"

namespace ippl {
    namespace mpi {
        template <std::contiguous_iterator InputIter, std::contiguous_iterator OutputIter>
        void waitall(InputIter req_first, InputIter req_last, OutputIter sta_first) {
            auto count = std::distance(req_first, req_last);
            MPI_Waitall(count, *req_first, *sta_first);
        }

        void wait(Request& request, Status& status) {
            MPI_Wait(request, status);
        }

    }  // namespace mpi
}  // namespace ippl

#endif
