//
// Global MPI Wait functions
//   Defines wait functions for non-blocking send/receive communication.
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
