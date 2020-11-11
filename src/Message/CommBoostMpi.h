#ifndef IPPL_COMM_MPI_H
#define IPPL_COMM_MPI_H

#include <boost/mpi/communicator.hpp>

// To be removed
#include "Message/Tags.h"
#include "Message/TagMaker.h"
class Message;

namespace ippl {
    class Communicate : boost::mpi::communicator
                      , public TagMaker
    {

    public:
        using kind_type = boost::mpi::comm_create_kind;

        Communicate() = default;

        Communicate(const MPI_Comm& comm = MPI_COMM_WORLD);


        ~Communicate() = default;


        [[deprecated]]
        int myNode() const noexcept {
            return this->rank();
        }

        [[deprecated]]
        int getNodes() const noexcept {
            return this->size();
        }


        [[deprecated]]
        const char *name() const noexcept {
            return "MPI";
        }


        [[deprecated]]
        int broadcast_others(Message *, int, bool = true) {
            return 0;
        }

        [[deprecated]]
        Message *receive_block(int& /*node*/, int& /*tag*/) {
            return nullptr;
        }
    };
}


#endif