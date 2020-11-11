#ifndef IPPL_COMM_MPI_H
#define IPPL_COMM_MPI_H

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

// To be removed
#include "Message/Tags.h"
#include "Message/TagMaker.h"
class Message;

namespace ippl {
    class Communicate : public TagMaker {

    public:
        using env_type = boost::mpi::environment;
        using comm_type = boost::mpi::communicator;
        using kind_type = boost::mpi::comm_create_kind;

        Communicate() = default;

        Communicate(int argc, char* argv[],
                    const MPI_Comm& comm = MPI_COMM_WORLD);


        ~Communicate();


        int myRank() const noexcept;

        int getSize() const noexcept;


        [[deprecated]]
        int myNode() const noexcept {
            return myRank();
        }

        [[deprecated]]
        int getNodes() const noexcept {
            return getSize();
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

    private:
        env_type env_m;
        comm_type world_m;

        int rank_m;
        int size_m;
    };



    inline
    int Communicate::myRank() const noexcept {
        return rank_m;
    }


    inline
    int Communicate::getSize() const noexcept {
        return size_m;
    }

}

#endif