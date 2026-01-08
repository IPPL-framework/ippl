
#include "Communicate/Communicator.h"

namespace ippl::mpi {

    Communicator::Communicator()
        : buffer_handlers_m(get_buffer_handler_instance())
        , comm_m(new MPI_Comm(MPI_COMM_WORLD)) {
        MPI_Comm_rank(*comm_m, &rank_m);
        MPI_Comm_size(*comm_m, &size_m);
    }

    Communicator::Communicator(MPI_Comm comm) {
        buffer_handlers_m = get_buffer_handler_instance();
        comm_m            = std::make_shared<MPI_Comm>(comm);
        MPI_Comm_rank(*comm_m, &rank_m);
        MPI_Comm_size(*comm_m, &size_m);
    }

    Communicator& Communicator::operator=(MPI_Comm comm) {
        buffer_handlers_m = get_buffer_handler_instance();
        comm_m            = std::make_shared<MPI_Comm>(comm);
        MPI_Comm_rank(*comm_m, &rank_m);
        MPI_Comm_size(*comm_m, &size_m);
        return *this;
    }

    Communicator Communicator::Communicator::split(int color, int key) const {
        MPI_Comm newcomm;
        MPI_Comm_split(*comm_m, color, key, &newcomm);
        return Communicator(newcomm);
    }

    void Communicator::probe(int source, int tag, Status& status) {
        MPI_Probe(source, tag, *comm_m, status);
    }

    bool Communicator::iprobe(int source, int tag, Status& status) {
        int flag = 0;
        MPI_Iprobe(source, tag, *comm_m, &flag, status);
        return (flag != 0);
    }

    // ---------------------------------------
    // singleton access to buffer manager
    // ---------------------------------------
    std::shared_ptr<Communicator::buffer_handler_type> Communicator::get_buffer_handler_instance() {
        static std::shared_ptr<Communicator::buffer_handler_type> comm_buff_handler_ptr{nullptr};
        if (comm_buff_handler_ptr == nullptr) {
            comm_buff_handler_ptr = std::make_shared<Communicator::buffer_handler_type>();
            SPDLOG_DEBUG("BufferHandler new: {}",
                         ippl::debug::print_type<Communicator::buffer_handler_type>());
        }
        return comm_buff_handler_ptr;
    }
}  // namespace ippl::mpi
