#include "Ippl.h"

#include <iostream>
#include <list>

#include "Communicate/Serializable.h"
#include "Communicate/Wait.h"
#include "Communicate/Window.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        //         int rank = -1;
        //         MPI_Comm comm = *ippl::Comm.get();
        //
        //         MPI_Comm_rank(comm, &rank);
        //
        //         std::cout << "First " << rank << " " << ippl::Comm->rank() << std::endl;
        //
        //         ippl::mpi::Communicator comm2;
        //         comm = comm2;
        //         MPI_Comm_rank(comm, &rank);
        //         std::cout << "Second " << rank << " " << ippl::Comm->rank() << std::endl;
        //
        //
        //         double value = 0;
        //
        //         if (ippl::Comm->rank() == 0) {
        //             value = 10;
        //             ippl::Comm->isend(value, 1, 1, 42, requests[0]);
        //         } else if (ippl::Comm->rank() == 1) {
        //             ippl::Comm->irecv(value, 1, 0, 42, requests[0]);
        //         }
        //
        //         ippl::mpi::waitall(requests.begin(), requests.end(), statuses.begin());
        //
        //         std::cout << ippl::Comm->rank() << " status tag    " << statuses[0].tag() <<
        //         std::endl; std::cout << ippl::Comm->rank() << " status source " <<
        //         statuses[0].source() << std::endl; int cnt = statuses[0].count<double>().value();
        //         std::cout << ippl::Comm->rank() << " status count " << cnt << std::endl;
        //
        //         if (ippl::Comm->rank() == 0) {
        //             std::cout << "rank 0 " << value << std::endl;
        //         } else {
        //             std::cout << "rank 1 " << value << std::endl;
        //         }

        //         std::vector<ippl::mpi::Request> requests(2);
        //         std::vector<ippl::mpi::Status> statuses(2);
        //
        //         int spaceProcs = 2;
        //         int spaceColor = ippl::Comm->rank() / spaceProcs;
        //         int timeColor = ippl::Comm->rank() % spaceProcs;
        //
        //         ippl::mpi::Communicator spaceComm = ippl::Comm->split(spaceColor,
        //         ippl::Comm->rank()); ippl::mpi::Communicator timeComm =
        //         ippl::Comm->split(timeColor, ippl::Comm->rank());
        //
        //         double value[2] = {0, 0};
        //
        //         if (spaceComm.rank() == 0) {
        //             value[0] = 10 * (ippl::Comm->rank() + 1);
        //             spaceComm.isend(value[0], 1, 1, 42, requests[0]);
        //         } else if (spaceComm.rank() == 1) {
        //             spaceComm.irecv(value[0], 1, 0, 42, requests[0]);
        //         }
        //
        //         if (timeComm.rank() == 0) {
        //             value[1] = 20 * (ippl::Comm->rank() + 1);
        //             timeComm.isend(value[1], 1, 1, 42, requests[1]);
        //         } else if (timeComm.rank() == 1) {
        //             timeComm.irecv(value[1], 1, 0, 42, requests[1]);
        //         }
        //
        //         ippl::mpi::waitall(requests.begin(), requests.end(), statuses.begin());
        //
        //         if (spaceComm.rank() == 0) {
        //             std::cout << "space comm rank 0 value = " << value[0] << std::endl;
        //         } else if (spaceComm.rank() == 1) {
        //             std::cout << "space comm rank 1 value = " << value[0] << std::endl;
        //         }
        //
        //         if (timeComm.rank() == 0) {
        //             std::cout << "time  comm rank 0 value = " << value[1] << std::endl;
        //         } else if (timeComm.rank() == 1) {
        //             std::cout << "time  comm rank 1 value = " << value[1] << std::endl;
        //         }

        //         ippl::Comm->barrier();
        //

        std::cout << ippl::mpi::is_serializable<std::vector<bool> >::value << std::endl;

        //         MPI_Win win;
        ippl::mpi::rma::Window<ippl::mpi::rma::Active> win;

        std::vector<double> mem(4);
        std::vector<double> mem2(4);

        if (ippl::Comm->rank() == 1) {
            mem2[0] = 42;
            mem2[1] = 88;
            mem2[2] = 12;
            mem2[3] = 3;
        }

        //         MPI_Win_allocate((MPI_Aint)(4 * sizeof(double)), sizeof(double),
        //                  MPI_INFO_NULL, MPI_COMM_WORLD, &mem, &win);

        //         win.create(*ippl::Comm, mem.begin(), mem.end());

        win.attach(*ippl::Comm, mem.begin(), mem.end());

        //         win.fence(0);

        //         if (ippl::Comm->rank() == 0) {
        //             win.get(mem.begin(), mem.end(), 1, 0);
        //         }

        //         win.fence(0);

        std::cout << ippl::Comm->rank() << " data: " << mem[0] << " " << mem[1] << " " << mem[2]
                  << " " << mem[3] << std::endl;

        ippl::Comm->barrier();

        win.fence(0);
        //         MPI_Win_fence(0, win);

        if (ippl::Comm->rank() == 1) {
            //             mem2[2] = 100;
            //             win.put<double>(mem2.data() + 2, 1, 2);
            win.put(mem2.begin(), mem2.end(), 0, 0);
            //             MPI_Put(mem2.data()+ 2, 1, MPI_DOUBLE, 1, 2, 1, MPI_DOUBLE, win);
        }

        win.fence(0);
        //         MPI_Win_fence(0, win);

        std::cout << "after put " << ippl::Comm->rank() << " data: " << mem[0] << " " << mem[1]
                  << " " << mem[2] << " " << mem[3] << std::endl;

        // // 0 data: 0 0 0 0
        // // after put 0 data: 0 0 0 0
        // // 1 data: 42 88 12 3
        // // after put 1 data: 42 88 100 3

        win.detach(mem.begin());
    }
    ippl::finalize();

    return 0;
}
