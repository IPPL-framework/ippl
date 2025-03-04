#include "Ippl.h"

#include <iostream>
#include <list>

#include "Communicate/Serializable.h"
#include "Communicate/Wait.h"
#include "Communicate/Window.h"
#include "Communicate/Communicator.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
      Inform msg("TestCommunicator:: ",INFORM_ALL_NODES);
      int rank, size;
      rank = ippl::Comm->rank(); size = ippl::Comm->size();
      
      constexpr int data_size = 1000;
      std::vector<double> send_data(data_size);
      std::vector<double> recv_data(data_size, 0.0);
      MPI_Request send_request, recv_request;

      // Determine source and destination ranks in a circular pattern
      int send_to = (rank + 1) % size;
      int recv_from = (rank - 1 + size) % size;
      
      // Initialize data
      for (int i = 0; i < data_size; ++i) {
        send_data[i] = static_cast<double>(rank * 1000 + i); // Unique data per rank
      }

      std::vector<MPI_Request> send_requests(1);
      
      int tag = 42;
      
      using memory_space=Kokkos::HostSpace;
	
      using buffer_type  = ippl::mpi::Communicator::buffer_type<memory_space>;

      buffer_type buf = ippl::Comm->template getBuffer<memory_space, double>(data_size);

      
      ippl::Comm->isend(send_to, tag, send_data.data(), *buf, send_requests.back(), data_size); // ippl::Comm->isend(send_data.data(), data_size, send_to, tag, send_requests.back());
      
      //      MPI_Isend(send_data.data(), data_size, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD, &send_request);
      //      MPI_Irecv(recv_data.data(), data_size, MPI_DOUBLE, recv_from, 0, MPI_COMM_WORLD, &recv_request);

      // Wait for both operations to complete
      //      MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      //      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

      // Print verification message
      msg << " Sent data to Rank " << send_to
	  << " and received data from Rank " << recv_from
	  << ". First received element: " << recv_data[0]
	  << ", Last: " << recv_data[data_size - 1] << endl;;
      
    }
    ippl::finalize();
}

/*
      int val = 0;
        
      if (ippl::Comm->rank() == 0) {
	val = 10;
	ippl::Comm->isend(val, 1, 1, 42, requests[0],1);
      } else if (ippl::Comm->rank() == 1) {
	ippl::Comm->irecv(val, 1, 0, 42, requests[0]);
      }
        
      ippl::mpi::waitall(requests.begin(), requests.end(), statuses.begin());
        
      std::cout << ippl::Comm->rank() << " status tag    " << statuses[0].tag() <<
	std::endl; std::cout << ippl::Comm->rank() << " status source " << statuses[0].source() << std::endl;
      int cnt = statuses[0].count<double>().value();
      std::cout << ippl::Comm->rank() << " status count " << cnt << std::endl;
      
      if (ippl::Comm->rank() == 0) {
	std::cout << "rank 0 " << val << std::endl;
      } else {
	std::cout << "rank 1 " << val << std::endl;
      }

        
      int spaceProcs = 2;
      int spaceColor = ippl::Comm->rank() / spaceProcs;
      int timeColor = ippl::Comm->rank() % spaceProcs;
        
      ippl::mpi::Communicator spaceComm = ippl::Comm->split(spaceColor, ippl::Comm->rank());
      ippl::mpi::Communicator timeComm =  ippl::Comm->split(timeColor,  ippl::Comm->rank());
        
      int value[2] = {0, 0};
        
      if (spaceComm.rank() == 0) {
	value[0] = 10 * (ippl::Comm->rank() + 1);
	spaceComm.isend(value[0], 1, 1, 42, requests[0],1);
      } else if (spaceComm.rank() == 1) {
	spaceComm.irecv(value[0], 1, 0, 42, requests[0]);
      }
      
      if (timeComm.rank() == 0) {
	value[1] = 20 * (ippl::Comm->rank() + 1);
	timeComm.isend(value[1], 1, 1, 42, requests[1],1);
      } else if (timeComm.rank() == 1) {
	timeComm.irecv(value[1], 1, 0, 42, requests[1]);
      }
      
      ippl::mpi::waitall(requests.begin(), requests.end(), statuses.begin());
        
      if (spaceComm.rank() == 0) {
	std::cout << "space comm rank 0 value = " << value[0] << std::endl;
      } else if (spaceComm.rank() == 1) {
	std::cout << "space comm rank 1 value = " << value[0] << std::endl;
      }
      
      if (timeComm.rank() == 0) {
	std::cout << "time  comm rank 0 value = " << value[1] << std::endl;
      } else if (timeComm.rank() == 1) {
	std::cout << "time  comm rank 1 value = " << value[1] << std::endl;
      }

      ippl::Comm->barrier();
    }
    ippl::finalize();

    return 0;
}
*/
