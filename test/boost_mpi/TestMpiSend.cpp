#include <iostream>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <Kokkos_Core.hpp>

namespace mpi = boost::mpi;

int main(int argc, char *argv[]) {

    mpi::environment env(argc, argv);


    Kokkos::initialize(argc,argv);
    {
        mpi::communicator world;
        typedef Kokkos::View<double*> view_type;


        if (world.rank() == 0) {
            view_type view_rank0("rank 0", 20);

            Kokkos::parallel_for("assign", 20, KOKKOS_LAMBDA(const int i) {
                view_rank0(i) = i;
            });

            for (int i = 1; i < world.size(); ++i) {
                world.send(i, 42 /*tag*/, view_rank0.data(), 20);
            }

        } else {
            view_type view_rank1("rank 1", 20);

            Kokkos::parallel_for("assign", 20, KOKKOS_LAMBDA(const int i) {
                view_rank1(i) = 0;
            });

            world.recv(0, 42 /*tag*/, view_rank1.data(), 20);

            for (size_t i = 0; i < 20; ++i) {
                std::cout << view_rank1(i) << std::endl;
            }
        }
    }
    Kokkos::finalize();


    return 0;
}
