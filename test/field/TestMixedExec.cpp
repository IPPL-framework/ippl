#include "Ippl.h"

#include <iostream>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        constexpr unsigned int dim = 3;

        ippl::Index I(16);
        ippl::NDIndex<dim> owned(I, I, I);

        // Specifies SERIAL, PARALLEL dims
        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        typedef ippl::BareField<double, dim, Kokkos::Cuda> cuda_field;
        typedef ippl::BareField<double, dim, Kokkos::OpenMP> omp_field;

        cuda_field cf{layout};
        omp_field of{layout};

        cf = 1;
        of = 1;
    }

    ippl::finalize();
    return 0;
}
