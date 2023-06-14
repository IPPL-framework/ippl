#include "Ippl.h"

#include <iostream>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    {
        constexpr unsigned int dim = 3;

        ippl::Index I(16);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag decomp[dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++) {
            decomp[d] = ippl::PARALLEL;
        }

        ippl::FieldLayout<dim> layout(owned, decomp);

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
