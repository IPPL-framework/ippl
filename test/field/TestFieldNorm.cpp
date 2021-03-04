#include "Ippl.h"

#include <iostream>
#include <typeinfo>

#include <cstdlib>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    int pt = 4;

    if (argc == 2) {
        pt = (int)strtol(argv[1], NULL, 10);
    }

    constexpr unsigned int dim = 3;


    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<dim> layout(owned,allParallel);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);


    typedef ippl::Field<double, dim> field_type;

    field_type field(mesh, layout);

    double pi = acos(-1.0);

    field = pi/4;

    Kokkos::Profiling::pushRegion("Single Kernel");
    double fastL2Norm = ippl::norm(field);
    Kokkos::Profiling::popRegion();

    std::cout << fastL2Norm << std::endl;

    Kokkos::Profiling::pushRegion("Double Kernel");
    field = field * field;
    double slowL2Norm = sqrt(field.sum());
    Kokkos::Profiling::popRegion();

    if (slowL2Norm == fastL2Norm) {
        std::cout << "Slow norm matches fast norm\n";
    } else {
        std::cout << slowL2Norm << std::endl;
    }

    return 0;
}
