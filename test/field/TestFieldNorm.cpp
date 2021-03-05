#include "Ippl.h"

#include <iostream>
#include <typeinfo>

#include <cstdlib>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    int pt = 4;

    if (argc == 2) {
        pt = 1 << (int)strtol(argv[1], NULL, 10);
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

    double l2 = sqrt(pow(pt, 3)) * pi / 4;
    double l1 = pow(pt, 3) * pi / 4;
    double linf = pi / 4;

    Kokkos::Profiling::pushRegion("L2 Norm");
    double compute_l2 = ippl::norm(field);
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("L1 Norm");
    double compute_l1 = ippl::norm(field, 1);
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("Max Norm");
    double compute_linf = ippl::norm(field, 0);
    Kokkos::Profiling::popRegion();

    if (abs(compute_l2 - l2) > 1e-6) {
        std::cerr << "L2 norm for N = " << pt << " does not match. Deviation: " << abs(l2 - compute_l2) << std::endl;
    }
    if (abs(compute_l1 - l1) > 1e-6) {
        std::cerr << "L1 norm for N = " << pt << " does not match. Deviation: " << abs(l1 - compute_l1) << std::endl;
    }
    if (compute_linf != linf) {
        std::cerr << "Max norm for N = " << pt << " does not match. Deviation: " << abs(linf - compute_linf) << std::endl;
    }

    return 0;
}
