#include "Ippl.h"

#include <iostream>
#include <typeinfo>

#include <cstdlib>

void checkError(double computed, double correct, int N, int p, double tolerance = 1e-16) {
    double relError = fabs(computed - correct) / correct;
    if (relError > tolerance) {
        std::cerr << "L" << p << " norm for N = " << N << " does not match.\n\tGot " << computed << ", expected " << correct << ". Relative error: " << relError << std::endl;
    }
}

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

    double l2 = pow(pt, 1.5) * pi / 4;
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

    checkError(compute_l2, l2, pt, 2);
    checkError(compute_l1, l1, pt, 1);
    checkError(compute_linf, linf, pt, 0);

    return 0;
}
