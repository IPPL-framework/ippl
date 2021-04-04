// Usage: ./TestFieldNorm (log_2 field size) --info 10
#include "Ippl.h"
#include "Utility/IpplTimings.h"

#include <iostream>
#include <typeinfo>

#include <cstdlib>

void checkError(double computed, double correct, int N, int p,
        Inform& mout, Inform& merr, double tolerance = 1e-16) {
    double absError = fabs(computed - correct);
    double relError = absError / correct;
    mout << "(" << N << ", L" << p << "): " << absError << "," << relError << endl;
    if (relError > tolerance) {
        merr << "L" << p << " norm for N = " << N << " does not match.\n\tGot "
            << computed << ", expected " << correct << ". Relative error: " << relError << endl;
    }
}

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    int pt = 4;

    if (argc >= 2) {
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

    IpplTimings::TimerRef l2Timer = IpplTimings::getTimer("L2"),
        l1Timer = IpplTimings::getTimer("L1"),
        l0Timer = IpplTimings::getTimer("Max");

    IpplTimings::startTimer(l2Timer);
    double compute_l2 = ippl::norm(field);
    IpplTimings::stopTimer(l2Timer);

    IpplTimings::startTimer(l1Timer);
    double compute_l1 = ippl::norm(field, 1);
    IpplTimings::stopTimer(l1Timer);

    IpplTimings::startTimer(l0Timer);
    double compute_linf = ippl::norm(field, 0);
    IpplTimings::stopTimer(l0Timer);

    IpplTimings::print("timings.dat");

    Inform m1("DATA");
    Inform m2("Deviation", std::cerr);

    checkError(compute_l2, l2, pt, 2, m1, m2);
    checkError(compute_l1, l1, pt, 1, m1, m2);
    checkError(compute_linf, linf, pt, 0, m1, m2);

    return 0;
}
