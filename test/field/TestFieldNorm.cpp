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

    ippl::Index I(pt);
    ippl::NDIndex<3> owned3(I, I, I);
    ippl::NDIndex<2> owned2(I, I);

    ippl::e_dim_tag d3[3], d2[2];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<3; d++) {
        d3[d] = ippl::PARALLEL;
        if (d < 2) d2[d] = ippl::PARALLEL;
    }

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<3> layout3(owned3,d3);
    ippl::FieldLayout<2> layout2(owned2,d2);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx3 = {dx, dx, dx};
    ippl::Vector<double, 3> origin3 = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh3(owned3, hx3, origin3);

    ippl::Vector<double, 2> hx2 = {dx, dx};
    ippl::Vector<double, 2> origin2 = {0, 0};
    ippl::UniformCartesian<double, 2> mesh2(owned2, hx2, origin2);


    typedef ippl::Field<double, 3> field3;
    typedef ippl::Field<double, 2> field2;

    field3 _3d(mesh3, layout3);
    field2 _2d(mesh2, layout2);

    double pi = acos(-1.0);

    _3d = pi/4;
    auto view2 = _2d.getView();
    Kokkos::parallel_for("assign 2D", _2d.getRangePolicy(),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
            view2(i, j) = pi / 4;
        }
    );

    double l2 = pow(pt, 1.5) * pi / 4;
    double l1 = pow(pt, 3) * pi / 4;
    double linf = pi / 4;

    double compute_l2 = ippl::norm(_3d);
    double compute_l1 = ippl::norm(_3d, 1);
    double compute_linf = ippl::norm(_3d, 0);

    Inform mD3("3D"), mD2("2D");
    Inform m2("Deviation", std::cerr);

    checkError(compute_l2, l2, pt, 2, mD3, m2);
    checkError(compute_l1, l1, pt, 1, mD3, m2);
    checkError(compute_linf, linf, pt, 0, mD3, m2);

    l2 = pt * pi / 4;
    l1 = pt * pt * pi / 4;

    compute_l2 = ippl::norm(_2d);
    compute_l1 = ippl::norm(_2d, 1);
    compute_linf = ippl::norm(_2d, 0);

    checkError(compute_l2, l2, pt, 2, mD2, m2);
    checkError(compute_l1, l1, pt, 1, mD2, m2);
    checkError(compute_linf, linf, pt, 0, mD2, m2);

    return 0;
}
