#include "Ippl.h"

#include <iostream>

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);

    Index I(10);
    NDIndex<3> domain(I, I, I);

    typedef ippl::Vector<double, 3> vector_type;
    typedef ippl::LField<vector_type, 3> vector_field_type;
    typedef ippl::LField<double, 3> scalar_field_type;

    vector_field_type lfield(domain);

    scalar_field_type lfield_s(domain);

    lfield = 1.0;

    lfield.write();

//     lfield = 5.0 + ((lfield + 4.0 * lfield) * (lfield + lfield))
//            / (lfield + lfield / 2.0 + lfield) - lfield - 1.0;


    //lfield = lfield * (1.0 + cross(4.0 * lfield, lfield));

    lfield_s = dot(lfield, lfield);

    lfield_s.write();

    return 0;
}
