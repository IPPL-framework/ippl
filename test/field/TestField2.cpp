#include "Ippl.h"

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;


    int pt = 4;
    ippl::Index I(pt);
    NDIndex<dim> owned(I, I, I);

    e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = SERIAL;

    // all parallel layout, standard domain, normal axis order
    FieldLayout<dim> layout(owned,allParallel);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);


    typedef ippl::Field<double, dim> field_type;

    field_type field(mesh, layout);

    double pi = acos(-1.0);

    field = pi/4;

    //field.write();

    field = fabs(7.0 * (sin(field) * cos(field))/(tan(field) * acos(field)) 
                - exp(field) + erf(field) + (asin(field) * cosh(field)) / (atan(field) 
                * sinh(field)) + tanh(field) * log(field)
               - log10(field) * sqrt(field) + floor(field) * ceil(field) + exp(2));

    field.write();

    field = -field;

    field.write();

    return 0;
}
