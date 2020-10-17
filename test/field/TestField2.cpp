#include "Ippl.h"

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;


    Index I(4);
    NDIndex<dim> owned(I, I, I);

    e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = SERIAL;

    // all parallel layout, standard domain, normal axis order
    FieldLayout<dim> layout(owned,allParallel, 1);


    typedef ippl::Field<double, dim> field_type;

    field_type field(layout);

    double pi = acos(-1.0);

    field = pi/4;

    //field.write();

    field = fabs(7.0 * (sin(field) * cos(field))/(tan(field) * acos(field)) 
                - exp(field) + erf(field) + (asin(field) * cosh(field)) / (atan(field) 
                * sinh(field)) + tanh(field) * log(field)
               - log10(field) * sqrt(field) + floor(field) * ceil(field));

    field.write();

    field = -field;

    field.write();

    ippl::UniformCartesian<double, 2> unif(I, I);


    return 0;
}
