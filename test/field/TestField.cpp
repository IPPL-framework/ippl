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
    typedef ippl::Field<ippl::Vector<double, dim>, dim> vector_field_type;

    field_type field(layout);

    vector_field_type vfield(layout);

    field = 1.0;

    vfield = grad(field);

    vfield.write();

    return 0;
}