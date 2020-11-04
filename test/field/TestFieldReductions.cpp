#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <boost/core/demangle.hpp>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;


    int pt = 4;
    Index I(pt);
    NDIndex<dim> owned(I, I, I);

    e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = SERIAL;

    // all parallel layout, standard domain, normal axis order
    FieldLayout<dim> layout(owned,allParallel, 1);

    //Unit box 
    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    double pi = acos(-1.0);

     typedef ippl::Field<double, dim> field_type;

    field_type field;

    field.initialize(mesh, layout);
    
    typename field_type::view_type& view = field.getView();


    Kokkos::parallel_for("Assign lfield", 
                          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                 {view.extent(0),
                                                                 view.extent(1),
                                                                 view.extent(2)}),
                          KOKKOS_LAMBDA(const int i, const int j, const int k){
            
                            double x = (i + 0.5) * hx[0] + origin[0];
                            double y = (j + 0.5) * hx[1] + origin[1];
                            double z = (k + 0.5) * hx[2] + origin[2];

                            //view(i, j, k) = 3.0 * x + 4.0 * y + 5.0 * z;
                            view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);


                          });

    double total = field.sum();
    double max = field.max();
    double min = field.min();
    double product = field.prod();

    field = fabs(field);
    double normInf = field.max();
    
    //const char* name = typeid(fabs(field)).name();

    //std::cout << name << std::endl;
    //std::cout << boost::core::demangle(name) << std::endl;
    field = pow(field,2);
    double norm2 = sqrt(field.sum());


    std::cout << "total: " << total << std::endl;
    std::cout << "max: " << max << std::endl;
    std::cout << "min: " << min << std::endl;
    std::cout << "Product: " << product << std::endl;
    std::cout << "norm2: " << norm2 << std::endl;
    std::cout << "normInf: " << normInf << std::endl;


    return 0;
}
