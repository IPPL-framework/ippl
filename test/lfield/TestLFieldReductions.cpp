#include "Ippl.h"

#include <iostream>

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);

    int pt = 4;
    Index I(pt);
    Index J(pt);

    Index K(pt);
    NDIndex<3> domain(I, J, K);
    
    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};

    typedef ippl::LField<double, 3> LField_t;

    LField_t lfield(domain);

    typename LField_t::view_type view = lfield.getView();

    double pi = acos(-1.0);

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

    //lfield = 1.0;

    double total = lfield.sum();
    double max = lfield.max();
    double min = lfield.min();
    double product = lfield.prod();

    std::cout << "total: " << total << std::endl;
    std::cout << "max: " << max << std::endl;
    std::cout << "min: " << min << std::endl;
    std::cout << "Product: " << product << std::endl;
    //lfield = 0.0;

    //lfield = total;
    
    //lfield.write();

    return 0;
}
