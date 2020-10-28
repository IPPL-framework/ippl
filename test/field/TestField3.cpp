#include "Ippl.h"

#include <iostream>
#include <typeinfo>

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
    typedef ippl::Field<ippl::Vector<double, dim>, dim> vector_field_type;

    field_type field, Lap;

    vector_field_type vfield;

    field.initialize(mesh, layout);
    
    vfield.initialize(mesh, layout);
    
    Lap.initialize(mesh,layout);

    typedef ippl::Field<double, dim> Field_t;

    typename Field_t::LField_t::view_type& view = field(0).getView();

    //typename Field_t::LField_t::view_type::HostMirror host_view = 
    //                                                  Kokkos::create_mirror_view(view);

    //int length = pt+2;

    //for (int i = 0; i < length; ++i) {
    //    for (int j = 0; j < length; ++j) {
    //        for (int k = 0; k < length; ++k) {
    //                
    //            double x = (i + 0.5) * hx[0] + origin[0];
    //            double y = (j + 0.5) * hx[1] + origin[1];
    //            double z = (k + 0.5) * hx[2] + origin[2];

    //            //host_view(i, j, k) = 3.0 * x + 4.0 * y + 5.0 * z;
    //            //host_view(i, j, k) = 3.0 * pow(x,2) + 4.0 * pow(y,2) + 5.0 * pow(z,2);
    //            host_view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);
    //        }
    //    }
    //}

    //Kokkos::deep_copy(view,host_view);


    Kokkos::parallel_for("Assign lfield", 
                          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                 {view.extent(0),view.extent(1),view.extent(2)}),
                          KOKKOS_LAMBDA(const int i, const int j, const int k){
            
                            double x = (i + 0.5) * hx[0] + origin[0];
                            double y = (j + 0.5) * hx[1] + origin[1];
                            double z = (k + 0.5) * hx[2] + origin[2];

                            //view(i, j, k) = 3.0 * x + 4.0 * y + 5.0 * z;
                            view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);


                          });



    field.write();

    vfield = grad(field);

    vfield.write();

    field = div(vfield);

    field.write();
    
    Lap = 0.0;
    
    Lap = laplace(field); 

    Lap.write();


    return 0;
}
