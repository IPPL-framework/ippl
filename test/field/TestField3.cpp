#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>
#include <fstream>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    int pt = 4;
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    ippl::e_dim_tag decomp[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        //decomp[d] = ippl::PARALLEL;
        decomp[d] = ippl::SERIAL;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<dim> layout(owned,decomp);

    //Unit box 
    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    double pi = acos(-1.0);


    typedef ippl::Field<double, dim> field_type;
    typedef ippl::Field<ippl::Vector<double, dim>, dim> vector_field_type;

    typedef ippl::Vector<double, dim> Vector_t;

    field_type field(mesh, layout);
    field_type Lap(mesh, layout);
    vector_field_type vfield(mesh, layout);

    typedef ippl::Field<double, dim> Field_t;

    typename Field_t::view_type& view = field.getView();
    typedef ippl::BConds<double, dim> bc_type; 
    typedef ippl::BConds<Vector_t, dim> vbc_type; 

    bc_type bcField;
    vbc_type vbcField;

    //X direction periodic BC
    for (unsigned int i=0; i < 6; ++i) {
        bcField[i] = std::make_shared<ippl::PeriodicFace<double, dim>>(i);
        vbcField[i] = std::make_shared<ippl::PeriodicFace<Vector_t, dim>>(i);
    }
    ////Lower Y face 
    //bcField[2] = std::make_shared<ippl::NoBcFace<double, dim>>(2);
    //vbcField[2] = std::make_shared<ippl::NoBcFace<Vector_t, dim>>(2);
    ////Higher Y face
    //bcField[3] = std::make_shared<ippl::ConstantFace<double, dim>>(3, 7.0);
    //vbcField[3] = std::make_shared<ippl::ConstantFace<Vector_t, dim>>(3, 7.0);
    ////Lower Z face
    //bcField[4] = std::make_shared<ippl::ZeroFace<double, dim>>(4);
    //vbcField[4] = std::make_shared<ippl::ZeroFace<Vector_t, dim>>(4);
    ////Higher Z face
    //bcField[5] = std::make_shared<ippl::ExtrapolateFace<double, dim>>(5, 0.0, 1.0);
    //vbcField[5] = std::make_shared<ippl::ExtrapolateFace<Vector_t, dim>>(5, 0.0, 1.0);

    field.setFieldBC(bcField);
    Lap.setFieldBC(bcField);
    vfield.setFieldBC(vbcField);

    const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
    const int nghost = field.getNghost();
    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    Kokkos::parallel_for("Assign field", 
                          mdrange_type({nghost, nghost, nghost},
                                       {view.extent(0) - nghost,
                                        view.extent(1) - nghost,
                                        view.extent(2) - nghost}),
                          KOKKOS_LAMBDA(const int i, 
                                        const int j, 
                                        const int k)
                          {
                            //local to global index conversion
                            const size_t ig = i + lDom[0].first() + nghost;
                            const size_t jg = j + lDom[1].first() + nghost;
                            const size_t kg = k + lDom[2].first() + nghost;
                            double x = (ig + 0.5) * hx[0];
                            double y = (jg + 0.5) * hx[1];
                            double z = (kg + 0.5) * hx[2];

                            //view(i, j, k) = 3.0*pow(x,1) + 4.0*pow(y,1) + 5.0*pow(z,1);
                            //view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);
                            view(i, j, k) = sin(pi * x) * sin(pi * y) * sin(pi * z);
                          });



    //field.write();

    //vfield = grad(field);

    //vfield.write();

    //field = div(vfield);

    //field.write();

    Lap = 0.0;

    Lap = laplace(field);

    int nRanks = Ippl::Comm->size();
    for (int rank = 0; rank < nRanks; ++rank) {
        if (rank == Ippl::Comm->rank()) {
            std::ofstream out("LaplacePeriodicBCSerial_" + 
                              std::to_string(rank) + 
                              ".dat", std::ios::out);
            Lap.write(out);
            out.close();
        }
        Ippl::Comm->barrier();
    }


    return 0;
}
