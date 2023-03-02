// Tests the application of various kinds of boundary conditions on fields
#include "Ippl.h"

#include <array>
#include <iostream>
#include <typeinfo>

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);

    constexpr unsigned int dim = 3;

    int pt = 4;
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    ippl::e_dim_tag allParallel[dim];  // Specifies SERIAL, PARALLEL dims
    for (unsigned int d = 0; d < dim; d++)
        allParallel[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned, allParallel);

    double dx                      = 1.0 / double(pt);
    ippl::Vector<double, 3> hx     = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};

    using Mesh_t = ippl::UniformCartesian<double, 3>;

    Mesh_t mesh(owned, hx, origin);

    typedef ippl::Field<double, dim> field_type;

    typedef ippl::BConds<double, dim> bc_type;

    bc_type bcField;

    // X direction periodic BC
    for (unsigned int i = 0; i < 2; ++i) {
        bcField[i] = std::make_shared<ippl::PeriodicFace<double, dim>>(i);
    }
    ////Lower Y face
    bcField[2] = std::make_shared<ippl::NoBcFace<double, dim>>(2);
    ////Higher Y face
    bcField[3] = std::make_shared<ippl::ConstantFace<double, dim>>(3, 7.0);
    ////Lower Z face
    bcField[4] = std::make_shared<ippl::ZeroFace<double, dim>>(4);
    ////Higher Z face
    bcField[5] = std::make_shared<ippl::ExtrapolateFace<double, dim>>(5, 0.0, 1.0);

    // std::cout << bcField << std::endl;
    std::cout << layout << std::endl;

    field_type field(mesh, layout, 1);

    field = 1.0;

    const ippl::NDIndex<dim>& lDom       = layout.getLocalNDIndex();
    const int nghost                     = field.getNghost();
    using mdrange_type                   = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    typename field_type::view_type& view = field.getView();

    Kokkos::parallel_for(
        "Assign field",
        mdrange_type({nghost, nghost, nghost},
                     {view.extent(0) - nghost, view.extent(1) - nghost, view.extent(2) - nghost}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            // local to global index conversion
            const size_t ig = i + lDom[0].first() - nghost;
            // const size_t jg = j + lDom[1].first() - nghost;
            // const size_t kg = k + lDom[2].first() - nghost;
            double x = (ig + 0.5) * hx[0] + origin[0];
            // double y = (jg + 0.5) * hx[1];
            // double z = (kg + 0.5) * hx[2];

            // view(i, j, k) = 3.0*pow(x,1) + 4.0*pow(y,1) + 5.0*pow(z,1);
            // view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);
            // view(i, j, k) = sin(pi * x) * sin(pi * y) * sin(pi * z);
            view(i, j, k) = x;
        });

    // field = field * 10.0;

    bcField.findBCNeighbors(field);

    unsigned int niter = 5;

    for (unsigned int i = 0; i < niter; ++i) {
        bcField.apply(field);
    }

    int nRanks = Ippl::Comm->size();
    for (int rank = 0; rank < nRanks; ++rank) {
        if (rank == Ippl::Comm->rank()) {
            std::string fname = "field_AllBC_" + std::to_string(rank) + ".dat";
            Inform out("Output", fname.c_str(), Inform::OVERWRITE, rank);
            field.write(out);
        }
        Ippl::Comm->barrier();
    }

    return 0;
}
