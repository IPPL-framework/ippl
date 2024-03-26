// Tests the application of various kinds of boundary conditions on fields
#include "Ippl.h"

#include <array>
#include <iostream>
#include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;

        int pt = 4;
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        double dx                        = 1.0 / double(pt);
        ippl::Vector<double, dim> hx     = dx;
        ippl::Vector<double, dim> origin = 0;

        using Mesh_t      = ippl::UniformCartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;

        Mesh_t mesh(owned, hx, origin);

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;

        typedef ippl::BConds<field_type, dim> bc_type;

        bc_type bcField;

        // X direction periodic BC
        for (unsigned int i = 0; i < 2; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
        }
        ////Lower Y face
        bcField[2] = std::make_shared<ippl::NoBcFace<field_type>>(2);
        ////Higher Y face
        bcField[3] = std::make_shared<ippl::ConstantFace<field_type>>(3, 7.0);
        ////Lower Z face
        bcField[4] = std::make_shared<ippl::ZeroFace<field_type>>(4);
        ////Higher Z face
        bcField[5] = std::make_shared<ippl::ExtrapolateFace<field_type>>(5, 0.0, 1.0);

        // std::cout << bcField << std::endl;
        std::cout << layout << std::endl;

        field_type field(mesh, layout, 1);

        field = 1.0;

        const ippl::NDIndex<dim>& lDom       = layout.getLocalNDIndex();
        const int nghost                     = field.getNghost();
        typename field_type::view_type& view = field.getView();

        Kokkos::parallel_for(
            "Assign field", field.getFieldRangePolicy(),
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

        int nRanks = ippl::Comm->size();
        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == ippl::Comm->rank()) {
                std::string fname = "field_AllBC_" + std::to_string(rank) + ".dat";
                Inform out("Output", fname.c_str(), Inform::OVERWRITE, rank);
                field.write(out);
            }
            ippl::Comm->barrier();
        }
    }
    ippl::finalize();

    return 0;
}
