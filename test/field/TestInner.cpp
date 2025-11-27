// Tests the Laplacian on a scalar field
#include "Ippl.h"

#include <array>
#include <iostream>
#include <sstream>
#include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform m("");

        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        int pt = std::stoi(argv[1]);
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        std::array<bool, dim> isParallel;
        isParallel.fill(true);  // Specifies SERIAL, PARALLEL dims

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // Unit box
        double dx                        = 2.0 / double(pt);
        ippl::Vector<double, dim> hx     = dx;
        ippl::Vector<double, dim> origin = -1;
        Mesh_t mesh(owned, hx, origin);

        double pi = acos(-1.0);

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> Field_t;
        typedef ippl::Field<ippl::Vector<double, dim>, dim, Mesh_t, Centering_t> vector_field_type;

        Field_t field(mesh, layout);

        typename Field_t::view_type& view       = field.getView();
        typedef ippl::BConds<Field_t, dim> bc_type;

        bc_type bcField;

        // Periodic BC
        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i]  = std::make_shared<ippl::PeriodicFace<Field_t>>(i);
        }
        field.setFieldBC(bcField);

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
        const int nghost               = field.getNghost();

        Kokkos::parallel_for(
            "Assign field", field.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // local to global index conversion
                const size_t ig = i + lDom[0].first() - nghost;
                const size_t jg = j + lDom[1].first() - nghost;
                const size_t kg = k + lDom[2].first() - nghost;
                double x        = (ig + 0.5) * hx[0] + origin[0];
                double y        = (jg + 0.5) * hx[1] + origin[1];
                double z        = (kg + 0.5) * hx[2] + origin[2];
                view(i, j, k)       = sin(pi * x) * sin(pi * y) * sin(pi * z);
            });

        static auto timer = IpplTimings::getTimer("innerProduct");
        IpplTimings::startTimer(timer);
        double field2 = innerProduct(field, field);
        IpplTimings::startTimer(timer);

        m << "inner product = " << field2 << endl;
    }
    ippl::finalize();

    return 0;
}
