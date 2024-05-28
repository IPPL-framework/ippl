// Tests the application for the Catalyst In-Situ Adaptor
// following environment variables do need to be set
//
// export CATALYST_IMPLEMENTATION_PATHS=<path-to-paraview-install>/lib/catalyst
// export CATALYST_IMPLEMENTATION_NAME=paraview
//
// on juwels these both are direclty set!
//
// RUN
// ./TestCatalystAdaptor ./<path-to-catalyst-pipeline>.py
//
// for dumping vtk files
// ./build/test/stream/TestCatalystAdaptor ./test/stream/catalyst_pipeline.py --info 5
//

#include "Ippl.h"

#include <iostream>

#include "Stream/InSitu/CatalystAdaptor.h"


int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim {3};

        CatalystAdaptor::Initialize_Adios(argc, argv);

        const int pt{2};
        ippl::Index Ix(pt);
        ippl::Index Iy{pt};
        ippl::Index Iz{pt};
        ippl::NDIndex<dim> owned(Ix, Iy, Iz);

        std::array<bool, dim> isParallel{true};  // Specifies SERIAL, PARALLEL dims

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        constexpr double dx            = {1.0 / double(pt)};
        constexpr double dy            = {1.0 / double(pt)};
        constexpr double dz            = {1.0 / double(pt)};
        ippl::Vector<double, 3> hx     = {dx, dy, dz};
        ippl::Vector<double, 3> origin = {0, 0, 0};

        using Mesh_t      = ippl::UniformCartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        using field_type = ippl::Field<double, dim, Mesh_t, Centering_t>;

        Mesh_t mesh(owned, hx, origin);

        std::cout << layout << std::endl;

        field_type field(mesh, layout);

        field = 1.0;

        const ippl::NDIndex<dim>& lDom       = layout.getLocalNDIndex();
        const int nghost                     = field.getNghost();
        using mdrange_type                   = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        const typename field_type::view_type& view = field.getView();

        double time           = {0.0};
        const double dt       = {1};
        const unsigned int nt = {10};
        for (unsigned int it = 0; it < nt; ++it) {
            Kokkos::parallel_for(
                "Assign field",
                mdrange_type(
                    {nghost, nghost, nghost},
                    {view.extent(0) - nghost, view.extent(1) - nghost, view.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    // local to global index conversion
                    // const size_t ig = i + lDom[0].first() - nghost;
                    const size_t jg = j + lDom[1].first() - nghost;
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    view(i, j, k) = y * time;
                });

            // call catalyst execute with 5th argument as nullopt or not specified

            std::optional<conduit_cpp::Node> node = std::nullopt;
            CatalystAdaptor::Execute_Field(it, time, ippl::Comm->rank(), field, node);
            // print should be same as field data
            time += dt;

            // dumpVTK only works with --info 5 and higher
            //        dumpVTK(field, field.get_mesh().getGridsize(0), field.get_mesh().getGridsize(1),
            //                field.get_mesh().getGridsize(2), it, field.get_mesh().getMeshSpacing(0),
            //                field.get_mesh().getMeshSpacing(1), field.get_mesh().getMeshSpacing(2));
        }

        int nRanks = ippl::Comm->size();
        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == ippl::Comm->rank()) {
                std::string fname = std::format("field_AllBC_{}.dat", std::to_string(rank));
                Inform out("Output", fname.c_str(), Inform::OVERWRITE, rank);
                field.write(out);
            }
            ippl::Comm->barrier();
        }
    }
    CatalystAdaptor::Finalize();
    ippl::finalize();
    return 0;
}
