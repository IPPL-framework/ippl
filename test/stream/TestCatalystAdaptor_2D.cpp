// Tests the application for the Catalyst In-Situ Adaptor
// following environment variables do need to be exported
//
// export CATALYST_IMPLEMENTATION_PATHS=<path-to-paraview-install>/lib/catalyst
// export CATALYST_IMPLEMENTATION_NAME=paraview
//
// on juwels these both are direclty set!
//
// export PARARVIEW_CATALYST_DIR=<path-to-paraview-install>/lib/catalyst
//
// RUN
// ./TestCatalystAdaptor ./<path-to-catalyst-pipeline>.py
//
// eg.
// ./build/test/stream/TestCatalystAdaptor ./test/stream/catalyst_pipeline.py

#include "Ippl.h"

#include <iostream>

#include "Stream/InSitu/CatalystAdaptor.h"

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    CatalystAdaptor::Initialize(argc, argv);

    constexpr unsigned int dim{2};

    const int pt{2};
    ippl::Index Ix(pt);
    ippl::Index Iy{pt};
    ippl::NDIndex<dim> owned(Ix, Iy);

    ippl::e_dim_tag allParallel[dim];  // Specifies SERIAL, PARALLEL dims
    for (auto& d : allParallel)
        d = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned, allParallel);

    constexpr double dx              = {1.0 / double(pt)};
    constexpr double dy              = {1.0 / double(pt)};
    ippl::Vector<double, dim> hx     = {dx, dy};
    ippl::Vector<double, dim> origin = {0, 0};

    using Mesh_t      = ippl::UniformCartesian<double, dim>;
    using Centering_t = Mesh_t::DefaultCentering;

    Mesh_t mesh(owned, hx, origin);

    typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;

    std::cout << layout << std::endl;

    field_type field(mesh, layout, 1);

    field = 1.0;

    const ippl::NDIndex<dim>& lDom       = layout.getLocalNDIndex();
    const int nghost                     = field.getNghost();
    using mdrange_type                   = Kokkos::MDRangePolicy<Kokkos::Rank<dim>>;
    typename field_type::view_type& view = field.getView();

    double time           = {0.0};
    const double dt       = {0.05};
    const unsigned int nt = {5};
    for (unsigned int it = 0; it < nt; ++it) {
        Kokkos::parallel_for(
            "Assign field",
            mdrange_type({nghost, nghost}, {view.extent(0) - nghost, view.extent(1) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
                // local to global index conversion
                // const size_t ig = i + lDom[0].first() - nghost;
                const size_t jg = j + lDom[1].first() - nghost;
                // const size_t kg = k + lDom[2].first() - nghost;
                // double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                // double z = (kg + 0.5) * hx[2];

                // view(i, j, k) = 3.0*pow(x,1) + 4.0*pow(y,1) + 5.0*pow(z,1);
                // view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);
                // view(i, j, k) = sin(pi * x) * sin(pi * y) * sin(pi * z);
                view(i, j) = y * time;
                // std::cout << view(i,j) << std::endl;
            });

        CatalystAdaptor::Execute(it, time, ippl.Comm.get()->rank(), field);  // field
        // print should be same as field data
        time += dt;
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

    CatalystAdaptor::Finalize();
    return 0;
}
