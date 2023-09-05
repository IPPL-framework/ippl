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
// ./build/test/stream/TestCatalystAdaptor ./test/stream/catalyst_pipeline.py --info 5
//
// when dumping of the vtk is required also --info 5 is needed

#include "Ippl.h"

#include <array>
#include <iostream>
#include <typeinfo>

#include "Stream/InSitu/CatalystAdaptor.h"

constexpr unsigned int dim{3};
using Mesh_t         = ippl::UniformCartesian<double, dim>;
using Centering_t    = Mesh_t::DefaultCentering;
using Field_t        = ippl::Field<double, dim, Mesh_t, Centering_t>;
const char* TestName = "CatalystAdaptor";

void dumpVTK(Field_t& rho, int nx, int ny, int nz, int iteration, double dx, double dy, double dz) {
    typename Field_t::view_type::host_mirror_type host_view = rho.getHostMirror();
    // auto view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    if (vtkout.openedSuccessfully()) {
        // start with header
        vtkout << "# vtk DataFile Version 2.0" << endl;
        vtkout << TestName << endl;
        vtkout << "ASCII" << endl;
        vtkout << "DATASET STRUCTURED_POINTS" << endl;
        vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
        vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
        vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
        vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

        vtkout << "SCALARS Rho float" << endl;
        vtkout << "LOOKUP_TABLE default" << endl;
        for (int z = 0; z < nz + 2; z++) {
            for (int y = 0; y < ny + 2; y++) {
                for (int x = 0; x < nx + 2; x++) {
                    vtkout << host_view(x, y, z) << endl;
                }
            }
        }
    } else {
        IpplException("TestCatalystAdaptor::dumpVTK", "opening of file was not successful");
    }
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    CatalystAdaptor::Initialize(argc, argv);

    // constexpr unsigned int dim {3};

    const int pt{2};
    ippl::Index Ix(pt);
    ippl::Index Iy{pt};
    ippl::Index Iz{pt};
    ippl::NDIndex<dim> owned(Ix, Iy, Iz);

    ippl::e_dim_tag allParallel[dim];  // Specifies SERIAL, PARALLEL dims
    for (unsigned int d = 0; d < dim; d++)
        allParallel[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned, allParallel);

    constexpr double dx            = {1.0 / double(pt)};
    constexpr double dy            = {1.0 / double(pt)};
    constexpr double dz            = {1.0 / double(pt)};
    ippl::Vector<double, 3> hx     = {dx, dy, dz};
    ippl::Vector<double, 3> origin = {0, 0, 0};

    // using Mesh_t      = ippl::UniformCartesian<double, 3>;
    // using Centering_t = Mesh_t::DefaultCentering;

    Mesh_t mesh(owned, hx, origin);

    typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;

    std::cout << layout << std::endl;

    field_type field(mesh, layout);

    field = 1.0;

    const ippl::NDIndex<dim>& lDom       = layout.getLocalNDIndex();
    const int nghost                     = field.getNghost();
    using mdrange_type                   = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    typename field_type::view_type& view = field.getView();

    double time           = {0.0};
    const double dt       = {0.05};
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
                // const size_t kg = k + lDom[2].first() - nghost;
                // double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                // double z = (kg + 0.5) * hx[2];

                // view(i, j, k) = 3.0*pow(x,1) + 4.0*pow(y,1) + 5.0*pow(z,1);
                // view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);
                // view(i, j, k) = sin(pi * x) * sin(pi * y) * sin(pi * z);
                view(i, j, k) = y * time;
            });

        CatalystAdaptor::Execute(it, time, ippl.Comm.get()->rank(), field);
        // print should be same as field data
        time += dt;
        dumpVTK(field, field.get_mesh().getGridsize(0), field.get_mesh().getGridsize(1),
                field.get_mesh().getGridsize(2), it, field.get_mesh().getMeshSpacing(0),
                field.get_mesh().getMeshSpacing(1), field.get_mesh().getMeshSpacing(2));
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
