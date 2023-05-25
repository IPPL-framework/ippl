// Tests the Laplacian on a scalar field
#include "Ippl.h"

#include <array>
#include <iostream>
#include <sstream>
#include <typeinfo>

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);

    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;

    int pt = std::stoi(argv[1]);
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    const int iterations = std::stoi(argv[2]);

    ippl::e_dim_tag decomp[dim];  // Specifies SERIAL, PARALLEL dims
    for (unsigned int d = 0; d < dim; d++) {
        decomp[d] = ippl::PARALLEL;
    }
    // decomp[d] = ippl::SERIAL;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<dim> layout(owned, decomp);

    // Unit box
    double dx                        = 2.0 / double(pt);
    ippl::Vector<double, dim> hx     = dx;
    ippl::Vector<double, dim> origin = -1;
    Mesh_t mesh(owned, hx, origin);

    double pi = acos(-1.0);

    typedef ippl::Field<double, dim, Mesh_t, Centering_t> Field_t;
    typedef ippl::Field<ippl::Vector<double, dim>, dim, Mesh_t, Centering_t> vector_field_type;

    typedef ippl::Field<double, dim, Mesh_t, Centering_t, Kokkos::OpenMP> Field_host_t;

    Field_t field(mesh, layout);
    Field_t Lap(mesh, layout);
    Field_t Lap_exact(mesh, layout);
    vector_field_type vfield(mesh, layout);

    Field_host_t field_host(mesh, layout);
    Field_host_t Lap_host(mesh, layout);
    Field_host_t Lap_host_exact(mesh, layout);

    typename Field_t::view_type& view       = field.getView();
    typename Field_t::view_type& view_exact = Lap_exact.getView();
    typedef ippl::BConds<Field_t, dim> bc_type;
    typedef ippl::BConds<vector_field_type, dim> vbc_type;

    typedef ippl::BConds<Field_host_t, dim> bc_host_type;

    bc_type bcField;
    vbc_type vbcField;

    bc_host_type bcField_host;

    // X direction periodic BC
    for (unsigned int i = 0; i < 6; ++i) {
        bcField[i]  = std::make_shared<ippl::PeriodicFace<Field_t>>(i);
        vbcField[i] = std::make_shared<ippl::PeriodicFace<vector_field_type>>(i);

        bcField_host[i] = std::make_shared<ippl::PeriodicFace<Field_host_t>>(i);
    }
    ////Lower Y face
    // bcField[2] = std::make_shared<ippl::NoBcFace<Field_t>>(2);
    // vbcField[2] = std::make_shared<ippl::NoBcFace<vector_field_type>>(2);
    ////Higher Y face
    // bcField[3] = std::make_shared<ippl::ConstantFace<Field_t>>(3, 7.0);
    // vbcField[3] = std::make_shared<ippl::ConstantFace<vector_field_type>>(3, 7.0);
    ////Lower Z face
    // bcField[4] = std::make_shared<ippl::ZeroFace<Field_t>>(4);
    // vbcField[4] = std::make_shared<ippl::ZeroFace<vector_field_type>>(4);
    ////Higher Z face
    // bcField[5] = std::make_shared<ippl::ExtrapolateFace<Field_t>>(5, 0.0, 1.0);
    // vbcField[5] = std::make_shared<ippl::ExtrapolateFace<vector_field_type>>(5, 0.0, 1.0);

    field.setFieldBC(bcField);
    Lap.setFieldBC(bcField);
    vfield.setFieldBC(vbcField);

    field_host.setFieldBC(bcField_host);
    Lap_host.setFieldBC(bcField_host);

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

            // view(i, j, k) = 3.0*pow(x,1) + 4.0*pow(y,1) + 5.0*pow(z,1);
            // view(i, j, k) = sin(pi * x) * cos(pi * y) * exp(z);
            view(i, j, k)       = sin(pi * x) * sin(pi * y) * sin(pi * z);
            view_exact(i, j, k) = -3.0 * pi * pi * sin(pi * x) * sin(pi * y) * sin(pi * z);
        });

    // set host field by copying
    auto mirror1 = field.getHostMirror();
    Kokkos::deep_copy(mirror1, view);
    Kokkos::deep_copy(field_host.getView(), mirror1);

    // set host exact laplacian
    auto mirror2 = Lap_exact.getHostMirror();
    Kokkos::deep_copy(mirror2, view_exact);
    Kokkos::deep_copy(Lap_host_exact.getView(), mirror2);

    // field.write();

    // vfield = grad(field);

    // vfield.write();

    // field = div(vfield);

    // field.write();

    Lap      = 0.0;
    Lap_host = 0.0;

    static auto timer = IpplTimings::getTimer("laplace");
    for (int i = 0; i < iterations; i++) {
        IpplTimings::startTimer(timer);
        Lap      = laplace(field);
        Lap_host = laplace(field_host);
        IpplTimings::stopTimer(timer);
        Ippl::fence();
    }

    Lap = Lap - Lap_exact;

    Lap          = pow(Lap, 2);
    Lap_exact    = pow(Lap_exact, 2);
    double error = sqrt(Lap.sum());
    error        = error / sqrt(Lap_exact.sum());

    if (Ippl::Comm->rank() == 0) {
        std::cout << "Error: " << error << std::endl;
    }
    std::stringstream ss;
    ss << "timing_" << pt << "pt_" << iterations << "iterations_" << Ippl::Comm->size()
       << "ranks.dat";
    IpplTimings::print(ss.str());

    return 0;
}
