#include "Ippl.h"

#include <iostream>
#include <typeinfo>

#include "P3MSolver.h"

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);

    constexpr unsigned int dim = 3;

    using Mesh_t      = ippl::UniformCartesian<double, 3>;
    using Centering_t = Mesh_t::DefaultCentering;

    typedef ippl::Field<double, dim, Mesh_t, Centering_t> Field_t;
    typedef ippl::Vector<double, 3> Vector_t;
    typedef ippl::Field<Vector_t, dim, Mesh_t, Centering_t> VField_t;
    typedef ippl::P3MSolver<Vector_t, double, dim, Mesh_t, Centering_t> Solver_t;

    // get the gridsize from the user
    ippl::Vector<int, dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

    // domain
    ippl::NDIndex<dim> owned;
    for (unsigned i = 0; i < dim; i++) {
        owned[i] = ippl::Index(nr[i]);
    }

    // specifies decomposition; here all dimensions are parallel
    ippl::e_dim_tag decomp[dim];
    for (unsigned int d = 0; d < dim; d++) {
        decomp[d] = ippl::PARALLEL;
    }

    // unit box
    double dx       = 1.0 / nr[0];
    double dy       = 1.0 / nr[1];
    double dz       = 1.0 / nr[2];
    Vector_t hr     = {dx, dy, dz};
    Vector_t origin = {-0.5, -0.5, -0.5};

    Mesh_t mesh(owned, hr, origin);
    ippl::FieldLayout<dim> layout(owned, decomp);

    Field_t field;
    field.initialize(mesh, layout);

    VField_t efield;
    efield.initialize(mesh, layout);

    ippl::ParameterList params;
    params.add("use_heffte_defaults", false);
    params.add("use_pencils", true);
    // params.add("use_reorder", false);
    params.add("use_gpu_aware", true);
    params.add("comm", ippl::a2av);
    params.add("r2c_direction", 0);

    // assign the rho field with 2.0
    typename Field_t::view_type view_rho = field.getView();
    const int nghost                     = field.getNghost();

    Kokkos::parallel_for(
        "Assign rho field", ippl::getRangePolicy<3>(view_rho, nghost),
        KOKKOS_LAMBDA(const int i, const int j, const int k) { view_rho(i, j, k) = 2.0; });

    std::cout << "Rho: " << std::endl;
    field.write();

    Solver_t solver(efield, field, params);

    solver.solve();

    std::cout << "Computed phi: " << std::endl;
    field.write();

    std::cout << "Efield: " << std::endl;
    efield.write();

    return 0;
}
