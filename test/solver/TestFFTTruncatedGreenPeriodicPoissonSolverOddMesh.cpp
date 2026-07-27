#include "Ippl.h"

#include <iostream>

#include "Utility/IpplException.h"

#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int status = 1;
    {
        constexpr unsigned dim = 3;
        using Mesh_t           = ippl::UniformCartesian<double, dim>;
        using Centering_t      = Mesh_t::DefaultCentering;
        using Field_t          = ippl::Field<double, dim, Mesh_t, Centering_t>;
        using Vector_t         = ippl::Vector<double, dim>;
        using VField_t         = ippl::Field<Vector_t, dim, Mesh_t, Centering_t>;
        using Solver_t         = ippl::FFTTruncatedGreenPeriodicPoissonSolver<VField_t, Field_t>;

        const ippl::Vector<int, dim> nr = {15, 16, 16};
        ippl::NDIndex<dim> owned;
        for (unsigned d = 0; d < dim; ++d) {
            owned[d] = ippl::Index(nr[d]);
        }

        std::array<bool, dim> isParallel;
        isParallel.fill(true);
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        const Vector_t hr     = {1.0 / nr[0], 1.0 / nr[1], 1.0 / nr[2]};
        const Vector_t origin = {-0.5, -0.5, -0.5};
        Mesh_t mesh(owned, hr, origin);
        Field_t rho(mesh, layout);
        rho = 0.0;

        ippl::ParameterList params;
        params.add("use_heffte_defaults", false);
        params.add("use_pencils", true);
        params.add("use_gpu_aware", true);
        params.add("comm", ippl::a2av);
        params.add("r2c_direction", 0);
        params.add("alpha", 2.0);
        params.add("force_constant", 1.0);

        Solver_t solver(rho, params);
        try {
            solver.solve();
        } catch (const IpplException&) {
            status = 0;
        }

        if (ippl::Comm->rank() == 0 && status != 0) {
            std::cerr << "Odd mesh was not rejected" << std::endl;
        }
    }
    ippl::finalize();
    return status;
}
