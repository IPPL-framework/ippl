// Integration test for preconditioner input validation behavior.
//
// Verifies two policies:
// 1) Unknown preconditioner types are rejected with an exception.
// 2) Known preconditioners with invalid tuning values are accepted by
//    falling back to default parameters (with warnings).
#include "Ippl.h"

#include <array>
#include <exception>
#include <limits>
#include <memory>

#include "PoissonSolvers/PoissonCG.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t                = ippl::UniformCartesian<double, dim>;
        using Centering_t           = Mesh_t::DefaultCentering;
        using Field_t               = ippl::Field<double, dim, Mesh_t, Centering_t>;
        using BConds_t              = ippl::BConds<Field_t, dim>;

        ippl::Vector<unsigned, dim> I(4);
        ippl::NDIndex<dim> domain(I);
        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, domain, isParallel);
        ippl::Vector<double, dim> hx     = 1.0 / 4.0;
        ippl::Vector<double, dim> origin = 0.0;
        Mesh_t mesh(domain, hx, origin);

        Field_t rhs(mesh, layout);
        Field_t lhs(mesh, layout);
        rhs = 0.0;
        lhs = 0.0;

        BConds_t bcField;
        for (unsigned int i = 0; i < 2 * dim; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<Field_t>>(i);
        }
        rhs.setFieldBC(bcField);
        lhs.setFieldBC(bcField);

        bool unknownTypeThrows = false;
        try {
            ippl::PoissonCG<Field_t> solver;
            ippl::ParameterList params;
            params.add("solver", "preconditioned");
            params.add("preconditioner_type", "my_preconditioner");
            solver.mergeParameters(params);
            solver.setRhs(rhs);
            solver.setLhs(lhs);
        } catch (const IpplException&) {
            unknownTypeThrows = true;
        } catch (...) {
            unknownTypeThrows = true;
        }

        if (!unknownTypeThrows) {
            ippl::finalize();
            return 1;
        }

        bool invalidKnownTypeAccepted = true;
        try {
            ippl::PoissonCG<Field_t> solver;
            ippl::ParameterList params;
            params.add("solver", "preconditioned");
            params.add("preconditioner_type", "ssor");
            params.add("newton_level", -1);
            params.add("chebyshev_degree", -1);
            params.add("gauss_seidel_inner_iterations", -2);
            params.add("gauss_seidel_outer_iterations", -3);
            params.add("richardson_iterations", -4);
            params.add("communication", 5);
            params.add("ssor_omega", std::numeric_limits<double>::quiet_NaN());
            solver.mergeParameters(params);
            solver.setRhs(rhs);
            solver.setLhs(lhs);
        } catch (...) {
            invalidKnownTypeAccepted = false;
        }

        if (!invalidKnownTypeAccepted) {
            ippl::finalize();
            return 2;
        }
    }
    ippl::finalize();
    return 0;
}
