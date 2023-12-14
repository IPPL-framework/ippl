// Tests the Mutigrid solver for Poisson problems
// by checking the relative error from the exact solution
// Usage:
//      TestMGSolver [size]
//      ./TestMGSolver 6 --info 5

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <iostream>
#include <typeinfo>
#include <string>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"
#include "Utility/ViewUtils.h"

#include "PoissonSolvers/PoissonMG.h"



int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, 3>;
        using Centering_t          = Mesh_t::DefaultCentering;

        int pt = 15, ptY = 15;
        bool isWeak = false;

        Inform info("Config");
        if (argc >= 2) {
            // First argument is the problem size (log2)
            double N = strtol(argv[1], NULL, 10);
            info << "Got " << N << " as size parameter" << endl;
            pt = ptY = (1 << (int)N)-1;
            if (argc >= 3) {
                if (argv[2][0] == 'w') {
                    // If weak scaling is specified, increase the problem size
                    // along the Y axis such that each rank has the same workload
                    // (the simplest enlargement method)
                    ptY = 1 << (5 + (int) N);
                    pt = 32;
                    info << "Performing weak scaling" << endl;
                    isWeak = true;
                }
            }
        }
        info << "Solver is Multigrid "<< endl;

        ippl::Index I(pt), Iy(ptY);
        ippl::NDIndex<dim> owned(I, Iy, I);

        ippl::e_dim_tag allParallel[dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++) {
            allParallel[d] = ippl::PARALLEL;
        }

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(owned, allParallel);

        double dx                        = 1.0 / double(pt);
        double dy                        = 1.0 / double(ptY);
        ippl::Vector<double, dim> hx     = {dx, dy, dx};
        ippl::Vector<double, dim> origin = 0.0;
        Mesh_t mesh(owned, hx, origin);

        double pi = Kokkos::numbers::pi_v<double>;

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;
        field_type rhs(mesh, layout), lhs(mesh, layout), solution(mesh, layout);

        typedef ippl::BConds<field_type, dim> bc_type;

        bc_type bcField;

        // Set Zero Dirichlet Boundary Conditions
        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] = std::make_shared<ippl::ZeroFace<field_type>>(i);
        }

        lhs.setFieldBC(bcField);

        typename field_type::view_type &viewRHS = rhs.getView(), viewSol = solution.getView();

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();

        using Kokkos::pow, Kokkos::sin, Kokkos::cos;

       // int shift1     = solution.getNghost();
        auto policySol = solution.getFieldRangePolicy();
        Kokkos::parallel_for(
            "Assign solution", policySol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first();
                const size_t jg = j + lDom[1].first();
                const size_t kg = k + lDom[2].first();
                double x        = ig  * hx[0];
                double y        = jg * hx[1];
                double z        = kg * hx[2];

                viewSol(i, j, k) = sin(pi * x) + sin(pi * y) + sin(pi * z);
            });

        //const int shift2 = rhs.getNghost();
        auto policyRHS   = rhs.getFieldRangePolicy();
        Kokkos::parallel_for(
            "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first();
                const size_t jg = j + lDom[1].first();
                const size_t kg = k + lDom[2].first();
                double x        = ig * hx[0];
                double y        = jg * hx[1];
                double z        = kg * hx[2];

                // https://gitlab.psi.ch/OPAL/Libraries/ippl-solvers/-/blob/5-fftperiodicpoissonsolver/test/TestFFTPeriodicPoissonSolver.cpp#L91
                viewRHS(i, j, k) =
                    -pow(pi, 2) * (sin(pi * x) + sin(pi * y) + sin(pi * z));
            });

        //ippl::detail::write<double , dim>(viewSol);
        constexpr int levels = 3;
        ippl::PoissonMG<field_type , field_type ,levels> lapsolver(lhs,rhs);

        ippl::ParameterList params;
        params.add("max_iterations", 2000);

        lapsolver.mergeParameters(params);

        lapsolver.setRhs(rhs);
        lapsolver.setLhs(lhs);

        lhs = 0;
        lapsolver.solve();
        //lapsolver.test(rhs);
        const char* name = isWeak ? "Convergence (weak)" : "Convergence";
        Inform m(name);

        field_type error(mesh, layout);
        // Solver solution - analytical solution
        error           = lhs - solution;
        double relError = norm(error) / norm(solution);

        // Laplace(solver solution) - rhs
        error          = -laplace(lhs) - rhs;
        double residue = norm(error) / norm(rhs);

        int size    = isWeak ? pt * pt * ptY : pt;
        int itCount = lapsolver.getIterationCount();
        m << size << "," << std::setprecision(16) << relError << "," << residue << "," << itCount
          << endl;

        IpplTimings::print("timings" + std::to_string(pt) + ".dat");
    }
    ippl::finalize();

    return 0;
}
