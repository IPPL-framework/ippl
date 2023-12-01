// Tests the FEM poison solver by // TODO

#include "Ippl.h"

#include "PoissonSolvers/FEMPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned dim = 3;  // 3D problem
        using Mesh_t           = ippl::UniformCartesian<double, 3>;
        using Centering_t      = Mesh_t::DefaultCentering;

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;

        int pt = 4, ptY = 4;
        bool isWeak = false;

        // start a timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        Inform info("Config");
        if (argc >= 2) {
            // First argument is the problem size (log2)
            double N = strtol(argv[1], NULL, 10);
            info << "Got " << N << " as size parameter" << endl;
            pt = ptY = 1 << (int)N;
            if (argc >= 3) {
                if (argv[2][0] == 'w') {
                    // If weak scaling is specified, increase the problem size
                    // along the Y axis such that each rank has the same workload
                    // (simplest enlargement method)
                    ptY = 1 << (5 + (int)N);
                    pt  = 32;
                    info << "Performing weak scaling" << endl;
                    isWeak = true;
                }
            }
        }

        // Define domain
        ippl::Index I(pt), Iy(ptY);
        ippl::NDIndex<dim> owned(I, Iy, I);

        ippl::e_dim_tag allParallel[dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++) {
            allParallel[d] = ippl::PARALLEL;
        }

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(owned, allParallel);

        // Unit box
        double dx                        = 2.0 / double(pt);
        double dy                        = 2.0 / double(ptY);
        ippl::Vector<double, dim> hx     = {dx, dy, dx};
        ippl::Vector<double, dim> origin = -1;
        Mesh_t mesh(owned, hx, origin);

        // Define fields for left hand side and right hand side and solution
        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;
        field_type rhs(mesh, layout), lhs(mesh, layout), solution(mesh, layout);

        // Define boundary conditions
        typedef ippl::BConds<field_type, dim> bc_type;

        bc_type bcField;

        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
        }

        lhs.setFieldBC(bcField);

        // prepare to compute solution and right-hand side
        double pi = Kokkos::numbers::pi_v<double>;

        typename field_type::view_type &viewRHS = rhs.getView(), viewSol = solution.getView();

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();

        using Kokkos::pow, Kokkos::sin, Kokkos::cos;

        // set solution
        int shift1     = solution.getNghost();
        auto policySol = solution.getFieldRangePolicy();
        Kokkos::parallel_for(
            "Assign solution", policySol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first() - shift1;
                const size_t jg = j + lDom[1].first() - shift1;
                const size_t kg = k + lDom[2].first() - shift1;
                double x        = (ig + 0.5) * hx[0];
                double y        = (jg + 0.5) * hx[1];
                double z        = (kg + 0.5) * hx[2];

                viewSol(i, j, k) = sin(sin(pi * x)) * sin(sin(pi * y)) * sin(sin(pi * z));
            });

        // set right hand side
        const int shift2 = rhs.getNghost();
        auto policyRHS   = rhs.getFieldRangePolicy();
        Kokkos::parallel_for(
            "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first() - shift2;
                const size_t jg = j + lDom[1].first() - shift2;
                const size_t kg = k + lDom[2].first() - shift2;
                double x        = (ig + 0.5) * hx[0];
                double y        = (jg + 0.5) * hx[1];
                double z        = (kg + 0.5) * hx[2];

                // https://gitlab.psi.ch/OPAL/Libraries/ippl-solvers/-/blob/5-fftperiodicpoissonsolver/test/TestFFTPeriodicPoissonSolver.cpp#L91
                viewRHS(i, j, k) =
                    pow(pi, 2)
                    * (cos(sin(pi * z)) * sin(pi * z) * sin(sin(pi * x)) * sin(sin(pi * y))
                       + (cos(sin(pi * y)) * sin(pi * y) * sin(sin(pi * x))
                          + (cos(sin(pi * x)) * sin(pi * x)
                             + (pow(cos(pi * x), 2) + pow(cos(pi * y), 2) + pow(cos(pi * z), 2))
                                   * sin(sin(pi * x)))
                                * sin(sin(pi * y)))
                             * sin(sin(pi * z)));
            });

        ippl::FEMPoissonSolver<field_type, field_type> solver(lhs, rhs);

        ippl::ParameterList params;
        params.add("max_iterations", 1000);
        solver.mergeParameters(params);

        lhs = 0.0;
        solver.solve();

        const char* name = isWeak ? "Convergence (weak)" : "Convergence";
        Inform m(name);

        field_type error(mesh, layout);
        error           = lhs - solution;  // sovler solution minus analytical solution
        double relError = norm(error) / norm(solution);

        // Laplace(solver solution) minus rhs
        error          = -laplace(lhs) - rhs;
        double residue = norm(error) / norm(rhs);

        int size    = isWeak ? pt * pt * ptY : pt;
        int itCount = solver.getIterationCount();
        m << size << "," << std::setprecision(16) << relError << "," << residue << "," << itCount
          << endl;

        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();
        IpplTimings::print("timings" + std::to_string(pt) + ".dat");
    }
    ippl::finalize();

    return 0;
}