
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <typeinfo>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/PoissonCG.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
	Inform m("");

        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, 3>;
        using Centering_t          = Mesh_t::DefaultCentering;

        unsigned pt = std::atoi(argv[1]);
        std::string solver = "not preconditioned";

        // start the timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        // start the timer
        static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
        IpplTimings::startTimer(initTimer);

        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++) {
            isParallel[d] = true;
        }

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // Unit box
        double dx                        = 2.0 / double(pt);
        ippl::Vector<double, dim> hx     = {dx, dx, dx};
        ippl::Vector<double, dim> origin = -1;
        Mesh_t mesh(owned, hx, origin);

        double pi = Kokkos::numbers::pi_v<double>;

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;
        field_type rhs(mesh, layout), lhs(mesh, layout), solution(mesh, layout);

        typedef ippl::BConds<field_type, dim> bc_type;
        bc_type bcField;
        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
        }
        lhs.setFieldBC(bcField);

        typename field_type::view_type &viewRHS = rhs.getView(), viewSol = solution.getView();

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();

        using Kokkos::pow, Kokkos::sin, Kokkos::cos;

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

        ippl::PoissonCG<field_type> lapsolver;

        ippl::ParameterList params;
        params.add("max_iterations", 2000);
        params.add("tolerance", 1e-13);
        params.add("solver", solver);

        lapsolver.mergeParameters(params);

        lapsolver.setRhs(rhs);
        lapsolver.setLhs(lhs);

        IpplTimings::stopTimer(initTimer);

        // start the timer
        static IpplTimings::TimerRef solveTimer = IpplTimings::getTimer("solve");
        for (int i = 0; i < 5; ++i) {
            lhs = 0;

            IpplTimings::startTimer(solveTimer);
            lapsolver.solve();
            IpplTimings::stopTimer(solveTimer);

	        field_type error(mesh, layout);
            // Solver solution - analytical solution
	        error           = lhs - solution;
	        double relError = norm(error) / norm(solution);

	        // Laplace(solver solution) - rhs
	        error          = -laplace(lhs) - rhs;
	        double residue = norm(error) / norm(rhs);

	        int itCount = lapsolver.getIterationCount();
            m << pt << "," << std::setprecision(16) << relError << "," << residue << "," 
              << itCount << endl;

            Kokkos::parallel_for(
            "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first() - shift2;
                const size_t jg = j + lDom[1].first() - shift2;
                const size_t kg = k + lDom[2].first() - shift2;
                double x        = (ig + 0.5) * hx[0];
                double y        = (jg + 0.5) * hx[1];
                double z        = (kg + 0.5) * hx[2];

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
        }

        IpplTimings::stopTimer(allTimer);

        IpplTimings::print();
        IpplTimings::print("timings.dat");
    }
    ippl::finalize();

    return 0;
}
