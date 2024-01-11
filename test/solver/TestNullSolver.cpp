//
// TestNullSolver
// This programs tests the NullSolver for a Gaussian source.
//   Usage:
//     srun ./TestNullSolver --info 5
//
//   Example:
//     srun ./TestNullSolver --info 5
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Utility/IpplTimings.h"

#include "PoissonSolvers/NullSolver.h"

// type declarations
using Mesh_t        = typename ippl::UniformCartesian<double, 3>;
using Centering_t   = typename Mesh_t::DefaultCentering;
using ScalarField_t = typename ippl::Field<double, 3, Mesh_t, Centering_t>;
using VectorField_t = typename ippl::Field<ippl::Vector<double, 3>, 3, Mesh_t, Centering_t>;
using Solver_t      = ippl::NullSolver<VectorField_t, ScalarField_t>;

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 0.05,
                                       double mu = 0.5) {
    double pi        = Kokkos::numbers::pi_v<double>;
    double prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");
        Inform msg2all("", INFORM_ALL_NODES);

        // start a timer to time the solver
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        int pt = std::atoi(argv[1]);

        msg << "TestNullSolver, problem size = " << pt << "^3" << endl;

        ippl::Index I(pt);
        ippl::NDIndex<3> owned(I, I, I);

        // specifies decomposition; here all dimensions are parallel
        std::array<bool, 3> isParallel;
        isParallel.fill(true);

        // unit box
        double dx                      = 1.0 / pt;
        ippl::Vector<double, 3> hx     = {dx, dx, dx};
        ippl::Vector<double, 3> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hx, origin);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        // define the R (rho) field
        ScalarField_t rho;
        rho.initialize(mesh, layout);
        // ScalarField_t exact;
        // exact.initialize(mesh, layout);

        // define the Vector field E and the exact E field
        VectorField_t fieldE;
        fieldE.initialize(mesh, layout);

        // assign the rho field with a gaussian
        auto view_rho = rho.getView();
        // auto view_exact  = exact.getView();
        const int nghost = rho.getNghost();
        const auto& ldom = layout.getLocalNDIndex();

        Kokkos::parallel_for(
            "Assign rho field", rho.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                // define the physical points (cell-centered)
                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                view_rho(i, j, k) = gaussian(x, y, z);
                // view_exact(i, j, k) = gaussian(x, y, z);
            });

        // define a NullSolver object
        Solver_t no_solver(fieldE, rho);

        // solve the Poisson equation -> rho contains the solution (phi) now
        no_solver.solve();

        // compute relative error norm for potential
        // rho   = rho - exact;
        double err = norm(rho);  // / norm(exact);

        // compute relative error norm for the E-field components
        ippl::Vector<double, 3> errE{0.0, 0.0, 0.0};
        auto view_fieldE = fieldE.getView();

        for (size_t d = 0; d < 3; ++d) {
            double temp = 0.0;
            Kokkos::parallel_reduce(
                "Vector errorNr reduce", fieldE.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {
                    double myVal = Kokkos::pow(view_fieldE(i, j, k)[d], 2);
                    valL += myVal;
                },
                Kokkos::Sum<double>(temp));

            double globaltemp = 0.0;

            ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<double>());
            errE[d] = std::sqrt(globaltemp);
        }

        msg << std::setprecision(16) << "Errors: " << err << " " << errE[0] << " " << errE[1] << " "
            << errE[2] << endl;

        // stop the timer
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
