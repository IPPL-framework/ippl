//
// TestSphere
// This programs tests the FFTPoissonSolver for the gravitational case.
// The source is a constant term which is 0 outside a certain radius,
// and the exact solution is the gravitational potential of a sphere.
//   Usage:
//     srun ./TestSphere <algorithm> --info 5
//     algorithm = "HOCKNEY" or "VICO", types of open BC algorithms
//
//     Example:
//       srun ./TestSphere HOCKNEY --info 5
//
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Solver/FFTPoissonSolver.h"

KOKKOS_INLINE_FUNCTION double source(double x, double y, double z, double density = 1.0,
                                     double R = 1.0, double mu = 1.2) {
    double pi = Kokkos::numbers::pi_v<double>;
    double G  = 6.674e-11;

    double r = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));
    bool checkInside = (r <= R);

    return double(checkInside) * 4.0 * pi * G * density;
}

KOKKOS_INLINE_FUNCTION double exact_fct(double x, double y, double z, double density = 1.0,
                                        double R = 1.0, double mu = 1.2) {
    double pi = Kokkos::numbers::pi_v<double>;
    double G  = 6.674e-11;

    double r = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));

    bool checkInside = (r <= R);
    return (double(checkInside) * (2.0 / 3.0) * pi * G * density * (3 * R * R - r * r))
           + ((1.0 - double(checkInside)) * (4.0 / 3.0) * pi * G * density * R * R * R / r);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        std::string algorithm = argv[1];

        // number of interations
        const int n = 4;

        // gridpoints to iterate over
        std::array<int, n> N = {16, 32, 64, 128};

        std::cout << "Spacing Error" << std::endl;

        for (int p = 0; p < n; ++p) {
            // domain
            int pt = N[p];
            ippl::Index I(pt);
            ippl::NDIndex<3> owned(I, I, I);

            // specifies decomposition; here all dimensions are parallel
            ippl::e_dim_tag decomp[3];
            for (unsigned int d = 0; d < 3; d++)
                decomp[d] = ippl::PARALLEL;

            using Mesh_t      = ippl::UniformCartesian<double, 3>;
            using Centering_t = Mesh_t::DefaultCentering;
            using Vector_t    = ippl::Vector<double, 3>;
            typedef ippl::Field<double, 3, Mesh_t, Centering_t> field;
            typedef ippl::Field<Vector_t, 3, Mesh_t, Centering_t> vfield;
            using Solver_t = ippl::FFTPoissonSolver<vfield, field>;

            // unit box
            double dx       = 2.4 / pt;
            Vector_t hx     = {dx, dx, dx};
            Vector_t origin = {0.0, 0.0, 0.0};
            Mesh_t mesh(owned, hx, origin);

            // all parallel layout, standard domain, normal axis order
            ippl::FieldLayout<3> layout(owned, decomp);

            // define the L (phi) and R (rho) fields
            field rho;
            rho.initialize(mesh, layout);

            // define the exact solution field
            field exact;
            exact.initialize(mesh, layout);

            // assign the rho field with its value
            typename field::view_type view_rho = rho.getView();
            const int nghost                   = rho.getNghost();
            const auto& ldom                   = layout.getLocalNDIndex();

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

                    view_rho(i, j, k) = source(x, y, z);
                });

            // assign the exact field with its values
            typename field::view_type view_exact = exact.getView();

            Kokkos::parallel_for(
                "Assign exact field", exact.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    double x = (ig + 0.5) * hx[0] + origin[0];
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    double z = (kg + 0.5) * hx[2] + origin[2];

                    view_exact(i, j, k) = exact_fct(x, y, z);
                });

            // set the solver parameters
            ippl::ParameterList params;

            // set the FFT parameters
            params.add("use_heffte_defaults", false);
            params.add("use_pencils", true);
            params.add("use_gpu_aware", true);
            params.add("comm", ippl::a2av);
            params.add("r2c_direction", 0);

            // set the algorithm
            if (algorithm == "HOCKNEY") {
                params.add("algorithm", Solver_t::HOCKNEY);
            } else if (algorithm == "VICO") {
                params.add("algorithm", Solver_t::VICO);
            } else {
                throw IpplException("TestGaussian.cpp main()", "Unrecognized algorithm type");
            }

            // define an FFTPoissonSolver object
            Solver_t FFTsolver(rho, params);

            // solve the Poisson equation -> rho contains the solution (phi) now
            FFTsolver.solve();

            // compute the relative error norm
            rho        = rho - exact;
            double err = norm(rho) / norm(exact);

            std::cout << std::setprecision(16) << dx << " " << err << std::endl;
        }
    }
    ippl::finalize();

    return 0;
}
