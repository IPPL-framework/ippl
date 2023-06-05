//
// TestGaussian
// This program tests the FFTPoissonSolver class with a Gaussian source.
// The solve is iterated 5 times for the purpose of timing studies.
//   Usage:
//     srun ./TestGaussian
//                  <nx> <ny> <nz> <reshape> <comm>
//                  <reorder> <algorithm> --info 5
//     nx        = No. cell-centered points in the x-direction
//     ny        = No. cell-centered points in the y-direction
//     nz        = No. cell-centered points in the z-direction
//     reshape   = "pencils" or "slabs" (heffte parameter)
//     comm      = "a2a", "a2av", "p2p", "p2p_pl" (heffte parameter)
//     reorder   = "reorder" or "no-reorder" (heffte parameter)
//     algorithm = "HOCKNEY" or "VICO", types of open BC algorithms
//
//     For more info on the heffte parameters, see:
//     https://github.com/icl-utk-edu/heffte
//
//     Example:
//       srun ./TestGaussian 64 64 64 pencils a2a no-reorder HOCKNEY --info 5
//
// Copyright (c) 2023, Sonali Mayani,
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "FFTPoissonSolver.h"

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 0.05,
                                       double mu = 0.5) {
    double pi        = Kokkos::numbers::pi_v<double>;
    double prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

KOKKOS_INLINE_FUNCTION double exact_fct(double x, double y, double z, double sigma = 0.05,
                                        double mu = 0.5) {
    double pi = Kokkos::numbers::pi_v<double>;
    double r  = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));

    return (1 / (4.0 * pi * r)) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
}

KOKKOS_INLINE_FUNCTION ippl::Vector<double, 3> exact_E(double x, double y, double z,
                                                       double sigma = 0.05, double mu = 0.5) {
    double pi = Kokkos::numbers::pi_v<double>;
    double r  = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));
    double factor =
        (1.0 / (4.0 * pi * r * r))
        * ((1.0 / r) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma))
           - Kokkos::sqrt(2.0 / pi) * (1.0 / sigma) * exp(-r * r / (2 * sigma * sigma)));

    ippl::Vector<double, 3> Efield = {(x - mu), (y - mu), (z - mu)};
    return factor * Efield;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        const unsigned int Dim = 3;

        using Mesh_t      = ippl::UniformCartesian<double, 3>;
        using Centering_t = Mesh_t::DefaultCentering;
        typedef ippl::Field<double, Dim, Mesh_t, Centering_t> field;
        typedef ippl::Field<ippl::Vector<double, Dim>, Dim, Mesh_t, Centering_t> fieldV;
        using Solver_t = ippl::FFTPoissonSolver<fieldV, field>;

        // start a timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        // get the gridsize from the user
        ippl::Vector<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        // get heffte parameters from the user
        std::string reshape       = argv[4];  // slabs or pencils
        std::string communication = argv[5];  // a2a or p2p
        std::string reordering    = argv[6];  // reorder or no-reorder

        // get the algorithm to be used
        std::string algorithm = argv[7];  // Hockney or Vico

        // print out info and title for the relative error (L2 norm)
        msg << "Test Gaussian, grid = " << nr << ", heffte params: " << reshape << " " << communication
            << " " << reordering << ", algorithm = " << algorithm << endl;
        msg << "Spacing Error ErrorEx ErrorEy ErrorEz" << endl;

        // domain
        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }

        // specifies decomposition; here all dimensions are parallel
        ippl::e_dim_tag decomp[Dim];
        for (unsigned int d = 0; d < Dim; d++) {
            decomp[d] = ippl::PARALLEL;
        }

        // unit box
        double dx                        = 1.0 / nr[0];
        double dy                        = 1.0 / nr[1];
        double dz                        = 1.0 / nr[2];
        ippl::Vector<double, Dim> hr     = {dx, dy, dz};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<Dim> layout(owned, decomp);

        // define the R (rho) field
        field exact, rho;
        exact.initialize(mesh, layout);
        rho.initialize(mesh, layout);

        // define the Vector field E (LHS)
        fieldV exactE, fieldE;
        exactE.initialize(mesh, layout);
        fieldE.initialize(mesh, layout);

        field Ex, Ey, Ez;
        Ex.initialize(mesh, layout);
        Ey.initialize(mesh, layout);
        Ez.initialize(mesh, layout);

        // assign the rho field with a gaussian
        typename field::view_type view_rho = rho.getView();
        const int nghost                   = rho.getNghost();
        const auto& ldom                   = layout.getLocalNDIndex();

        Kokkos::parallel_for(
            "Assign rho field", ippl::getRangePolicy(view_rho, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                // define the physical points (cell-centered)
                double x = (ig + 0.5) * hr[0] + origin[0];
                double y = (jg + 0.5) * hr[1] + origin[1];
                double z = (kg + 0.5) * hr[2] + origin[2];

                view_rho(i, j, k) = gaussian(x, y, z);
            });

        // assign the exact field with its values (erf function)
        typename field::view_type view_exact = exact.getView();

        Kokkos::parallel_for(
            "Assign exact field", ippl::getRangePolicy(view_exact, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                double x = (ig + 0.5) * hr[0] + origin[0];
                double y = (jg + 0.5) * hr[1] + origin[1];
                double z = (kg + 0.5) * hr[2] + origin[2];

                view_exact(i, j, k) = exact_fct(x, y, z);
            });

        // assign the exact E field
        auto view_exactE = exactE.getView();

        Kokkos::parallel_for(
            "Assign exact E-field", ippl::getRangePolicy(view_exactE, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                double x = (ig + 0.5) * hr[0] + origin[0];
                double y = (jg + 0.5) * hr[1] + origin[1];
                double z = (kg + 0.5) * hr[2] + origin[2];

                view_exactE(i, j, k)[0] = exact_E(x, y, z)[0];
                view_exactE(i, j, k)[1] = exact_E(x, y, z)[1];
                view_exactE(i, j, k)[2] = exact_E(x, y, z)[2];
            });

        // Parameter List to pass to solver
        ippl::ParameterList params;

        // set the FFT parameters
        if (reshape == "pencils") {
            params.add("use_pencils", true);
        } else if (reshape == "slabs") {
            params.add("use_pencils", false);
        } else {
            throw IpplException("TestGaussian.cpp main()", "Unrecognized heffte parameter");
        }

        if (communication == "a2a") {
            params.add("comm", ippl::a2a);
        } else if (communication == "a2av") {
            params.add("comm", ippl::a2av);
        } else if (communication == "p2p") {
            params.add("comm", ippl::p2p);
        } else if (communication == "p2p_pl") {
            params.add("comm", ippl::p2p_pl);
        } else {
            throw IpplException("TestGaussian.cpp main()", "Unrecognized heffte parameter");
        }

        if (reordering == "reorder") {
            params.add("use_reorder", true);
        } else if (reordering == "no-reorder") {
            params.add("use_reorder", false);
        } else {
            throw IpplException("TestGaussian.cpp main()", "Unrecognized heffte parameter");
        }
        params.add("use_heffte_defaults", false);
        params.add("use_gpu_aware", true);
        params.add("r2c_direction", 0);

        // set the algorithm
        if (algorithm == "HOCKNEY") {
            params.add("algorithm", Solver_t::HOCKNEY);
        } else if (algorithm == "VICO") {
            params.add("algorithm", Solver_t::VICO);
        } else {
            throw IpplException("TestGaussian.cpp main()", "Unrecognized algorithm type");
        }

        // add output type
        params.add("output_type", Solver_t::SOL_AND_GRAD);

        // define an FFTPoissonSolver object
        Solver_t FFTsolver(fieldE, rho, params);

        // iterate over 5 timesteps
        for (int times = 0; times < 5; ++times) {
            // solve the Poisson equation -> rho contains the solution (phi) now
            FFTsolver.solve();

            const int nghostE = fieldE.getNghost();
            auto Eview        = fieldE.getView();

            auto viewEx = Ex.getView();
            auto viewEy = Ey.getView();
            auto viewEz = Ez.getView();

            Kokkos::parallel_for(
                "Vector E reduce", ippl::getRangePolicy(Eview, nghostE),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    viewEx(i, j, k) = Eview(i, j, k)[0];
                    viewEy(i, j, k) = Eview(i, j, k)[1];
                    viewEz(i, j, k) = Eview(i, j, k)[2];
                });

            // compute relative error norm for potential
            rho        = rho - exact;
            double err = norm(rho) / norm(exact);

            // compute relative error norm for the E-field components
            ippl::Vector<double, Dim> errE{0.0, 0.0, 0.0};
            fieldE           = fieldE - exactE;
            auto view_fieldE = fieldE.getView();

            for (size_t d = 0; d < Dim; ++d) {
                double temp = 0.0;
                Kokkos::parallel_reduce(
                    "Vector errorNr reduce", ippl::getRangePolicy(view_fieldE, nghost),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {
                        double myVal = pow(view_fieldE(i, j, k)[d], 2);
                        valL += myVal;
                    },
                    Kokkos::Sum<double>(temp));

                double globaltemp = 0.0;
                MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
                double errorNr = std::sqrt(globaltemp);

                temp = 0.0;
                Kokkos::parallel_reduce(
                    "Vector errorDr reduce", ippl::getRangePolicy(view_exactE, nghost),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {
                        double myVal = pow(view_exactE(i, j, k)[d], 2);
                        valL += myVal;
                    },
                    Kokkos::Sum<double>(temp));

                globaltemp = 0.0;
                MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
                double errorDr = std::sqrt(globaltemp);

                errE[d] = errorNr / errorDr;
            }

            msg << std::setprecision(16) << dx << " " << err << " " << errE[0] << " " << errE[1] << " "
                << errE[2] << endl;

            // reassign the correct values to the fields for the loop to work
            Kokkos::parallel_for(
                "Assign rho field", ippl::getRangePolicy(view_rho, nghost),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    // go from local to global indices
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    // define the physical points (cell-centered)
                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];

                    view_rho(i, j, k) = gaussian(x, y, z);
                });

            Kokkos::parallel_for(
                "Assign exact field", ippl::getRangePolicy(view_exact, nghost),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];

                    view_exact(i, j, k) = exact_fct(x, y, z);
                });

            Kokkos::parallel_for(
                "Assign exact E-field", ippl::getRangePolicy(view_exactE, nghost),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];

                    view_exactE(i, j, k)[0] = exact_E(x, y, z)[0];
                    view_exactE(i, j, k)[1] = exact_E(x, y, z)[1];
                    view_exactE(i, j, k)[2] = exact_E(x, y, z)[2];
                });
        }

        // stop the timers
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
