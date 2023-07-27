//
// TestGaussian_convergence
// This programs tests the FFTPoissonSolver for a Gaussian source.
// Different problem sizes are used for the purpose of convergence tests.
//   Usage:
//     srun ./TestGaussian_convergence <algorithm> <precision> --info 5
//     algorithm = "HOCKNEY" or "VICO", types of open BC algorithms
//     precision = "DOUBLE" or "SINGLE", precision of the fields
//
//     Example:
//       srun ./TestGaussian_convergence HOCKNEY DOUBLE --info 5
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


// TODO:
// Add this file to cmake lists once pack, unpack and the header file sre sorted
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "Solver/FFTPoissonSolver.h"

template <typename T>
using Mesh_t = typename ippl::UniformCartesian<T, 2>;

template <typename T>
using Centering_t = typename Mesh_t<T>::DefaultCentering;

template <typename T>
using ScalarField_t = typename ippl::Field<T, 2, Mesh_t<T>, Centering_t<T>>;

template <typename T>
using VectorField_t = typename ippl::Field<ippl::Vector<T, 2>, 2, Mesh_t<T>, Centering_t<T>>;

template <typename T>
using Solver_t = ippl::FFTPoissonSolver<VectorField_t<T>, ScalarField_t<T>>;

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian(T x, T y, T sigma = 0.05, T mu = 0.5) {
    T pi        = Kokkos::numbers::pi_v<T>;
    T prefactor = (1 / (2 * pi * sigma * sigma));
    T r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu);

    return prefactor * exp(-r2 /(2 * sigma * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T exact_fct(T x, T y, T sigma = 0.05, T mu = 0.5) {
    T pi = Kokkos::numbers::pi_v<T>;

    T r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu);
    // the argument of expint is -ve because std::expint is defined as the integral from -x to inf 
    // the equation for the potential in Zou et al (2021) defines it from +x to inf, hence the factor of -1
    return (1 / (4.0 * pi)) * (std::expint(-r2/(2 * sigma * sigma)) - std::log(r2));

}

template <typename T>
KOKKOS_INLINE_FUNCTION ippl::Vector<T, 2> exact_E(T x, T y, T sigma = 0.05, T mu = 0.5) {
    T pi     = Kokkos::numbers::pi_v<T>;
    T r2     = (x - mu) * (x - mu) + (y - mu) * (y - mu);
    T factor = ((1 / (4 * pi)) * (2 * (1 - exp(-r2 / (2 * sigma * sigma))))) / (r2);
    //T factor = (1 / (2 * pi)) * (1 / r2) * (exp((-r2) / (2 * sigma * sigma)) - 1);
    
    //T factor = - exp(-r2/(2*sigma*sigma)) / (r2/(2*sigma*sigma));

    
    ippl::Vector<T, 2> Efield = {(x - mu), (y - mu)};
    
    //ippl::Vector<T, 2> Efield = {(1/(4*pi)) * (2*(x - mu)/r2  - factor), (1/(4*pi)) * (2*(y - mu)/r2 - factor)};

    return Efield * factor;
}

/*
// Define vtk dump function for plotting the fields
template <typename T>
void dumpVTK(std::string path, ScalarField_t<T>& rho, int nx, int ny, int iteration, T dx,
             T dy) {
    typename ScalarField_t<T>::view_type::host_mirror_type host_view = rho.getHostMirror();
    Kokkos::deep_copy(host_view, rho.getView());
    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << path;
    fname << "/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    if (!vtkout) {
        std::cout << "couldn't open" << std::endl;
    }
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "GaussianSource" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx + 1 << " " << ny + 1 << std::endl;
    vtkout << "ORIGIN " << 0.0 << " " << 0.0 << std::endl;
    vtkout << "SPACING " << dx << " " << dy << std::endl;
    vtkout << "CELL_DATA " << (nx) * (ny) << std::endl;

    vtkout << "SCALARS Phi float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int y = 1; y < ny + 1; y++) {
        for (int x = 1; x < nx + 1; x++) {
            vtkout << host_view(x, y) << std::endl;
        }
    }
    }

    // close the output file for this iteration:
    vtkout.close();
*/

template <typename T>
void compute_convergence(std::string algorithm, int pt) {
    Inform errorMsg("");
    Inform errorMsg2all("", INFORM_ALL_NODES);

    ippl::Index I(pt);
    ippl::NDIndex<2> owned(I, I);

    // specifies decomposition; here all dimensions are parallel
    ippl::e_dim_tag decomp[2];
    for (unsigned int d = 0; d < 2; d++)
        decomp[d] = ippl::PARALLEL;

    // unit box
    T dx                      = 1.0 / pt;
    ippl::Vector<T, 2> hx     = {dx, dx};
    ippl::Vector<T, 2> origin = {0.0, 0.0};
    Mesh_t<T> mesh(owned, hx, origin);

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<2> layout(owned, decomp);

    // define the R (rho) field
    ScalarField_t<T> rho;
    rho.initialize(mesh, layout);

    // define the exact solution field
    ScalarField_t<T> exact;
    exact.initialize(mesh, layout);

    // define the Vector field E and the exact E field
    VectorField_t<T> exactE, fieldE;
    exactE.initialize(mesh, layout);
    fieldE.initialize(mesh, layout);

    // assign the rho field with a gaussian
    auto view_rho    = rho.getView();
    const int nghost = rho.getNghost();
    const auto& ldom = layout.getLocalNDIndex();

    Kokkos::parallel_for(
        "Assign rho field", rho.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const int i, const int j) {
            // go from local to global indices
            const int ig = i + ldom[0].first() - nghost;
            const int jg = j + ldom[1].first() - nghost;

            // define the physical points (cell-centered)
            T x = (ig + 0.5) * hx[0] + origin[0];
            T y = (jg + 0.5) * hx[1] + origin[1];

            view_rho(i, j) = gaussian(x, y);
        });

    // assign the exact field with its values (erf function)
    typename ScalarField_t<T>::view_type view_exact = exact.getView();

    Kokkos::parallel_for(
        "Assign exact field",  exact.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const int i, const int j) {
            const int ig = i + ldom[0].first() - nghost;
            const int jg = j + ldom[1].first() - nghost;

            T x = (ig + 0.5) * hx[0] + origin[0];
            T y = (jg + 0.5) * hx[1] + origin[1];

            view_exact(i, j) = exact_fct(x, y);
        });

    // assign the exact E field
    auto view_exactE = exactE.getView();

    Kokkos::parallel_for(
        "Assign exact E-field", exactE.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const int i, const int j) {
            const int ig = i + ldom[0].first() - nghost;
            const int jg = j + ldom[1].first() - nghost;

            T x = (ig + 0.5) * hx[0] + origin[0];
            T y = (jg + 0.5) * hx[1] + origin[1];

            view_exactE(i, j)[0] = exact_E(x, y)[0];
            view_exactE(i, j)[1] = exact_E(x, y)[1];
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
        params.add("algorithm", Solver_t<T>::HOCKNEY);
    } else {
        throw IpplException("TestGaussian_convergence.cpp main()", "Unrecognized algorithm type - for a 2D solve only HOCKEY is available");
    }

    // add output type
    params.add("output_type", Solver_t<T>::SOL_AND_GRAD);

    // define an FFTPoissonSolver object
    Solver_t<T> FFTsolver(fieldE, rho, params);

    // solve the Poisson equation -> rho contains the solution (phi) now   
    FFTsolver.solve();

    // compute relative error norm for potential
    rho   = rho - exact;
    
    T err = norm(rho) / norm(exact);

    // compute relative error norm for the E-field components
    ippl::Vector<T, 2> errE{0.0, 0.0};
    fieldE           = fieldE - exactE;
    auto view_fieldE = fieldE.getView();

    for (size_t d = 0; d < 2; ++d) {
        T temp = 0.0;
        Kokkos::parallel_reduce(
            "Vector errorNr reduce", ippl::getRangePolicy(view_fieldE, nghost),
            KOKKOS_LAMBDA(const size_t i, const size_t j, T& valL) {
                T myVal = pow(view_fieldE(i, j)[d], 2);
                valL += myVal;
            },
            Kokkos::Sum<T>(temp));

        T globaltemp = 0.0;

        MPI_Datatype mpi_type = get_mpi_datatype<T>(temp);
        MPI_Allreduce(&temp, &globaltemp, 1, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());
        T errorNr = std::sqrt(globaltemp);

        temp = 0.0;
        Kokkos::parallel_reduce(
            "Vector errorDr reduce", ippl::getRangePolicy(view_exactE, nghost),
            KOKKOS_LAMBDA(const size_t i, const size_t j, T& valL) {
                T myVal = pow(view_exactE(i, j)[d], 2);
                valL += myVal;
            },
            Kokkos::Sum<T>(temp));

        globaltemp = 0.0;
        MPI_Allreduce(&temp, &globaltemp, 1, mpi_type, MPI_SUM, ippl::Comm->getCommunicator());
        T errorDr = std::sqrt(globaltemp);

        errE[d] = errorNr / errorDr;
    }

    errorMsg << std::setprecision(16) << dx << " " << err << " " << errE[0] << " " << errE[1] << endl;

    return;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");
        Inform msg2all("", INFORM_ALL_NODES);

        std::string algorithm = argv[1];
        std::string precision = argv[2];

        if (precision != "DOUBLE" && precision != "SINGLE") {
            throw IpplException("TestGaussian_convergence",
                                "Precision argument must be DOUBLE or SINGLE.");
        }

        // start a timer to time the FFT Poisson solver
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        // number of interations
        const int n = 9;

        // number of gridpoints to iterate over
        std::array<int, n> N = {4, 8, 16, 32, 64, 128, 256, 512, 1024};

        msg << "Spacing Error ErrorEx ErrorEy" << endl;

        for (int p = 0; p < n; ++p) {
            if (precision == "DOUBLE") {
                compute_convergence<double>(algorithm, N[p]);
            } else {
                compute_convergence<float>(algorithm, N[p]);
            }
        }

        // stop the timer
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
