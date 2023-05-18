// This program tests the FFTPoissonSolver class with a Gaussian source.
// Different problem sizes are used for the purpose of convergence tests.
// The algorithm used is chosen by the user:
//     srun ./TestGaussian_convergence HOCKNEY --info 10
// OR  srun ./TestGaussian_convergence VICO --info 10

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Utility/IpplTimings.h"

#include "FFTPoissonSolver.h"

using Mesh_t        = ippl::UniformCartesian<double, 3>;
using Centering_t   = Mesh_t::DefaultCentering;
using ScalarField_t = ippl::Field<double, 3, Mesh_t, Centering_t>;
using VectorField_t = ippl::Field<ippl::Vector<double, 3>, 3, Mesh_t, Centering_t>;

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 0.05,
                                       double mu = 0.5) {
    double pi        = Kokkos::numbers::pi_v<double>;
    double prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return -prefactor * exp(-r2 / (2 * sigma * sigma));
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

// Define vtk dump function for plotting the fields
void dumpVTK(std::string path, ScalarField_t& rho, int nx, int ny, int nz, int iteration, double dx,
             double dy, double dz) {
    typename ScalarField_t::view_type::host_mirror_type host_view = rho.getHostMirror();
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
    vtkout << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << nz + 1 << std::endl;
    vtkout << "ORIGIN " << 0.0 << " " << 0.0 << " " << 0.0 << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "CELL_DATA " << (nx) * (ny) * (nz) << std::endl;

    vtkout << "SCALARS Phi float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int z = 1; z < nz + 1; z++) {
        for (int y = 1; y < ny + 1; y++) {
            for (int x = 1; x < nx + 1; x++) {
                vtkout << host_view(x, y, z) << std::endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    Inform msg("");
    Inform msg2all("", INFORM_ALL_NODES);

    std::string algorithm = argv[1];

    // start a timer to time the FFT Poisson solver
    static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
    IpplTimings::startTimer(allTimer);

    // number of interations
    const int n = 6;

    // number of gridpoints to iterate over
    std::array<int, n> N = {4, 8, 16, 32, 64, 128};

    msg << "Spacing Error ErrorEx ErrorEy ErrorEz" << endl;

    for (int p = 0; p < n; ++p) {
        // domain
        int pt = N[p];
        ippl::Index I(pt);
        ippl::NDIndex<3> owned(I, I, I);

        // specifies decomposition; here all dimensions are parallel
        ippl::e_dim_tag decomp[3];
        for (unsigned int d = 0; d < 3; d++)
            decomp[d] = ippl::PARALLEL;

        // unit box
        double dx                      = 1.0 / pt;
        ippl::Vector<double, 3> hx     = {dx, dx, dx};
        ippl::Vector<double, 3> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hx, origin);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(owned, decomp);

        // define the R (rho) field
        ScalarField_t rho;
        rho.initialize(mesh, layout);

        // define the exact solution field
        ScalarField_t exact;
        exact.initialize(mesh, layout);

        // define the Vector field E and the exact E field
        VectorField_t exactE, fieldE;
        exactE.initialize(mesh, layout);
        fieldE.initialize(mesh, layout);

        // assign the rho field with a gaussian
        typename ScalarField_t::view_type view_rho = rho.getView();
        const int nghost                           = rho.getNghost();
        const auto& ldom                           = layout.getLocalNDIndex();

        Kokkos::parallel_for(
            "Assign rho field", ippl::getRangePolicy<3>(view_rho, nghost),
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
            });

        // assign the exact field with its values (erf function)
        typename ScalarField_t::view_type view_exact = exact.getView();

        Kokkos::parallel_for(
            "Assign exact field", ippl::getRangePolicy<3>(view_exact, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                view_exact(i, j, k) = exact_fct(x, y, z);
            });

        // assign the exact E field
        auto view_exactE = exactE.getView();

        Kokkos::parallel_for(
            "Assign exact E-field", ippl::getRangePolicy<3>(view_exactE, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                view_exactE(i, j, k)[0] = exact_E(x, y, z)[0];
                view_exactE(i, j, k)[1] = exact_E(x, y, z)[1];
                view_exactE(i, j, k)[2] = exact_E(x, y, z)[2];
            });

        // set the FFT parameters

        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", false);
        fftParams.add("use_pencils", true);
        fftParams.add("use_gpu_aware", true);
        fftParams.add("comm", ippl::a2av);
        fftParams.add("r2c_direction", 0);
        // define an FFTPoissonSolver object
        ippl::FFTPoissonSolver<ippl::Vector<double, 3>, double, 3, Mesh_t, Centering_t> FFTsolver(
            fieldE, rho, fftParams, algorithm);

        // solve the Poisson equation -> rho contains the solution (phi) now
        FFTsolver.solve();

        // compute relative error norm for potential
        rho        = rho - exact;
        double err = norm(rho) / norm(exact);

        // compute relative error norm for the E-field components
        ippl::Vector<double, 3> errE{0.0, 0.0, 0.0};
        fieldE           = fieldE - exactE;
        auto view_fieldE = fieldE.getView();

        for (size_t d = 0; d < 3; ++d) {
            double temp = 0.0;
            Kokkos::parallel_reduce(
                "Vector errorNr reduce", ippl::getRangePolicy<3>(view_fieldE, nghost),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {
                    double myVal = pow(view_fieldE(i, j, k)[d], 2);
                    valL += myVal;
                },
                Kokkos::Sum<double>(temp));

            double globaltemp = 0.0;
            MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
            double errorNr = std::sqrt(globaltemp);

            temp = 0.0;
            Kokkos::parallel_reduce(
                "Vector errorDr reduce", ippl::getRangePolicy<3>(view_exactE, nghost),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {
                    double myVal = pow(view_exactE(i, j, k)[d], 2);
                    valL += myVal;
                },
                Kokkos::Sum<double>(temp));

            globaltemp = 0.0;
            MPI_Allreduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
            double errorDr = std::sqrt(globaltemp);

            errE[d] = errorNr / errorDr;
        }

        msg << std::setprecision(16) << dx << " " << err << " " << errE[0] << " " << errE[1] << " "
            << errE[2] << endl;
    }

    // stop the timer
    IpplTimings::stopTimer(allTimer);
    IpplTimings::print(std::string("timing.dat"));

    return 0;
}
