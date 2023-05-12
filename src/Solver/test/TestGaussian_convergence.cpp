// This program tests the FFTPoissonSolver class with a Gaussian source.
// Different problem sizes are used for the purpose of convergence tests.
// The test can be ran either in single or T precision.
// The algorithm used is chosen by the user:
//     srun ./TestGaussian_convergence HOCKNEY [DOUBLE|SINGLE] --info 10
// OR  srun ./TestGaussian_convergence VICO [DOUBLE|SINGLE] --info 10

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Utility/IpplTimings.h"
#include "Utility/IpplException.h"

#include "FFTPoissonSolver.h"

template<typename T>
using Mesh_t        = typename ippl::UniformCartesian<T, 3>;

template<typename T>
using Centering_t   = typename Mesh_t<T>::DefaultCentering;

template<typename T>
using ScalarField_t = typename ippl::Field<T, 3, Mesh_t<T>, Centering_t<T>>;

template<typename T>
using VectorField_t = typename ippl::Field<ippl::Vector<T, 3>, 3, Mesh_t<T>, Centering_t<T>>;

template<typename T>
KOKKOS_INLINE_FUNCTION T gaussian(T x, T y, T z, T sigma = 0.05,
                                       T mu = 0.5) {
    T pi        = Kokkos::numbers::pi_v<T>;
    T prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    T r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return -prefactor * exp(-r2 / (2 * sigma * sigma));
}

template<typename T>
KOKKOS_INLINE_FUNCTION T exact_fct(T x, T y, T z, T sigma = 0.05,
                                        T mu = 0.5) {
    T pi = Kokkos::numbers::pi_v<T>;

    T r  = std::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));

    return (1 / (4.0 * pi * r)) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
}

template<typename T>
KOKKOS_INLINE_FUNCTION ippl::Vector<T, 3> exact_E(T x, T y, T z,
                                                       T sigma = 0.05, T mu = 0.5) {
    T pi     = Kokkos::numbers::pi_v<T>;
    T r      = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));
    T factor =
        (1.0 / (4.0 * pi * r * r))
        * ((1.0 / r) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma))
           - Kokkos::sqrt(2.0 / pi) * (1.0 / sigma) * exp(-r * r / (2 * sigma * sigma)));

    ippl::Vector<T, 3> Efield = {(x - mu), (y - mu), (z - mu)};
    return factor * Efield;
}

// Define vtk dump function for plotting the fields
template<typename T>
void dumpVTK(std::string path, ScalarField_t<T>& rho, int nx, int ny, int nz, int iteration, T dx,
             T dy, T dz) {
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


template<typename T>
void compute_convergence(std::string algorithm, int pt){
	Inform errorMsg("");
	Inform errorMsg2all("", INFORM_ALL_NODES);

        ippl::Index I(pt);
        ippl::NDIndex<3> owned(I, I, I);

        // specifies decomposition; here all dimensions are parallel
        ippl::e_dim_tag decomp[3];
        for (unsigned int d = 0; d < 3; d++)
            decomp[d] = ippl::PARALLEL;

        // unit box
        T dx                      = 1.0 / pt;
        ippl::Vector<T, 3> hx     = {dx, dx, dx};
        ippl::Vector<T, 3> origin = {0.0, 0.0, 0.0};
        Mesh_t<T> mesh(owned, hx, origin);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(owned, decomp);

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
        typename ScalarField_t<T>::view_type view_rho = rho.getView();
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
                T x = (ig + 0.5) * hx[0] + origin[0];
                T y = (jg + 0.5) * hx[1] + origin[1];
                T z = (kg + 0.5) * hx[2] + origin[2];

                view_rho(i, j, k) = gaussian(x, y, z);
            });

        // assign the exact field with its values (erf function)
        typename ScalarField_t<T>::view_type view_exact = exact.getView();

        Kokkos::parallel_for(
            "Assign exact field", ippl::getRangePolicy<3>(view_exact, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                T x = (ig + 0.5) * hx[0] + origin[0];
                T y = (jg + 0.5) * hx[1] + origin[1];
                T z = (kg + 0.5) * hx[2] + origin[2];

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

                T x = (ig + 0.5) * hx[0] + origin[0];
                T y = (jg + 0.5) * hx[1] + origin[1];
                T z = (kg + 0.5) * hx[2] + origin[2];

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
        ippl::FFTPoissonSolver<ippl::Vector<T, 3>, T, 3, Mesh_t<T>, Centering_t<T>> FFTsolver(
            fieldE, rho, fftParams, algorithm);

        // solve the Poisson equation -> rho contains the solution (phi) now
        FFTsolver.solve();

        // compute relative error norm for potential
        rho        = rho - exact;
        T err = norm(rho) / norm(exact);

        // compute relative error norm for the E-field components
        ippl::Vector<T, 3> errE{0.0, 0.0, 0.0};
        fieldE           = fieldE - exactE;
        auto view_fieldE = fieldE.getView();

        for (size_t d = 0; d < 3; ++d) {
            T temp = 0.0;
            Kokkos::parallel_reduce(
                "Vector errorNr reduce", ippl::getRangePolicy<3>(view_fieldE, nghost),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, T& valL) {
                    T myVal = pow(view_fieldE(i, j, k)[d], 2);
                    valL += myVal;
                },
                Kokkos::Sum<T>(temp));

            T globaltemp = 0.0;

            MPI_Datatype mpi_type = get_mpi_datatype<T>(temp);
            MPI_Allreduce(&temp, &globaltemp, 1, mpi_type, MPI_SUM, Ippl::getComm());
            T errorNr = std::sqrt(globaltemp);

            temp = 0.0;
            Kokkos::parallel_reduce(
                "Vector errorDr reduce", ippl::getRangePolicy<3>(view_exactE, nghost),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, T& valL) {
                    T myVal = pow(view_exactE(i, j, k)[d], 2);
                    valL += myVal;
                },
                Kokkos::Sum<T>(temp));

            globaltemp = 0.0;
            MPI_Allreduce(&temp, &globaltemp, 1, mpi_type, MPI_SUM, Ippl::getComm());
            T errorDr = std::sqrt(globaltemp);

            errE[d] = errorNr / errorDr;

        }

        errorMsg << std::setprecision(16) << dx << " " << err << " " << errE[0] << " " << errE[1] << " "
            << errE[2] << endl;

    return;
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    Inform msg("");
    Inform msg2all("", INFORM_ALL_NODES);

    std::string algorithm = argv[1];
    std::string precision = argv[2];

    if( precision != "DOUBLE" && precision != "SINGLE" ){
        throw IpplException( "TestGaussian_convergence", "Precision argument must be DOUBLE or SINGLE." );
    }

    // start a timer to time the FFT Poisson solver
    static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
    IpplTimings::startTimer(allTimer);

    // number of interations
    const int n = 6;

    // number of gridpoints to iterate over
    std::array<int, n> N = {4, 8, 16, 32, 64, 128};

    msg << "Spacing Error ErrorEx ErrorEy ErrorEz" << endl;

    for (int p = 0; p < n; ++p) {

        if( precision == "DOUBLE" ) {
            compute_convergence<double>(algorithm, N[p]);
        }
        else {
            compute_convergence<float>(algorithm, N[p]);
         }

    }

    // stop the timer
    IpplTimings::stopTimer(allTimer);
    IpplTimings::print(std::string("timing.dat"));

    return 0;
}
