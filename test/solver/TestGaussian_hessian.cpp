//
// TestGaussian_hessian
// This programs tests the FFTOpenPoissonSolver for a Gaussian source.
// More specifically, it tests also the Hessian calculation functionality.
// Different problem sizes are used for the purpose of convergence tests.
//   Usage:
//     srun ./TestGaussian_hessian <algorithm> <precision> --info 5
//     algorithm = "HOCKNEY" or "VICO", types of open BC algorithms
//     precision = "DOUBLE" or "SINGLE", precision of the fields
//
//     Example:
//       srun ./TestGaussian_hessian HOCKNEY DOUBLE --info 5
//
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/FFTOpenPoissonSolver.h"

template <typename T>
using Mesh_t = typename ippl::UniformCartesian<T, 3>;

template <typename T>
using Matrix_t = typename Mesh_t<T>::matrix_type;

template <typename T>
using Centering_t = typename Mesh_t<T>::DefaultCentering;

template <typename T>
using ScalarField_t = typename ippl::Field<T, 3, Mesh_t<T>, Centering_t<T>>;

template <typename T>
using VectorField_t = typename ippl::Field<ippl::Vector<T, 3>, 3, Mesh_t<T>, Centering_t<T>>;

template <typename T>
using MField_t = typename ippl::Field<Matrix_t<T>, 3, Mesh_t<T>, Centering_t<T>>;

template <typename T>
using Solver_t = ippl::FFTOpenPoissonSolver<VectorField_t<T>, ScalarField_t<T>>;

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian(T x, T y, T z, T sigma = 0.05, T mu = 0.5) {
    T pi        = Kokkos::numbers::pi_v<T>;
    T prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    T r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T exact_fct(T x, T y, T z, T sigma = 0.05, T mu = 0.5) {
    T pi = Kokkos::numbers::pi_v<T>;

    T r = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));

    return (1 / (4.0 * pi * r)) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION ippl::Vector<T, 3> exact_E(T x, T y, T z, T sigma = 0.05, T mu = 0.5) {
    T pi     = Kokkos::numbers::pi_v<T>;
    T r      = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));
    T factor = (1.0 / (4.0 * pi * r * r))
               * ((1.0 / r) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma))
                  - Kokkos::sqrt(2.0 / pi) * (1.0 / sigma) * exp(-r * r / (2 * sigma * sigma)));

    ippl::Vector<T, 3> Efield = {(x - mu), (y - mu), (z - mu)};
    return factor * Efield;
}

template <typename T>
KOKKOS_INLINE_FUNCTION Matrix_t<T> exact_H(T x, T y, T z, T sigma = 0.05, T mu = 0.5) {
    T pi = Kokkos::numbers::pi_v<T>;

    x -= mu;
    y -= mu;
    z -= mu;

    T r        = Kokkos::sqrt(x * x + y * y + z * z);
    T errorfct = Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
    T exponent = exp(-r * r / (2 * sigma * sigma));

    Matrix_t<T> exactH;

    exactH[0][0] = ((3.0 * x * x * errorfct) / (4.0 * pi * Kokkos::pow(r, 5)))
                   - (errorfct / (4.0 * pi * Kokkos::pow(r, 3)))
                   - ((3.0 * x * x * exponent)
                      / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * Kokkos::pow(r, 4)))
                   + (exponent / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * r * r))
                   - ((x * x * exponent)
                      / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * Kokkos::pow(sigma, 3) * r * r));

    exactH[1][1] = ((3.0 * y * y * errorfct) / (4.0 * pi * Kokkos::pow(r, 5)))
                   - (errorfct / (4.0 * pi * Kokkos::pow(r, 3)))
                   - ((3.0 * y * y * exponent)
                      / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * Kokkos::pow(r, 4)))
                   + (exponent / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * r * r))
                   - ((y * y * exponent)
                      / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * Kokkos::pow(sigma, 3) * r * r));

    exactH[2][2] = ((3.0 * z * z * errorfct) / (4.0 * pi * Kokkos::pow(r, 5)))
                   - (errorfct / (4.0 * pi * Kokkos::pow(r, 3)))
                   - ((3.0 * z * z * exponent)
                      / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * Kokkos::pow(r, 4)))
                   + (exponent / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * r * r))
                   - ((z * z * exponent)
                      / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * Kokkos::pow(sigma, 3) * r * r));

    exactH[0][1] =
        x * y
        * (((3.0 * errorfct) / (4.0 * pi * Kokkos::pow(r, 5)))
           - ((3.0 * exponent)
              / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * Kokkos::pow(r, 4)))
           - (exponent / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * Kokkos::pow(sigma, 3) * r * r)));

    exactH[0][2] =
        x * z
        * (((3.0 * errorfct) / (4.0 * pi * Kokkos::pow(r, 5)))
           - ((3.0 * exponent)
              / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * Kokkos::pow(r, 4)))
           - (exponent / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * Kokkos::pow(sigma, 3) * r * r)));

    exactH[1][2] =
        z * y
        * (((3.0 * errorfct) / (4.0 * pi * Kokkos::pow(r, 5)))
           - ((3.0 * exponent)
              / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * sigma * Kokkos::pow(r, 4)))
           - (exponent / (Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi) * Kokkos::pow(sigma, 3) * r * r)));

    exactH[1][0] = exactH[0][1];
    exactH[2][0] = exactH[0][2];
    exactH[2][1] = exactH[1][2];

    return exactH;
}

// Define vtk dump function for plotting the fields
template <typename T>
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

template <typename T>
void compute_convergence(std::string algorithm, int pt) {
    Inform errorMsg("");
    Inform errorMsg2all("", INFORM_ALL_NODES);

    ippl::Index I(pt);
    ippl::NDIndex<3> owned(I, I, I);

    // specifies decomposition; here all dimensions are parallel
    std::array<bool, 3> isParallel;
    isParallel.fill(true);

    // unit box
    T dx                      = 1.0 / pt;
    ippl::Vector<T, 3> hx     = {dx, dx, dx};
    ippl::Vector<T, 3> origin = {0.0, 0.0, 0.0};
    Mesh_t<T> mesh(owned, hx, origin);

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

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

    // define the Matrix field for the Hessian
    MField_t<T> exactH;
    MField_t<T>* fieldH = nullptr;
    exactH.initialize(mesh, layout);

    // assign the rho field with a gaussian
    auto view_rho    = rho.getView();
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
            T x = (ig + 0.5) * hx[0] + origin[0];
            T y = (jg + 0.5) * hx[1] + origin[1];
            T z = (kg + 0.5) * hx[2] + origin[2];

            view_rho(i, j, k) = gaussian(x, y, z);
        });

    // assign the exact field with its values (erf function)
    auto view_exact = exact.getView();

    Kokkos::parallel_for(
        "Assign exact field", exact.getFieldRangePolicy(),
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
        "Assign exact E-field", exactE.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int ig = i + ldom[0].first() - nghost;
            const int jg = j + ldom[1].first() - nghost;
            const int kg = k + ldom[2].first() - nghost;

            T x = (ig + 0.5) * hx[0] + origin[0];
            T y = (jg + 0.5) * hx[1] + origin[1];
            T z = (kg + 0.5) * hx[2] + origin[2];

            view_exactE(i, j, k) = exact_E(x, y, z);
        });

    // assign the exact Hessian field
    auto view_exactH = exactH.getView();

    Kokkos::parallel_for(
        "Assign exact Matrix field", exactH.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int ig = i + ldom[0].first() - nghost;
            const int jg = j + ldom[1].first() - nghost;
            const int kg = k + ldom[2].first() - nghost;

            T x = (ig + 0.5) * hx[0] + origin[0];
            T y = (jg + 0.5) * hx[1] + origin[1];
            T z = (kg + 0.5) * hx[2] + origin[2];

            view_exactH(i, j, k) = exact_H(x, y, z);
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
    } else if (algorithm == "VICO") {
        params.add("algorithm", Solver_t<T>::VICO);
    } else if (algorithm == "DCT_VICO") {
        params.add("algorithm", Solver_t<T>::DCT_VICO);
    } else {
        throw IpplException("TestGaussian_convergence.cpp main()", "Unrecognized algorithm type");
    }

    // add output type
    params.add("output_type", Solver_t<T>::SOL_AND_GRAD);

    // add hessian flag parameter
    params.add("hessian", true);

    // define an FFTOpenPoissonSolver object
    Solver_t<T> FFTsolver(fieldE, rho, params);

    // solve the Poisson equation -> rho contains the solution (phi) now
    FFTsolver.solve();

    fieldH = FFTsolver.getHessian();

    // compute relative error norm for potential
    rho   = rho - exact;
    T err = norm(rho) / norm(exact);

    // compute relative error norm for the E-field components
    ippl::Vector<T, 3> errE{0.0, 0.0, 0.0};
    fieldE           = fieldE - exactE;
    auto view_fieldE = fieldE.getView();

    for (size_t d = 0; d < 3; ++d) {
        T temp = 0.0;
        Kokkos::parallel_reduce(
            "Vector errorNr reduce", fieldE.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, T& valL) {
                T myVal = Kokkos::pow(view_fieldE(i, j, k)[d], 2);
                valL += myVal;
            },
            Kokkos::Sum<T>(temp));

        T globaltemp = 0.0;

        ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<T>());
        T errorNr = std::sqrt(globaltemp);

        temp = 0.0;
        Kokkos::parallel_reduce(
            "Vector errorDr reduce", exactE.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, T& valL) {
                T myVal = Kokkos::pow(view_exactE(i, j, k)[d], 2);
                valL += myVal;
            },
            Kokkos::Sum<T>(temp));

        globaltemp = 0.0;
        ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<T>());
        T errorDr = std::sqrt(globaltemp);

        errE[d] = errorNr / errorDr;
    }

    // compute relative error for hessian components
    Matrix_t<T> errH;
    auto view_fieldH = fieldH->getView();

    for (size_t m = 0; m < 3; ++m) {
        for (size_t n = 0; n < 3; ++n) {
            T diffNorm  = 0;
            T exactNorm = 0;
            Kokkos::parallel_reduce(
                "MFieldError", fieldH->getFieldRangePolicy(),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, T& diffVal,
                              T& exactVal) {
                    diffVal +=
                        Kokkos::pow(view_fieldH(i, j, k)[m][n] - view_exactH(i, j, k)[m][n], 2);
                    exactVal += Kokkos::pow(view_exactH(i, j, k)[m][n], 2);
                },
                Kokkos::Sum<T>(diffNorm), Kokkos::Sum<T>(exactNorm));

            T global_diff  = 0.0;
            T global_exact = 0.0;

            ippl::Comm->allreduce(diffNorm, global_diff, 1, std::plus<T>());
            ippl::Comm->allreduce(exactNorm, global_exact, 1, std::plus<T>());

            errH[m][n] = Kokkos::sqrt(global_diff / global_exact);
        }
    }

    errorMsg << std::setprecision(16) << dx << " " << err << " " << errE[0] << " " << errH[0][0]
             << " " << errH[0][1] << endl;

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

        // gridsizes to iterate over
        std::array<int, 6> N = {4, 8, 16, 32, 64, 128};

        msg << "Spacing Error ErrorEx ErrorDxx ErrorDxy" << endl;

        for (int pt : N) {
            if (precision == "DOUBLE") {
                compute_convergence<double>(algorithm, pt);
            } else {
                compute_convergence<float>(algorithm, pt);
            }
        }

        // stop the timer
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
