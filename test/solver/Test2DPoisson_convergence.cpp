//
// Test2DPoisson_convergence
// This programs tests the FFTOpenPoissonSolver for a 2D Gaussian source.
// Different problem sizes are used for the purpose of convergence tests.
//   Usage:
//     srun ./Test2DPoisson_convergence <algorithm> <precision> --info 5
//     algorithm = "HOCKNEY" (only one supported in 2D currently)
//     precision = "DOUBLE" or "SINGLE", precision of the fields
//
//     Example:
//       srun ./Test2DPoisson_convergence HOCKNEY DOUBLE --info 5
//
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/FFTOpenPoissonSolver.h"

constexpr unsigned int Dim = 2;

template <typename T>
using Mesh_t = typename ippl::UniformCartesian<T, Dim>;

template <typename T>
using Centering_t = typename Mesh_t<T>::DefaultCentering;

template <typename T>
using ScalarField_t = typename ippl::Field<T, Dim, Mesh_t<T>, Centering_t<T>>;

template <typename T>
using VectorField_t = typename ippl::Field<ippl::Vector<T, Dim>, Dim, Mesh_t<T>, Centering_t<T>>;

template <typename T>
using Solver_t = ippl::FFTOpenPoissonSolver<VectorField_t<T>, ScalarField_t<T>>;

// Exponential integral Ei(x) = integral_{-inf}^{x} (e^t / t) dt (SciPy expi convention).
template <typename T>
KOKKOS_INLINE_FUNCTION T Ei(T x) {
    const T gamma = Kokkos::numbers::egamma_v<T>;

    if (x < T(4)) {
        T sum  = gamma + Kokkos::log(x);
        T term = x;
        for (int n = 1; n <= 20; ++n) {
            sum += term / T(n);
            term = term * x / T(n + 1);
        }
        return sum;
    }

    T sum  = T(1);
    T fact = T(1);
    T xpow = T(1);
    for (int k = 1; k <= 6; ++k) {
        fact *= T(k);
        xpow *= x;
        sum += fact / xpow;
    }
    return Kokkos::exp(x) * sum / x;
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian(T x, T y, T sigma = 0.05, T mu = 0.5) {
    T pi        = Kokkos::numbers::pi_v<T>;
    T prefactor = (1 / (2 * pi * sigma * sigma));
    T r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu);

    return prefactor * Kokkos::exp(-r2 / (2 * sigma * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T exact_fct(T x, T y, T sigma = 0.05, T mu = 0.5) {
    T pi = Kokkos::numbers::pi_v<T>;

    T r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu);

    return (1 / (4.0 * pi)) * (std::expint(-r2 / (2.0 * sigma * sigma)) - Kokkos::log(r2));
}

template <typename T>
KOKKOS_INLINE_FUNCTION ippl::Vector<T, Dim> exact_E(T x, T y, T sigma = 0.05, T mu = 0.5) {
    T pi     = Kokkos::numbers::pi_v<T>;
    T r2     = (x - mu) * (x - mu) + (y - mu) * (y - mu);
    T factor = (1.0 / (2.0 * pi)) * (1 - Kokkos::exp(-r2 / (2.0 * sigma * sigma))) / r2;

    ippl::Vector<T, Dim> Efield = {(x - mu), (y - mu)};
    return factor * Efield;
}

template <typename T>
void compute_convergence(std::string algorithm, int pt) {
    Inform errorMsg("");
    Inform errorMsg2all("", INFORM_ALL_NODES);

    ippl::Index I(pt);
    ippl::NDIndex<Dim> owned(I, I);

    // specifies decomposition; here all dimensions are parallel
    std::array<bool, 2> isParallel;
    isParallel.fill(true);

    // unit box
    T dx                      = 1.0 / pt;
    ippl::Vector<T, 2> hx     = {dx, dx};
    ippl::Vector<T, 2> origin = {0.0, 0.0};
    Mesh_t<T> mesh(owned, hx, origin);

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<2> layout(MPI_COMM_WORLD, owned, isParallel);

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

    // assign the exact field with its values (Ei function)
    auto view_exact = exact.getView();

    Kokkos::parallel_for(
        "Assign exact field", exact.getFieldRangePolicy(),
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

            view_exactE(i, j) = exact_E(x, y);
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
        throw IpplException("Test2DPoisson_convergence.cpp main()",
                            "Only HOCKNEY supported for 2D test!");
    }

    // add output type
    params.add("output_type", Solver_t<T>::SOL_AND_GRAD);

    // define an FFTOpenPoissonSolver object
    Solver_t<T> FFTsolver(fieldE, rho, params);

    // solve the Poisson equation -> rho contains the solution (phi) now
    FFTsolver.solve();

    // compute relative error norm for potential
    rho   = rho - exact;
    T err = norm(rho) / norm(exact);

    // compute relative error norm for the E-field components
    ippl::Vector<T, Dim> errE{0.0, 0.0};
    fieldE           = fieldE - exactE;
    auto view_fieldE = fieldE.getView();

    for (size_t d = 0; d < Dim; ++d) {
        T temp = 0.0;
        Kokkos::parallel_reduce(
            "Vector errorNr reduce", fieldE.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const size_t i, const size_t j, T& valL) {
                T myVal = Kokkos::pow(view_fieldE(i, j)[d], 2);
                valL += myVal;
            },
            Kokkos::Sum<T>(temp));

        T globaltemp = 0.0;

        ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<T>());
        T errorNr = std::sqrt(globaltemp);

        temp = 0.0;
        Kokkos::parallel_reduce(
            "Vector errorDr reduce", exactE.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const size_t i, const size_t j, T& valL) {
                T myVal = Kokkos::pow(view_exactE(i, j)[d], 2);
                valL += myVal;
            },
            Kokkos::Sum<T>(temp));

        globaltemp = 0.0;
        ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<T>());
        T errorDr = std::sqrt(globaltemp);

        errE[d] = errorNr / errorDr;
    }

    errorMsg << std::setprecision(16) << dx << " " << err << " " << errE[0] << " "
             << errE[1] << endl;

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
            throw IpplException("Test2DPoisson_convergence",
                                "Precision argument must be DOUBLE or SINGLE.");
        }

        // start a timer to time the FFT Poisson solver
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        // gridsizes to iterate over
        std::array<int, 8> N = {4, 8, 16, 32, 64, 128, 256, 512};

        msg << "Spacing Error ErrorEx ErrorEy" << endl;

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
