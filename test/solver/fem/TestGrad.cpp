// Tests the FEM Poisson solver WITH PRECONDITIONING
// by solving the problem:
//
// -Laplacian(u) = 1.0,
// where x in [0,1] and u(0) = u(1) = 1.0. 
//
// The exact solution is 
// u(x) = 1 + x/2 - (x^2)/2.
//
// BCs: Dirichlet BCs (Constant Face = 1).
// This is only 1D!
//
// The test prints out the relative error as we refine
// the mesh spacing i.e. it is a convergence study. 
// The order of convergence should be 2. 
//
// Usage:
//     ./TestNonHomDirichlet_1d_preconditioned --info 5

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "PoissonSolvers/FEMPoissonSolver.h"

template <typename T>
struct AnalyticSol {
    KOKKOS_FUNCTION const T operator()(ippl::Vector<T, 1> x_vec) const {
        return (1.0 + 0.5*x_vec[0] - 0.5*(x_vec[0]*x_vec[0]));
    }
};

template <typename T>
struct EfieldSol {
    KOKKOS_FUNCTION const ippl::Vector<T,1> operator()(ippl::Vector<T, 1> x_vec) const {
        return {-(0.5 - x_vec[0])};
    }
};

template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    Inform m("");
    Inform msg2all("", INFORM_ALL_NODES);

    using Mesh_t   = ippl::UniformCartesian<T, Dim>;
    using Field_t  = ippl::Field<T, Dim, Mesh_t, Cell>;
    using VField_t = ippl::Field<ippl::Vector<T, Dim>, Dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, Dim>;

    const unsigned numCellsPerDim = numNodesPerDim - 1;
    const unsigned numGhosts      = 1;

    // Domain: [-1, 1]
    const ippl::Vector<unsigned, Dim> nodesPerDimVec(numNodesPerDim);
    ippl::NDIndex<Dim> domain(nodesPerDimVec);
    ippl::Vector<T, Dim> cellSpacing((domain_end - domain_start) / static_cast<T>(numCellsPerDim));
    ippl::Vector<T, Dim> origin(domain_start);
    Mesh_t mesh(domain, cellSpacing, origin);

    // specifies decomposition; here all dimensions are parallel
    std::array<bool, Dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, isParallel);
    Field_t lhs(mesh, layout, numGhosts);  // left hand side (updated in the algorithm)
    Field_t rhs(mesh, layout, numGhosts);  // right hand side (set once)
    VField_t grad(mesh, layout, numGhosts);

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ConstantFace<Field_t>>(i, 1.0);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);
    bcField.apply(lhs);
    bcField.apply(rhs);

    // set rhs
    rhs = 1.0;

    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::FEMPoissonSolver<Field_t, Field_t, 1, 2> solver(lhs, rhs);
    solver.setGradient(grad);

    // turn on computation of grad
    ippl::ParameterList params;
    params.add("output_type", ippl::FEMPoissonSolver<Field_t, Field_t, 1, 2>::SOL_AND_GRAD);

    solver.mergeParameters(params);

    // solve the problem
    solver.solve();

    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);

    // Compute the error
    AnalyticSol<T> analytic;
    const T relError = solver.getL2Error(analytic);

    // Compute the error of the Efield
    EfieldSol<T> analyticE;
    const T relErrorE = solver.getL2ErrorGrad(analyticE);

    // norm error Efield

    // assign the exact E field
    VField_t exactE(mesh, layout, numGhosts);
    EfieldSol<T> efield;
    auto view_exactE = exactE.getView();
    auto ldom        = layout.getLocalNDIndex();

    Kokkos::RangePolicy<> range(numGhosts, view_exactE.extent(0) - numGhosts); // - 1);
    Kokkos::parallel_for(
        "Assign exact E-field", range,
        KOKKOS_LAMBDA(const int i) {
            const int ig = i + ldom[0].first() - numGhosts;
            //const T x_mid = (ig + 0.5)*cellSpacing[0] + origin[0];
            const T x_mid = (ig)*cellSpacing[0] + origin[0];

            view_exactE(i) = efield(x_mid);
        });
    Kokkos::fence();

    /*
    // print outs (debugging)
    std::cout << "spacing = " << cellSpacing << std::endl;

    std::cout << "solution phi = " << std::endl;
    lhs.write();

    std::cout << "gradient E = " << std::endl;
    grad.write();

    std::cout << "analytic E = " << std::endl;
    exactE.write();
    */

    // compute relative error
    grad = grad - exactE;
    auto view_grad = grad.getView();

    T temp = 0.0;
    Kokkos::parallel_reduce(
        "Vector errorNr reduce", grad.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const size_t i, T& valL) {
            T myVal = Kokkos::pow(view_grad(i)[0], 2);
            valL += myVal;
        },
        Kokkos::Sum<T>(temp));

    T globaltemp = 0.0;

    ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<T>());
    T errorNr = std::sqrt(globaltemp);

    temp = 0.0;
    Kokkos::parallel_reduce(
        "Vector errorDr reduce", exactE.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const size_t i, T& valL) {
            T myVal = Kokkos::pow(view_exactE(i)[0], 2);
            valL += myVal;
        },
        Kokkos::Sum<T>(temp));

    globaltemp = 0.0;
    ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<T>());
    T errorDr = std::sqrt(globaltemp);

    T normErrorE = errorNr / errorDr;

    m << std::setw(10) << numNodesPerDim;
    m << std::setw(25) << std::setprecision(16) << cellSpacing[0];
    m << std::setw(25) << std::setprecision(16) << relError;
    m << std::setw(25) << std::setprecision(16) << relErrorE;
    m << std::setw(25) << std::setprecision(16) << normErrorE;
    m << std::setw(25) << std::setprecision(16) << solver.getResidue();
    m << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
    m << endl;

    IpplTimings::stopTimer(errorTimer);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");

        using T = double;

        // start the timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        msg << std::setw(10) << "Size";
        msg << std::setw(25) << "Spacing";
        msg << std::setw(25) << "Relative Error";
        msg << std::setw(25) << "Error E";
        msg << std::setw(25) << "Norm Error E";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        for (unsigned n = 1 << 2; n <= 1 << 10; n = n << 1) {
            testFEMSolver<T, 1>(n, 0.0, 1.0);
        }

        // stop the timer
        IpplTimings::stopTimer(allTimer);

        // print the timers
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
