// Tests the FEM Poisson solver by solving the 2d problem:
//
// curl(curl(E)) + E = {(1+k^2)sin(k*y), (1+k^2)sin(k*x)} in [-1,1]^2
// n x E = 0 on boundary 
//
// Exact solution is E = {sin(k*y),sin(k*x)}
//
// BCs: Zero dirichlet bc.
//

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "MaxwellSolvers/FEMMaxwellDiffusionSolver.h"

#include <fstream>


template <typename T>
void saveToFile(size_t nx, size_t ny, ippl::Vector<T,2> cellSpacing, ippl::Vector<T,2> origin,
    ippl::FEMVector<ippl::Vector<T,2>> data, const std::string& filename) {
    
    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    
    std::ofstream file(filename);
    file << "x,y,z,vx,vy,vz\n";
    for (size_t i = 0; i < hView.extent(0); ++i) {
        size_t yOffset = i / (2*nx - 1);
        size_t xOffset = i - (2*nx -1)*yOffset;
        T x = xOffset*cellSpacing[0];
        T y = yOffset*cellSpacing[1];
        if (xOffset < (nx-1)) {
            // we are parallel to the x axis
            x += cellSpacing[0]/2.;
        } else {
            // we are parallel to the y axis
            y += cellSpacing[1]/2.;
            x -= (nx-1)*cellSpacing[0];
        }
        x += origin[0];
        y += origin[1];

        file << x << "," << y << ",0," << hView(i)[0] << "," << hView(i)[1] << ",0\n";
    }

    file.close();
}

template <typename T>
void saveToFile(size_t nx, size_t ny, ippl::Vector<T,2> cellSpacing, ippl::Vector<T,2> origin,
    ippl::FEMVector<T> data, const std::string& filename) {
    
    auto view = data.getView();
    auto hView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hView, view);
    
    std::ofstream file(filename);
    file << "x,y,z,val\n";
    for (size_t i = 0; i < hView.extent(0); ++i) {
        size_t yOffset = i / (2*nx - 1);
        size_t xOffset = i - (2*nx -1)*yOffset;
        T x = xOffset*cellSpacing[0];
        T y = yOffset*cellSpacing[1];
        if (xOffset < (nx-1)) {
            // we are parallel to the x axis
            x += cellSpacing[0]/2.;
        } else {
            // we are parallel to the y axis
            y += cellSpacing[1]/2.;
            x -= (nx-1)*cellSpacing[0];
        }
        x += origin[0];
        y += origin[1];

        file << x << "," << y << ",0," << hView(i) << "\n";
    }

    file.close();
}


template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    Inform m("");
    Inform msg2all("", INFORM_ALL_NODES);

    using Mesh_t   = ippl::UniformCartesian<T, Dim>;
    using Field_t  = ippl::Field<ippl::Vector<T,Dim>, Dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, Dim>;
    using point_t =  ippl::Vector<T, Dim>;

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

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    // Generate the rhs FEMVector
    size_t nx = numNodesPerDim;
    size_t ny = numNodesPerDim;
    ippl::FEMVector<ippl::Vector<T,Dim> > rhsVector(nx*(ny-1) + ny*(nx-1));
    auto viewRhs = rhsVector.getView();

    ippl::FEMVector<ippl::Vector<T,Dim> > solutionVector(nx*(ny-1) + ny*(nx-1));
    auto viewSolution = solutionVector.getView();

    T k = 3.14159265359;

    auto rhsFunc = [k](const point_t& pos) -> point_t {
        point_t sol(0);
        sol[0] = (1. + k*k)*Kokkos::sin(k*pos[1]);
        sol[1] = (1. + k*k)*Kokkos::sin(k*pos[0]);
        return sol;
    };

    auto analytical = [k](const point_t& pos) -> point_t {
        point_t sol(0);
        sol[0] = Kokkos::sin(k*pos[1]);
        sol[1] = Kokkos::sin(k*pos[0]);
        return sol;
    };

    Kokkos::parallel_for("Assign RHS", rhsVector.size(),
        KOKKOS_LAMBDA(size_t i) {

            size_t yOffset = i / (2*nx - 1);
            size_t xOffset = i - (2*nx -1)*yOffset;
            T x = xOffset*cellSpacing[0];
            T y = yOffset*cellSpacing[1];
            if (xOffset < (nx-1)) {
                // we are parallel to the x axis
                x += cellSpacing[0]/2.;
            } else {
                // we are parallel to the y axis
                y += cellSpacing[1]/2.;
                x -= (nx-1)*cellSpacing[0];
            }
            x += origin[0];
            y += origin[1];
            
            viewRhs(i)[0] = (1. + k*k)*Kokkos::sin(k*y);
            viewRhs(i)[1] = (1. + k*k)*Kokkos::sin(k*x);

            viewSolution(i)[0] = Kokkos::sin(k*y);
            viewSolution(i)[1] = Kokkos::sin(k*x);
        }
    );

    saveToFile(nx, ny, cellSpacing, origin, rhsVector, "rhs.csv");
    saveToFile(nx, ny, cellSpacing, origin, solutionVector, "solution.csv");

    IpplTimings::stopTimer(initTimer);


    // initialize the solver
    ippl::FEMMaxwellDiffusionSolver<Field_t> solver(lhs, rhs, rhsVector, rhsFunc);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 2000);
    solver.mergeParameters(params);

    // solve the problem
    ippl::FEMVector<ippl::Vector<T,Dim> > result = solver.solve();
    ippl::FEMVector<ippl::Vector<T,Dim> > diff = result.template skeletonCopy<ippl::Vector<T,Dim>>();
    diff  = result - solutionVector;
    saveToFile(nx, ny, cellSpacing, origin, result, "result.csv");
    saveToFile(nx, ny, cellSpacing, origin, diff, "diff.csv");
    saveToFile(nx, ny, cellSpacing, origin, *(solver.rhsVector_m), "solver_rhs.csv");
    saveToFile(nx, ny, cellSpacing, origin, *(solver.lhsVector_m), "solver_lhs.csv");

    ippl::FEMVector<ippl::Vector<T,Dim> > dummy = result.template skeletonCopy<ippl::Vector<T,Dim>>(); 

    auto resultView = result.getView();
    auto solutionView = solutionVector.getView();
    auto hResultView = Kokkos::create_mirror_view(resultView);
    auto hSolutionView = Kokkos::create_mirror_view(solutionView);
    Kokkos::deep_copy(hResultView, resultView);
    Kokkos::deep_copy(hSolutionView, solutionView);

    T s = 0;
    for (size_t i = 0; i < hSolutionView.extent(0); ++i) {
        auto a = hResultView(i) - hSolutionView(i);
        s += dot(a,a).apply();
    }

    s = Kokkos::sqrt(s)/(Dim*nx);
    
    T error = solver.getL2Error(result, analytical);
    T coefError = solver.getL2ErrorCoeff(*(solver.lhsVector_m), analytical);
    
    if (ippl::Comm->rank() == 0) {
        std::cout << std::setw(10) << "num nodes" << std::setw(25) << "cell spacing"
                  << std::setw(25) << "value error" << std::setw(25) << "interp error"
                  << std::setw(25) << "interp error coef"
                  << std::setw(25) << "solver residue"
                  << std::setw(15) << "num it\n";
        std::cout << std::setw(10) << numNodesPerDim;
        std::cout << std::setw(25) << std::setprecision(16) << cellSpacing[0];
        std::cout << std::setw(25) << std::setprecision(16) << s;
        std::cout << std::setw(25) << std::setprecision(16) << error;
        std::cout << std::setw(25) << std::setprecision(16) << coefError;
        std::cout << std::setw(25) << std::setprecision(16) << solver.getResidue();
        std::cout << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
        std::cout << "\n";
    }

    /*
    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);

    // Compute the error
    AnalyticSol<T, Dim> analytic;
    const T relError = solver.getL2Error(analytic);

    if (ippl::Comm->rank() == 0) {
        std::cout << std::setw(10) << numNodesPerDim;
        std::cout << std::setw(25) << std::setprecision(16) << cellSpacing[0];
        std::cout << std::setw(25) << std::setprecision(16) << relError;
        std::cout << std::setw(25) << std::setprecision(16) << solver.getResidue();
        std::cout << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
        std::cout << "\n";
    }
    IpplTimings::stopTimer(errorTimer);
    */
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");

        using T = double;

        unsigned dim = 3;

        if (argc > 1 && std::atoi(argv[1]) == 1) {
            dim = 1;
        } else if (argc > 1 && std::atoi(argv[1]) == 2) {
            dim = 2;
        }

        // start the timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        msg << std::setw(10) << "Size";
        msg << std::setw(25) << "Spacing";
        msg << std::setw(25) << "Relative Error";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        /*
        if (dim == 1) {
            // 1D Sinusoidal
            for (unsigned n = 1 << 5; n <= 1 << 12; n = n << 1) {
                testFEMSolver<T, 1>(n, 1.0, 3.0);
            }
        } else if (dim == 2) {
            // 2D Sinusoidal
            for (unsigned n = 1 << 5; n <= 1 << 12; n = n << 1) {
                testFEMSolver<T, 2>(n, 1.0, 3.0);
            }
        } else {
            // 3D Sinusoidal
            for (unsigned n = 1 << 5; n <= 1 << 12; n = n << 1) {
                testFEMSolver<T, 3>(n, 1.0, 3.0);
            }
        }
        */
        
        
        for (unsigned n = 1 << 5; n <= 1 << 8; n = n << 1) {
            testFEMSolver<T, 2>(n, 1.0, 3.0);
        }
            
         
        
        
        //testFEMSolver<T, 2>(100, 0.0, 3.0);

        // stop the timer
        IpplTimings::stopTimer(allTimer);

        // print the timers
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
