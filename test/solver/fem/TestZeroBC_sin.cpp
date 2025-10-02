// Tests the FEM Poisson solver by solving the problem:
//
// -Laplacian(u) = pi^2 * sin(pi * x), x in [-1,1]
// u(-1) = u(1) = 0
//
// Exact solution is u(x) = sin(pi * x)
//
// BCs: Homogeneous Dirichlet BCs (Zero).
//
// The test prints out the relative error as we refine
// the mesh spacing i.e. it is a convergence study. 
// The order of convergence should be 2. 
//
// The test is available in 1D (problem above),
// as well as 2D and 3D with analogous test cases.
//
// Usage:
//    ./TestZeroBC_sin <dim> --info 5

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "PoissonSolvers/FEMPoissonSolver.h"

// only works for 3D...
template <typename VField, unsigned Dim>
void dumpVTK(VField& E, ippl::Vector<unsigned, Dim> nr, int iteration, ippl::Vector<double, Dim> hr,
             ippl::Vector<double, Dim> origin) {

    typename VField::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "Grad-Exact" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;

    ippl::Vector<unsigned, 3> nr_vtk = {1, 1, 1};
    ippl::Vector<double, 3> hr_vtk = {1.0, 1.0, 1.0};
    ippl::Vector<double, 3> origin_vtk = {1.0, 1.0, 1.0};
    nr_vtk[0] = nr[0] + 3;
    hr_vtk[0] = hr[0];
    origin_vtk[0] = origin[0];
    if constexpr(Dim > 1) {
        nr_vtk[1] = nr[1] + 3;
        hr_vtk[1] = hr[1];
        origin_vtk[1] = origin[1];
    }
    if constexpr(Dim > 2) {
        nr_vtk[2] = nr[2] + 3;
        hr_vtk[2] = hr[2];
        origin_vtk[2] = origin[2];
    }

    vtkout << "DIMENSIONS " << nr_vtk[0] << " " << nr_vtk[1] << " " << nr_vtk[2] << endl;
    vtkout << "ORIGIN " << origin_vtk[0] << " " << origin_vtk[1] << " " << origin_vtk[2]  << endl;;
    vtkout << "SPACING " << hr_vtk[0] << " " << hr_vtk[1] << " " << hr_vtk[2] << endl;
    int mult = 1;
    for (unsigned d = 0; d < Dim; ++d) {
        mult *= (nr[d] + 2);
    }
    vtkout << "CELL_DATA " << mult << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for (unsigned z = 0; z < nr[2] + 2; z++) {
        for (unsigned y = 0; y < nr[1] + 2; y++) {
            for (unsigned x = 0; x < nr[0] + 2; x++) {
                if constexpr(Dim == 1) {
                    vtkout << host_view(x)[0] << endl;
                } else if constexpr(Dim == 2) {
                    vtkout << host_view(x, y)[0] << "\t" << host_view(x, y)[1] << "\t" << endl;
                } else {
                    vtkout << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                           << host_view(x, y, z)[2] << endl;
                }
            }
        }
    }
}

template <typename T, unsigned Dim>
KOKKOS_INLINE_FUNCTION T sinusoidalRHSFunction(ippl::Vector<T, Dim> x_vec) {
    const T pi = Kokkos::numbers::pi_v<T>;

    T val = 1.0;
    for (unsigned d = 0; d < Dim; d++) {
        val *= Kokkos::sin(pi * x_vec[d]);
    }

    return Dim * pi * pi * val;
}

template <typename T, unsigned Dim>
struct AnalyticSol {
    const T pi = Kokkos::numbers::pi_v<T>;

    KOKKOS_FUNCTION const T operator()(ippl::Vector<T, Dim> x_vec) const {
        T val = 1.0;
        for (unsigned d = 0; d < Dim; d++) {
            val *= Kokkos::sin(pi * x_vec[d]);
        }
        return val;
    }
};

template <typename T, unsigned Dim>
struct EfieldSol {
    const T pi = Kokkos::numbers::pi_v<T>;

    KOKKOS_FUNCTION const ippl::Vector<T, 1> operator()(ippl::Vector<T, 1> x_vec) const {
        ippl::Vector<T, 1> val;
        val[0] = -pi * Kokkos::cos(pi * x_vec[0]);
        return val;
    }

    KOKKOS_FUNCTION const ippl::Vector<T, 2> operator()(ippl::Vector<T, 2> x_vec) const {
        ippl::Vector<T, 2> val;
        val[0] = -pi * Kokkos::cos(pi * x_vec[0]) * Kokkos::sin(pi * x_vec[1]);
        val[1] = -pi * Kokkos::cos(pi * x_vec[1]) * Kokkos::sin(pi * x_vec[0]);
        return val;
    }

    KOKKOS_FUNCTION const ippl::Vector<T, 3> operator()(ippl::Vector<T, 3> x_vec) const {
        ippl::Vector<T, 3> val;
        val[0] = -pi * Kokkos::cos(pi * x_vec[0]) * Kokkos::sin(pi * x_vec[1]) * 
                 Kokkos::sin(pi * x_vec[2]);
        val[1] = -pi * Kokkos::cos(pi * x_vec[1]) * Kokkos::sin(pi * x_vec[0]) *
                 Kokkos::sin(pi * x_vec[2]);
        val[2] = -pi * Kokkos::cos(pi * x_vec[2]) * Kokkos::sin(pi * x_vec[0]) *
                 Kokkos::sin(pi * x_vec[1]);
        return val;
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
    VField_t grad(mesh, layout, numGhosts); // vector field for grad

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    // set rhs
    auto view_rhs = rhs.getView();
    auto ldom     = layout.getLocalNDIndex();

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "Assign RHS", rhs.getFieldRangePolicy(), KOKKOS_LAMBDA(const index_array_type& args) {
            ippl::Vector<int, Dim> iVec = args - numGhosts;
            for (unsigned d = 0; d < Dim; ++d) {
                iVec[d] += ldom[d].first();
            }

            const ippl::Vector<T, Dim> x = (iVec)*cellSpacing + origin;

            apply(view_rhs, args) = sinusoidalRHSFunction<T, Dim>(x);
        });

    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::FEMPoissonSolver<Field_t, Field_t> solver(lhs, rhs);
    solver.setGradient(grad);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 2000);
    params.add("output_type", ippl::FEMPoissonSolver<Field_t, Field_t, 1, 2>::SOL_AND_GRAD);
    solver.mergeParameters(params);

    // solve the problem
    solver.solve();

    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);

    // Compute the error
    AnalyticSol<T, Dim> analytic;
    const T relError = solver.getL2Error(analytic);

    // Compute the error of the Efield
    EfieldSol<T, Dim> analyticE;
    const T relErrorE = solver.getL2ErrorGrad(analyticE);

    m << std::setw(10) << numNodesPerDim;
    m << std::setw(25) << std::setprecision(16) << cellSpacing[0];
    m << std::setw(25) << std::setprecision(16) << relError;
    m << std::setw(25) << std::setprecision(16) << relErrorE;
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
        msg << std::setw(25) << "ErrorE";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        if (dim == 1) {
            // 1D Sinusoidal
            for (unsigned n = 1 << 2; n <= 1 << 10; n = n << 1) {
                testFEMSolver<T, 1>(n, -1.0, 1.0);
            }
        } else if (dim == 2) {
            // 2D Sinusoidal
            for (unsigned n = 1 << 2; n <= 1 << 8; n = n << 1) {
                testFEMSolver<T, 2>(n, -1.0, 1.0);
            }
        } else {
            // 3D Sinusoidal
            for (unsigned n = 1 << 2; n <= 1 << 7; n = n << 1) {
                testFEMSolver<T, 3>(n, -1.0, 1.0);
            }
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
