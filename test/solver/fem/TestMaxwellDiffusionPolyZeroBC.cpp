// Tests the FEM Maxwell Diffusion solver by solving the problem:
//
// curl(curl(E)) + E = f in [-1,1]^dim
// n x E = 0 on boundary 
// with dim = 2 or 3,
// and f = {2 - (y^2 - 1), 2 - (x^2 - 1)} for 2d
//       = {-2*(z^2 - 1) - 2*(y^2 - 1) + (y^2 - 1)*(z^2 - 1),
//          -2*(x^2 - 1) - 2*(z^2 - 1) + (x^2 - 1)*(z^2 - 1),
//          -2*(y^2 - 1) - 2*(x^2 - 1) + (x^2 - 1)*(y^2 - 1)} for 3d
//
// Exact solution is E = {-(y^2 - 1), -(x^2 - 1)} for 2d
//                     = {(y^2 - 1)*(z^2 - 1),
//                        (x^2 - 1)*(z^2 - 1),
//                        (x^2 - 1)*(y^2 - 1)} for 3d
//
// BCs: Zero dirichlet bc.
//

#include "Ippl.h"
#include <Kokkos_Random.hpp>
#include <chrono>

#include "Meshes/Centering.h"
#include "MaxwellSolvers/FEMMaxwellDiffusionSolver.h"



template <typename T, unsigned Dim>
struct Analytical{
    using point_t =  ippl::Vector<T, Dim>;

    Analytical() {}

    KOKKOS_FUNCTION const point_t operator() (const point_t& pos) const{
        point_t sol(0);
        if constexpr (Dim == 2) {
            T x = pos[0];
            T y = pos[1];
            sol[0] = -(y*y - 1);
            sol[1] = -(x*x - 1);
        } else {
            T x = pos[0];
            T y = pos[1];
            T z = pos[2];
            sol[0] = (y*y - 1)*(z*z - 1);
            sol[1] = (x*x - 1)*(z*z - 1);
            sol[2] = (x*x - 1)*(y*y - 1);
        }
        return sol;
    }
};


template <typename T, unsigned Dim>
struct Rhs {
    using point_t = ippl::Vector<T, Dim>;

    Rhs() {}

    KOKKOS_FUNCTION const point_t operator() (const point_t& pos) const {
        point_t sol(0);
        if constexpr (Dim == 2) {
            T x = pos[0];
            T y = pos[1];
            sol[0] = 2 - (y*y - 1);
            sol[1] = 2 - (x*x - 1);
        }
        if constexpr (Dim == 3) {
            T x = pos[0];
            T y = pos[1];
            T z = pos[2];
            sol[0] = -2*(z*z - 1) - 2*(y*y - 1) + (y*y - 1)*(z*z - 1);
            sol[1] = -2*(x*x - 1) - 2*(z*z - 1) + (x*x - 1)*(z*z - 1);
            sol[2] = -2*(y*y - 1) - 2*(x*x - 1) + (x*x - 1)*(y*y - 1);
        }
        return sol;
    }        
};


template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    using Mesh_t   = ippl::UniformCartesian<T, Dim>;
    using Field_t  = ippl::Field<ippl::Vector<T,Dim>, Dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, Dim>;
    using point_t =  ippl::Vector<T, Dim>;
    using indices_t =  ippl::Vector<size_t, Dim>;

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


    // Get our rhs and solution functors
    Rhs<T, Dim> rhsFunc;
    Analytical<T, Dim> analytical;

    // create the FEMVector for the rhs
    size_t n = 1;
    auto ldom = layout.getLocalNDIndex();
    indices_t extents = (ldom.last() - ldom.first()) + 3;
        
    if constexpr (Dim == 2) {
        size_t nx = extents(0);
        size_t ny = extents(1);
        n = nx*(ny-1) + ny*(nx-1);
    } else {
        size_t nx = extents(0);
        size_t ny = extents(1);
        size_t nz = extents(2);
        n = (nz-1)*(nx*(ny-1) + ny*(nx-1) + nx*ny) + nx*(ny-1) + ny*(nx-1);
    }
    ippl::FEMVector<point_t> rhsVector(n);
    auto viewRhs = rhsVector.getView();

    if constexpr (Dim == 2) {
        Kokkos::parallel_for("Assign RHS", rhsVector.size(),
            KOKKOS_LAMBDA(size_t i) {
                size_t nx = extents(0);

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
                // have to subtract one cellspacing for the ghost cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];
                
                viewRhs(i) = rhsFunc(point_t(x,y));
            }
        );

    } else {
        Kokkos::parallel_for("Assign RHS", rhsVector.size(),
            KOKKOS_LAMBDA(size_t i) {
                size_t nx = extents(0);
                size_t ny = extents(1);
                
                size_t zOffset = i / (nx*(ny-1) + ny*(nx-1) + nx*ny);
                T z = zOffset*cellSpacing[2];
                T x = 0;
                T y = 0;

                if (i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset >= (nx*(ny-1) + ny*(nx-1))) {
                    // we are parallel to z axis
                    size_t f = i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset
                        - (nx*(ny-1) + ny*(nx-1));
                    z += cellSpacing[2]/2;
                    
                    size_t yOffset = f / nx;
                    y = yOffset*cellSpacing[1];

                    size_t xOffset = f % nx;
                    x = xOffset*cellSpacing[0];
                } else {
                    // are parallel to one of the other axes
                    size_t f = i - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset;
                    size_t yOffset = f / (2*nx - 1);
                    size_t xOffset = f - (2*nx - 1)*yOffset;

                    x = xOffset*cellSpacing[0];
                    y = yOffset*cellSpacing[1];

                    if (xOffset < (nx-1)) {
                        // we are parallel to the x axis
                        x += cellSpacing[0]/2.;
                    } else {
                        // we are parallel to the y axis
                        y += cellSpacing[1]/2.;
                        x -= (nx-1)*cellSpacing[0];
                    }
                }

                // have to subtract one cellspacing for the ghost cells
                x += origin[0] + ldom.first()[0]*cellSpacing[0] - cellSpacing[0];
                y += origin[1] + ldom.first()[1]*cellSpacing[1] - cellSpacing[1];
                z += origin[2] + ldom.first()[2]*cellSpacing[2] - cellSpacing[2];
                
                viewRhs(i) = rhsFunc(point_t(x,y,z));

            }
        );
    }


    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::FEMMaxwellDiffusionSolver<Field_t> solver(lhs, rhs, rhsVector);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 10e-13);
    params.add("max_iterations", 10000);
    solver.mergeParameters(params);
    
    // solve the problem
    solver.solve();

    // retrieve values at random positions
    // we will take 100 points out of which 97 will be random and the last 3
    // will be manually chosen. The last three are manually chosen to lie on the
    // origin of the domain, the end of the domain and on an edge, to also check
    // these situations.
    Kokkos::Random_XorShift64_Pool<> randomPool(42);
    Kokkos::View<point_t*> positions("positions", 100);

    Kokkos::parallel_for("assign positions", 97,
        KOKKOS_LAMBDA(size_t i) {
  	    (void) positions;            // add (void) to make clang happy i.e. no unused variable warning
            (void) ldom;
            (void) domain_start;
            auto gen = randomPool.get_state();
            T cellWidth = (domain_end - domain_start) / numCellsPerDim;
            if constexpr (Dim == 2) {
                positions(i) =
                    point_t(T(gen.drand(ldom.first()<:0:>*cellWidth+domain_start,
                            ldom.last()<:0:>*cellWidth+domain_start)),
                        T(gen.drand(ldom.first()<:1:>*cellWidth+domain_start,
                            ldom.last()<:1:>*cellWidth+domain_start)));
            } else {
                positions(i) =
                    point_t(T(gen.drand(ldom.first()<:0:>*cellWidth+domain_start,
                            ldom.last()<:0:>*cellWidth+domain_start)),
                        T(gen.drand(ldom.first()<:1:>*cellWidth+domain_start,
                            ldom.last()<:1:>*cellWidth+domain_start)),
                        T(gen.drand(ldom.first()<:2:>*cellWidth+domain_start,
                            ldom.last()<:2:>*cellWidth+domain_start)));
            }


            if (i == 96) {
                positions(97) = ldom.first()*cellWidth+domain_start;
                positions(98) = ldom.last()*cellWidth+domain_start;
                // This point belongs to an edge
                if constexpr (Dim == 2) {
                positions(99) =
                    point_t(ldom.first()<:0:>*cellWidth+domain_start + cellWidth,
                        ldom.first()<:1:>*cellWidth+domain_start + 0.5*cellWidth);
                } else {
                    positions(99) =
                        point_t(ldom.first()<:0:>*cellWidth+domain_start + cellWidth,
                            ldom.first()<:1:>*cellWidth+domain_start + 0.5*cellWidth,
                            ldom.first()<:2:>*cellWidth+domain_start + 0.5*cellWidth);
                }
            }
            randomPool.free_state(gen);
        }
    );

    
    Kokkos::View funcVals = solver.reconstructToPoints(positions);

    // calculate the error
    T linfError = 0;
    Kokkos::parallel_reduce( "calc l infinity", 100,
        KOKKOS_LAMBDA (size_t i, T& lmax) {
            point_t dif = funcVals(i) - analytical(positions(i));
            T val = Kokkos::sqrt(dif.dot(dif));
            if( val > lmax ) lmax = val; 
        },
        Kokkos::Max<T>(linfError)
    );

    
    // print information and error
    T coefError = solver.getL2Error(analytical);
    Inform m(0, 0);
    m << std::setw(10) << numNodesPerDim;
    m << std::setw(25) << std::setprecision(16) << cellSpacing[0];
    m << std::setw(25) << std::setprecision(16) << coefError;
    m << std::setw(25) << std::setprecision(16) << linfError;
    m << std::setw(25) << std::setprecision(16) << solver.getResidue();
    m << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
    m << endl;

}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(0, 0);

        using T = double;

        unsigned dim = 3;

        if (argc > 1 && std::atoi(argv[1]) == 2) {
            dim = 2;
        }

        // start the timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        msg << std::setw(10) << "num_nodes"
            << std::setw(25) << "cell_spacing"
            << std::setw(25) << "interp_error_coef"
            << std::setw(25) << "recon_error"
            << std::setw(25) << "solver_residue"
            << std::setw(15) << "num_it"
            << endl;
        
        
        if (argc > 2 && std::atoi(argv[2]) != 0) {
            size_t n = std::atoi(argv<:2:>);
            if (dim == 2) {
                // 2D
                testFEMSolver<T, 2>(n, -1.0, 1.0);
            } else {
                // 3D
                testFEMSolver<T, 3>(n, -1.0, 1.0);
            }
        } else {
            if (dim == 2) {
                // 2D Sinusoidal
                for (unsigned n = 16; n <= 1024; n = std::sqrt(2)*n) {
                    testFEMSolver<T, 2>(n, -1.0, 1.0);
                }
            } else {
                // 3D Sinusoidal
                for (unsigned n = 16; n <= 1024; n = std::sqrt(2)*n) {
                    testFEMSolver<T, 3>(n, -1.0, 1.0);
                }
            }
        }

        // stop the timer
        IpplTimings::stopTimer(allTimer);

        // write the timers
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
