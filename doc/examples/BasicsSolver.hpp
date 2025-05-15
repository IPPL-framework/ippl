/**
@page basic_solver Basics: Solver {#Solver} 

@section solvers Solvers

Different types of Poisson solvers and a biharmonic solver are available in IPPL.

Poisson solvers are numerical method used to solve Poisson's equation, a PDE that appears in many
areas of physics. The equation is of the form:

\f[
    - \nabla^2 \phi = f
\f]

where \f$\nabla^2 \f$ is the Laplace operator, \f$\phi\f$ is the unknown function, and \f$ f \f$ is
a given function.

IPPL provides different variants:

- ippl::FFTPeriodicPoissonSolver: Solves Poisson equation with periodic bcs spectrally
- ippl::FFTOpenPoissonSolver: Solves Poisson equation with Open BCs using FFTs (subtypes: Hockney
(2nd order solver), Vico (spectral solver))
- ippl::P3MSolver: Solves Poisson equation with periodic bcs, based on FFT
- ippl::PoissonCG: Solves Poison equation with Conjugate Gradient method

All these inherit from the Poisson class.

The FFT based solvers are not available for 1D as heFFTe doesn't support 1D FFTs.

For Poisson equation solved with these different types of solvers see:

- test/solver/TestFFTPeriodicPoissonSolver.cpp (for FFTPeriodicPoissonSolver)
- test/solver/TestGaussian_convergence.cpp (for FFTOpenPoissonSolver)
- test/solver/TestP3MSolver.cpp (for P3MSolver)
- test/solver/TestCGSolver.cpp (for CG Poisson solver)

@subsection example_poisson_solver Example: Poisson solver

This section shows how to use the solvers. This example uses the FFTOpenPoissonSolver. The concepts
used here are the same for the other solvers.
___

To start we define the mesh and the field types:
@code
using Mesh_t      = ippl::UniformCartesian<double, 3>;
using Centering_t = Mesh_t::DefaultCentering;
typedef ippl::Field<double, Dim, Mesh_t, Centering_t> field;
typedef ippl::Field<ippl::Vector<double, Dim>, Dim, Mesh_t, Centering_t> fieldV;


// .... Define the mesh and the field types .... //


// define the R (rho) field
field exact, rho;
exact.initialize(mesh, layout);
rho.initialize(mesh, layout);

// define the Vector field E (LHS)
fieldV exactE, fieldE;
exactE.initialize(mesh, layout);
fieldE.initialize(mesh, layout);

@endcode

Then we need to define the solver type we want to use:
@code
using Solver_t = ippl::FFTOpenPoissonSolver<fieldV, field>
@endcode
We define the parameters to pass to the solver. Consider the not declared variable to be your choice
for your own simulation:
@code
// Parameter List to pass to solver
ippl::ParameterList params;

// Set the parameters
params.add("use_pencils", true); // can be true or false
params.add("comm", ippl::a2a); // can be ippl::a2a, ippl::a2av, ippl::p2p, ippl::p2p_pl
params.add("use_reorder", true); // can be true or false
params.add("use_heffte_defaults", false); // can be true or false
params.add("use_gpu_aware", true); // can be true or false
params.add("r2c_direction", 0); // can be 0, 1, 2
params.add("algorithm", Solver_t::HOCKNEY); // can be Solver_t::HOCKNEY or Solver_t::VICO
params.add("output_type", Solver_t::SOL_AND_GRAD); // can be Solver_t::SOL_AND_GRAD or
Solver_t::SOL_ONLY
@endcode




Now we can define the solver object and solve the Poisson equation:
@code
Solver_t FFTsolver(fieldE, rho, params);
FFTsolver.solve();
@endcode
The potential is stored in the rho Field. The E-Field is stored in the fieldE Field.


## [Optional] Using a Preconditioner 

If you want to precondition the solver you can add following parameters to the parameter list:
@code
// Define the preconditioner type (jacobi, newton, chebyshev, richardson or gauss_seidel)
params.add("preconditioner_type", preconditioner_type);
// Define the gauss_seidel parameters
params.add("gauss_seidel_inner_iterations", gauss_seidel_inner_iterations);
params.add("gauss_seidel_outer_iterations", gauss_seidel_outer_iterations);
// Define the newton parameters
params.add("newton_level", newton_level);
// Define the chebyshev parameters
params.add("chebyshev_degree", chebyshev_degree);
// Define the richardson parameters
params.add("richardson_iterations", richardson_iterations);
// Define the communication parameters (needed for richardson and gauss_seidel)
params.add("communication", communication);
// Merge the parameters
solver.mergeParameters(params);
@endcode

*/
