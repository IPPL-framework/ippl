/**
 * @page LandauDamping LandauDamping

 * This is an example demonstrating the setup and execution of a Landau damping simulation using the
IPPL framework.
 *
 * ### Usage
 * The executable can be run with the following command-line arguments:
 * ```
 * srun ./LandauDamping <nx> [<ny>...] <Np> <Nt> <stype> <lbthres> <tintegr> --overallocate
<ovfactor> --info 10
 * ```
 * - **nx**: Number of cell-centered points in the x-direction.
 * - **ny...**: Number of cell-centered points in the y-, z-, ...-direction.
 * - **Np**: Total number of macro-particles in the simulation.
 * - **Nt**: Number of time steps to execute.
 * - **stype**: Field solver type, with FFT and CG (Conjugate Gradient) being supported.
 * - **lbthres**: Load balancing threshold. For example, a value of 0.01 means load balancing occurs
if the load imbalance exceeds 1%.
 * - **tintegr**: Type of time integrator. For example Leapfrog.
 * - **ovfactor**: Over-allocation factor for buffers used in communication. A value of 1.0 means no
over-allocation.
 *
 * ### Example Command
 * ```
 * srun ./LandauDamping 128 128 128 10000 10 FFT 0.01 LeapFrog --overallocate 2.0 --info 10
 * ```
 * This command runs a simulation with a 128x128x128 grid, 10,000 particles, for 10 time steps,
using the FFT solver,
 * a load balancing threshold of 1%, with LeapFrog integration, overallocating buffers by a factor
of 2.
 *
 * ### Implementation Detail
 * The main function initializes the Ippl framework, sets up the simulation parameters from
command-line arguments,
 * and then creates and configures an instance of `LandauDampingManager` to manage the simulation.
It performs pre-run
 * operations to set up the simulation environment and then runs the simulation for the specified
number of time steps.
 * Timing information is collected and output for performance analysis.
 * @code
constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "LandauDamping";

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "datatypes.h"

#include "Utility/IpplTimings.h"

#include "LandauDampingManager.h"
#include "Manager/PicManager.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {

        Inform msg(TestName);
        Inform msg2all(TestName, INFORM_ALL_NODES);
        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);


//Parses command-line arguments to set up simulation parameters.
// This segment of code iterates through the command-line arguments provided to the application,
// assigning each parameter to the corresponding variable. These parameters define the
// simulation's grid size, number of particles, number of time steps, solver type, load balancing
// threshold, and overallocation factor.
//
        int arg = 1;
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }
        size_type totalP = std::atoll(argv[arg++]);
        int nt  = std::atoi(argv[arg++]);
        std::string solver = argv[arg++];
        double lbt = std::atof(argv[arg++]);
        std::string step_method = argv[arg++];

//Sets up and executes the Landau damping simulation.
//  After parsing the input parameters, this code initializes the LandauDampingManager with the
//  specified configuration. It then calls pre_run to prepare the simulation environment, sets
//  the simulation start time, and executes the simulation for the defined number of time steps.


        // Create an instance of a manger for the considered application
        LandauDampingManager<T, Dim> manager(totalP, nt, nr, lbt, solver, step_method);
        // Perform pre-run operations, including creating mesh, particles,...
        manager.pre_run();
        manager.setTime(0.0);
        msg << "Starting iterations ..." << endl;
        manager.run(manager.getNt());
        msg << "End." << endl;


//Collects timing information and finalizes the simulation.
//  Marks the end of the simulation execution, stops the main timer, and outputs timing
//  information. This helps in analyzing the performance of the simulation. Finally, it
//  calls `ippl::finalize()` to clean up the Ippl framework and gracefully exit the application.

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
* @endcode
*/