// StructureFormation.cpp
// The StructureFormation simulation is designed to model the formation and evolution of cosmic
// structures, such as galaxies and clusters of galaxies, in the universe. It uses initial
// conditions, including particle positions and velocities, to simulate the gravitational
// interactions and dynamics of a large number of particles over time. The goal is to understand how
// initial density fluctuations grow and evolve under the influence of gravity, leading to the
// large-scale structure observed in the universe today. The old mc-4 initializer based in Zaria's
// old and new initializer is added. 
//
//   Usage:
//     srun ./StructureFormation
//                  <inputfile> <tfFn> <outFn> <outDir> <fsType> 
//                  <lbthres> <integrator> --overallocate <ovfactor> --info 10
//     inputfile = describes the simulation
//     tfFn      = transfer function
//     outFn     = base file name for snapshots
//     outDir    = base directory for output data
//     fsType    = Field solver type (FFT and CG supported)
//     lbthres   = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                 percentage which can be tolerated and beyond which
//                 particle load balancing occurs. A value of 0.01 is good for many typical
//                 simulations.
//     integrator= LeapFrog
//     ovfactor  = Over-allocation factor for the buffers used in the communication. Typical
//                 values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun StructureFormation input.par tf.dat out.data datadir FFT 1.0 LeapFrog --overallocate 1.0 --info 5

/*  Example of input file

// Run:

np=3
nt=1
box_size=64.0   // In Mpc/h
seed=9854373
z_in=50.0
z_fi=0.0

// Cosmology:
hubble=0.7
Omega_m=0.25   // Total matter content; 1-this will be DE
Omega_nu=0.0
Omega_bar=0.04

Sigma_8=0.8
n_s=0.9
w_de=-1.0
N_nu=3
nu_pairs=4
f_NL=0.0
TFFlag=2

// Code stuff:
PrintFormat=0

// new
Omega_L=0.7  

*/



constexpr unsigned Dim = 3;
using T                = double;

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

#include "Utility/IpplTimings.h"

#include "Manager/PicManager.h"
#include "StructureFormationManager.h"

#include "mc-4-Initializer/DataBase.h"
#include "mc-4-Initializer/InputParser.h"

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

	std::string indatName = argv[1];
	std::string tfName = argv[2];
	std::string outBase = argv[3];
        std::string ic_folder = argv[4];
       	
	int arg = 5;
	initializer::InputParser par(indatName);
	initializer::GlobalStuff::instance().GetParameters(par);
		
        // Number of gridpoints in each dimension
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
	  nr[d] = initializer::GlobalStuff::instance().ngrid;
        }
        // Total number of particles
        size_type totalP = nr[0]*nr[1]*nr[2];

        // Number of time steps
        int nt = 0;
	par.getByName("nt", nt);

	int readInParticles = 0;
	par.getByName("ReadInParticles", readInParticles);
	bool readICs = (readInParticles == 0);
	
        // Solver method
        std::string solver = argv[arg++];
        // Check if the solver type is valid
        if (solver != "CG" && solver != "FFT") {
            throw std::invalid_argument("Invalid solver type. Supported types are 'CG' and 'FFT'.");
        }
        // Load Balance Threshold
        double lbt = std::atof(argv[arg++]);
        // Time stepping method
        std::string step_method = argv[arg++];

        // Create an instance of a manager for the considered application
        StructureFormationManager<T, Dim> manager(totalP, nt, nr, lbt, solver, step_method, par, tfName, readICs);

        // set initial conditions folder
        manager.setIC(ic_folder);

        // Perform pre-run operations, including creating mesh, particles,...
        manager.pre_run();

        msg << "Starting iterations ... up to " << manager.getNt() << endl;

        manager.run(manager.getNt());

        msg << "End." << endl;

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();

        std::stringstream ss;
        ss << "timing_" << ippl::Comm->size() << ".dat";
        std::string filename = ss.str();

        IpplTimings::print(manager.folder + filename);
    }
    ippl::finalize();

    return 0;
}
