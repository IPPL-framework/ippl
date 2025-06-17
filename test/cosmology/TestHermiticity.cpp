constexpr unsigned Dim = 3;
using T = double;

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Complex.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "Manager/PicManager.h"

#include "mc-4-Initializer/DataBase.h"
#include "mc-4-Initializer/InputParser.h"

#include "StructureFormationManager.h"

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

template<class Manager>
void hermiticityTest(Manager& manager)
{
    auto cview = manager.getCView();
    auto particles = manager.getNr();
    const int Nx = particles[0], Ny = particles[1], Nz = particles[2];
    int global_kx0 = 2, global_ky0 = 1, global_kz0 = 1; // global k for the main mode

    const int ngh = manager.getGhostCells();
    const ippl::NDIndex<Dim>& lDom = manager.getLayout().getLocalNDIndex();

    // Convert global_kx0, global_ky0, global_kz0 to local view indices
    int l_kx0 = global_kx0 - lDom[0].first() + ngh;
    int l_ky0 = global_ky0 - lDom[1].first() + ngh;
    int l_kz0 = global_kz0 - lDom[2].first() + ngh;

    // Compute the GLOBAL negative indices for global_kx0, global_ky0, global_kz0
    int global_kx_neg = (global_kx0 == 0 ? 0 : Nx - global_kx0);
    int global_ky_neg = (global_ky0 == 0 ? 0 : Ny - global_ky0);
    int global_kz_neg = (global_kz0 == 0 ? 0 : Nz - global_kz0);

    // Convert these GLOBAL negative indices to LOCAL view indices
    int l_kx_neg = global_kx_neg - lDom[0].first() + ngh;
    int l_ky_neg = global_ky_neg - lDom[1].first() + ngh;
    int l_kz_neg = global_kz_neg - lDom[2].first() + ngh;

    // --- Test Case 1: Proper Hermitian pair ---
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0)); // Clear the field

    Kokkos::parallel_for("HermitianSingleMode_Test1", 1, KOKKOS_LAMBDA(int) {
        Kokkos::complex<double> val_pos = Kokkos::complex<double>(2.0, 3.0);
        Kokkos::complex<double> val_neg = Kokkos::complex<double>(2.0, -3.0);

        cview(l_kx0, l_ky0, l_kz0) = val_pos;
        cview(l_kx_neg, l_ky_neg, l_kz_neg) = val_neg;

    });

    std::cout << "[1] Hermitian pair: "
              << (manager.isHermitian() ? "TRUE" : "FALSE") << '\n';

    // --- Test Case 2: Non-Hermitian single coefficient ---
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0)); // Clear the field again

    Kokkos::parallel_for("NonHermitianSingleMode_Test2", 1, KOKKOS_LAMBDA(int) {
        Kokkos::complex<double> val = Kokkos::complex<double>(2.0, 3.0);
        cview(l_kx0, l_ky0, l_kz0) = val; // only +k
    });

    std::cout << "[2] non-Hermitian single: "
              << (manager.isHermitian() ? "TRUE" : "FALSE (expected)") << '\n';
}

int main(int argc,char** argv)
{

  ippl::initialize(argc,argv);
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

        // Create an instance of a manager for the hermiticity test
        StructureFormationManager<T, Dim> manager(totalP, nt, nr, lbt, solver, step_method, par, tfName, readICs);

        // set initial conditions folder
        manager.setIC(ic_folder);
        
        // Perform pre-run operations, including creating mesh, particles,...
        manager.pre_run();

        hermiticityTest(manager);
  }
  ippl::finalize();
  return 0;
}
