// TestHermiticity.cpp
//
// A dedicated test to verify the Hermitian symmetry check function 
// within the cosmo-app of the IPPL framework.
//
// To perform these checks, a full instance of the `StructureFormationManager` class is initialized.
// This is because the `isHermitian()` method and the field view (`cview`) depend on the manager's 
// internal setup, including its `FieldLayout`, mesh, and ghost cell configurations, which are 
// established during the manager's construction and `pre_run()` method.
//
// Usage:
//   This test is designed to be run on a single MPI rank for simplicity and directness
//   of data manipulation, as it directly sets specific Fourier coefficients.
//
//   srun -np 1 ./TestHermiticity <inputfile> <tfFn> <outFn> <outDir> <fsType> <lbthres> <integrator>
//
//   <inputfile>: describes the simulation parameters (e.g. np, box_size, cosmology)
//   <tfFn>:      transfer function filename
//   <outFn>:     base file name for snapshots (can be a dummy for this test)
//   <outDir>:    base directory for output data (can be a dummy for this test)
//   <fsType>:    Field solver type (e.g. "FFT" or "CG")
//   <lbthres>:   Load balancing threshold (e.g. 1)
//   <integrator>:Time stepping method (e.g. "LeapFrog")
//
// Example:
//   srun -np 1 ./TestHermiticity input.par tf.dat out.data datadir FFT 0.01 LeapFrog
//
// Example input.par content:
/*
np=128
nt=0
box_size=64.0
seed=9854373
z_in=50.0
z_fi=0.0

// Cosmology:
hubble=0.7
Omega_m=0.25
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

// Read from file (0) or create (1)
ReadInParticles=1
*/

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
using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

/**
 * @brief Performs a series of sanity checks to verify the Hermitian symmetry check
 * of Fourier-space fields within the StructureFormationManager. Designed for one 
 * MPI rank only.
 *
 * The test cases include:
 * 1. Setting a pair of Fourier modes (k and -k) with values that satisfy
 * Hermitian symmetry (F(-k) = conj(F(k))). `isHermitian()` is expected 
 * to return 'TRUE'.
 * 2. Setting a single Fourier mode (k) without its corresponding Hermitian pair.
 * `isHermitian()` is expected to return 'FALSE' as the field explicitly violates 
 * Hermitian symmetry.
 *
 * @tparam Manager The type of the manager class, expected to be StructureFormationManager<T, Dim>.
 * @param manager An instance of the StructureFormationManager class, providing
 * access to the Fourier field (`cfield_m`), grid dimensions,
 * ghost cell information, and the `isHermitian()` check.
 */
template<class Manager>
void hermiticityTest1(Manager& manager)
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

    // Test Case 1: Proper Hermitian pair
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0)); // Clear the field

    Kokkos::parallel_for("HermitianSingleMode_Test1", 1, KOKKOS_LAMBDA(int) {
        Kokkos::complex<double> val_pos = Kokkos::complex<double>(2.0, 3.0);
        Kokkos::complex<double> val_neg = Kokkos::complex<double>(2.0, -3.0);

        cview(l_kx0, l_ky0, l_kz0) = val_pos;
        cview(l_kx_neg, l_ky_neg, l_kz_neg) = val_neg;

    });

    std::cout << "[1] Hermitian pair: "
              << (manager.isHermitian() ? "TRUE" : "FALSE") << '\n';

    // Test Case 2: Non-Hermitian single coefficient
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0)); // Clear the field again

    Kokkos::parallel_for("NonHermitianSingleMode_Test2", 1, KOKKOS_LAMBDA(int) {
        Kokkos::complex<double> val = Kokkos::complex<double>(2.0, 3.0);
        cview(l_kx0, l_ky0, l_kz0) = val; // only +k
    });

    std::cout << "[2] non-Hermitian single: "
              << (manager.isHermitian() ? "TRUE" : "FALSE (expected)") << '\n';
}

/**
 * @brief Performs a multi-rank test by initializing a Fourier field with a
 * real cosine wave and verifying its Hermitian symmetry.
 *
 * This test uses a specific global k-vector to define the cosine wave.
 * The Fourier transform of a real cosine wave is known to be real and
 * symmetric about k=0 (i.e., F(k) = F(-k), and both are real).
 *
 * @tparam Manager The type of the manager class, expected to be StructureFormationManager<T, Dim>.
 * @param manager An instance of the StructureFormationManager class, providing
 * access to the Fourier field (`cfield_m`), grid dimensions,
 * ghost cell information, and the `isHermitian()` check.
 */
template<class Manager>
void hermiticityTest2(Manager& manager)
{
    auto cview = manager.getCView(); // Local view of the distributed field
    auto particles = manager.getNr(); // Global dimensions (Nx, Ny, Nz)
    const int Nx = particles[0], Ny = particles[1], Nz = particles[2];

    const int ngh = manager.getGhostCells();
    const ippl::NDIndex<Dim>& lDom = manager.getLayout().getLocalNDIndex(); // Local domain for this rank

    const int myrank = ippl::Comm->rank(); // Get current MPI rank

    // Define the global k-vector for our cosine wave
    // Choose a non-zero k-vector that is unlikely to be self-conjugate
    const int global_kx0 = 2;
    const int global_ky0 = 1;
    const int global_kz0 = 0;

    ippl::Comm->barrier();

    // Clear the entire distributed field to zeros
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0));

    // Initialize the Fourier density field for a real cosine wave
    // Each rank computes its local contribution based on global k-vectors
    ippl::parallel_for("RealCosineFourierField", ippl::getRangePolicy(cview, ngh),
        KOKKOS_LAMBDA(const index_array_type& idx) {
            
            // Compute global coordinates (i,j,k) for this local index
            int i = idx[0] - ngh + lDom[0].first();
            int j = idx[1] - ngh + lDom[1].first();
            int k = idx[2] - ngh + lDom[2].first();

            // Its negative (in the positive frequency space mapping)
            int global_kx_neg = (global_kx0 == 0 ? 0 : Nx - global_kx0);
            int global_ky_neg = (global_ky0 == 0 ? 0 : Ny - global_ky0);
            int global_kz_neg= (global_kz0 == 0 ? 0 : Nz - global_kz0);

            // Value to set for the cosine mode components.
            // For A*cos(k.x), the Fourier components at k and -k are both A/2, and purely real.
            Kokkos::complex<double> cos_val = Kokkos::complex<double>(0.5, 0.0);

            // Check if this global (i,j,k) corresponds to our target k-vector or its negative
            if (i == global_kx0 && j == global_ky0 && k == global_kz0) {
                cview(idx[0], idx[1], idx[2]) = cos_val;
            } else if (i == global_kx_neg && j == global_ky_neg && k == global_kz_neg) {
                cview(idx[0], idx[1], idx[2]) = cos_val; // For real cosine, conjugate is same value
            }
        });

    if (myrank == 0) {
        std::cout << "[3] Real Cosine Fourier field (Multi-Rank Init): "
                  << (manager.isHermitian() ? "TRUE" : "FALSE") << '\n';
        std::cout << "--- End hermiticityTest2 ---\n";
    }
    ippl::Comm->barrier(); 
}

/**
 * @brief Performs a multi-rank test by initializing a random gaussian field that fills
 * the entire fourier space to check that the hermiticity check works correctly.
 *
 * 
 *
 * @tparam Manager The type of the manager class, expected to be StructureFormationManager<T, Dim>.
 * @param manager An instance of the StructureFormationManager class, providing
 * access to the Fourier field (`cfield_m`), grid dimensions,
 * ghost cell information, and the `isHermitian()` check.
 */
template<class Manager>
void hermiticityTest3(Manager& manager)
{
    auto cview = manager.getCView(); // Local view of the distributed field
    auto particles = manager.getNr(); // Global dimensions (Nx, Ny, Nz)
    const int Nx = particles[0], Ny = particles[1], Nz = particles[2];

    const int ngh = manager.getGhostCells();
    const ippl::NDIndex<Dim>& lDom = manager.getLayout().getLocalNDIndex(); // Local domain for this rank

    const int myrank = ippl::Comm->rank(); // Get current MPI rank

    ippl::Comm->barrier();

    // Clear the entire distributed field to zeros
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0));
    
    const uint64_t global_seed = 12345ULL;  // Shared global seed for reproducibility

    // Initialize the Fourier density field with Gaussian random modes (Hermitian symmetric)
    ippl::parallel_for("InitDeltaField", ippl::getRangePolicy(cview, ngh),
		       KOKKOS_LAMBDA(const index_array_type& idx) {
			 const double pi = Kokkos::numbers::pi_v<double>;
			 // Compute global coordinates (i,j,k) for this local index
			 int i = idx[0] - ngh + lDom[0].first();
			 int j = idx[1] - ngh + lDom[1].first();
			 int k = idx[2] - ngh + lDom[2].first();

			 // DC mode (k=0 vector) set to 0 (no DC offset)
			 if (i == 0 && j == 0 && k == 0) {
			   ippl::apply(cview, idx) = Kokkos::complex<double>(0.0, 0.0);
			 } else {
			   // Compute the global “negative” indices for Hermitian pair
			   int i_neg = (i == 0 ? 0 : Nx - i);
			   int j_neg = (j == 0 ? 0 : Ny - j);
			   int k_neg = (k == 0 ? 0 : Nz - k);
			   
			   // Determine if this index is its own conjugate (self-Hermitian case)
			   bool self = (i_neg == i && j_neg == j && k_neg == k);
			   // Determine lexicographically which of (i,j,k) and its negative is smaller
			   bool is_conjugate = (!self && 
						(i_neg < i || (i_neg == i && j_neg < j) || 
						 (i_neg == i && j_neg == j && k_neg < k)));
			   // Choose the "key" coordinates (the global smaller of the pair) for random generation
			   int key_i = is_conjugate ? i_neg : i;
			   int key_j = is_conjugate ? j_neg : j;
			   int key_k = is_conjugate ? k_neg : k;

			   // Deterministically generate two uniform [0,1) random numbers from the key
			   uint64_t key_index = ((uint64_t)key_i * Ny + key_j) * Nz + key_k;
			   uint64_t x = key_index ^ global_seed;
			   if (x == 0ull) x = 1ull;  // avoid zero state
			   // XorShift64 steps to produce pseudorandom 64-bit values
			   x ^= x << 13;  x ^= x >> 7;  x ^= x << 17;
			   uint64_t r1 = x;
			   x ^= x << 13;  x ^= x >> 7;  x ^= x << 17;
			   uint64_t r2 = x;
			   // Map the 53 most significant bits of r1,r2 to double in [0,1)
			   const double norm = 1.0 / 9007199254740992.0;  // 1/2^53
			   double u1 = (double)(r1 >> 11) * norm;
			   double u2 = (double)(r2 >> 11) * norm;
			   
			   // Convert uniforms to Gaussian via Box-Muller
			   double R     = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
			   double theta = 2.0 * pi * u2;
			   double gauss_re = R * Kokkos::cos(theta);   // Gaussian(0,1) for real part
			   double gauss_im = R * Kokkos::sin(theta);   // Gaussian(0,1) for imaginary part
			   double Pk = 1;

			   // Set amplitude: for self-conjugate modes use sqrt(Pk), otherwise sqrt(Pk/2)
			   double amp = self ? Kokkos::sqrt(Pk) : Kokkos::sqrt(Pk / 2.0);
			   double val_re = amp * gauss_re;
			   double val_im = amp * gauss_im;
			   if (self) {
			     // For self-conjugate (Nyquist) modes, enforce the mode is real
			     val_im = 0.0;
			   } else if (is_conjugate) {
			     // If this index is the "conjugate partner" (lexicographically larger), flip the imaginary sign
			     val_im = -val_im;
			   }
			   // Assign the complex value to this local mode
			   ippl::apply(cview, idx) = Kokkos::complex<double>(val_re, val_im);
                           }
		       });

    if (myrank == 0) {
        std::cout << "[4] Random Gaussian Field (Multi-Rank Init): "
                  << (manager.isHermitian() ? "TRUE" : "FALSE") << '\n';
        std::cout << "--- End hermiticityTest3 ---\n";
    }
    ippl::Comm->barrier(); 
}


int main(int argc,char** argv)
{

  ippl::initialize(argc,argv);
  {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);
        
        if (ippl::Comm->size() == 1){
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

		hermiticityTest1(manager);
		hermiticityTest2(manager);
		hermiticityTest3(manager);
	} else {
	    std::cerr << "Error: Attempting to run multirank. This test is designed to only work on 1 rank / CPU node." << std::endl;
	} 
  }
  ippl::finalize();
  return 0;
}
