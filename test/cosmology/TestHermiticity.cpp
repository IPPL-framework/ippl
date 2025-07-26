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
 * @brief Performs a parallel hermiticity test by initializing a
 * Fourier field with two points at index k and -k which are each
 * others complex conjugate and then calling the isHermitian function
 * from the cosmology StructureFormationManager class.
 *
 * @tparam Manager The type of the manager class, expected to be StructureFormationManager<T, Dim>.
 * @param manager An instance of the StructureFormationManager class, providing
 * access to the Fourier field (`cfield_m`), grid dimensions,
 * ghost cell information, and the `isHermitian()` check.
 */
template<class Manager>
void hermiticityTest1(Manager& manager)
{
    Inform msg("hermiticityTest1");
    auto cview = manager.getCView(); // Local view of the distributed field
    auto particles = manager.getNr(); // Global dimensions (Nx, Ny, Nz)
    const int Nx = particles[0], Ny = particles[1], Nz = particles[2];

    const int ngh = manager.getGhostCells();
    const ippl::NDIndex<Dim>& lDom = manager.getLayout().getLocalNDIndex(); // Local domain for this rank

    const int myrank = ippl::Comm->rank(); // Get current MPI rank

    // Define the global k-vector for our cosine wave
    // Choose an arbitrary non-zero k-vector
    const int global_kx0 = 2;
    const int global_ky0 = 1;
    const int global_kz0 = 0;

    ippl::Comm->barrier();

    // Clear the entire distributed field to zeros
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0));

    // Initialize the Fourier density field with two real numbers which are
    // complex conjugate points 
    // Each rank computes its local contribution based on global k-vectors
    ippl::parallel_for("InitHermitianField", ippl::getRangePolicy(cview, ngh),
        KOKKOS_LAMBDA(const index_array_type& idx) {
            // Compute global coordinates (i,j,k) for this local index
            int i = idx[0] - ngh + lDom[0].first();
            int j = idx[1] - ngh + lDom[1].first();
            int k = idx[2] - ngh + lDom[2].first();

            // Its negative (in the positive frequency space mapping)
            int global_kx_neg = (global_kx0 == 0 ? 0 : Nx - global_kx0);
            int global_ky_neg = (global_ky0 == 0 ? 0 : Ny - global_ky0);
            int global_kz_neg= (global_kz0 == 0 ? 0 : Nz - global_kz0);

            // Value to set for the mode components.
            Kokkos::complex<double> val = Kokkos::complex<double>(0.5, 0.0);

            // Check if this global (i,j,k) corresponds to our target k-vector or its negative
            if (i == global_kx0 && j == global_ky0 && k == global_kz0) {
                cview(idx[0], idx[1], idx[2]) = val;
            } else if (i == global_kx_neg && j == global_ky_neg && k == global_kz_neg) {
                cview(idx[0], idx[1], idx[2]) = val; // for real value, conjugate is same value
            }
        });

    auto hermitian1 = manager.isHermitian();

    ippl::Comm->barrier();  

    if (myrank == 0) {
        msg << "[1/4] Real Fourier field : "
                  << (hermitian1 ? "TRUE" : "FALSE") << endl ;
    }
    
    // Clear the entire distributed field to zeros
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0));
    
    // Initialize a non-hermitian field
    ippl::parallel_for("InitNonHermitianField", ippl::getRangePolicy(cview, ngh),
        KOKKOS_LAMBDA(const index_array_type& idx) {
            
            // Compute global coordinates (i,j,k) for this local index
            int i = idx[0] - ngh + lDom[0].first();
            int j = idx[1] - ngh + lDom[1].first();
            int k = idx[2] - ngh + lDom[2].first();

            // Its negative (in the positive frequency space mapping)
            int global_kx_neg = (global_kx0 == 0 ? 0 : Nx - global_kx0);
            int global_ky_neg = (global_ky0 == 0 ? 0 : Ny - global_ky0);
            int global_kz_neg= (global_kz0 == 0 ? 0 : Nz - global_kz0);

            Kokkos::complex<double> val = Kokkos::complex<double>(0.5, 0.0);

            // Check if this global (i,j,k) corresponds to our target k-vector or its negative
            if (i == global_kx0 && j == global_ky0 && k == global_kz0) {
                cview(idx[0], idx[1], idx[2]) = val;
            } else if (i == global_kx_neg && j == global_ky_neg && k == global_kz_neg) {
                cview(idx[0], idx[1], idx[2]) = -2 * val; // Make sure conjugate does not match
            }
        });

    auto hermitian2 = manager.isHermitian();

    if (myrank == 0) {
        msg << "[2/4] Real Fourier field : "
                  << (hermitian2 ? "TRUE" : "FALSE (expected)") << endl;
    }
    
    ippl::Comm->barrier();  
}

/**
 * @brief Performs a (parallel) test by initializing a random gaussian field that fills
 * the entire fourier space and then uses the isHermitian() function from the 
 * StructureFormationManager class to check hermiticity of this field.
 *
 * @tparam Manager The type of the manager class, expected to be StructureFormationManager<T, Dim>.
 * @param manager An instance of the StructureFormationManager class, providing
 * access to the Fourier field (`cfield_m`), grid dimensions,
 * ghost cell information, and the `isHermitian()` check.
 */
template<class Manager>
void hermiticityTest2(Manager& manager)
{
    Inform msg("hermiticityTest2");
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
    ippl::parallel_for("InitGaussianHermitian", ippl::getRangePolicy(cview, ngh),
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

    auto hermitian3 = manager.isHermitian();

    if (myrank == 0) {
        msg << "[3/4] Random Gaussian Field : "
                  << (hermitian3 ? "TRUE" : "FALSE") << endl;
    }
    
    ippl::Comm->barrier(); 
    
    // Clear the entire distributed field to zeros
    Kokkos::deep_copy(cview, Kokkos::complex<double>(0, 0));

    // Initialize the Fourier density field with Gaussian random modes (Hermitian symmetric)
    ippl::parallel_for("InitGaussianNonHermitian", ippl::getRangePolicy(cview, ngh),
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
			     val_im = -val_im * 2; // multiply by 2 to ensure the field is not hermitian
			   }
			   // Assign the complex value to this local mode
			   ippl::apply(cview, idx) = Kokkos::complex<double>(val_re, val_im);
                           }
		       });

    auto hermitian4 = manager.isHermitian();
    
    if (myrank == 0) {
        msg << "[4/4] Random Gaussian Field : "
                  << (hermitian4 ? "TRUE" : "FALSE (expected) ") << endl;
    }
    
    ippl::Comm->barrier(); 
}


int main(int argc,char** argv)
{

  ippl::initialize(argc,argv);
  {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

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
  }

  ippl::finalize();
  return 0;
}
