#define IPPL_ENABLE_TESTS 

#include "Ippl.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <Kokkos_Complex.hpp>

#include "StructureFormationManager.h"

// field accessor becomes visible


// 1. Generate simple Fourier-space single mode field 
template<class ComplexView>
void generateHermitianSingleMode(ComplexView delta,
                                 int kx0, int ky0, int kz0)
{
  const int Nx = delta.extent(0);
  const int Ny = delta.extent(1);
  const int Nz = delta.extent(2);

  Kokkos::parallel_for("HermitianSingleMode", 1, KOKKOS_LAMBDA(int) {
    delta(kx0, ky0, kz0) = Kokkos::complex<double>( 2.0,  3.0);      // A
    int kx1 = (kx0 == 0 ? 0 : Nx - kx0);
    int ky1 = (ky0 == 0 ? 0 : Ny - ky0);
    int kz1 = (kz0 == 0 ? 0 : Nz - kz0);
    delta(kx1, ky1, kz1) = Kokkos::complex<double>( 2.0, -3.0);      // A*
  });
}

// Generate simple fourier-space single mode field *without* its conjugate
template<class ComplexView>
void generateNonHermitianSingleMode(ComplexView delta,
                                    int kx0, int ky0, int kz0)
{
  Kokkos::parallel_for("NonHermitianSingleMode", 1, KOKKOS_LAMBDA(int) {
    delta(kx0, ky0, kz0) = Kokkos::complex<double>( 2.0, 3.0);       // only +k
  });
}

// 3. Real-space cosine field
template<class RealView>
void generateRealCosineField(RealView field)
{
  const int Nx = field.extent(0);
  const int Ny = field.extent(1);
  const int Nz = field.extent(2);
  constexpr double pi = 3.141592653589793;

  using MDPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
  Kokkos::parallel_for("RealCosineField",
                       MDPolicy({0,0,0}, {Nx,Ny,Nz}),
                       KOKKOS_LAMBDA(int i, int j, int k)
  {
    double x = static_cast<double>(i)/Nx;
    double y = static_cast<double>(j)/Ny;
    double z = static_cast<double>(k)/Nz;
    field(i,j,k) = std::cos(2*pi*x) + std::cos(2*pi*y) + std::cos(2*pi*z);
  });
}

// Deliberately break one Fourier coefficient in given field
template<class ComplexView>
void breakHermiticityInPlace(ComplexView delta,
                             int kx, int ky, int kz)
{
  Kokkos::parallel_for("BreakHermiticity", 1, KOKKOS_LAMBDA(int) {
    delta(kx,ky,kz) *= Kokkos::complex<double>(-1.0, 0.5);   // destroy symmetry
  });
}

template<class Manager>
void hermiticityTest(Manager& manager)
{
  auto cview = manager.getCView();              // view into cfield_m
  int kx0=2, ky0=1, kz0=1;

  // 1. proper Hermitian pair
  Kokkos::deep_copy(cview,Kokkos::complex<double>(0,0));
  generateHermitianSingleMode(cview, kx0, ky0, kz0);
  std::cout<<"[1] Hermitian pair: "
           <<(manager.isHermitian()?"PASS":"FAIL")<<'\n';

  // 2. non-Hermitian single coefficient
  Kokkos::deep_copy(cview,Kokkos::complex<double>(0,0));
  generateNonHermitianSingleMode(cview,kx0,ky0,kz0);
  std::cout<<"[2] non-Hermitian single: "
           <<(manager.isHermitian()?"FAIL":"PASS (expected)")<<'\n';

  // 3. real cosine field  (imag part = 0) – should be Hermitian
  Kokkos::deep_copy(cview,Kokkos::complex<double>(0,0));
  generateRealCosineField(cview);
  std::cout<<"[3] cosine field: "
           <<(manager.isHermitian()?"PASS":"FAIL")<<'\n';

  // 4. break symmetry
  breakHermiticityInPlace(cview,1,0,0);
  std::cout<<"[4] broken cosine field: "
           <<(manager.isHermitian()?"FAIL":"PASS (expected)")<<'\n';
}

using size_type = ippl::detail::size_type;
constexpr unsigned Dim = 3;
using T = double;
using Vector_t = ippl::Vector<int,Dim>;

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
