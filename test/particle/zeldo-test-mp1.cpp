// Test zeldo-test-mp1
//   This test program has the following assumptions
//   box size [0...1]^3 and Pk = 1 and const. The field is distributed.
//   The \delta(k) == \delta(-k)* is checked
//   Purpose:
//   . creates a random gaussian field in k-space (InitDeltaField) 
//   . compute per dimension the displacement in k-space (ComputeDisplacementComponentK) 
//   . FFT^-1 
//   . apply the displacement to particle coordinated and initialize verlocity field (ComputeWorldCoordinates)
//
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

/*
salloc --nodes=64  --partition=standard-g --time=00:30:00 --account=project_465001705 --gres=gpu:8 --ntasks-per-node=8 --gpus-per-node=8
srun -N64 --ntasks-per-node=8 --gpus-per-node=8 ./zeldo-test-mp1 4096 4096 4096  --info 5
*/

#include "Ippl.h"
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#define str(x)  #x
#define xstr(x) str(x)

// dimension of our positions
#define DIM     3
constexpr unsigned Dim          = DIM;
constexpr const char* PROG_NAME = "zeldo1-" xstr(DIM) "d";

// some typedefs
typedef ippl::ParticleSpatialLayout<double, Dim> PLayout_t;
typedef ippl::UniformCartesian<double, Dim> Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;
typedef Mesh_t::DefaultCentering Centering_t;

template <typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim>
using Field = ippl::Field<T, Dim, Mesh_t, Centering_t>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<float, Dim>    SField_t;
typedef Field<Vector_t, Dim> VField_t;
typedef Field<Kokkos::complex<double>, Dim> CField_t;

double pi = Kokkos::numbers::pi_v<double>;

typedef ippl::Field<Kokkos::complex<double>, Dim, Mesh_t, Centering_t> field_type;

typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;

using gen_t = Kokkos::Random_XorShift64_Pool<>::generator_type;

template <class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:

  field_type cfield_m; 

  Vector<int, Dim> nr_m;
    
  Vector_t hr_m;
  Vector_t rmin_m;
  Vector_t rmax_m;
  std::array<bool, Dim> decomp_m;
  float M_m;

  std::unique_ptr<FFT_type> fft_m;

  ParticleAttrib<float> M;                                       
  typename ippl::ParticleBase<PLayout>::particle_position_type V;  // particle velocity

    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the update function invokes this
    */
    ChargedParticles(PLayout& pl)
        : ippl::ParticleBase<PLayout>(pl) {
        // register the particle attributes
        this->addAttribute(M);
        this->addAttribute(V);
    }

  ChargedParticles(PLayout& pl, FieldLayout_t& fl, Vector_t hr, Vector_t rmin, Vector_t rmax,
                     std::array<bool, Dim> decomp, float m)
        : ippl::ParticleBase<PLayout>(pl)
        , hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
	, decomp_m(decomp)
        , M_m(m) {
        // register the particle attributes
        this->addAttribute(M);
        this->addAttribute(V);
        setupBCs();

	ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", true);
	fft_m = std::make_unique<FFT_type>(fl, fftParams);
    }

  void setupBCs() { setBCAllPeriodic(); }

  void updateLayout(FieldLayout_t& fl, Mesh_t& mesh, ChargedParticles<PLayout>& buffer) {
    // Update local fields
    static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
    IpplTimings::startTimer(tupdateLayout);
    this->cfield_m.updateLayout(fl);

    // Update layout with new FieldLayout
    PLayout& layout = this->getLayout();
    layout.updateLayout(fl, mesh);
    IpplTimings::stopTimer(tupdateLayout);
    static IpplTimings::TimerRef tupdatePLayout = IpplTimings::getTimer("updatePB");
    IpplTimings::startTimer(tupdatePLayout);
    layout.update(*this, buffer);
    IpplTimings::stopTimer(tupdatePLayout);
  }


  ~ChargedParticles() {}

  bool checkHerm(field_type& cfield, 
		 FieldLayout_t& fl) 
{
  Inform msg("checkHerm ");

  auto devView  = cfield.getView();
  auto hostView = Kokkos::create_mirror_view(devView);
  Kokkos::deep_copy(hostView, devView);
  
  // 2. Get sizes and local index range
  auto gdom = cfield.getDomain();

  const unsigned int gNx = gdom[0].length(), gNy = gdom[1].length(), gNz = gdom[2].length();

  auto lDom = fl.getLocalNDIndex();
  const unsigned int li_min = lDom[0].first(), li_max = lDom[0].last();
  const unsigned int lj_min = lDom[1].first(), lj_max = lDom[1].last();
  const unsigned int lk_min = lDom[2].first(), lk_max = lDom[2].last();

  bool ok = true; const double tol = 1e-3;
  // 3. Loop over local points
  for(int i=li_min; i<=li_max && ok; ++i) {
    for(int j=lj_min; j<=lj_max && ok; ++j) {
      for(int k=lk_min; k<=lk_max && ok; ++k) {
        // Map to host-view indices
        int io = i - li_min;
        int jo = j - lj_min;
        int ko = k - lk_min;
        auto value = hostView(io, jo, ko);

        // Compute conjugate index (periodic)
        int ic = (gNx - i) % gNx;
        int jc = (gNy - j) % gNy;
        int kc = (gNz - k) % gNz;

        // Check if conjugate index is on the same rank
        if(ic>=li_min && ic<=li_max && jc>=lj_min && jc<=lj_max && kc>=lk_min && kc<=lk_max) {
          // Local compare
          int ico = ic - li_min;
          int jco = jc - lj_min;
          int kco = kc - lk_min;
          auto value_c = hostView(ico, jco, kco);
	  auto diff = value - Kokkos::conj(value_c);
	  double diffMag = Kokkos::sqrt(diff.real()*diff.real() + diff.imag()*diff.imag());
	  ok = (diffMag<tol);
	}
      }
    }
  }
  return ok;
}  

  void LinearZeldoInitMP(FieldLayout_t& fl, unsigned int nloc) {
    // After creating the field layout (cfield_m) and determining global grid sizes Nx, Ny, Nz:

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    using index_type       = typename ippl::RangePolicy<Dim>::index_type;
    
    typename CField_t::view_type& view = cfield_m.getView();

    auto rView = this->R.getView();
    auto vView = this->V.getView();

    const int ngh = cfield_m.getNghost();
    const ippl::NDIndex<Dim>& lDom = fl.getLocalNDIndex();

    index_type lgridsize = 1;
    for (unsigned d = 0; d < Dim; d++) {
      lgridsize *= lDom[d].length();
    }
    const uint64_t global_seed = 12345ULL;  // Shared global seed for reproducibility

    const int Nx = nr_m[0];
    const int Ny = nr_m[1];
    const int Nz = nr_m[2];
    const double Lx = rmax_m[0];
    const double Ly = rmax_m[1];
    const double Lz = rmax_m[2];

    static IpplTimings::TimerRef fourDenTimer = IpplTimings::getTimer("Fourier Density");
    IpplTimings::startTimer(fourDenTimer);
    // Initialize the Fourier density field with Gaussian random modes (Hermitian symmetric)
    ippl::parallel_for("InitDeltaField", ippl::getRangePolicy(view, ngh),
		       KOKKOS_LAMBDA(const index_array_type& idx) {
			 const double pi = Kokkos::numbers::pi_v<double>;
			 // Compute global coordinates (i,j,k) for this local index
			 int i = idx[0] - ngh + lDom[0].first();
			 int j = idx[1] - ngh + lDom[1].first();
			 int k = idx[2] - ngh + lDom[2].first();

			 // DC mode (k=0 vector) set to 0 (no DC offset)
			 if (i == 0 && j == 0 && k == 0) {
			   ippl::apply(view, idx) = Kokkos::complex<double>(0.0, 0.0);
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
			   double Pk = 1.0;  // power spectrum P(k) (assumed constant = 1 here)
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
			   ippl::apply(view, idx) = Kokkos::complex<double>(val_re, val_im);
			 }
		       });

    IpplTimings::stopTimer(fourDenTimer);
    
#ifdef DOHECHECK    
    bool hermOK = checkHerm(cfield_m, fl);
    if (!hermOK) {
      Inform msg("checkHerm ");
      msg << "failed" << endl;
    }
#endif
    
    static IpplTimings::TimerRef fourDisplTimer = IpplTimings::getTimer("Fourier Displacement");

    // Store delta(k) for reuse 
    auto tmpcfield = cfield_m; 
    typename CField_t::view_type& viewtmpcfield = tmpcfield.getView();

    // 2–4. Loop over displacement components x(0), y(1), z(2)
    for (int dim = 0; dim < 3; ++dim) {
      IpplTimings::startTimer(fourDisplTimer);
      // Compute displacement component in k-space
      ippl::parallel_for("ComputeDisplacementComponentK", ippl::getRangePolicy(view, ngh),
			 KOKKOS_LAMBDA(const index_array_type& idx) {
			   const double pi = Kokkos::numbers::pi_v<double>;
			   int i = idx[0] - ngh + lDom[0].first();
			   int j = idx[1] - ngh + lDom[1].first();
			   int k = idx[2] - ngh + lDom[2].first();
			   
			   int kx_i = (i <= Nx / 2) ? i : i - Nx;
			   int ky_i = (j <= Ny / 2) ? j : j - Ny;
			   int kz_i = (k <= Nz / 2) ? k : k - Nz;
			   
			   double kx = 2.0 * pi * kx_i / Lx;
			   double ky = 2.0 * pi * ky_i / Ly;
			   double kz = 2.0 * pi * kz_i / Lz;
			   double k2 = kx * kx + ky * ky + kz * kz;
			   
			   Kokkos::complex<double> delta = ippl::apply(viewtmpcfield, idx);
			   Kokkos::complex<double> I(0.0, 1.0);
			   double k_comp = (dim == 0) ? kx : (dim == 1) ? ky : kz;
			   Kokkos::complex<double> result = (k2 == 0.0) ? Kokkos::complex<double>(0.0, 0.0)
			     : I * (k_comp / k2) * delta;
			   ippl::apply(view, idx) = result;
			 });
	
	// Inverse FFT to real space
	fft_m->transform(ippl::BACKWARD, cfield_m);
	IpplTimings::stopTimer(fourDisplTimer);


	static IpplTimings::TimerRef posvelInitTimer = IpplTimings::getTimer("Position/Velocisty Init");
	IpplTimings::startTimer(posvelInitTimer);
	const unsigned int nx = lDom[0].length();
	const unsigned int ny = lDom[1].length();
	const Vector_t hr = hr_m;
	
	Kokkos::parallel_for("ComputeWorldCoordinates", lgridsize, KOKKOS_LAMBDA(const index_type n) {
	    // Convert 1D index n back to 3D indices (i, j, k)
	    const unsigned int i = n % nx;
	    const unsigned int j = (n / nx) % ny;
	    const unsigned int k = n / (nx * ny);
	    double disp = view(i, j, k).real();
	    unsigned int idx = (dim == 0) ? i : (dim == 1) ? j : k;
	    rView(n)[dim] = ((idx + 0.5) * hr[dim]) + disp;
	    vView(n)[dim] = disp;
	  });
	IpplTimings::stopTimer(posvelInitTimer);
      }	        
  }
  
  void LinearZeldoInit(FieldLayout_t& fl, unsigned int nloc) {
      
      auto rView = this->R.getView();
      auto vView = this->V.getView();
      
      typename CField_t::view_type& view = cfield_m.getView();
      const int ngh = cfield_m.getNghost();
      ippl::NDIndex<Dim> domain = cfield_m.getDomain();
      
      const ippl::NDIndex<Dim>& lDom = fl.getLocalNDIndex();
	
      Kokkos::Random_XorShift64_Pool<> rand_pool(12345); // Seed for reproducibility
      using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
      using index_type       = typename ippl::RangePolicy<Dim>::index_type;

      for (unsigned int i = 0; i < Dim; i++) {
	nr_m[i] = domain[i].length();
      }

      const double Lx = rmax_m[0];
      const double Ly = rmax_m[1];
      const double Lz = rmax_m[2];
      const int Nx = nr_m[0];
      const int Ny = nr_m[1];
      const int Nz = nr_m[2];

      unsigned int gridpoints = 1;
      for (unsigned d = 0; d < Dim; d++) {
	gridpoints *= nr_m[d];
      }
    
      if (nloc * ippl::Comm->size() != gridpoints) {
	Inform m("initPositions ");
	m << "Particle count must match gridpoint count to use gridpoint "
	  "locations. Switching to uniform distribution."
	  << endl;
      }

      index_type lgridsize = 1;
      for (unsigned d = 0; d < Dim; d++) {
	lgridsize *= lDom[d].length();
      }
    
      // 1. Initialize delta(k) in cfield_m
      ippl::parallel_for("InitDeltaField", ippl::getRangePolicy(view, ngh),
			 KOKKOS_LAMBDA(const index_array_type& idx) {
			   const double pi = Kokkos::numbers::pi_v<double>;
			   int i = idx[0] - ngh + lDom[0].first();
			   int j = idx[1] - ngh + lDom[1].first();
			   int k = idx[2] - ngh + lDom[2].first();

			   if (i == 0 && j == 0 && k == 0) {
			     ippl::apply(view, idx) = Kokkos::complex<double>(0.0, 0.0);
			   } else {
			     auto state = rand_pool.get_state();
			     double u1 = Kokkos::rand<gen_t, double>::draw(state);
			     double u2 = Kokkos::rand<gen_t, double>::draw(state);
			     double R = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
			     double theta = 2.0 * pi * u2;
			     double re = R * Kokkos::cos(theta);
			     double im = R * Kokkos::sin(theta);
			     double Pk = 1.0;
			     double amp = Kokkos::sqrt(Pk / 2.0);
			     ippl::apply(view, idx) = amp * Kokkos::complex<double>(re, im);
			     rand_pool.free_state(state);
			   }
			 });

#ifdef DOHECHECK
      bool hermOK = checkHerm(cfield_m, fl);
      if (!hermOK) {
	Inform msg("checkHerm ");
	msg << "failed" << endl;
      }
#endif
      
      // Store delta(k) for reuse 
      auto tmpcfield = cfield_m; 
      typename CField_t::view_type& viewtmpcfield = tmpcfield.getView();
                  
      // 2–4. Loop over displacement components x(0), y(1), z(2)
      for (int dim = 0; dim < 3; ++dim) {
	// Compute displacement component in k-space
	ippl::parallel_for("ComputeDisplacementComponentK", ippl::getRangePolicy(view, ngh),
			   KOKKOS_LAMBDA(const index_array_type& idx) {
			     const double pi = Kokkos::numbers::pi_v<double>;
			     int i = idx[0] - ngh + lDom[0].first();
			     int j = idx[1] - ngh + lDom[1].first();
			     int k = idx[2] - ngh + lDom[2].first();
			     
			     int kx_i = (i <= Nx / 2) ? i : i - Nx;
			     int ky_i = (j <= Ny / 2) ? j : j - Ny;
			     int kz_i = (k <= Nz / 2) ? k : k - Nz;
			     
			     double kx = 2.0 * pi * kx_i / Lx;
			     double ky = 2.0 * pi * ky_i / Ly;
			     double kz = 2.0 * pi * kz_i / Lz;
			     double k2 = kx * kx + ky * ky + kz * kz;
			     
			     Kokkos::complex<double> delta = ippl::apply(viewtmpcfield, idx);
			     Kokkos::complex<double> I(0.0, 1.0);
			     double k_comp = (dim == 0) ? kx : (dim == 1) ? ky : kz;
			     Kokkos::complex<double> result = (k2 == 0.0) ? Kokkos::complex<double>(0.0, 0.0)
			       : I * (k_comp / k2) * delta;
			     ippl::apply(view, idx) = result;
			   });
	
	// Inverse FFT to real space
	fft_m->transform(ippl::BACKWARD, cfield_m);
	
	const unsigned int nx = lDom[0].length();
	const unsigned int ny = lDom[1].length();
	const Vector_t hr = hr_m;
	
	Kokkos::parallel_for("ComputeWorldCoordinates", lgridsize, KOKKOS_LAMBDA(const index_type n) {
	    // Convert 1D index n back to 3D indices (i, j, k)
	    const unsigned int i = n % nx;
	    const unsigned int j = (n / nx) % ny;
	    const unsigned int k = n / (nx * ny);
	    double disp = view(i, j, k).real();
	    unsigned int idx = (dim == 0) ? i : (dim == 1) ? j : k;
	    rView(n)[dim] = ((idx + 0.5) * hr[dim]) + disp;
	    vView(n)[dim] = disp;
	  });
      }	        
}
  
private:
  void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(PROG_NAME);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        ippl::Comm->setDefaultOverallocation(1.5);

        int arg = 1;

        size_t volume = 1;
        ippl::Vector<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            volume *= nr[d] = std::atoi(argv[arg++]);
        }

        // Each rank must have a minimal volume of 8
        if (volume < 8 * ippl::Comm->size()) {
            msg << "!!! Ranks have not enough volume for proper working !!! (Minimal volume per "
                   "rank: "
                   "8)"
                << endl;
        }

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
        IpplTimings::startTimer(mainTimer);
        const size_t totalP = std::pow(nr[0],3);


        msg << "particle test " << PROG_NAME << endl
            << "np= " << totalP << " grid = " << nr << endl;

        using bunch_type = ChargedParticles<PLayout_t>;

        std::unique_ptr<bunch_type> univ;

        Vector_t rmin(0.0);
        Vector_t rmax(1.0);

        // create mesh and layout objects for this problem domain
        Vector_t hr;
        ippl::NDIndex<Dim> domain;

	std::array<bool, Dim> decomp;
        decomp.fill(true);
	
        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = ippl::Index(nr[d]);
            hr[d]     = rmax[d] / nr[d];
        }

        Vector_t origin = rmin;

	const bool isAllPeriodic = true;

        Mesh_t mesh(domain, hr, origin);
        FieldLayout_t FL(MPI_COMM_WORLD, domain, decomp, isAllPeriodic);
        PLayout_t PL(FL, mesh);

        msg << "field layout created " << endl;
	msg << "FIELD LAYOUT (INITIAL)" << endl;
        msg << FL << endl;

	float  M = 1.0;

	univ = std::make_unique<bunch_type>(PL, FL, hr, rmin, rmax, decomp, M);

	unsigned long int nloc = totalP / ippl::Comm->size();
        int rest = (int)(totalP - nloc * ippl::Comm->size());
        if (ippl::Comm->rank() < rest) {
            ++nloc;
        }

        univ->create(nloc);
        double totalParticles = 0.0;
        double localParticles = univ->getLocalNum();
        MPI_Reduce(&localParticles, &totalParticles, 1, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
        msg << "total particles " << totalParticles << " created" << endl;

        static IpplTimings::TimerRef LinearZeldoInitTimer = IpplTimings::getTimer("LinearZeldoInit");
        IpplTimings::startTimer(LinearZeldoInitTimer);
	univ->cfield_m.initialize(mesh, FL);
        msg << "fields initialized " << endl;
	univ->LinearZeldoInitMP(FL,nloc);
	IpplTimings::stopTimer(LinearZeldoInitTimer);
		
        msg << "field test " << PROG_NAME << ": End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
	IpplTimings::print(std::string("timing" + std::to_string(ippl::Comm->size()) + "r_"
				       + std::to_string(nr[0]) + "c.dat"));
    }
    ippl::finalize();

    return 0;
}
