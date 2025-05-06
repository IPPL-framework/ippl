// Test 
//   This test program sets up 
//
//   Usage:
//     srun ./univ-2 128 128 128 
//
// Copyright (c) 2020, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//


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

// #include "StructureFormationManager.h"

#define str(x)  #x
#define xstr(x) str(x)

// dimension of our positions
#define DIM     3
constexpr unsigned Dim          = DIM;
constexpr const char* PROG_NAME = "univ-3-" xstr(DIM) "d";

// some typedefs : these are all the same as the ones in the datatypes.h cosmology code
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
typedef Field<float, Dim>    SField_t; // currently not included in the cosmology code
typedef Field<Vector_t, Dim> VField_t;
typedef Field<Kokkos::complex<double>, Dim> CField_t; // currently not included in the cosmology code

double pi = Kokkos::numbers::pi_v<double>;


typedef ippl::Field<Kokkos::complex<double>, Dim, Mesh_t, Centering_t> field_type; // currently not included in the cosmology code

// the cosmology code uses a periodic poisson solver - what is the difference here?
typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type; // currently not included in the costmology code

// This class is equivalent to the fieldContainer class in cosmology
template <class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:

  field_type cfield_m; 

  Vector<int, Dim> nr_m;
    
  Vector_t hr_m;
  Vector_t rmin_m;
  Vector_t rmax_m;
  std::array<bool, Dim> decomp_m;
  float M_m; // ? not in cosmology

  std::unique_ptr<FFT_type> fft_m; // not in cosmology
  // this is in the particleContainer class in cosmology, velocity, mass and gravity
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
    // this is equivalent in cosmology
    void setupBCs() { setBCAllPeriodic(); }
    // this is almost equivalent in cosmology, exists inuder the same name updateLayout
    // with slightly different arguments
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
  
  // These are the test functions that you should use
  // This function can remain the same because it's just setting up the fields
  void initFields() {
    Inform msg ("FFTTEst ");
    
    ippl::NDIndex<Dim> domain = cfield_m.getDomain(); // this can stay the same, it's just getting the domain

    for (unsigned int i = 0; i < Dim; i++) {
      nr_m[i] = domain[i].length();
    }

    double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    double dx = Lx / nr_m[0];
    double dy = Ly / nr_m[1];
    double dz = Lz / nr_m[2];
    int kx = 3, ky = 2, kz = 1;
	
    typename CField_t::view_type& view        = cfield_m.getView();

    Kokkos::Random_XorShift64_Pool<> rand_pool(12345); // Seed for reproducibility
    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

    ippl::parallel_for(
		       "Compute cfield_m", ippl::getRangePolicy(view),
		       KOKKOS_LAMBDA(const index_array_type& args) {
			 auto pi =  Kokkos::numbers::pi_v<double>;
			 int i = args[0];
			 int j = args[1];
			 int k = args[2];
			 double x = i * dx;
			 double y = j * dy;
			 double z = k * dz;
			 double val = Kokkos::sin(2.0 * pi * kx * x / Lx) * Kokkos::sin(2.0 * pi * ky * y / Ly) * Kokkos::sin(2.0 * pi * kz * z / Lz);
			 ippl::apply(view, args) = Kokkos::complex<double>(val, 0.0);
		       });

    fft_m->transform(ippl::FORWARD, cfield_m);

    // Copy to host to inspect
    auto fft_result_host = cfield_m.getHostMirror();
    Kokkos::deep_copy(fft_result_host, cfield_m.getView());

    // Check if the correct mode has large magnitude and others are near zero
    double max_mag = 0.0;
    int found_i = -1, found_j = -1, found_k = -1;

    for (int i = 0; i < nr_m[0] ; ++i) {
      for (int j = 0; j < nr_m[1]; ++j) {
	for (int k = 0; k < nr_m[2]; ++k) {
	  double mag = Kokkos::abs(fft_result_host(i, j, k));
	  if (mag > max_mag) {
	    max_mag = mag;
	    found_i = i;
	    found_j = j;
	    found_k = k;
	  }
	}
      }
    }
    
    msg << "Max FFT magnitude found at: (" << found_i << ", "
	<< found_j << ", " << found_k << "), magnitude = " << max_mag << endl;
    

    // Expected locations for sin-sin-sin (due to real input, expect symmetrical peaks)
    int expected_i = kx;
    int expected_j = ky;
    int expected_k = kz;

    assert((found_i == expected_i || found_i == nr_m[0] - expected_i) &&
           (found_j == expected_j || found_j == nr_m[1] - expected_j) &&
           (found_k == expected_k || found_k == nr_m[2] - expected_k) &&
           "FFT peak not at expected position!");
    }
    // this function is just setting up the lagrangian positions, need to make sure it matches the 
    // cosmology values too
    
    void initPositions(FieldLayout_t& fl, Vector_t& hr, unsigned int nloc) {
      
      auto rView              = this->R.getView(); // I don't fully understand where this R and getView() method come from  
      auto dom                = fl.getDomain();
      unsigned int gridpoints = 1;
      for (unsigned d = 0; d < Dim; d++) {
	gridpoints *= dom[d].length();
      }
      
      if (nloc * ippl::Comm->size() != gridpoints) {
	Inform m("initPositions ");
	m << "Particle count must match gridpoint count to use gridpoint "
	  "locations. Switching to uniform distribution."
	  << endl;
      }

      const ippl::NDIndex<Dim>& lDom = fl.getLocalNDIndex();
      using index_type               = typename ippl::RangePolicy<Dim>::index_type;

      index_type lgridsize = 1;
      for (unsigned d = 0; d < Dim; d++) {
	lgridsize *= lDom[d].length();
      }
	
      const unsigned int nx = lDom[0].length();
      const unsigned int ny = lDom[1].length();
    
      Kokkos::parallel_for("ComputeWorldCoordinates", lgridsize, KOKKOS_LAMBDA(const index_type n) {
	    // Convert 1D index n back to 3D indices (i, j, k)
	  const unsigned int i = n % nx;
	  const unsigned int j = (n / nx) % ny;
	  const unsigned int k = n / (nx * ny);
	  // Compute world coordinates	    
	  rView(n)[0] = (i + 0.5) * hr[0];  // x-coordinate
	  rView(n)[1] = (j + 0.5) * hr[1];  // y-coordinate
	  rView(n)[2] = (k + 0.5) * hr[2];  // z-coordinate
	});
      Kokkos::deep_copy(this->R.getView(), rView);
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

        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);
        univ->create(nloc);
        double totalParticles = 0.0;
        double localParticles = univ->getLocalNum();
        MPI_Reduce(&localParticles, &totalParticles, 1, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
        msg << "total particles " << totalParticles << " initialized" << endl;
	univ->initPositions(FL, hr, nloc);
        IpplTimings::stopTimer(particleCreation);

        static IpplTimings::TimerRef initFieldsTimer = IpplTimings::getTimer("initFields");
        IpplTimings::startTimer(initFieldsTimer);
	univ->cfield_m.initialize(mesh, FL);
        msg << "fields initialized " << endl;
	univ->initFields();
	IpplTimings::stopTimer(initFieldsTimer);
		
        msg << "field test " << PROG_NAME << ": End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
	IpplTimings::print(std::string("timing" + std::to_string(ippl::Comm->size()) + "r_"
				       + std::to_string(nr[0]) + "c.dat"));
    }
    ippl::finalize();

    return 0;
}
