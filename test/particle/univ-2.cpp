// Test PICnd
//   This test program sets up a simple sine-wave electric field in N dimensions,
//   creates a population of particles with random positions and and velocities,
//   and then tracks their motions in the static
//   electric field using cloud-in-cell interpolation and periodic particle BCs.
//
//   This test also provides a base for load-balancing using a domain-decomposition
//   based on an ORB.
//
//   Usage:
//     srun ./PICnd 128 128 128 --info 10
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
/*

salloc --nodes=64  --partition=standard-g --time=00:30:00 --account=project_465001705 --gres=gpu:8 --ntasks-per-node=8 --gpus-per-node=8
srun -N64 --ntasks-per-node=8 --gpus-per-node=8 ./PICnd 4096 4096 4096  --info 5
*/

#include "Ippl.h"
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
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
constexpr const char* PROG_NAME = "PIC" xstr(DIM) "d";

// some typedefs
typedef ippl::ParticleSpatialLayout<double, Dim> PLayout_t;
typedef ippl::UniformCartesian<double, Dim> Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;
typedef Mesh_t::DefaultCentering Centering_t;

template <typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim>
using Field = ippl::Field<T, Dim, Mesh_t, Centering_t>;

typedef ippl::OrthogonalRecursiveBisection<Field<double, Dim>> ORB;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<float, Dim>    SField_t;
typedef Field<Vector_t, Dim> VField_t;

double pi = Kokkos::numbers::pi_v<double>;

template <class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
  
    Field<Vector<double, Dim>, Dim> forcF_m;
    Field<float, Dim> densF_m;

    Vector<int, Dim> nr_m;

    unsigned int decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    float M_m;

public:

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

    ChargedParticles(PLayout& pl, Vector_t hr, Vector_t rmin, Vector_t rmax,
                     unsigned int decomp[Dim], double m)
        : ippl::ParticleBase<PLayout>(pl)
        , hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , M_m(m) {
        // register the particle attributes
        this->addAttribute(M);
        this->addAttribute(V);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++) {
            decomp_m[i] = decomp[i];
        }
    }

    void setupBCs() { setBCAllPeriodic(); }

    void updateLayout(FieldLayout_t& fl, Mesh_t& mesh, ChargedParticles<PLayout>& buffer) {
        // Update local fields
        static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
        IpplTimings::startTimer(tupdateLayout);
        this->forcF_m.updateLayout(fl);
        this->densF_m.updateLayout(fl);

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



    void initFields() {
        static IpplTimings::TimerRef initFieldsTimer = IpplTimings::getTimer("initFields");
        IpplTimings::startTimer(initFieldsTimer);
        Inform m("initFields ");

        ippl::NDIndex<Dim> domain = forcF_m.getDomain();

        for (unsigned int i = 0; i < Dim; i++) {
            nr_m[i] = domain[i].length();
        }

        double phi0 = 0.1;
        double pi   = Kokkos::numbers::pi_v<double>;
        // scale_fact so that particles move more
        double scale_fact = 1e5;  // 1e6

        Vector_t hr = hr_m;

        typename VField_t::view_type& view = forcF_m.getView();
        const FieldLayout_t& layout        = forcF_m.getLayout();
        const ippl::NDIndex<Dim>& lDom     = layout.getLocalNDIndex();
        const int nghost                   = forcF_m.getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "Assign forcF_m", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                // local to global index conversion
                Vector_t vec = (0.5 + args + lDom.first() - nghost) * hr;

                ippl::apply(view, args)[0] = -scale_fact * 2.0 * pi * phi0;
                for (unsigned d1 = 0; d1 < Dim; d1++) {
                    ippl::apply(view, args)[0] *= Kokkos::cos(2 * ((d1 + 1) % 3) * pi * vec[d1]);
                }
                for (unsigned d = 1; d < Dim; d++) {
                    ippl::apply(view, args)[d] = scale_fact * 4.0 * pi * phi0;
                    for (int d1 = 0; d1 < (int)Dim - 1; d1++) {
                        ippl::apply(view, args)[d] *=
                            Kokkos::sin(2 * ((d1 + 1) % 3) * pi * vec[d1]);
                    }
                }
            });

        densF_m = dot(forcF_m, forcF_m);
        densF_m = sqrt(densF_m);
        IpplTimings::stopTimer(initFieldsTimer);
    }

    void initPositions(FieldLayout_t& fl, Vector_t& hr, unsigned int nloc) {
      
      auto rView = this->R.getView();            
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



        msg << "field test " << PROG_NAME << ": End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
	IpplTimings::print(std::string("timing" + std::to_string(ippl::Comm->size()) + "r_"
				       + std::to_string(nr[0]) + "c.dat"));
    }
    ippl::finalize();

    return 0;
}
