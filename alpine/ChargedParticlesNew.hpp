// ChargedParticles header file
//   Defines a particle attribute for charged particles to be used in
//   test programs
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include <csignal>
#include <thread>

#include "Utility/TypeUtils.h"

#include "Solver/ElectrostaticsCG.h"
#include "Solver/FFTPeriodicPoissonSolver.h"
#include "Solver/FFTPoissonSolver.h"
#include "Solver/P3MSolver.h"

#include "PContainer/PContainer.hpp"

template <class PLayout, typename T, unsigned Dim = 3>
class ChargedParticles : public PContainer<PLayout, T, Dim> {

public:

    double rhoNorm_m;

    double Qtot_m;

    ParticleAttrib<double> q;                 // charge
    typename PContainer<PLayout, T, Dim>::Base::particle_position_type P;  // particle velocity
    typename PContainer<PLayout, T, Dim>::Base::particle_position_type E;  // electric field at particle position

    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the bunch buffer uses this
     */

     ChargedParticles(PLayout& pl)
         : PContainer<PLayout, T, Dim>(pl) {
       registerAttributes();
       setPotentialBCs();
     }

     ChargedParticles(PLayout& pl, Vector_t<double, Dim> hr, Vector_t<double, Dim> rmin,
                      Vector_t<double, Dim> rmax, ippl::e_dim_tag decomp[Dim], double Q,
                      std::string solver)
       : PContainer<PLayout, T, Dim>(pl, hr, rmin, rmax, decomp, solver)
       , Qtot_m(Q) {
         registerAttributes();
         for (unsigned int i = 0; i < Dim; i++) {
             PContainer<PLayout, T, Dim>::decomp_m[i] = decomp[i];
         }
         setupBCs();
         setPotentialBCs();
     }

     void setPotentialBCs() {
         // CG requires explicit periodic boundary conditions while the periodic Poisson solver
         // simply assumes them
         if (PContainer<PLayout, T, Dim>::stype_m == "CG") {
             for (unsigned int i = 0; i < 2 * Dim; ++i) {
                 PContainer<PLayout, T, Dim>::allPeriodic[i] = std::make_shared<ippl::PeriodicFace<Field<T, Dim>>>(i);
             }
         }
     }

    void setupBCs() { PContainer<PLayout, T, Dim>::setBCAllPeriodic(); }
  
    void registerAttributes() {
      // register the particle attributes
      this->addAttribute(q);
      this->addAttribute(P);
      this->addAttribute(E);
    }

     ~ChargedParticles() {}


  /*
    Gather/Scatter are the interfaces to Solver via PContainer
   */
  
    void gatherCIC() {gather(this->E, PContainer<PLayout, T, Dim>::F_m, this->R); }

    void scatterCIC(size_type totalP, unsigned int iteration, Vector_t<double, Dim>& hrField) {
        Inform m("scatter ");

        PContainer<PLayout, T, Dim>::rhs_m = 0.0;
        scatter(q, PContainer<PLayout, T, Dim>::rhs_m, this->R);

        static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("Check");
        IpplTimings::startTimer(sumTimer);
        double Q_grid = PContainer<PLayout, T, Dim>::rhs_m.sum();

        size_type Total_particles = 0;
        size_type local_particles = this->getLocalNum();

        MPI_Reduce(&local_particles, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        double rel_error = std::fabs((Qtot_m - Q_grid) / Qtot_m);
        m << "Rel. error in charge conservation = " << rel_error << endl;

        if (ippl::Comm->rank() == 0) {
            if (Total_particles != totalP || rel_error > 1e-10) {
                m << "Time step: " << iteration << endl;
                m << "Total particles in the sim. " << totalP << " "
                  << "after update: " << Total_particles << endl;
                m << "Rel. error in charge conservation: " << rel_error << endl;
                ippl::Comm->abort();
            }
        }

        double cellVolume =
            std::reduce(hrField.begin(), hrField.end(), 1., std::multiplies<double>());
        PContainer<PLayout, T, Dim>::rhs_m = PContainer<PLayout, T, Dim>::rhs_m / cellVolume;

        rhoNorm_m = norm(PContainer<PLayout, T, Dim>::rhs_m);
        IpplTimings::stopTimer(sumTimer);
	/*
	Connector::dumpVTK( PContainer<PLayout, T, Dim>::rhs_m,
			    PContainer<PLayout, T, Dim>::nr_m[0],
			    PContainer<PLayout, T, Dim>::nr_m[1],
			    PContainer<PLayout, T, Dim>::nr_m[2], iteration, hrField[0], hrField[1], hrField[2]);
	*/
        // rho = rho_e - rho_i (only if periodic BCs)
        if ( PContainer<PLayout, T, Dim>::stype_m != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *=  PContainer<PLayout, T, Dim>::rmax_m[d] -  PContainer<PLayout, T, Dim>::rmin_m[d];
            }
            PContainer<PLayout, T, Dim>::rhs_m = PContainer<PLayout, T, Dim>::rhs_m - (Qtot_m / size);
        }
    }

};

