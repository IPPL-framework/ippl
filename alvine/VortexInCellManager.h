#ifndef IPPL_VORTEX_IN_CELL_MANAGER_H
#define IPPL_VORTEX_IN_CELL_MANAGER_H

#include <memory>

#include "AlvineManager.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randu.h"
#include "VortexDistributions.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using host_type = typename ippl::ParticleAttrib<T>::host_mirror_type; /*using host_type = typename ippl::ParticleAttrib<T>::HostMirror;*/


template <typename T, unsigned Dim, typename VortexDistribution>
class VortexInCellManager : public AlvineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>; 
    using LoadBalancer_t      = LoadBalancer<T, Dim>;    
    FieldLayout_t<Dim> FL_m;    // Store the field layout
    Mesh_t<Dim> mesh_m;          // Store the mesh

    // Constructor declaration
    VortexInCellManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_, 
                        std::string& solver_, int dump_freq_,
                        Vector_t<double, Dim> rmin_ = 0.0,
                        Vector_t<double, Dim> rmax_ = 10.0,
                        Vector_t<double, Dim> origin_ = 0.0,
                        FieldLayout_t<Dim>& FL_ = nullptr,
                        Mesh_t<Dim>& mesh_ = nullptr)
    : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_) {
        this->rmin_m = rmin_;
        this->rmax_m = rmax_;
        this->origin_m = origin_;
        this->FL_m = FL_;          // Store the layout
        this->mesh_m = mesh_;      // Store the mesh
    }

    ~VortexInCellManager() {}

void pre_run() override {

      Inform csvout(NULL, "particles.csv", Inform::OVERWRITE);
      csvout.precision(16);
      csvout.setf(std::ios::scientific, std::ios::floatfield);

      if constexpr (Dim == 2) {
          csvout << "time,index,pos_x,pos_y,vorticity" << endl;
      } else {
          csvout << "time,index,pos_x,pos_y,pos_z" << endl;
      }

      for (unsigned i = 0; i < Dim; i++) {
          this->domain_m[i] = ippl::Index(this->nr_m[i]);
      }


      Vector_t<double, Dim> dr = this->rmax_m - this->rmin_m;

      this->hr_m = dr / this->nr_m;

      // Courant condition
      this->dt_m = 0.05;//std::min(0.05, 0.5 * ( *std::min_element(this->hr_m.begin(), this->hr_m.end()) ) );

      this->it_m = 0;
      this->time_m = 0.0;

      //this->np_m = 10000; //this->nr_m[0] * this->nr_m[0];

      this->decomp_m.fill(true);
      this->isAllPeriodic_m = true;

      this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
            this->isAllPeriodic_m));

      this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));
        
      this->fcontainer_m->initializeFields();

      this->setFieldSolver( std::make_shared<FieldSolver_t>( this->solver_m, &this->fcontainer_m->getOmegaField()) );
      
      this->fsolver_m->initSolver();

      //this->setLoadBalancer( std::make_shared<LoadBalancer_t>( this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m) );

      initializeParticles();

      this->par2grid();
      auto omega0 = this->fcontainer_m->getOmegaField().deepCopy();

      this->fsolver_m->runSolver();
      this->computeVelocityField();
      logEnergyDiagnostics();
      Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega0.getView());
      logEnstrophyDiagnostics();
      logDivergenceDiagnostics();
      this->grid2par();

      std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

      pc->R_old = pc->R;
      pc->R = pc->R_old + pc->P * this->dt_m;
      pc->update();

    }

 
void initializeParticles() {
    auto* mesh = &this->fcontainer_m->getMesh();
    auto* FL   = &this->fcontainer_m->getFL();
    std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
    ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
    const bool isFEM = (this->solver_m == "FEM") || (this->solver_m == "FEM_PRECON");
    rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(*FL, *mesh, isFEM);
    const ippl::NDIndex<Dim>& local = FL->getLocalNDIndex();

    // 1. Global lattice dimensions based on user‑supplied particle count
    unsigned nxp_global = static_cast<unsigned>(std::sqrt(this->np_m));
    unsigned nyp_global = this->np_m / nxp_global;
    size_type totalP_global = nxp_global * nyp_global;

    // 2. Physical bounds and spacing
    double xmin_global = this->rmin_m[0];
    double xmax_global = this->rmax_m[0];
    double ymin_global = this->rmin_m[1];
    double ymax_global = this->rmax_m[1];
    double ymin_band = (ymin_global + ymax_global) / 2.0 - 1.0;
    double ymax_band = (ymin_global + ymax_global) / 2.0 + 1.0;

    double dxp = (xmax_global - xmin_global) / nxp_global;
    double dyp = (ymax_band - ymin_band) / nyp_global;

    // 3. Local domain from grid decomposition
    int local_start_x = local[0].first();
    int local_end_x   = local[0].last();
    int local_start_y = local[1].first();
    int local_end_y   = local[1].last();

    double xmin_local = xmin_global + local_start_x * this->hr_m[0];
    double xmax_local = xmin_global + (local_end_x + 1) * this->hr_m[0];
    double ymin_local = ymin_global + local_start_y * this->hr_m[1];
    double ymax_local = ymin_global + (local_end_y + 1) * this->hr_m[1];

    // 4. Intersect rank's physical rectangle with the vortex band
    double y_low  = std::max(ymin_local, ymin_band);
    double y_high = std::min(ymax_local, ymax_band);
    // If the intersection is empty, the rank gets no particles
    if (y_low >= y_high) {
        pc->create(0);
        return;
    }

    // 5. Find the range of lattice indices that fall inside the rank's rectangle
    int ix_start = static_cast<int>(std::ceil((xmin_local - xmin_global - 0.5 * dxp) / dxp));
    int ix_end   = static_cast<int>(std::floor((xmax_local - xmin_global - 0.5 * dxp) / dxp));
    ix_start = std::max(0, ix_start);
    ix_end   = std::min(static_cast<int>(nxp_global - 1), ix_end);

    int iy_start = static_cast<int>(std::ceil((y_low - ymin_band - 0.5 * dyp) / dyp));
    int iy_end   = static_cast<int>(std::floor((y_high - ymin_band - 0.5 * dyp) / dyp));
    iy_start = std::max(0, iy_start);
    iy_end   = std::min(static_cast<int>(nyp_global - 1), iy_end);

    unsigned nxp_local = ix_end - ix_start + 1;
    unsigned nyp_local = iy_end - iy_start + 1;
    size_type nlocal = nxp_local * nyp_local;

    // 6. Create particles
    pc->create(nlocal);

    // 7. Random number generator for jitter
    int seed = 42;
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed + 100 * ippl::Comm->rank());

    // 8. Fill positions on the lattice (device side)
    auto R_view = pc->R.getView();

    Kokkos::parallel_for(
        "init_particle_positions",
        nlocal,
        KOKKOS_LAMBDA(const int i) {
            unsigned ix_local = i % nxp_local;
            unsigned iy_local = i / nxp_local;
            unsigned ix_global = ix_start + ix_local;
            unsigned iy_global = iy_start + iy_local;

            auto rand_gen = rand_pool.get_state();
            double jitter_x = (rand_gen.drand() - 0.5) * dxp * 0.2;
            double jitter_y = (rand_gen.drand() - 0.5) * dyp * 0.2;
            rand_pool.free_state(rand_gen);

            double x = xmin_global + (ix_global + 0.5) * dxp + jitter_x;
            double y = ymin_band + (iy_global + 0.5) * dyp + jitter_y;

            R_view(i)[0] = x;
            R_view(i)[1] = y;
        }
    );

// 9. Particle circulation strength (2D VIC)
auto omega_view = pc->omega.getView();
double omega0 = 1.0;         // physical vorticity amplitude
double Ap = dxp * dyp;       // particle area

Kokkos::parallel_for(
    "init_particle_vorticity",
    nlocal,
    KOKKOS_LAMBDA(const int i) {
        omega_view(i) = omega0 * Ap;
    }
);

    Kokkos::fence();

}

void logEnergyDiagnostics() {
    double energy = this->computeKineticEnergy();

    if (!this->energy_initialized_m) {
        this->energy0_m = energy;
        this->energy_initialized_m = true;

        if (ippl::Comm->rank() == 0) {
            std::ofstream out("energy.csv", std::ios::out);
            out << "step,time,energy,rel_error\n";
            out.close();
        }
        ippl::Comm->barrier();
    }

    double relErr = this->relativeError(energy, this->energy0_m);

    if (ippl::Comm->rank() == 0) {
        Inform m("energy ");
        m << "kinetic energy = " << energy
          << ", relError = " << relErr << endl;

        std::ofstream out("energy.csv", std::ios::app);
        out.precision(16);
        out.setf(std::ios::scientific, std::ios::floatfield);
        out << this->it_m << ","
            << this->time_m << ","
            << energy << ","
            << relErr << "\n";
        out.close();
    }
}

void logEnstrophyDiagnostics() {
    double enstrophy = this->computeEnstrophy();

    if (!this->enstrophy_initialized_m) {
        this->enstrophy0_m = enstrophy;
        this->enstrophy_initialized_m = true;

        if (ippl::Comm->rank() == 0) {
            std::ofstream out("enstrophy.csv", std::ios::out);
            out << "step,time,enstrophy,rel_error\n";
            out.close();
        }
        ippl::Comm->barrier();
    }

    double relErr = this->relativeError(enstrophy, this->enstrophy0_m);

    if (ippl::Comm->rank() == 0) {
        Inform m("enstrophy ");
        m << "enstrophy = " << enstrophy
          << ", relError = " << relErr << endl;

        std::ofstream out("enstrophy.csv", std::ios::app);
        out.precision(16);
        out.setf(std::ios::scientific, std::ios::floatfield);
        out << this->it_m << ","
            << this->time_m << ","
            << enstrophy << ","
            << relErr << "\n";
        out.close();
    }
}


void logDivergenceDiagnostics() {
    double divL2 = this->computeDivergenceL2();

    if (ippl::Comm->rank() == 0) {
        Inform m("divergence ");
        m << "L2 = " << divL2 << endl;

        std::ofstream out("divergence.csv", std::ios::app);

        if (this->it_m == 0) {
            out << "step,time,div_l2\n";
        }

        out.precision(16);
        out.setf(std::ios::scientific, std::ios::floatfield);

        out << this->it_m << ","
            << this->time_m << ","
            << divL2 << "\n";
        out.close();
    }
}
    void advance() override {
      LeapFrogStep();     
    }

    void LeapFrogStep() {

      static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
      static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
      static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
      static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");
      static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("par2grid");
      static IpplTimings::TimerRef grid2parTimer = IpplTimings::getTimer("grid2par");
      
      std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

      // scatter the vorticity to the underlying grid
      IpplTimings::startTimer(par2gridTimer);	
      this->par2grid();
      IpplTimings::stopTimer(par2gridTimer);	
      auto omega_n = this->fcontainer_m->getOmegaField().deepCopy();

      // claculate stream function
      IpplTimings::startTimer(SolveTimer);
      this->fsolver_m->runSolver();
      IpplTimings::stopTimer(SolveTimer);

      // calculate velocity from stream function
      IpplTimings::startTimer(PTimer);
      this->computeVelocityField();
      logEnergyDiagnostics();
      Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega_n.getView());
      logEnstrophyDiagnostics();
      logDivergenceDiagnostics();
      IpplTimings::stopTimer(PTimer);

      // gather velocity field
      IpplTimings::startTimer(grid2parTimer);	
      this->grid2par();
      IpplTimings::stopTimer(grid2parTimer);	

      //drift
      IpplTimings::startTimer(RTimer);
      typename ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>::particle_position_type R_old_temp = pc->R_old;

      pc->R_old = pc->R;
      pc->R = R_old_temp + 2 * pc->P * this->dt_m;
      IpplTimings::stopTimer(RTimer);


      IpplTimings::startTimer(updateTimer);
      pc->update();
      IpplTimings::stopTimer(updateTimer);

    }
#include <fstream>
#include <sstream>
#include <memory>

void dumpParticleDataPerRank() {
    auto pc = this->pcontainer_m;

    auto R_host = pc->R.getHostMirror();
    auto omega_host = pc->omega.getHostMirror();

    Kokkos::deep_copy(R_host, pc->R.getView());
    Kokkos::deep_copy(omega_host, pc->omega.getView());

    std::stringstream fname;
    fname << "particles_rank_" << ippl::Comm->rank() << ".csv";

    bool write_header = (this->it_m == 1);

    std::ofstream csvout;
    if (write_header) {
        csvout.open(fname.str(), std::ios::out);
    } else {
        csvout.open(fname.str(), std::ios::app);
    }

    if constexpr (Dim == 2) {
        if (write_header) {
            csvout << "time,index,pos_x,pos_y,vorticity\n";
        }

        for (size_type i = 0; i < pc->getLocalNum(); i++) {
            csvout << this->it_m << "," << i
                   << "," << R_host(i)[0]
                   << "," << R_host(i)[1]
                   << "," << omega_host(i) << "\n";
        }
    } else {
        if (write_header) {
            csvout << "time,index,pos_x,pos_y,pos_z\n";
        }

        for (size_type i = 0; i < pc->getLocalNum(); i++) {
            csvout << this->it_m << "," << i
                   << "," << R_host(i)[0]
                   << "," << R_host(i)[1]
                   << "," << R_host(i)[2] << "\n";
        }
    }

    csvout.close();
    ippl::Comm->barrier();
}

  /*  void dump() override {
      static IpplTimings::TimerRef dumpTimer = IpplTimings::getTimer("dump");
      IpplTimings::startTimer(dumpTimer);
      dumpParticleDataPerRank();
      IpplTimings::stopTimer(dumpTimer);
       
    }*/

};
#endif
