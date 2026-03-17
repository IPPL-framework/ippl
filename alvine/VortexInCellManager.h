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

      this->fsolver_m->runSolver();
      this->computeVelocityField();

      this->grid2par();

      std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

      pc->R_old = pc->R;
      pc->R = pc->R_old + pc->P * this->dt_m;
      pc->update();

    }

 /*   void initializeParticles() {

      std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

      // Create np_m particles in container
      size_type totalP = this->np_m;
      pc->create(totalP); // TODO: local number of particles? from kokkos?
      
      view_type* R = &(pc->R.getView()); // Position vector
      host_type omega_host = pc->omega.getHostMirror(); // Vorticity values
        
      // Random number generator
      int seed         = 42;
      Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

      double rmin[Dim];
      double rmax[Dim];
      for(unsigned int i=0; i<Dim; i++){
	  if(i==0) {
          	rmin[i] = this->rmin_m[i];
          	rmax[i] = this->rmax_m[i];
	  }
	  else {
		rmin[i] = (this->rmin_m[i] + this->rmax_m[i]) / 2.0 - 1.0;
		rmax[i] = (this->rmin_m[i] + this->rmax_m[i]) / 2.0 + 1.0;
	  }
      }

      // Sample from uniform distribution
      Kokkos::parallel_for(totalP, ippl::random::randu<double, Dim>(*R, rand_pool64, rmin, rmax));

      // Assign vorticity based on radius from center
      Kokkos::parallel_for(totalP,
        VortexDistribution(*R, omega_host, this->rmin_m, this->rmax_m, this->origin_m, this->np_m));
    
      Kokkos::deep_copy(pc->omega.getView(), omega_host);

      Kokkos::fence();
      ippl::Comm->barrier();

    }*/
void initializeParticles() {
    auto* mesh = &this->fcontainer_m->getMesh();
    auto* FL   = &this->fcontainer_m->getFL();
    std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
    ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
    const bool isFEM = (this->solver_m == "FEM") || (this->solver_m == "FEM_PRECON");
    rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(*FL, *mesh, isFEM);
    const ippl::NDIndex<Dim>& local = FL->getLocalNDIndex();
    // For rank 0 : local = { [0,3], [0,7] }
    //          x: 0 to 3   y: 0 to 7

    int local_start_x = local[0].first(); // For rank 0, this is 0;
    int local_end_x   = local[0].last(); // For rank 0, this is 3; (use last(), not second())
    int local_start_y = local[1].first(); // For rank 0, this is 0;
    int local_end_y   = local[1].last(); // For rank 0, this is 7; (use last(), not second())
    
    // Number of local grid cells in each direction
    unsigned nxp_local = local[0].length(); // Number of x-cells this rank owns
    unsigned nyp_local = local[1].length(); // Number of y-cells this rank owns
    
    // Each grid cell gets one particle at its center
    size_type nlocal = nxp_local * nyp_local; // Local number of particles for this rank

    pc->create(nlocal); // Each rank will only initialize its local portion of these particles.

    auto* R = &(pc->R.getView());
    auto omega_host = pc->omega.getHostMirror();

    double xmin = this->rmin_m[0];
    double xmax = this->rmax_m[0];

    // ===== VORTEX BAND - BASED ON GLOBAL Y COORDINATES =====
    double ymin_band = (this->rmin_m[1] + this->rmax_m[1]) / 2.0 - 1.0;
    double ymax_band = (this->rmin_m[1] + this->rmax_m[1]) / 2.0 + 1.0;

    // Calculate local physical bounds in x (still need this for x positions)
    double xmin_local = xmin + local_start_x * this->hr_m[0];
    double xmax_local = xmin + (local_end_x + 1) * this->hr_m[0];

    // For y, we use the GLOBAL band bounds, not local!
    // But we still need local y indices to know which particles belong to this rank
    double dy_global = (ymax_band - ymin_band) / (this->nr_m[1]); // Global y spacing in band

    // Local grid spacing in x (based on local domain)
    double dx_local = (xmax_local - xmin_local) / nxp_local;

    int seed = 42;
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed + 100 * ippl::Comm->rank());

    Kokkos::parallel_for(
        "init_particle_positions",
        nlocal,
        KOKKOS_LAMBDA(const int i) {

            //Convert 1D index to 2D grid indices
            unsigned ix_local = i % nxp_local; // Local x index within this rank's domain
            unsigned iy_local = i / nxp_local; // Local y index within this rank's domain

            // Convert local y index to global y index
            unsigned iy_global = local_start_y + iy_local;

            auto rand_gen = rand_pool.get_state();// Get random generator for this thread

            // Add jitter to avoid particles being exactly on grid points
            double jitter_x = (rand_gen.drand() - 0.5) * dx_local * 0.2;
            double jitter_y = (rand_gen.drand() - 0.5) * dy_global * 0.2; // Use global dy for jitter

            rand_pool.free_state(rand_gen); // free the random generator state

            // Calculate particle position:
            // X: based on LOCAL domain (rank-specific)
            // Y: based on GLOBAL vortex band (same for all ranks)
            (*R)(i)[0] = xmin_local + (ix_local + 0.5) * dx_local + jitter_x;
            (*R)(i)[1] = ymin_band + (iy_global + 0.5) * dy_global + jitter_y;
        }
    );

    Kokkos::parallel_for(
        "init_particle_vorticity",
        nlocal,
        VortexDistribution(*R, omega_host,
                           this->rmin_m, this->rmax_m,
                           this->origin_m, nlocal)
    );

    Kokkos::deep_copy(pc->omega.getView(), omega_host);

    Kokkos::fence();
    ippl::Comm->barrier();
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

      // claculate stream function
      IpplTimings::startTimer(SolveTimer);
      this->fsolver_m->runSolver();
      IpplTimings::stopTimer(SolveTimer);

      // calculate velocity from stream function
      IpplTimings::startTimer(PTimer);
      this->computeVelocityField();
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

    void dump() override {
      static IpplTimings::TimerRef dumpTimer = IpplTimings::getTimer("dump");
      IpplTimings::startTimer(dumpTimer);
      std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

      Inform csvout(NULL, "particles.csv", Inform::APPEND);

      for (unsigned i = 0; i < this->np_m; i++) {
        csvout << this->it_m << "," << i; 
        for (unsigned d = 0; d < Dim; d++) {
          csvout << "," << pc->R(i)[d];
        }
        csvout << "," << pc->omega(i) << endl;
      }
      dumpParticleDataPerRank();
      IpplTimings::stopTimer(dumpTimer);
       
    }

};
#endif
