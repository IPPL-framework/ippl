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
class FSLManager : public AlvineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>; 
    using LoadBalancer_t      = LoadBalancer<T, Dim>;    
    FieldLayout_t<Dim> FL_m;    // Store the field layout
    Mesh_t<Dim> mesh_m;          // Store the mesh

    // Constructor declaration
    FSLManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_, 
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

    ~FSLManager() {}

void pre_run() override {
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

      //initializeParticles();

      //this->par2grid();

      initializeGridVorticity();
      auto omega0 = this->fcontainer_m->getOmegaField().deepCopy();
      this->fsolver_m->runSolver();
      this->computeVelocityField();
      logEnergyDiagnostics();
      Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega0.getView());
      logEnstrophyDiagnostics();
      logDivergenceDiagnostics();
//      this->grid2par();
	double omega_init = computeOmegaL2();

if (ippl::Comm->rank() == 0) {
    Inform m("debug ");
    m << "omega L2 after initialization = " << omega_init << endl;
}

    }

 
void initializeVirtualParticles(){
    clearVirtualParticles();
    auto* FL = &this->fcontainer_m->getFL();
    auto local = FL->getLocalNDIndex();

    int i0 = local[0].first();
    int i1 = local[0].last();
    int j0 = local[1].first();
    int j1 = local[1].last();

    size_type nx = i1 - i0 + 1;
    size_type ny = j1 - j0 + 1;
    size_type nlocal = nx * ny;
    const int nghost = this->fcontainer_m->getOmegaField().getNghost();

    auto pc = this->pcontainer_m;
    pc->create(nlocal);

    auto R_view = pc->R.getView();
    auto omega_p = pc->omega.getView();
    auto omega_g = this->fcontainer_m->getOmegaField().getView();

    Vector_t<double, Dim> rmin = this->rmin_m;
    Vector_t<double, Dim> hr   = this->hr_m;
    double dA = hr[0] * hr[1];

    Kokkos::parallel_for(
        "init_virtual_particles_from_grid",
        nlocal,
        KOKKOS_LAMBDA(const int p) {
            int ii = p % nx;
            int jj = p / nx;

            int i = i0 + ii;
            int j = j0 + jj;
            int li = ii + nghost;
            int lj = jj + nghost;

            R_view(p)[0] = rmin[0] + (i+0.5) * hr[0];
            R_view(p)[1] = rmin[1] + (j+0.5) * hr[1];

            omega_p(p) = omega_g(li, lj) * dA;
        }
    );

    Kokkos::fence();
}

double computeOmegaL2() {
    auto& omegaField = this->fcontainer_m->getOmegaField();
    auto omega = omegaField.getView();
    double local = 0.0;
    const int nghost = omegaField.getNghost();

    Kokkos::parallel_reduce(
        "omega_l2",
        ippl::getRangePolicy(omega, nghost),
        KOKKOS_LAMBDA(const int i, const int j, double& sum) {
            sum += omega(i, j) * omega(i, j);
        },
        local
    );

    double global = 0.0;
    ippl::Comm->reduce(local, global, 1, std::plus<double>());

    return std::sqrt(global);
}

void logPushDebug() {
    auto pc = this->pcontainer_m;

    auto P_view = pc->P.getView();

    double dt = this->dt_m;
    double dx = this->hr_m[0];

    double localMaxVel = 0.0;
    double localMaxDx  = 0.0;
    double localMaxDy  = 0.0;

    Kokkos::parallel_reduce(
        "push_debug",
        pc->getLocalNum(),
        KOKKOS_LAMBDA(const int p,
                      double& maxVel,
                      double& maxDx,
                      double& maxDy) {

            double ux = P_view(p)[0];
            double uy = P_view(p)[1];

            double vel = sqrt(ux * ux + uy * uy);

            double dispX = fabs(ux * dt);
            double dispY = fabs(uy * dt);

            if (vel   > maxVel) maxVel = vel;
            if (dispX > maxDx)  maxDx  = dispX;
            if (dispY > maxDy)  maxDy  = dispY;

        },
        Kokkos::Max<double>(localMaxVel),
        Kokkos::Max<double>(localMaxDx),
        Kokkos::Max<double>(localMaxDy)
    );

    double globalMaxVel = localMaxVel;
    double globalMaxDx  = localMaxDx;
    double globalMaxDy  = localMaxDy;

    if (ippl::Comm->rank() == 0) {
        Inform m("push_debug ");

        m << "step = " << this->it_m
          << ", max velocity = " << globalMaxVel
          << ", max |dx| = " << globalMaxDx
          << ", max |dy| = " << globalMaxDy
          << ", max displacement/dx = "
          << std::max(globalMaxDx, globalMaxDy) / dx
          << endl;
    }
}

void logVelocityRatioDebug() {
    auto pc = this->pcontainer_m;
    auto P_view = pc->P.getView();

    double localMaxRatio = 0.0;
    double localSumRatio = 0.0;
    size_type localN = pc->getLocalNum();

    Kokkos::parallel_reduce(
        "velocity_ratio_debug",
        localN,
        KOKKOS_LAMBDA(const int p, double& maxRatio, double& sumRatio) {
            double ux = fabs(P_view(p)[0]);
            double uy = fabs(P_view(p)[1]);

            double ratio = uy / (ux + 1.0e-30);

            if (ratio > maxRatio) maxRatio = ratio;
            sumRatio += ratio;
        },
        Kokkos::Max<double>(localMaxRatio),
        localSumRatio
    );

    double globalMaxRatio = 0.0;
    double globalSumRatio = 0.0;
    size_type globalN = 0;

    MPI_Allreduce(&localMaxRatio, &globalMaxRatio, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localSumRatio, &globalSumRatio, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localN, &globalN, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (ippl::Comm->rank() == 0) {
        Inform m("velocity_ratio ");
        m << "step = " << this->it_m
          << ", max |uy|/|ux| = " << globalMaxRatio
          << ", avg |uy|/|ux| = " << globalSumRatio / globalN
          << endl;
    }
}

void initializeGridVorticity() {
    auto& omegaField = this->fcontainer_m->getOmegaField();
    auto omega_view = omegaField.getView();

    auto localND = this->fcontainer_m->getFL().getLocalNDIndex();

    int i0 = localND[0].first();
    int i1 = localND[0].last();
    int j0 = localND[1].first();
    int j1 = localND[1].last();
    const int nghost = omegaField.getNghost();

    Vector_t<double, Dim> rmin = this->rmin_m;
    Vector_t<double, Dim> hr   = this->hr_m;

    Kokkos::parallel_for(
        "initialize_grid_vorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({nghost, nghost},
                                               {nghost + i1 - i0 + 1,
                                                nghost + j1 - j0 + 1}),
        KOKKOS_LAMBDA(const int li, const int lj) {
            int i = i0 + li - nghost;
            int j = j0 + lj - nghost;

            double xc = 5.0;
            double yc = 5.0;
            double sigma = 0.5;

            double x = rmin[0] + (i + 0.5) * hr[0];
            double y = rmin[1] + (j + 0.5) * hr[1];

            double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);

            omega_view(li, lj) = exp(-r2 / (2.0 * sigma * sigma));
        }
    );

    Kokkos::fence();
}

void pushVirtualParticlesForward() {
    auto pc = this->pcontainer_m;

    // 1. Gather velocity from grid to particles
    this->grid2par();   // fills pc->P using uField

    // Debug before push: checks expected displacement = P * dt
    logPushDebug();
    logVelocityRatioDebug();
    // 2. Move particles forward
    auto R_view = pc->R.getView();
    auto P_view = pc->P.getView();

    double dt = this->dt_m;

    Kokkos::parallel_for(
        "push_virtual_particles_forward",
        pc->getLocalNum(),
        KOKKOS_LAMBDA(const int p) {
            R_view(p) = R_view(p) + P_view(p) * dt;
        }
    );

    Kokkos::fence();

    // 3. Apply particle boundary conditions / update ownership
    pc->update();
}

void clearVirtualParticles() {
    auto pc = this->pcontainer_m;
    size_type nlocal = pc->getLocalNum();

    if (nlocal == 0) {
        return;
    }

    Kokkos::View<bool*> invalid("invalid", nlocal);

    Kokkos::parallel_for(
        "mark_all_particles_invalid",
        nlocal,
        KOKKOS_LAMBDA(const int p) {
            invalid(p) = true;
        }
    );

    Kokkos::fence();

    pc->destroy(invalid, nlocal);
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
      advectForward();     
    }

void advectForward() {
    static IpplTimings::TimerRef PTimer =
        IpplTimings::getTimer("pushVelocity");
    static IpplTimings::TimerRef RTimer =
        IpplTimings::getTimer("pushPosition");
    static IpplTimings::TimerRef SolveTimer =
        IpplTimings::getTimer("solve");
    static IpplTimings::TimerRef par2gridTimer =
        IpplTimings::getTimer("par2grid");

    double omega_before = computeOmegaL2();
    auto omega_n = this->fcontainer_m->getOmegaField().deepCopy();

    // 1. Compute velocity u^n from omega^n
    // The FFT solver writes the Poisson solution into omegaField. Restore the
    // saved vorticity before creating/remapping virtual particles.
    IpplTimings::startTimer(SolveTimer);
    this->fsolver_m->runSolver();
    IpplTimings::stopTimer(SolveTimer);

    IpplTimings::startTimer(PTimer);
    this->computeVelocityField();
    IpplTimings::stopTimer(PTimer);
    Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega_n.getView());

    // 2. Create virtual particles from omega^n
    initializeVirtualParticles();

    // 3. Push particles using u^n
    IpplTimings::startTimer(RTimer);
    pushVirtualParticlesForward();
    IpplTimings::stopTimer(RTimer);

    // 4. Scatter particles to form omega^{n+1}
    IpplTimings::startTimer(par2gridTimer);
    this->par2grid();
    IpplTimings::stopTimer(par2gridTimer);

    // 5. Delete temporary particles
    clearVirtualParticles();

    double omega_after = computeOmegaL2();

    if (ippl::Comm->rank() == 0) {
        Inform m("debug ");
        m << "omega L2 before = " << omega_before
          << ", omega L2 after = " << omega_after << endl;
    }

    // 6. Compute velocity from omega^{n+1} for diagnostics
    auto omega_np1 = this->fcontainer_m->getOmegaField().deepCopy();

    IpplTimings::startTimer(SolveTimer);
    this->fsolver_m->runSolver();
    IpplTimings::stopTimer(SolveTimer);

    IpplTimings::startTimer(PTimer);
    this->computeVelocityField();

    logEnergyDiagnostics();
    Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega_np1.getView());
    logEnstrophyDiagnostics();
    logDivergenceDiagnostics();

    IpplTimings::stopTimer(PTimer);
}
};
#endif
