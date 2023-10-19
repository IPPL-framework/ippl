#ifndef IPPL_PENNING_TRAP_MANAGER_H
#define IPPL_PENNING_TRAP_MANAGER_H

#include <memory>
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Random/Distribution.h"
#include "Random/NormalDistribution.h"
#include "Random/InverseTransformSampling.h"

using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

const char* TestName = "PenningTrap";

class PenningTrapManager : public ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>, LoadBalancer<double, 3>> {
public:
    double loadbalancethreshold_m;
    double time_m;
    PenningTrapManager()
        : ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>, LoadBalancer<double, 3>>(),totalP(0), nt(0), lbt(0), dt(0),  step_method("LeapFrog"){
    }
    Vector_t<int, Dim> nr;
    size_type totalP;
    int nt;
    double lbt;
    double dt;
    int it;
    std::string step_method;
 public:
     using ParticleContainer_t = ParticleContainer<T, Dim>;
     using FieldContainer_t = FieldContainer<T, Dim>;
     using FieldSolver_t= FieldSolver<T, Dim>;
     using LoadBalancer_t= LoadBalancer<T, Dim>;

    std::string solver;
    Vector_t<double, Dim> rmin;
    Vector_t<double, Dim> rmax;
    Vector_t<double, Dim> length;
    Vector_t<double, Dim> hr;
    double Q;
    double Bext;
    Vector_t<double, Dim> origin;
    unsigned int nrMax;
    double dxFinest;
    bool isAllPeriodic;
    bool isFirstRepartition;
    double alpha;
    double DrInv;
    double rhoNorm_m;
private:
    ippl::NDIndex<Dim> domain;
    ippl::e_dim_tag decomp[Dim];

public:
     void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }
    void post_step() override {
        // Update time
        this->time_m += this->dt;
        this->it ++;
        // wrtie solution to output file
        this->dump();
        
        Inform m("Post-step:");
        m << "Finished time step: " << this->it << " time: " << this->time_m << endl;
         
    }
    void pre_run() override {
         Inform m("Pre Run");
         for (unsigned i = 0; i < Dim; i++) {
            this->domain[i] = ippl::Index(this->nr[i]);
        }
        for (unsigned d = 0; d < Dim; ++d) {
            this->decomp[d] = ippl::PARALLEL;
        }

        this->rmin = 0;
        this->rmax = 20;

        this->length = this->rmax - this->rmin;
        this->hr = this->length / this->nr;

        this->Q    = -1562.5;
        this->Bext = 5.0;
        this->origin = this->rmin;

        this->nrMax    = 2048;  // Max grid size in our studies
        this->dxFinest = this->length[0] / this->nrMax;
        this->dt       = 0.5 * this->dxFinest;  // size of timestep
        
        this->it = 0;
        
        this->alpha = -0.5 * this->dt;
        this->DrInv = 1.0 / (1 + (std::pow((this->alpha * this->Bext), 2)));
        
        m << "Discretization:" << endl
            << "nt " << this->nt << " Np= " << this->totalP << " grid = " << this->nr << endl;
        
        this->isAllPeriodic = true;
        Mesh_t<Dim> *mesh = new Mesh_t<Dim>(this->domain, this->hr, this->origin);
        FieldLayout_t<Dim> *FL = new FieldLayout_t<Dim>(this->domain, this->decomp, this->isAllPeriodic);
        PLayout_t<T, Dim> *PL = new PLayout_t<T, Dim>(*FL, *mesh);
        
        this->pcontainer_m = std::make_shared<ParticleContainer_t>(*PL);
        this->fcontainer_m = std::make_shared<FieldContainer_t>(this->hr, this->rmin, this->rmax, this->decomp);
        this->fcontainer_m->initializeFields(*mesh, *FL);
        
        this->fsolver_m = std::make_shared<FieldSolver_t>(this->solver, this->fcontainer_m->rho_m, this->fcontainer_m->E_m);
        this->fsolver_m->initSolver();
        this->loadbalancer_m = std::make_shared<LoadBalancer_t>(this->lbt, this->fcontainer_m, this->pcontainer_m, this->fsolver_m);

        this->setParticleContainer(pcontainer_m);
        this->setFieldContainer(fcontainer_m);
        this->setFieldSolver(fsolver_m);
        this->setLoadBalancer(loadbalancer_m);
        
        this ->initializeParticles(*mesh, *FL);
        this->fcontainer_m->rho_m = 0.0;
        this->fsolver_m->runSolver();
        this->par2grid();
        this->fsolver_m->runSolver();
        this->grid2par();
        m << "Done";
    }

    void initializeParticles(Mesh_t<Dim>& mesh_m, FieldLayout_t<Dim>& FL_m){
        Inform m("Initialize Particles");

        Vector_t<double, Dim> mu, sd;
        for (unsigned d = 0; d < Dim; d++) {
            mu[d] = 0.5 * this->length[d] + this->origin[d];
        }
        sd[0] = sqrt( 0.15 * this->length[0] );
        sd[1] = sqrt( 0.05 * this->length[1] );
        sd[2] = sqrt( 0.20 * this->length[2] );
        
        using DistR_t = ippl::random::NormalDistribution<double, Dim>;
        const double parR[2*Dim] = {mu[0], sd[0], mu[1], sd[1], mu[2], sd[2]};
        DistR_t distR(parR);

        Vector_t<double, Dim> hr_m = this->hr;
        Vector_t<double, Dim> origin_m = this->origin;

        if ((this->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            this->isFirstRepartition             = true;
            const ippl::NDIndex<Dim>& lDom = FL_m.getLocalNDIndex();
            const int nghost               = this->fcontainer_m->rho_m.getNghost();
            auto rhoview                   = this->fcontainer_m->rho_m.getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", this->fcontainer_m->rho_m.getFieldRangePolicy(),
                KOKKOS_LAMBDA (const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * hr_m + origin_m;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = distR.full_pdf(xvec);
                });

            Kokkos::fence();

            this->loadbalancer_m->initializeORB(FL_m, mesh_m);
            this->loadbalancer_m->repartition(FL_m, mesh_m, this->isFirstRepartition);
        }

        // Sample particle positions:
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>( FL_m, mesh_m );
        
        size_type totalP_m = this->totalP;
        int seed = 42;
        using size_type = ippl::detail::size_type;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        using samplingR_t = ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace, DistR_t>;
        Vector_t<double, Dim> rmin_m = rmin;
        Vector_t<double, Dim> rmax_m = rmax;
        samplingR_t samplingR(distR, rmax_m, rmin_m, rlayout, totalP_m);
        size_type nlocal = samplingR.getLocalNum();
        
        this->pcontainer_m->create(nlocal);

        view_type* R_m = &this->pcontainer_m->R.getView();
        samplingR.generate(*R_m, rand_pool64);

        view_type* P_m = &this->pcontainer_m->P.getView();

        double muP[Dim] = {0.0, 0.0, 0.0};
        double sdP[Dim] = {1.0, 1.0, 1.0};
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P_m, rand_pool64, muP, sdP));

        Kokkos::fence();
        ippl::Comm->barrier();

        this->pcontainer_m->q = this->Q / this->totalP;
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
            if (this->step_method == "LeapFrog"){
                LeapFrogStep();
            }
    }
    void LeapFrogStep(){
          // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
          // Here, we assume a constant charge-to-mass ratio of -1 for
          // all the particles hence eliminating the need to store mass as
          // an attribute
          Inform m("LeapFrog");

          double alpha_m = this->alpha;
          double Bext_m = this->Bext;
          double DrInv_m = this->DrInv;
          double V0  = 30 * this->length[2];
          Vector_t<double, Dim> length_m = this->length;
          Vector_t<double, Dim> origin_m = origin;
          double dt_m = this->dt;
          std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
          std::shared_ptr<FieldContainer_t> fc = this->fcontainer_m;
          int unsigned totalP_m = this->totalP;
          int it_m = this->it;
          bool isFirstRepartition_m = false;

          m << "0" << endl;
          auto Rview = pc->R.getView();
          auto Pview = pc->P.getView();
          auto Eview = pc->E.getView();
          Kokkos::parallel_for(
               "Kick1", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
                double Eext_x = -(Rview(j)[0] - origin_m[0] - 0.5 * length_m[0])
                                * (V0 / (2 * Kokkos::pow(length_m[2], 2)));
                double Eext_y = -(Rview(j)[1] - origin_m[1] - 0.5 * length_m[1])
                                * (V0 / (2 * Kokkos::pow(length_m[2], 2)));
                double Eext_z = (Rview(j)[2] - origin_m[2] - 0.5 * length_m[2])
                                * (V0 / (Kokkos::pow(length_m[2], 2)));

                Eext_x += Eview(j)[0];
                Eext_y += Eview(j)[1];
                Eext_z += Eview(j)[2];

                Pview(j)[0] += alpha_m * (Eext_x + Pview(j)[1] * Bext_m);
                Pview(j)[1] += alpha_m * (Eext_y - Pview(j)[0] * Bext_m);
                Pview(j)[2] += alpha_m * Eext_z;
          });
          Kokkos::fence();

          m << "1" << endl;
          // drift
          pc->R = pc->R + dt_m * pc->P;

          m << "2" << endl;
          // Since the particles have moved spatially update them to correct processors
          pc->update();

          /* m << "3" << endl;
          if (loadbalancer_m->balance(totalP_m, it_m + 1)) {
                auto mesh = fc->rho_m.get_mesh();
                auto FL = fc->getLayout();
                loadbalancer_m->repartition(FL, mesh, isFirstRepartition_m);
          }
          Kokkos::fence();*/

          m << "4" << endl;
          // scatter the charge onto the underlying grid
          this->par2grid();

          m << "5" << endl;
          // Field solve
          this->fsolver_m->runSolver();

          m << "6" << endl;
          // gather E field
          this->grid2par();

          m << "7" << endl;
          auto R2view = pc->R.getView();
          auto P2view = pc->P.getView();
          auto E2view = pc->E.getView();
          Kokkos::parallel_for(
             "Kick2", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
             double Eext_x = -(R2view(j)[0] - origin_m[0] - 0.5 * length_m[0])
                           * (V0 / (2 * Kokkos::pow(length_m[2], 2)));
             double Eext_y = -(R2view(j)[1] - origin_m[1] - 0.5 * length_m[1])
                            * (V0 / (2 * Kokkos::pow(length_m[2], 2)));
             double Eext_z = (R2view(j)[2] - origin_m[2] - 0.5 * length_m[2])
                           * (V0 / (Kokkos::pow(length_m[2], 2)));

             Eext_x += Eview(j)[0];
             Eext_y += Eview(j)[1];
             Eext_z += Eview(j)[2];

             P2view(j)[0] = DrInv_m * (P2view(j)[0] + alpha_m * (Eext_x + P2view(j)[1] * Bext_m + alpha_m * Bext_m * Eext_y));
             P2view(j)[1] = DrInv_m * (P2view(j)[1] + alpha_m * (Eext_y - P2view(j)[0] * Bext_m - alpha_m * Bext_m * Eext_x));
             P2view(j)[2] += alpha_m * Eext_z;
          });
          Kokkos::fence();
          m << "8" << endl;
    }

    void par2grid() override {
        scatterCIC();
    }

    void grid2par() override {
        gatherCIC();
    }

    void gatherCIC() {
        using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        Base::particle_position_type *E_p = &this->pcontainer_m->E;
        Base::particle_position_type *R_m = &this->pcontainer_m->R;
        VField_t<T, Dim> *E_f = &this->fcontainer_m->E_m;
        gather(*E_p, *E_f, *R_m);
    }

    void scatterCIC() {
        Inform m("scatter ");
        this->fcontainer_m->rho_m = 0.0;

        using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        ippl::ParticleAttrib<double> *q_m = &this->pcontainer_m->q;
        Base::particle_position_type *R_m = &this->pcontainer_m->R;
        Field_t<Dim> *rho_m = &this->fcontainer_m->rho_m;
        double Q_m = this->Q;
        Vector_t<double, Dim> rmin_m = rmin;
        Vector_t<double, Dim> rmax_m = rmax;
        Vector_t<double, Dim> hr_m = hr;

        scatter(*q_m, *rho_m, *R_m);
        m << std::fabs((Q_m - (*rho_m).sum()) / Q_m) << endl;

        size_type Total_particles = 0;
        size_type local_particles = pcontainer_m->getLocalNum();

        MPI_Reduce(&local_particles, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        double cellVolume =
            std::reduce(hr_m.begin(), hr_m.end(), 1., std::multiplies<double>());
        (*rho_m) = (*rho_m) / cellVolume;

        this->rhoNorm_m = norm(*rho_m);
        
        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->stype_m != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax_m[d] - rmin_m[d];
            }
            *rho_m = *rho_m - (Q_m / size);
        }
    }

    void dump() {} //{ dumpData(); }
    
    void dumpData() {
        auto Pview = this->pcontainer_m->P.getView();
        double kinEnergy = 0.0;
        double potEnergy = 0.0;
        this->fcontainer_m->rho_m     = dot(this->fcontainer_m->E_m, this->fcontainer_m->E_m);
        potEnergy = 0.5 * this->hr[0] * this->hr[1] * this->hr[2] * this->fcontainer_m->rho_m.sum();

        Kokkos::parallel_reduce(
            "Particle Kinetic Energy", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(Pview(i), Pview(i)).apply();
                valL += myVal;
            },
            Kokkos::Sum<double>(kinEnergy));

        kinEnergy *= 0.5;
        double gkinEnergy = 0.0;

        MPI_Reduce(&kinEnergy, &gkinEnergy, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        const int nghostE = this->fcontainer_m->E_m.getNghost();
        auto Eview        = this->fcontainer_m->E_m.getView();
        Vector_t<T, Dim> normE;

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        for (unsigned d = 0; d < Dim; ++d) {
            T temp = 0.0;
            ippl::parallel_reduce(
                "Vector E reduce", ippl::getRangePolicy(Eview, nghostE),
                KOKKOS_LAMBDA(const index_array_type& args, T& valL) {
                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    T myVal = std::pow(ippl::apply(Eview, args)[d], 2);
                    valL += myVal;
                },
                Kokkos::Sum<T>(temp));
            Kokkos::fence();
            T globaltemp          = 0.0;
            MPI_Datatype mpi_type = get_mpi_datatype<T>(temp);
            MPI_Reduce(&temp, &globaltemp, 1, mpi_type, MPI_SUM, 0, ippl::Comm->getCommunicator());
            normE[d] = std::sqrt(globaltemp);
            ippl::Comm->barrier();
        }

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/ParticleField_";
            fname << ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, Potential energy, Kinetic energy, Total energy, Rho_norm2";
                for (unsigned d = 0; d < Dim; d++) {
                    csvout << ", E" << static_cast<char>((Dim <= 3 ? 'x' : '1') + d) << "_norm2";
                }
                csvout << endl;
            }

            csvout << time_m << " " << potEnergy << " " << gkinEnergy << " "
                   << potEnergy + gkinEnergy << " " << this->rhoNorm_m << " ";
            for (unsigned d = 0; d < Dim; d++) {
                csvout << normE[d] << " ";
            }
            csvout << endl;
            csvout.flush();
        }
        ippl::Comm->barrier();
    }
    
    template <typename View>
    void dumpLandau(const View& Eview) {
        const int nghostE = fcontainer_m->E_m.getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localEx2 = 0, localExNorm = 0;
        ippl::parallel_reduce(
            "Ex stats", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& E2, double& ENorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(Eview, args)[0];
                double e2  = Kokkos::pow(val, 2);
                E2 += e2;

                double norm = Kokkos::fabs(ippl::apply(Eview, args)[0]);
                if (norm > ENorm) {
                    ENorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

        double globaltemp = 0.0;
        MPI_Reduce(&localEx2, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());
        double fieldEnergy =
            std::reduce(fcontainer_m->hr_m.begin(), fcontainer_m->hr_m.end(), globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        MPI_Reduce(&localExNorm, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, ippl::Comm->getCommunicator());

        
        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << ippl::Comm->size();
            fname<<"_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (time_m == 0.0) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
