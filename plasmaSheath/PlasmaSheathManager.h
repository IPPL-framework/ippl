#ifndef IPPL_PLASMA_SHEATH_MANAGER_H
#define IPPL_PLASMA_SHEATH_MANAGER_H

#include <memory>

#include "AlpineManager.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_typeR = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using view_typeP = typename ippl::detail::ViewType<ippl::Vector<double, 3>, 1>::view_type;
using view_typeQ = typename ippl::detail::ViewType<double, 1>::view_type;

template <typename T, unsigned Dim>
class PlasmaSheathManager : public AlpineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;

    PlasmaSheathManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                         std::string& solver_, std::string& stepMethod_,
                         double L = 1, T phiWall = 1, Vector_t<T, 3> Bext = {0,0,0})
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_) {
            setup(L, phiWall, Bext);
        }

    PlasmaSheathManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                         std::string& solver_, std::string& stepMethod_,
                         std::vector<std::string> preconditioner_params_, 
                         double L = 1, T phiWall = 1, Vector_t<T, 3> Bext = {0,0,0})
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_,
                                preconditioner_params_) {
            setup(L, phiWall, Bext);
        }

    ~PlasmaSheathManager() {}

    void setup(T L, T phiWall, Vector_t<T, 3> Bext) {
        Inform m("Setup");

        if ((this->solver_m != "CG") && (this->solver_m != "PCG")) {
            throw IpplException("PlasmaSheath",
                                "Open boundaries solver incompatible with this simulation!");
        }

        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);

        this->rmin_m   = 0.0;
        this->rmax_m   = L; // L = size of domain
        this->origin_m = this->rmin_m;
        this->hr_m     = L / this->nr_m;

        this->phiWall_m = phiWall; // Dirichlet BC for phi at wall
        this->Bext_m = Bext; // External magnetic field

        this->dt_m = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m   = 0;
        this->time_m = 0.0;

        // total charge is 0 since quasineutral; 
        // if total no. of particles is odd, will have 1 electron more
        this->Q_m = 0.0 - (this->totalP_m % 2);

        m << "Discretization:" << endl
          << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid=" << this->nr_m
          << " dt=" << this->dt_m << endl;
    }

    void pre_run() override {
        Inform m("Pre Run");

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
            this->phiWall_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        this->fcontainer_m->initializeFields(this->solver_m);

        if (this->getSolver() == "PCG") {
            this->setFieldSolver(std::make_shared<FieldSolver_t>(
                this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(),
                &this->fcontainer_m->getPhi(), this->fcontainer_m->getPhiWall(),
                this->preconditioner_params_m));
        } else {
            this->setFieldSolver(std::make_shared<FieldSolver_t>(
                this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(),
                &this->fcontainer_m->getPhi(), this->fcontainer_m->getPhiWall()));
        }

        this->fsolver_m->initSolver();

        this->setLoadBalancer(std::make_shared<LoadBalancer_t>(
            this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m));

        initializeParticles();

        static IpplTimings::TimerRef DummySolveTimer = IpplTimings::getTimer("solveWarmup");
        IpplTimings::startTimer(DummySolveTimer);

        this->fcontainer_m->getRho() = 0.0;
        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(DummySolveTimer);

        this->par2grid();

        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        IpplTimings::startTimer(SolveTimer);

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(SolveTimer);

        this->grid2par();

        this->dump();

        m << "Done";
    }

    void initializeParticles() {
        Inform m("Initialize Particles");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particleCreation");

        IpplTimings::startTimer(particleCreation);

        const double pi = Kokkos::numbers::pi_v<T>;

        // divide particles equally among ranks
        size_type totalP = this->totalP_m;
        size_type nlocal = totalP / ippl::Comm->size();
        int rest = (int)(totalP - nlocal * ippl::Comm->size());
        if (ippl::Comm->rank() < rest)
            ++nlocal;

        // create particles on each rank
        this->pcontainer_m->create(nlocal);

        // TODO: put correct physical values
        // there are two species: electrons and ions
        double q_i = 1.0; // ion charge
        double q_e = -1.0; // electron charge
        T m_i = 1836; // ion mass
        T m_e = 1; // electron mass
        
        // electron distribution: Maxwellian (normal distribution)
        Vector_t<double, 3> v0 = {0.0, 0.0, 0.0}; // avg velocity
        double T_e = 1; // temperature
        double n_e = 1; // n_e

        double prefactor_e = n_e * m_e / (2 * pi * T_e);
        double stdeviation_e = Kokkos::sqrt(T_e/m_e); 

        double muE[3] = {v0[0], v0[1], v0[2]};
        double sdE[3] = {stdeviation_e, stdeviation_e, stdeviation_e};

        // ion distribution: (normal distribution)
        double parallel_v = 1; // v_parallel
        double v_thi = 1; // thermal velocity of ions
        double K = 1; // constant K

        double prefactor_i = K * Kokkos::sqrt(2 * pi) * parallel_v * parallel_v / v_thi;
        double stdeviation_i = v_thi; 

        double muI[3] = {0.0, 0.0, 0.0};
        double sdI[3] = {stdeviation_i, stdeviation_i, stdeviation_i};

        // assign the particle attributes

        // particles are initially sampled at x=0 (bulk plasma)
        this->pcontainer_m->R = 0;

        // charge and mass are species dependent
        view_typeQ Qview = this->pcontainer_m->q.getView();
        view_typeQ Mview = this->pcontainer_m->m.getView();

        // velocity is sampled from the species' respective distribution
        view_typeP Pview = this->pcontainer_m->P.getView();

        int seed = 42;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        // half the particles are ions, half are electrons
        // we do this approximate division by checking whether even or odd ID
        Kokkos::parallel_for("Set attributes", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i) {
                bool odd = (i % 2);
                
                Qview(i) = ((!odd) * q_e) + (odd * q_i);
                Mview(i) = ((!odd) * m_e) + (odd * m_i);

                auto rand_gen = rand_pool64.get_state();

                // accept only those which have velocity_x > 0 (moving towards wall)
                double v_x = 0.0;
                while (v_x <= 0.0) {
                    v_x = (!odd) * prefactor_e * (muE[0] + sdE[0] * rand_gen.normal(0.0, 1.0))
                          + odd * prefactor_i * (muI[0] + sdI[0] * rand_gen.normal(0.0, 1.0));
                }
                Pview(i)[0] = v_x;
                for (unsigned d = 1; d < 3; ++d) {
                    Pview(i)[d] = (!odd) * prefactor_e * (muE[d] + sdE[d] * rand_gen.normal(0.0, 1.0))
                                  + odd * prefactor_i * (muI[d] + sdI[d] * rand_gen.normal(0.0, 1.0));
                }

                rand_pool64.free_state(rand_gen);
            });
        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        m << "particles created and initial conditions assigned " << endl;
        this->dump();
    }

    void advance() override {
        if (this->stepMethod_m == "Boris") {
            BorisStep();
        } else {
            throw IpplException(TestName, "Step method is not set/recognized!");
        }
    }

    void BorisStep() {
        // Boris pusher: half-step, kick, rotation, half-step
        static IpplTimings::TimerRef ETimer              = IpplTimings::getTimer("kickVelocity");
        static IpplTimings::TimerRef BTimer              = IpplTimings::getTimer("rotateVelocity");
        static IpplTimings::TimerRef RTimer              = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer         = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer          = IpplTimings::getTimer("solve");

        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        // push position (half step)
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + (0.5 * dt * pc->P);
        IpplTimings::stopTimer(RTimer);

        // remove particles which have hit the wall (either side of domain)
        // and resample to insert them from plasma boundary
        
        // TODO: put correct physical values
        const double pi = Kokkos::numbers::pi_v<T>;
        T m_e = 1; // electron mass
        
        // electron distribution: Maxwellian (normal distribution)
        Vector_t<double, 3> v0 = {0.0, 0.0, 0.0}; // avg velocity
        double T_e = 1; // temperature
        double n_e = 1; // n_e

        double prefactor_e = n_e * m_e / (2 * pi * T_e);
        double stdeviation_e = Kokkos::sqrt(T_e/m_e); 

        double muE[3] = {v0[0], v0[1], v0[2]};
        double sdE[3] = {stdeviation_e, stdeviation_e, stdeviation_e};

        // ion distribution: (normal distribution)
        double parallel_v = 1; // v_parallel
        double v_thi = 1; // thermal velocity of ions
        double K = 1; // constant K

        double prefactor_i = K * Kokkos::sqrt(2 * pi) * parallel_v * parallel_v / v_thi;
        double stdeviation_i = v_thi; 

        double muI[3] = {0.0, 0.0, 0.0};
        double sdI[3] = {stdeviation_i, stdeviation_i, stdeviation_i};

        auto rmin = this->rmin_m;
        auto rmax = this->rmax_m;

        view_typeR Rview = this->pcontainer_m->R.getView();
        view_typeP Pview = this->pcontainer_m->P.getView();

        int seed = 42;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        Kokkos::parallel_for("Remove particle", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i) {
                bool outside = false;

                for (unsigned int d = 0; d < Dim; ++d) {
                    if ((Rview(i)[d] > rmax[d]) || (Rview(i)[d] < rmin[d])) {
                        outside = true;
                    }
                }

                if (outside) {
                    Rview(i) = 0;
                    bool odd = (i % 2);
                    auto rand_gen = rand_pool64.get_state();

                    // accept only those which have velocity_x > 0 (moving towards wall)
                    double v_x = 0.0;
                    while (v_x <= 0.0) {
                        v_x = (!odd) * prefactor_e * (muE[0] + sdE[0] * rand_gen.normal(0.0, 1.0))
                              + odd * prefactor_i * (muI[0] + sdI[0] * rand_gen.normal(0.0, 1.0));
                    }
                    Pview(i)[0] = v_x;
                    for (unsigned d = 1; d < 3; ++d) {
                        Pview(i)[d] = (!odd) * prefactor_e * (muE[d] + sdE[d] * rand_gen.normal(0.0, 1.0))
                                      + odd * prefactor_i * (muI[d] + sdI[d] * rand_gen.normal(0.0, 1.0));
                    }

                    rand_pool64.free_state(rand_gen);
                }
            });
        Kokkos::fence();
        ippl::Comm->barrier();

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        size_type totalP        = this->totalP_m;
        int it                  = this->it_m;
        bool isFirstRepartition = false;
        if (this->loadbalancer_m->balance(totalP, it + 1)) {
            IpplTimings::startTimer(domainDecomposition);
            auto* mesh = &fc->getRho().get_mesh();
            auto* FL   = &fc->getFL();
            this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            IpplTimings::stopTimer(domainDecomposition);
        }

        // scatter the charge onto the underlying grid
        this->par2grid();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        this->grid2par();

        // kick (velocity update)

        // half acceleration
        IpplTimings::startTimer(ETimer);
        pc->P = pc->P + 0.5 * dt * (pc->q/pc->m) * pc->E;
        IpplTimings::stopTimer(ETimer);

        // rotation
        IpplTimings::startTimer(BTimer);

        Vector_t<T, 3> Bext = this->Bext_m;
        view_typeQ Qview = this->pcontainer_m->q.getView();
        view_typeQ Mview = this->pcontainer_m->m.getView();

        Kokkos::parallel_for("Apply rotation", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i) {
                Vector_t<T, 3> const t = 0.5 * dt * (Qview(i)/Mview(i)) * Bext;
                Vector_t<T, 3> const w = Pview(i) + cross(Pview(i), t).apply();
                Vector_t<T, 3> const s = (2.0 / (1 + dot(t, t).apply())) * t;
                Pview(i) = Pview(i) + cross(w, s);
            });
        IpplTimings::stopTimer(BTimer);

        // half acceleration
        IpplTimings::startTimer(ETimer);
        pc->P = pc->P + 0.5 * dt * (pc->q/pc->m) * pc->E;
        IpplTimings::stopTimer(ETimer);

        // push position (half step)
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + (0.5 * dt * pc->P);
        IpplTimings::stopTimer(RTimer);

        // since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        this->dump();
    }

    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);

        // dump for each particle the particles attributes (q, m, R, P)
        dumpParticleData();

        IpplTimings::stopTimer(dumpDataTimer);
    }

    void dumpParticleData() {
        typename ParticleAttrib<Vector_t<T, Dim>>::HostMirror R_host = this->pcontainer_m->R.getHostMirror();
        typename ParticleAttrib<Vector_t<T, 3>>::HostMirror P_host = this->pcontainer_m->P.getHostMirror();
        typename ParticleAttrib<double>::HostMirror q_host = this->pcontainer_m->q.getHostMirror();
        typename ParticleAttrib<T>::HostMirror m_host = this->pcontainer_m->m.getHostMirror();

        Kokkos::deep_copy(R_host, this->pcontainer_m->R.getView());
        Kokkos::deep_copy(P_host, this->pcontainer_m->P.getView());
        Kokkos::deep_copy(q_host, this->pcontainer_m->q.getView());
        Kokkos::deep_copy(m_host, this->pcontainer_m->m.getView());

        std::stringstream pname;
        pname << "data/ParticleIC_";
        pname << ippl::Comm->rank();
        pname << ".csv";
        Inform pcsvout(NULL, pname.str().c_str(), Inform::APPEND, ippl::Comm->rank());
        pcsvout.precision(10);
        pcsvout.setf(std::ios::scientific, std::ios::floatfield);

        pcsvout << "q, m, R_x, V_x, V_y, V_z" << endl;

        for (size_type i = 0; i < this->pcontainer_m->getLocalNum(); i++) {
            pcsvout << q_host(i) << " ";
            pcsvout << m_host(i) << " ";

            for (unsigned d = 0; d < Dim; d++) {
                pcsvout << R_host(i)[d] << " ";
            }

            for (unsigned d = 0; d < 3; d++) {
                pcsvout << P_host(i)[d] << " ";
            }
            pcsvout << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
