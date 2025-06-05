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
#include "input.h"

#ifdef IPPL_ENABLE_CATALYST
#include <optional>
#include "Stream/InSitu/CatalystAdaptor.h"
#endif

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
    using RNG                 = typename Kokkos::Random_XorShift64_Pool<>;

    struct ParticleGen {
        RNG rand_pool64;
        // need to save these locally as not constexpr
        // so need a local copy for device code
        double v_th_e;
        double v_trunc_e;

        enum Species {
            Electrons,
            Ions
        };

        ParticleGen(double v_th_e_, double v_trunc_e_)
            : rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()))
            , v_th_e(v_th_e_)
            , v_trunc_e(v_trunc_e_) {}

        KOKKOS_FUNCTION Vector<T, 3> fieldaligned_to_wallaligned(double vpar, double vperpx,
                                                                 double vperpy) const {
            return {
                vperpx * Kokkos::cos(params::alpha) - vpar * Kokkos::sin(params::alpha),
                vperpy,
                vperpx * Kokkos::sin(params::alpha) + vpar * Kokkos::cos(params::alpha),
            };
        }

        KOKKOS_FUNCTION Vector<T, 3> sample_v3(Species s) const {
            RNG::generator_type rand_gen = rand_pool64.get_state();
            Vector<T, 3> v3;

            while (true) {
                // TODO: use samples from the Gamma distribution generated using the normal
                // distribution samples
                // 1. sample in field-aligned coordinates
                // 1.a. sample vpar from the modified half-maxwellian
                // note that by coincidence, the normalization constant for beta = 0 and beta = 2
                // (i.e. vpar² prefactor) are the same, and evaluate to 2/√(2π)
                const double stdpar  = s == Electrons ? v_th_e : params::v_th_i,
                             v_trunc = s == Electrons ? v_trunc_e : params::v_trunc_i;

                double vpar;
                while (true) {
                    vpar           = rand_gen.normal(0.0, stdpar);
                    const double R = double(vpar > 0.0) * double(vpar < v_trunc) * 2.0
                                     * ((s == Electrons) ? 1.0 : vpar * vpar / (v_trunc * v_trunc));
                    if (rand_gen.drand(0.0, 1.0) < R)
                        break;
                }

                // 1.b. sample vperp coordinates
                const double stdperp =
                    s == Electrons ? v_th_e : params::v_th_i * params::nu;
                const double vperpx = rand_gen.normal(0.0, stdperp),
                             vperpy = rand_gen.normal(0.0, stdperp);

                // 2. convert to wall-aligned coordinates
                v3 = fieldaligned_to_wallaligned(vpar, vperpx, vperpy);

                // 3. only keep velocities for which v_x < 0 and v_x > -v_trunc (for the CFL
                // condition) !!
                if (v3[0] < 0.0 && v3[0] > -v_trunc)
                    break;
            }

            rand_pool64.free_state(rand_gen);
            return v3;
        }

        KOKKOS_FUNCTION Vector<T, 3> generate_ion() const { return sample_v3(Ions); }

        KOKKOS_FUNCTION Vector<T, 3> generate_electron() const { return sample_v3(Electrons); }
    };

    int n_timeavg;

    PlasmaSheathManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                        std::string& solver_, std::string& stepMethod_)
        : n_timeavg(1)
        , AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_) {
        setup();
    }

    PlasmaSheathManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                        std::string& solver_, std::string& stepMethod_,
                        std::vector<std::string> preconditioner_params_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_,
                                preconditioner_params_) {
        setup();
    }

    ~PlasmaSheathManager() {}

    void setup() {
        Inform m("Setup");

        if ((this->solver_m != "CG") && (this->solver_m != "PCG")) {
            throw IpplException("PlasmaSheath",
                                "Open boundaries solver incompatible with this simulation!");
        }

        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);

        // the particles are spawned at x=L (injection from bulk plasma),
        // and x=0 is the wall
        this->rmin_m   = 0.0;
        this->rmax_m   = params::L;  // L = size of domain
        this->origin_m = this->rmin_m;
        this->hr_m     = params::dx;

        this->phiWall_m = params::phi0;  // Dirichlet BC for phi at wall (x=0)

        // normalized B-field - vector for direction
        this->Bext_m = {-Kokkos::cos(params::alpha), 0.0,
                        Kokkos::sin(params::alpha)};  // External magnetic field

        this->dt_m   = params::dt;
        this->it_m   = 0;
        this->time_m = 0.0;

        if (params::kinetic_electrons) {
            // total charge is 0 since quasineutral;
            // if total no. of particles is odd, will have 1 electron more
            this->Q_m = 0.0 - params::Z_e * (this->totalP_m % 2);
        } else {
            this->Q_m = this->totalP_m * params::Z_i;
        }

        m << "Discretization:" << endl
          << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid=" << this->nr_m
          << " dt=" << this->dt_m << " kinetic electrons? " << params::kinetic_electrons << endl;
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

        if (!params::kinetic_electrons) {
            // is the electrons are adiabatic, then we have a background
            // charge density field which is given by exp(phi) where
            // phi is the previous iteration's solution (electric potential)
            this->fcontainer_m->getRho() =
                this->fcontainer_m->getRho()
                + exp(this->fcontainer_m->getPhi()) * params::Z_e * params::n_e0;
        }
        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(SolveTimer);

        this->grid2par();

        // save the rho and phi for computation for the rolling average of the fields
        resetPlasmaAverage();

        // dump particle ICs
        this->dump();

        m << "Done";
    }

    void initializeParticles() {
        Inform m("Initialize Particles");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particleCreation");

        IpplTimings::startTimer(particleCreation);

        // divide particles equally among ranks
        size_type totalP = this->totalP_m;
        size_type nlocal = totalP / ippl::Comm->size();
        int rest         = (int)(totalP - nlocal * ippl::Comm->size());
        if (ippl::Comm->rank() < rest)
            ++nlocal;

        // create particles on each rank
        this->pcontainer_m->create(nlocal);

        // particle velocity sampler
        ParticleGen pgen(params::v_th_e, params::v_trunc_e);

        // particles are initially sampled at x = L (bulk plasma)
        // the wall is at x = 0
        this->pcontainer_m->R = this->rmax_m;

        if (params::kinetic_electrons) {
            // charge and mass are species dependent
            view_typeQ Qview = this->pcontainer_m->q.getView();
            view_typeQ Mview = this->pcontainer_m->m.getView();

            // velocity is sampled from the species' respective distribution
            view_typeP Pview = this->pcontainer_m->P.getView();

            // TODO check what limits of velocity to put on vy and vz
            // half the particles are ions, half are electrons
            // we do this approximate division by checking whether even or odd ID
            Kokkos::parallel_for(
                "Set attributes", this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const int i) {
                    bool odd = (i % 2);

                    Qview(i) = ((!odd) * params::Z_e) + (odd * params::Z_i);
                    Mview(i) = ((!odd) * params::m_e) + (odd * params::m_i);

                    // accept only those which have velocity_x > 0 (moving towards wall)
                    if (odd) {
                        Pview(i) = pgen.generate_ion();
                    } else {
                        Pview(i) = pgen.generate_electron();
                    }
                });
        } else {
            // single species: ions
            // adiabatic electrons are taken care of in the fieldsolver
            this->pcontainer_m->q = params::Z_i;
            this->pcontainer_m->m = params::m_i;

            // velocity is sampled from the ion distribution
            view_typeP Pview = this->pcontainer_m->P.getView();
            Kokkos::parallel_for(
                "Set attributes", this->pcontainer_m->getLocalNum(),
                KOKKOS_LAMBDA(const int i) { Pview(i) = pgen.generate_ion(); });
        }
        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        m << "particles created and initial conditions assigned " << endl;
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

        // particle velocity sampler
        ParticleGen pgen(params::v_th_e, params::v_trunc_e);

        auto rmin = this->rmin_m;
        auto rmax = this->rmax_m;

        view_typeR Rview = this->pcontainer_m->R.getView();
        view_typeP Pview = this->pcontainer_m->P.getView();

        int seed = 42;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        Kokkos::parallel_for(
            "Remove particle", this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const int i) {
                bool outside = false;
                for (unsigned int d = 0; d < Dim; ++d) {
                    if ((Rview(i)[d] > rmax[d]) || (Rview(i)[d] < rmin[d])) {
                        outside = true;
                    }
                }
                if (outside) {
                    Rview(i) = rmax;
                    bool odd = (i % 2);
                    if (params::kinetic_electrons) {
                        if (odd) {
                            Pview(i) = pgen.generate_ion();
                        } else {
                            Pview(i) = pgen.generate_electron();
                        }
                    } else {
                        Pview(i) = pgen.generate_ion();
                    }
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
        #ifdef IPPL_ENABLE_CATALYST
        std::optional<conduit_cpp::Node> node = std::nullopt;
        CatalystAdaptor::Execute_Particle_1d(it, this->time_m, ippl::Comm->rank(),  pc, node);
        //auto *rho               = &this->fcontainer_m->getRho();
        //CatalystAdaptor::Execute_Field(it, this->time_m, ippl::Comm->rank(),  *rho, node);
        //auto *E               = &this->fcontainer_m->getE();
        //CatalystAdaptor::Execute_Field(it, this->time_m, ippl::Comm->rank(),  *E, node);
        //CatalystAdaptor::Execute_Field_Particle(it, this->time_m, ippl::Comm->rank(),  *E, pc);
        #endif 
        // Field solve
        IpplTimings::startTimer(SolveTimer);
        if (!params::kinetic_electrons) {
            // is the electrons are adiabatic, then we have a background
            // charge density field which is given by exp(phi) where
            // phi is the previous iteration's solution (electric potential)
            this->fcontainer_m->getRho() =
                this->fcontainer_m->getRho()
                + exp(this->fcontainer_m->getPhi()) * params::Z_e * params::n_e0;
        }
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        this->grid2par();

        // kick (velocity update)

        // half acceleration
        // 1/tau factor in front of (q/m)*E for physics
        IpplTimings::startTimer(ETimer);
        pc->P = pc->P + 0.5 * dt * params::tau * (pc->q / pc->m) * pc->E;
        IpplTimings::stopTimer(ETimer);

        // rotation
        IpplTimings::startTimer(BTimer);

        Vector_t<T, 3> Bext = this->Bext_m;
        view_typeQ Qview    = this->pcontainer_m->q.getView();
        view_typeQ Mview    = this->pcontainer_m->m.getView();

        // 1/D_C factor in front of (q/m)*(v x B) for physics
        Kokkos::parallel_for(
            "Apply rotation", this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const int i) {
                Vector_t<T, 3> const t =
                    (1.0 / params::D_C) * 0.5 * dt * (Qview(i) / Mview(i)) * Bext;
                Vector_t<T, 3> const w = Pview(i) + cross(Pview(i), t).apply();
                Vector_t<T, 3> const s = (2.0 / (1 + dot(t, t).apply())) * t;
                Pview(i)               = Pview(i) + cross(w, s);
            });
        IpplTimings::stopTimer(BTimer);

        // half acceleration
        IpplTimings::startTimer(ETimer);
        pc->P = pc->P + 0.5 * dt * params::tau * (pc->q / pc->m) * pc->E;
        IpplTimings::stopTimer(ETimer);

        // push position (half step)
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + (0.5 * dt * pc->P);
        IpplTimings::stopTimer(RTimer);

        // since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        // update the incremental average
        updatePlasmaAverage();

        if ((this->it_m % params::dump_interval) == 1) {
            dumpPlasma();
        }
    }

    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);

        // dump for each particle the particles attributes (q, m, R, P)
        dumpParticleData();

        IpplTimings::stopTimer(dumpDataTimer);
    }

    void dumpParticleData() {
        typename ParticleAttrib<Vector_t<T, Dim>>::HostMirror R_host =
            this->pcontainer_m->R.getHostMirror();
        typename ParticleAttrib<Vector_t<T, 3>>::HostMirror P_host =
            this->pcontainer_m->P.getHostMirror();
        typename ParticleAttrib<double>::HostMirror q_host = this->pcontainer_m->q.getHostMirror();
        typename ParticleAttrib<T>::HostMirror m_host      = this->pcontainer_m->m.getHostMirror();

        Kokkos::deep_copy(R_host, this->pcontainer_m->R.getView());
        Kokkos::deep_copy(P_host, this->pcontainer_m->P.getView());
        Kokkos::deep_copy(q_host, this->pcontainer_m->q.getView());
        Kokkos::deep_copy(m_host, this->pcontainer_m->m.getView());

        std::stringstream pname;
        pname << "data_" << params::kinetic_electrons << "/ParticleIC_";
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

    void resetPlasmaAverage() {
        this->fcontainer_m->getPhiTimeavg() = this->fcontainer_m->getPhi();
        this->fcontainer_m->getRhoTimeavg() = this->fcontainer_m->getRho();
        n_timeavg                           = 1;
    }

    void updatePlasmaAverage() {
        this->fcontainer_m->getPhiTimeavg() =
            this->fcontainer_m->getPhiTimeavg()
            + (this->fcontainer_m->getPhi() - this->fcontainer_m->getPhiTimeavg()) / n_timeavg;
        this->fcontainer_m->getRhoTimeavg() =
            this->fcontainer_m->getRhoTimeavg()
            + (this->fcontainer_m->getRho() - this->fcontainer_m->getRhoTimeavg()) / n_timeavg;

        n_timeavg += 1;
    }

    void dumpPlasma() {
        typename Field_t<Dim>::view_type::host_mirror_type host_view_rho =
            this->fcontainer_m->getRho().getHostMirror();
        typename Field<T, Dim>::view_type::host_mirror_type host_view_phi =
            this->fcontainer_m->getPhi().getHostMirror();
        typename Field_t<Dim>::view_type::host_mirror_type host_view_rho_timeavg =
            this->fcontainer_m->getRhoTimeavg().getHostMirror();
        typename Field<T, Dim>::view_type::host_mirror_type host_view_phi_timeavg =
            this->fcontainer_m->getPhiTimeavg().getHostMirror();

        Kokkos::deep_copy(host_view_rho, this->fcontainer_m->getRho().getView());
        Kokkos::deep_copy(host_view_phi, this->fcontainer_m->getPhi().getView());
        Kokkos::deep_copy(host_view_rho_timeavg, this->fcontainer_m->getRhoTimeavg().getView());
        Kokkos::deep_copy(host_view_phi_timeavg, this->fcontainer_m->getPhiTimeavg().getView());

        const int nghost    = this->fcontainer_m->getRho().getNghost();
        const int nx        = this->nr_m[0];
        const double hx     = this->hr_m[0];
        const double orig_x = this->origin_m[0];

        std::stringstream fname;
        fname << "data_" << params::kinetic_electrons << "/Fields_";
        fname << this->it_m;
        fname << ".csv";
        Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
        csvout.precision(16);
        csvout.setf(std::ios::scientific, std::ios::floatfield);
        csvout << "x rho(x) rho_timeavg(x) phi(x) phi_timeavg(x)" << endl;

        for (int i = nghost; i < nx + nghost; ++i) {
            double x = (i + 0.5) * hx + orig_x;
            csvout << x << " ";
            csvout << host_view_rho(i) << " ";
            csvout << host_view_rho_timeavg(i) << " ";
            csvout << host_view_phi(i) << " ";
            csvout << host_view_phi_timeavg(i) << endl;
        }

        resetPlasmaAverage();

        ippl::Comm->barrier();
    }
};
#endif
