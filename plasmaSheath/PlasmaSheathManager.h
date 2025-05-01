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
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type   = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using view_type3D = typename ippl::detail::ViewType<ippl::Vector<double, 3>, 1>::view_type;

// define functions used in sampling particles
struct CustomDistributionFunctions {
    struct CDF {
        KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d,
                                                 const double* params_p) const {
            return x
                   + (params_p[d * 2 + 0] / params_p[d * 2 + 1])
                         * Kokkos::sin(params_p[d * 2 + 1] * x);
        }
    };

    struct PDF {
        KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d,
                                                 double const* params_p) const {
            return 1.0 + params_p[d * 2 + 0] * Kokkos::cos(params_p[d * 2 + 1] * x);
        }
    };

    struct Estimate {
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d,
                                                 double const* params_p) const {
            return u + params_p[d] * 0.;
        }
    };
};

template <typename T, unsigned Dim>
class PlasmaSheathManager : public AlpineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;

    PlasmaSheathManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                         std::string& solver_, std::string& stepMethod_,
                         double L = 1, double phiWall = 1, Vector_t<double, 3> Bext = {0,0,0})
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_) {
            setup(L, phiWall, Bext);
        }

    PlasmaSheathManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                         std::string& solver_, std::string& stepMethod_,
                         std::vector<std::string> preconditioner_params_, 
                         double L = 1, double phiWall = 1, Vector_t<double, 3> Bext = {0,0,0})
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_,
                                preconditioner_params_) {
            setup(L, phiWall, Bext);
        }

    ~PlasmaSheathManager() {}

    void setup(double L, double phiWall, Vector_t<double, 3> Bext) {
        Inform m("Setup");

        if ((this->solver_m != "CG") || (this->solver_m != "PCG")) {
            throw IpplException("PlasmaSheath",
                                "Open boundaries solver incompatible with this simulation!");
        }

        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);

        this->rmin_m   = 0.0;
        this->origin_m = this->rmin_m;
        this->rmax_m   = L; // L = size of domain

        this->phiWall_m = phiWall; // Dirichlet BC for phi at wall
        this->Bext_m = Bext; // External magnetic field

        this->hr_m = this->rmax_m / this->nr_m;
        this->dt_m = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m   = 0;
        this->time_m = 0.0;

        // Q = -\int\int f dx dv
        this->Q_m =
            std::reduce(this->rmax_m.begin(), this->rmax_m.end(), -1., std::multiplies<double>());

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

        // TODO change to correct distribution function here

        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL   = &this->fcontainer_m->getFL();

        Vector_t<double, Dim> hr                         = this->hr_m;
        Vector_t<double, Dim> origin                     = this->origin_m;

        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);

        // Sample particle positions:
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(*FL, *mesh);

        // unsigned int
        size_type totalP = this->totalP_m;
        int seed         = 42;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        using DistR_t =
            ippl::random::Distribution<double, Dim, 2 * Dim, CustomDistributionFunctions>;
        double parR[2 * Dim];
        for (unsigned int i = 0; i < Dim; i++) {
            parR[i * 2]     = 0.5;
            parR[i * 2 + 1] = 0.5;
        }

        DistR_t distR(parR);
        using samplingR_t =
            ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace,
                                                   DistR_t>;
        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> rmax = this->rmax_m;
        samplingR_t samplingR(distR, rmax, rmin, rlayout, totalP);
        size_type nlocal = samplingR.getLocalSamplesNum();

        this->pcontainer_m->create(nlocal);

        view_type* R = &(this->pcontainer_m->R.getView());
        samplingR.generate(*R, rand_pool64);

        view_type3D* P = &(this->pcontainer_m->P.getView());

        double mu[3];
        double sd[3];
        for (unsigned int i = 0; i < 3; i++) {
            mu[i] = 0.0;
            sd[i] = 1.0;
        }
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, 3>(*P, rand_pool64, mu, sd));
        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        this->pcontainer_m->q = this->Q_m / totalP;
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
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        static IpplTimings::TimerRef ETimer              = IpplTimings::getTimer("kickVelocity");
        static IpplTimings::TimerRef BTimer              = IpplTimings::getTimer("rotateVelocity");
        static IpplTimings::TimerRef RTimer              = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer         = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer          = IpplTimings::getTimer("solve");

        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        // TODO check whether this is doing the correct thing (adding only P_x to the 1-dimensional
        // position R i.e. taking only first component of 3D vector P).
        // push position (half step)
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + (0.5 * dt * pc->P);
        IpplTimings::stopTimer(RTimer);

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

        // TODO change the -1.0 factor to q/m for each particle (2 species here)
        // half acceleration
        IpplTimings::startTimer(ETimer);
        pc->P = pc->P + 0.5 * dt * (-1.0) * pc->E;
        IpplTimings::stopTimer(ETimer);

        // rotation
        IpplTimings::startTimer(BTimer);
        Vector_t<double, 3> Bext = this->Bext_m;
        view_type3D Pview = this->pcontainer_m->P.getView();
        Kokkos::parallel_for("Apply rotation", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i) {
                // TODO change the -1.0 factor to q/m for each particle (2 species here)
                Vector_t<double, 3> const t = 0.5 * dt * (-1.0) * Bext;
                Vector_t<double, 3> const w = Pview(i) + cross(Pview(i), t).apply();
                Vector_t<double, 3> const s = (2.0 / (1 + dot(t, t).apply())) * t;
                Pview(i) = Pview(i) + cross(w, s);
            });
        IpplTimings::stopTimer(BTimer);

        // TODO change the -1.0 factor to q/m for each particle (2 species here)
        // half acceleration
        IpplTimings::startTimer(ETimer);
        pc->P = pc->P + 0.5 * dt * (-1.0) * pc->E;
        IpplTimings::stopTimer(ETimer);

        // push position (half step)
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + (0.5 * dt * pc->P);
        IpplTimings::stopTimer(RTimer);

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);
    }

    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);

        // dump for each particle the particles attributes (q, m, R, P)

        IpplTimings::stopTimer(dumpDataTimer);
    }
};
#endif
