#ifndef IPPL_LANDAU_DAMPING_MANAGER_H
#define IPPL_LANDAU_DAMPING_MANAGER_H

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

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

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
class LandauDampingManager : public AlpineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;

    LandauDampingManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                         std::string& solver_, std::string& stepMethod_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_) {}

    LandauDampingManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                         std::string& solver_, std::string& stepMethod_,
                         std::vector<std::string> preconditioner_params_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_,
                                preconditioner_params_) {}

    ~LandauDampingManager() {}

    void pre_run() override {
        Inform m("Pre Run");

	const double pi = Kokkos::numbers::pi_v<T>;
	
        if (this->solver_m == "OPEN") {
            throw IpplException("LandauDamping",
                                "Open boundaries solver incompatible with this simulation!");
        }

        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        this->kw_m    = 0.5;
        this->alpha_m = 0.05;
        this->rmin_m  = 0.0;
        this->rmax_m  = 2 * pi / this->kw_m;

        this->hr_m = this->rmax_m / this->nr_m;
        // Q = -\int\int f dx dv
        this->Q_m =
            std::reduce(this->rmax_m.begin(), this->rmax_m.end(), -1., std::multiplies<double>());
        this->origin_m = this->rmin_m;
        this->dt_m   = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m   = 0;
        this->time_m = 0.0;

        m << "Discretization:" << endl
          << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid = " << this->nr_m << endl;

        this->isAllPeriodic_m = true;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
            this->isAllPeriodic_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        this->fcontainer_m->initializeFields(this->solver_m);

        if (this->getSolver() == "PCG") {
            this->setFieldSolver(std::make_shared<FieldSolver_t>(
                this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(),
                &this->fcontainer_m->getPhi(), this->preconditioner_params_m));
        } else {
            this->setFieldSolver(std::make_shared<FieldSolver_t>(
                this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(),
                &this->fcontainer_m->getPhi()));
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

        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL   = &this->fcontainer_m->getFL();
        using DistR_t =
            ippl::random::Distribution<double, Dim, 2 * Dim, CustomDistributionFunctions>;
        double parR[2 * Dim];
        for (unsigned int i = 0; i < Dim; i++) {
            parR[i * 2]     = this->alpha_m;
            parR[i * 2 + 1] = this->kw_m[i];
        }
        DistR_t distR(parR);

        Vector_t<double, Dim> kw                         = this->kw_m;
        Vector_t<double, Dim> hr                         = this->hr_m;
        Vector_t<double, Dim> origin                     = this->origin_m;
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            IpplTimings::startTimer(domainDecomposition);
            this->isFirstRepartition_m     = true;
            const ippl::NDIndex<Dim>& lDom = FL->getLocalNDIndex();
            const int nghost               = this->fcontainer_m->getRho().getNghost();
            auto rhoview                   = this->fcontainer_m->getRho().getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF",
                this->fcontainer_m->getRho().getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * hr + origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = distR.getFullPdf(xvec);
                });

            Kokkos::fence();

            this->loadbalancer_m->initializeORB(FL, mesh);
            this->loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition_m);
            IpplTimings::stopTimer(domainDecomposition);
        }

        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);

        // Sample particle positions:
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(*FL, *mesh);

        // unsigned int
        size_type totalP = this->totalP_m;
        int seed         = 42;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

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

        view_type* P = &(this->pcontainer_m->P.getView());

        double mu[Dim];
        double sd[Dim];
        for (unsigned int i = 0; i < Dim; i++) {
            mu[i] = 0.0;
            sd[i] = 1.0;
        }
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P, rand_pool64, mu, sd));
        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        this->pcontainer_m->q = this->Q_m / totalP;
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
        if (this->stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        } else {
            throw IpplException(TestName, "Step method is not set/recognized!");
        }
    }

    void LeapFrogStep() {
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        static IpplTimings::TimerRef PTimer              = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer              = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer         = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer          = IpplTimings::getTimer("solve");

        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        IpplTimings::startTimer(PTimer);
        pc->P = pc->P - 0.5 * dt * pc->E;
        IpplTimings::stopTimer(PTimer);

        // drift
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + dt * pc->P;
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

        // kick
        IpplTimings::startTimer(PTimer);
        pc->P = pc->P - 0.5 * dt * pc->E;
        IpplTimings::stopTimer(PTimer);
    }

    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);
        dumpLandau(this->fcontainer_m->getE().getView());
        IpplTimings::stopTimer(dumpDataTimer);
    }

    template <typename View>
    void dumpLandau(const View& Eview) {
        const int nghostE = this->fcontainer_m->getE().getNghost();

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
        ippl::Comm->reduce(localEx2, globaltemp, 1, std::plus<double>());

        double fieldEnergy =
            std::reduce(this->fcontainer_m->getHr().begin(), this->fcontainer_m->getHr().end(),
                        globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        ippl::Comm->reduce(localExNorm, ExAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (std::fabs(this->time_m) < 1e-14) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << this->time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
