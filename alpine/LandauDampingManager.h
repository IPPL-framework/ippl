#ifndef IPPL_LANDAU_DAMPING_MANAGER_H
#define IPPL_LANDAU_DAMPING_MANAGER_H

#include <nvtx3/nvToolsExt.h>

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "AlpineManager.h"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

// define functions used in sampling particles
struct CustomDistributionFunctions {
  struct CDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, const double *params_p) const {
           return x + (params_p[d * 2 + 0] / params_p[d * 2 + 1]) * Kokkos::sin(params_p[d * 2 + 1] * x);
       }
  };

  struct PDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params_p) const {
           return 1.0 + params_p[d * 2 + 0] * Kokkos::cos(params_p[d * 2 + 1] * x);
       }
  };

  struct Estimate{
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params_p) const {
            return u + params_p[d] * 0.;
	}
  };
};

template <typename T, unsigned Dim>
class LandauDampingManager : public AlpineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;

    LandauDampingManager(size_type totalP_, int nt_, Vector_t<int, Dim> &nr_,
                       double lbt_, std::string& solver_, std::string& stepMethod_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_){}

    ~LandauDampingManager(){}

    void pre_run() override {
        Inform m("Pre Run");

        if (this->solver_m == "OPEN") {
            throw IpplException("LandauDamping", "Open boundaries solver incompatible with this simulation!");
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
        this->Q_m = std::reduce(this->rmax_m.begin(), this->rmax_m.end(), -1., std::multiplies<double>());
        this->origin_m = this->rmin_m;
        this->dt_m     = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m     = 0;
        this->time_m   = 0.0;

        m << "Discretization:" << endl
          << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid = " << this->nr_m << endl;

        this->isAllPeriodic_m = true;

        this->setFieldContainer( std::make_shared<FieldContainer_t>( this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m, this->isAllPeriodic_m) );

        this->setParticleContainer( std::make_shared<ParticleContainer_t>( this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()) );

        this->fcontainer_m->initializeFields(this->solver_m);

        this->setFieldSolver( std::make_shared<FieldSolver_t>( this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(), &this->fcontainer_m->getPhi()) );

        this->fsolver_m->initSolver();

        this->setLoadBalancer( std::make_shared<LoadBalancer_t>( this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m) );

        initializeParticles();

        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
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

    void initializeParticles(){
        Inform m("Initialize Particles");

        auto *mesh = &this->fcontainer_m->getMesh();
        auto *FL = &this->fcontainer_m->getFL();
        using DistR_t = ippl::random::Distribution<double, Dim, 2 * Dim, CustomDistributionFunctions>;
        double parR[2 * Dim];
        for(unsigned int i=0; i<Dim; i++){
            parR[i * 2   ]  = this->alpha_m;
            parR[i * 2 + 1] = this->kw_m[i];
        }
        DistR_t distR(parR);

        Vector_t<double, Dim> kw     = this->kw_m;
        Vector_t<double, Dim> hr     = this->hr_m;
        Vector_t<double, Dim> origin = this->origin_m;
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            IpplTimings::startTimer(domainDecomposition);
            this->isFirstRepartition_m           = true;
            const ippl::NDIndex<Dim>& lDom = FL->getLocalNDIndex();
            const int nghost               = this->fcontainer_m->getRho().getNghost();
            auto rhoview                   = this->fcontainer_m->getRho().getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", this->fcontainer_m->getRho().getFieldRangePolicy(),
                KOKKOS_LAMBDA (const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr + origin;

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
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>( *FL, *mesh );

        // unsigned int
        size_type totalP = this->totalP_m;
        int seed           = 42;
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
        for(unsigned int i=0; i<Dim; i++){
            mu[i] = 0.0;
            sd[i] = 1.0;
        }
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P, rand_pool64, mu, sd));
        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        this->pcontainer_m->q = this->Q_m/totalP;
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
        if (this->stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        }
	else{
            throw IpplException(TestName, "Step method is not set/recognized!");
        }
    }

    void LeapFrogStep(){
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        static IpplTimings::TimerRef PTimer              = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer              = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer         = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer          = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef SortTimer           = IpplTimings::getTimer("sort");
        static IpplTimings::TimerRef PermuteTimer        = IpplTimings::getTimer("permute");

        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        IpplTimings::startTimer(PTimer);
        nvtxRangePush("pushVelocity1");
        pc->P = pc->P - 0.5 * dt * pc->E;
        nvtxRangePop();
        IpplTimings::stopTimer(PTimer);

        // drift
        IpplTimings::startTimer(RTimer);
        nvtxRangePush("pushPosition");
        pc->R = pc->R + dt * pc->P;
        nvtxRangePop();
        IpplTimings::stopTimer(RTimer);

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        nvtxRangePush("update");
        pc->update();
        nvtxRangePop();
        IpplTimings::stopTimer(updateTimer);

        size_type totalP        = this->totalP_m;
        int it                  = this->it_m;
        bool isFirstRepartition = false;
        if (this->loadbalancer_m->balance(totalP, it + 1)) {
                IpplTimings::startTimer(domainDecomposition);
                auto* mesh = &fc->getRho().get_mesh();
                auto* FL = &fc->getFL();
                this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
                IpplTimings::stopTimer(domainDecomposition);
        }

        Kokkos::fence();

        // sort the particle positions for over-decomposition
        IpplTimings::startTimer(SortTimer);
        nvtxRangePush("sort");
        auto *FL = &this->fcontainer_m->getFL();
        const ippl::NDIndex<Dim>& ldom = FL->getLocalNDIndex();
        int ncell = 1;
        for (unsigned int i = 0; i < Dim; ++i) {
            ncell *= ldom[i].length();
        }
        Kokkos::View<int*> index("index", pc->getLocalNum());
        Kokkos::View<int*> start("start", ncell+1);
        this->sort(index, start, ncell);
        nvtxRangePop();
        IpplTimings::stopTimer(SortTimer);

        IpplTimings::startTimer(PermuteTimer);
        nvtxRangePush("permute");
        auto Rview = pc->R.getView();
        this->permute<Kokkos::View<ippl::Vector<double, Dim>*>>(Rview, index);
        nvtxRangePop();
        IpplTimings::stopTimer(PermuteTimer);

        // scatter the charge onto the underlying grid
        nvtxRangePush("scatter");
        this->par2grid();
        nvtxRangePop();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        nvtxRangePush("solve");
        this->fsolver_m->runSolver();
        nvtxRangePop();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        nvtxRangePush("gather");
        this->grid2par();
        nvtxRangePop();

        // kick
        IpplTimings::startTimer(PTimer);
        nvtxRangePush("pushVelocity2");
        pc->P = pc->P - 0.5 * dt * pc->E;
        nvtxRangePop();
        IpplTimings::stopTimer(PTimer);
    }

    void sort(Kokkos::View<int*> index, Kokkos::View<int*> start, int ncell) {
        // given a particle container with positions, return the index in the grid

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        int npart  = pc->getLocalNum();
        auto Rview = pc->R.getView();

        Vector_t<double, Dim> hr     = this->hr_m;
        Vector_t<double, Dim> origin = this->origin_m;

        auto *FL = &this->fcontainer_m->getFL();
        const ippl::NDIndex<Dim>& ldom = FL->getLocalNDIndex();

        Kokkos::View<int*> cell("c", npart), sorted("s",npart);

        int ncell_ = ncell;

        Kokkos::parallel_for("Get cell index", npart,
            KOKKOS_LAMBDA(size_t idx) {
                Vector_t<double, Dim> pos = Rview(idx);
                size_t serialized = ((pos[0] - origin[0]) * (1.0 / hr[0]));

                for (unsigned int i = 1; i < Dim; ++i) {
                    // compute index for each dim
                    size_t index_i = ((pos[i] - origin[i]) * (1.0 / hr[i]));
                    index_i = index_i - ldom[i].first();

                    // serialize to get global cell index
                    size_t length = 1;
                    for (unsigned int j = 0; j < i; ++j) {
                        length *= ldom[j].length();
                    }
                    serialized += index_i * length;
                }

                if ((serialized < 0) || (serialized >= ncell_)) {
                    printf("wrong cell id!\n");
                }
                cell(idx) = serialized;
        });

        Kokkos::deep_copy(start,0);

        Kokkos::parallel_for(npart,
            KOKKOS_LAMBDA (int i) {
                int c=cell(i);
                Kokkos::atomic_add(&start(c+1),1);
        });
        
        Kokkos::parallel_scan(ncell,
            KOKKOS_LAMBDA(int i, int &sum, bool is_final) {
                int tmp;
                if (is_final) {
                    tmp = sum;
                }
                sum += start(i+1);
                if (is_final) {
                    start(i+1) = tmp;
                }
        });

        int ncells = ncell;

        Kokkos::parallel_for(npart,
            KOKKOS_LAMBDA (int i) {
                int loc = Kokkos::atomic_fetch_add(&start(cell(i)+1),1);
                index(i)=loc;
        });
        
        Kokkos::parallel_for(npart,
            KOKKOS_LAMBDA (int i) {
                sorted(index(i))=cell(i);
        });
    }

    template <typename Attrib>
    void permute(Attrib orig_attrib, Kokkos::View<int*> index) {
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        int npart  = pc->getLocalNum();

        // size of temp should be (overalloc factor * npart) to match Rview
        int overalloc = ippl::Comm->getDefaultOverallocation();
        Attrib temp("temp view", overalloc * npart);

        Kokkos::parallel_for("Permute", npart, 
            KOKKOS_LAMBDA(int i) {
                temp(index(i)) = orig_attrib(i);
            });
        Kokkos::fence();

        Kokkos::deep_copy(orig_attrib, temp);
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
            std::reduce(this->fcontainer_m->getHr().begin(), this->fcontainer_m->getHr().end(), globaltemp, std::multiplies<double>());

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
            if ( std::fabs(this->time_m) < 1e-14 ) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << this->time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
