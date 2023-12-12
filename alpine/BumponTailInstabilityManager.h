#ifndef IPPL_BUMPON_TAIL_INSTABILITY_MANAGER_H
#define IPPL_BUMPON_TAIL_INSTABILITY_MANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/UniformDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

constexpr bool EnablePhaseDump = false;

// define functions used in sampling particles
struct CustomDistributionFunctions {
  struct CDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, const double *params_p) const {
           if( d == Dim - 1)
               	return x + (params_p[d * 2 + 0] / params_p[d * 2 + 1]) * Kokkos::sin(params_p[d * 2 + 1] * x);
           else
                return ippl::random::uniform_cdf_func<double>(x);
       }
  };

  struct PDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params_p) const {
           if( d == Dim - 1)
               return  (1.0 + params_p[d * 2 + 0] * Kokkos::cos(params_p[d * 2 + 1] * x));
           else
               return ippl::random::uniform_pdf_func<double>();
       }
  };

  struct Estimate{
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params_p) const {
            return u + params_p[d] * 0.;
	}
  };
};

class BumponTailInstabilityManager
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                              LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;
    struct PhaseDump;
private:
    size_type totalP;
    int nt;
    Vector_t<int, Dim> nr;
    double lbt;
    std::string solver;
    std::string step_method;
    std::shared_ptr<PhaseDump> phase;
public:
    BumponTailInstabilityManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_, std::string& solver_, std::string& step_method_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>>()
        , totalP(totalP_)
        , nt(nt_)
        , nr(nr_)
        , lbt(lbt_)
        , solver(solver_)
        , step_method(step_method_){
            phase = std::make_shared<PhaseDump>();
        }
    ~BumponTailInstabilityManager(){}

private:
    double loadbalancethreshold_m;
    double time_m;
    double dt;
    int it;
    Vector_t<double, Dim> kw;
    Vector_t<double, Dim> rmin;
    Vector_t<double, Dim> rmax;
    Vector_t<double, Dim> hr;
    double Q;
    Vector_t<double, Dim> origin;
    bool isAllPeriodic;
    bool isFirstRepartition;
    ippl::NDIndex<Dim> domain;
    std::array<bool, Dim> decomp;
    double sigma;
    double  muBulk;
    double  muBeam;
    double epsilon;
    double delta;
public:
    size_type getTotalP() const { return totalP; }

    void setTotalP(size_type totalP_) { totalP = totalP_; }

    int getNt() const { return nt; }

    void setNt(int nt_) { nt = nt_; }

    const std::string& getSolver() const { return solver; }

    void setSolver(const std::string& solver_) { solver = solver_; }

    double getLoadBalanceThreshold() const { return lbt; }

    void setLoadBalanceThreshold(double lbt_) { lbt = lbt_; }

    const std::string& getStepMethod() const { return step_method; }

    void setStepMethod(const std::string& step_method_) { step_method = step_method_; }

    const Vector_t<int, Dim>& getNr() const { return nr; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }

    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }
    void post_step() override {
        // Update time
        this->time_m += this->dt;
        this->it++;

        // wrtie solution to output file
        this->dump();
        Inform m("Post-step:");
        m << "Finished time step: " << this->it << " time: " << this->time_m << endl;
    }
    void pre_run() override {
        Inform m("Pre Run");
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        decomp.fill(true);


        if (std::strcmp(TestName, "TwoStreamInstability") == 0) {
            // Parameters for two stream instability as in
            //  https://www.frontiersin.org/articles/10.3389/fphy.2018.00105/full
            kw      = 0.5;
            sigma   = 0.1;
            epsilon = 0.5;
            muBulk  = -pi / 2.0;
            muBeam  = pi / 2.0;
            delta   = 0.01;
        } else if (std::strcmp(TestName, "BumponTailInstability") == 0) {
            kw      = 0.21;
            sigma   = 1.0 / std::sqrt(2.0);
            epsilon = 0.1;
            muBulk  = 0.0;
            muBeam  = 4.0;
            delta   = 0.01;
        } else {
            // Default value is two stream instability
            kw      = 0.5;
            sigma   = 0.1;
            epsilon = 0.5;
            muBulk  = -pi / 2.0;
            muBeam  = pi / 2.0;
            delta   = 0.01;
        }

        rmin(0.0);
        rmax = 2 * pi / kw;
        hr = rmax / nr;
        // Q = -\int\int f dx dv
        Q = std::reduce(rmax.begin(), rmax.end(), -1., std::multiplies<double>());
        origin = rmin;
        dt     = std::min(.05, 0.5 * *std::min_element(hr.begin(), hr.end()));
        it     = 0;
        time_m = 0.0;

        m << "Discretization:" << endl
          << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

        std::shared_ptr<Mesh_t<Dim>> mesh = std::make_shared<Mesh_t<Dim>>(domain, hr, origin);

        isAllPeriodic = true;

        std::shared_ptr<FieldLayout_t<Dim>> FL = std::make_shared<FieldLayout_t<Dim>>(MPI_COMM_WORLD, domain, decomp, isAllPeriodic);

        std::shared_ptr<PLayout_t<T, Dim>> PL = std::make_shared<PLayout_t<T, Dim>>(*FL, *mesh);

        if (solver == "OPEN") {
            throw IpplException("BumpOnTailInstability", "Open boundaries solver incompatible with this simulation!");
        }

        setParticleContainer( std::make_shared<ParticleContainer_t>(PL) );

        setFieldContainer( std::make_shared<FieldContainer_t>(hr, rmin, rmax, decomp) );

        fcontainer_m->initializeFields(mesh, FL, solver);

        setFieldSolver( std::make_shared<FieldSolver_t>(this->solver, &fcontainer_m->getRho(), &fcontainer_m->getE(), &fcontainer_m->getPhi()) );

        fsolver_m->initSolver();

        setLoadBalancer( std::make_shared<LoadBalancer_t>( lbt, fcontainer_m, pcontainer_m, fsolver_m) );

        initializeParticles(mesh, FL);

        if constexpr (EnablePhaseDump) {
            if (ippl::Comm->size() != 1) {
                m << "Phase dump only supported on one rank" << endl;
                ippl::Comm->abort();
            }
            phase->initialize(*std::max_element(nr.begin(), nr.end()),
                             *std::max_element(rmax.begin(), rmax.end()));
        }

        fcontainer_m->getRho() = 0.0;

        fsolver_m->runSolver();

        par2grid();

        fsolver_m->runSolver();

        grid2par();

        dump();

        m << "Done";
    }

    void initializeParticles(std::shared_ptr<Mesh_t<Dim>> mesh_m, std::shared_ptr<FieldLayout_t<Dim>> FL_m){
        Inform m("Initialize Particles");

        using DistR_t = ippl::random::Distribution<double, Dim, 2 * Dim, CustomDistributionFunctions>;
        double parR[2 * Dim];
        for(unsigned int i=0; i<Dim; i++){
            parR[i * 2   ]  = delta;
            parR[i * 2 + 1] = kw[i];
        }
        DistR_t distR(parR);

        Vector_t<double, Dim> hr_m     = hr;
        Vector_t<double, Dim> origin_m = origin;
        if ((this->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            this->isFirstRepartition       = true;
            const ippl::NDIndex<Dim>& lDom = FL_m->getLocalNDIndex();
            const int nghost               = this->fcontainer_m->getRho().getNghost();
            auto rhoview                   = this->fcontainer_m->getRho().getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", this->fcontainer_m->getRho().getFieldRangePolicy(),
                KOKKOS_LAMBDA (const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr_m + origin_m;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = distR.getFullPdf(xvec);
                });

            Kokkos::fence();

            this->loadbalancer_m->initializeORB(FL_m.get(), mesh_m.get());
            this->loadbalancer_m->repartition(FL_m.get(), mesh_m.get(), this->isFirstRepartition);
        }

        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>( *FL_m, *mesh_m );

        size_type totalP_m = this->totalP;
        int seed           = 42;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        using samplingR_t =
            ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace, DistR_t>;

        Vector_t<double, Dim> rmin_m = rmin;
        Vector_t<double, Dim> rmax_m = rmax;
        samplingR_t samplingR(distR, rmax_m, rmin_m, rlayout, totalP_m);
        size_type nlocal = samplingR.getLocalSamplesNum();

        double factorVelBulk      = 1.0 - epsilon;
        double factorVelBeam      = 1.0 - factorVelBulk;
        size_type nlocBulk        = (size_type)(factorVelBulk * nlocal);
        size_type nlocBeam        = (size_type)(factorVelBeam * nlocal);
        nlocal                    = nlocBulk + nlocBeam;

        int rank = ippl::Comm->rank();
        size_type nglobal = 0;
        MPI_Allreduce(&nlocal, &nglobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, ippl::Comm->getCommunicator());
        int rest = (int)(totalP - nglobal);
        if (rank < rest) {
            ++nlocal;
        }

        this->pcontainer_m->create(nlocal);

        view_type* R_m = &(this->pcontainer_m->R.getView());
        samplingR.generate(*R_m, rand_pool64);

        view_type* P_m = &(this->pcontainer_m->P.getView());

        double mu[Dim];
        double sd[Dim];
        for(unsigned int i=0; i<Dim; i++){
           mu[i] = 0.0;
           sd[i] = sigma;
        }
        // sample first nlocBulk with muBulk as mean velocity
        mu[Dim-1] = muBulk;
        Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, nlocBulk), ippl::random::randn<double, Dim>(*P_m, rand_pool64, mu, sd));

        // sample remaining with muBeam as mean velocity
        mu[Dim-1] = muBeam;
        Kokkos::parallel_for(Kokkos::RangePolicy<int>(nlocBulk, nlocal), ippl::random::randn<double, Dim>(*P_m, rand_pool64, mu, sd));

        Kokkos::fence();
        ippl::Comm->barrier();

        this->pcontainer_m->q = Q/totalP;
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
        if (this->step_method == "LeapFrog") {
            LeapFrogStep();
        }
    }

    void LeapFrogStep(){
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute

        double dt_m                             = this->dt;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        pc->P = pc->P - 0.5 * dt_m * pc->E;

        // drift
        pc->R = pc->R + dt_m * pc->P;

        // Since the particles have moved spatially update them to correct processors
        pc->update();

        size_type totalP_m        = this->totalP;
        int it_m                  = this->it;
        bool isFirstRepartition_m = false;
        if (loadbalancer_m->balance(totalP_m, it_m + 1)) {
                auto* mesh = &fc->getRho().get_mesh();
                auto* FL = &fc->getFL();
                loadbalancer_m->repartition(FL, mesh, isFirstRepartition_m);
        }

        // scatter the charge onto the underlying grid
        this->par2grid();

        // Field solve
        this->fsolver_m->runSolver();

        // gather E field
        this->grid2par();

        // kick
        pc->P = pc->P - 0.5 * dt_m * pc->E;
    }

    void par2grid() override { scatterCIC(); }

    void grid2par() override { gatherCIC(); }

    void gatherCIC() {
        using Base                        = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        Base::particle_position_type *E_p = &this->pcontainer_m->E;
        Base::particle_position_type *R_m = &this->pcontainer_m->R;
        VField_t<T, Dim> *E_f             = &this->fcontainer_m->getE();
        gather(*E_p, *E_f, *R_m);
    }

    void scatterCIC() {
        Inform m("scatter ");
        this->fcontainer_m->getRho() = 0.0;

        using Base                        = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        ippl::ParticleAttrib<double> *q_m = &this->pcontainer_m->q;
        Base::particle_position_type *R_m = &this->pcontainer_m->R;
        Field_t<Dim> *rho_m               = &this->fcontainer_m->getRho();
        double Q_m                        = this->Q;
        Vector_t<double, Dim> rmin_m      = rmin;
        Vector_t<double, Dim> rmax_m      = rmax;
        Vector_t<double, Dim> hr_m        = hr;

        scatter(*q_m, *rho_m, *R_m);
        double rel_error = std::fabs((Q_m-(*rho_m).sum())/Q_m);

        m << rel_error << endl;

        size_type Total_particles = 0;
        size_type local_particles = pcontainer_m->getLocalNum();

        ippl::Comm->reduce(local_particles, Total_particles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            if (Total_particles != this->totalP || rel_error > 1e-10) {
                m << "Time step: " << this->it << endl;
                m << "Total particles in the sim. " << this->totalP << " "
                  << "after update: " << Total_particles << endl;
                m << "Rel. error in charge conservation: " << rel_error << endl;
                ippl::Comm->abort();
            }
        }

        double cellVolume = std::reduce(hr_m.begin(), hr_m.end(), 1., std::multiplies<double>());
        (*rho_m)          = (*rho_m) / cellVolume;

        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax_m[d] - rmin_m[d];
            }
            *rho_m = *rho_m - (Q_m / size);
        }
    }

    struct PhaseDump {
        void initialize(size_t nr_, double domain_) {
           ippl::Index I(nr_);
           ippl::NDIndex<2> owned(I, I);
           layout = FieldLayout_t<2>(MPI_COMM_WORLD, owned, isParallel);

           Vector_t<double, 2> hx = {domain_ / nr_, 16. / nr_};
           Vector_t<double, 2> orgn{0, -8};

           mesh = Mesh_t<2>(owned, hx, orgn);
           phaseSpace.initialize(mesh, layout);
           if (ippl::Comm->rank() == 0) {
               phaseSpaceBuf.initialize(mesh, layout);
           }
           std::cout << ippl::Comm->rank() << ": " << phaseSpace.getOwned() << std::endl;
        }

        void dump(int it_, std::shared_ptr<ParticleContainer_t> pc, bool allDims = false) {
           const auto pcount = pc->getLocalNum();
           phase.realloc(pcount);
           auto& Ri = pc->R;
           auto& Pi = pc->P;
           for (unsigned d = allDims ? 0 : Dim - 1; d < Dim; d++) {
               Kokkos::parallel_for(
                   "Copy phase space", pcount, KOKKOS_CLASS_LAMBDA(const size_t i) {
                       phase(i) = {Ri(i)[d], Pi(i)[d]};
                   });
               phaseSpace = 0;
               Kokkos::fence();
               scatter(pc->q, phaseSpace, phase);
               auto& view = phaseSpace.getView();
               MPI_Reduce(view.data(), phaseSpaceBuf.getView().data(), view.size(), MPI_DOUBLE,
                       MPI_SUM, 0, ippl::Comm->getCommunicator());
               if (ippl::Comm->rank() == 0) {
                   std::stringstream fname;
                   fname << "PhaseSpace_t=" << it_ << "_d=" << d << ".csv";

                   Inform out("Phase Dump", fname.str().c_str(), Inform::OVERWRITE, 0);
                   phaseSpaceBuf.write(out);

                   auto max = phaseSpaceBuf.max();
                   auto min = phaseSpaceBuf.min();
                   if (max > maxValue) {
                       maxValue = max;
                   }
                   if (min < minValue) {
                       minValue = min;
                   }
               }
               ippl::Comm->barrier();
           }
           MPI_Bcast(&maxValue, 1, MPI_DOUBLE, 0, ippl::Comm->getCommunicator());
           MPI_Bcast(&minValue, 1, MPI_DOUBLE, 0, ippl::Comm->getCommunicator());
        }

        double maxRecorded() const { return maxValue; }
        double minRecorded() const { return minValue; }

    private:
        std::array<bool, 2> isParallel = {false, false};
        FieldLayout_t<2> layout;
        Mesh_t<2> mesh;
        Field_t<2> phaseSpace, phaseSpaceBuf;
        ippl::ParticleAttrib<Vector_t<double, 2>> phase;

        double maxValue = 0, minValue = 0;
    };

    void dump() {
        dumpBumponTailInstability(fcontainer_m->getE().getView());
        if constexpr (EnablePhaseDump) {
                phase->dump(it, pcontainer_m);
        }
    }

    template <typename View>
    void dumpBumponTailInstability(const View& Eview) {
        const int nghostE = fcontainer_m->getE().getNghost();
        double fieldEnergy, EzAmp;

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double temp            = 0.0;

        ippl::parallel_reduce(
            "Ex inner product", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                // ippl::apply accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double myVal = std::pow(ippl::apply(Eview, args)[Dim - 1], 2);
                valL += myVal;
            },
            Kokkos::Sum<double>(temp));

        double globaltemp = 0.0;
        ippl::Comm->reduce(temp, globaltemp, 1, std::plus<double>());

        fieldEnergy =
            std::reduce(fcontainer_m->getHr().begin(), fcontainer_m->getHr().end(), globaltemp, std::multiplies<double>());

        double tempMax = 0.0;
        ippl::parallel_reduce(
            "Ex max norm", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                // ippl::apply accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double myVal = std::fabs(ippl::apply(Eview, args)[Dim - 1]);
                if (myVal > valL) {
                    valL = myVal;
                }
            },
            Kokkos::Max<double>(tempMax));

        EzAmp = 0.0;
        ippl::Comm->reduce(tempMax, EzAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldBumponTail_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if ( std::fabs(time_m) < 1e-14 ) {
                csvout << "time, Ez_field_energy, Ez_max_norm" << endl;
            }

            csvout << time_m << " " << fieldEnergy << " " << EzAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
