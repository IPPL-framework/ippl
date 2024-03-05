#ifndef IPPL_ADJOINT_MANAGER_H
#define IPPL_ADJOINT_MANAGER_H

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

//using Vector_t = ippl::Vector<double, Dim>;

using Matrix_t = ippl::Vector< ippl::Vector<double, Dim>, Dim>;

// phi_ext = a * exp(-(x-b)^2)
// dphi_ext/dxi = - a * 2 * (x_i-b_i) * exp(-(x-b)^2)
// dphi_ext/da = exp(-(x-b)^2)
// dphi_ext/db = a * 2 * (x-b) * exp(-(x-b)^2)
/*
template <unsigned Dim>
struct ExternalForceField{
       KOKKOS_INLINE_FUNCTION double operator()(auto x, unsigned int d, auto params) const {
           double y = -params[0]* 2.0 * ( x[d]-params[d+1] );

           for(unsigned int dim=0; dim<Dim; dim++){
               y *= Kokkos::exp(-Kokkos::pow(x[dim]-params[1+dim], 2) );
           }
           return y;
      }
};
template <unsigned Dim>
struct DExternalForceFieldDparms{
       KOKKOS_INLINE_FUNCTION double operator()(auto x, unsigned int d, auto params) const {
           double y = 0.;
           if(d==0){
               y = 1.0;
               for(unsigned int dim=0; dim<Dim; dim++){
                   y *= Kokkos::exp(-Kokkos::pow(x[dim]-params[1+dim], 2) );
               }
           }
           else{
               y =  params[0] * 2. * ( x[d-1]-params[d] );
               for(unsigned int dim=0; dim<Dim; dim++){
                    y *= Kokkos::exp(-Kokkos::pow(x[dim]-params[1+dim], 2) );
               }
           }
           return y;
       }
};*/

// phi_ext = p0*x0 + p1*x1 + p2*x2 + p3*x0*x1 + p4*x0*x2 + p5*x1*x2 + p6*x0^2 + p7*x1^2 + p8*x2^2
// dphi_ext/dx0 = p0 + p3*x1 + p4*x2 + 2*p6*x0
// dphi_ext/dx1 = p1 + p3*x0 + p5*x2 + 2*p7*x1
// dphi_ext/dx2 = p2 + p4*x0 + p5*x1 + 2*p8*x2
// dphi_ext/dpi = [x0, x1, x2, x0*x1, x0*x2, x1*x2, x0^2, x1^2, x2^2
template <unsigned Dim>
struct ExternalForceField{
       KOKKOS_INLINE_FUNCTION double operator()(auto x, unsigned int d, auto p) const {
           if(d==0)
		return p[0] + p[3]*x[1] + p[4]*x[2] + 2.*p[6]*x[0];
           else if(d==1)
		return p[1] + p[3]*x[0] + p[5]*x[2] + 2*p[7]*x[1];
           else if(d==2)
		return p[2] + p[4]*x[0] + p[5]*x[1] + 2*p[8]*x[2];
           else
                return 0.0;
      }
};
template <unsigned Dim>
struct DExternalForceFieldDparms{
       KOKKOS_INLINE_FUNCTION double operator()(auto x, unsigned int d, auto p) const {
           if(d==0)
                return x[0] + 0.*p[0];
           else if(d==1)
                return x[1];
           else if(d==2)
                return x[2];
           else if(d==3)
                return x[0]*x[1];
           else if(d==4)
                return x[0]*x[2];
           else if(d==5)
                return x[1]*x[2];
           else if(d==6)
                return x[0]*x[0];
           else if(d==7)
                return x[1]*x[1];
           else if(d==8)
                return x[2]*x[2];
           else
                return 0.0;
       }
};

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
class AdjointManager : public AlpineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;

    AdjointManager(size_type totalP_, int nt_, Vector_t<int, Dim> &nr_,
                       double lbt_, std::string& solver_, std::string& stepMethod_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_){}

    ~AdjointManager(){}

    int nparams_m;
    Vector_t<T, 9> params_m;
    Vector_t<T, 9> dparams_m;

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
        this->rmin_m  =  0.0;//-2 * pi / this->kw_m;
        this->rmax_m  =   2 * pi / this->kw_m;

        this->hr_m = (this->rmax_m-this->rmin_m) / this->nr_m;
        // Q = -\int\int f dx dv
        this->Q_m =  std::reduce(this->rmax_m.begin(), this->rmax_m.end(), -1., std::multiplies<double>());
        this->Q_m -= std::reduce(this->rmin_m.begin(), this->rmin_m.end(), -1., std::multiplies<double>());
        this->origin_m = this->rmin_m;
        this->dt_m     = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m     = 0;
        this->time_m   = 0.0;

        //this->nparams_m = 4;
        //this->params_m = 1.e-4;
        //this->dparams_m = 0.0;

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

        this->externalForceField();

        this->dump();

        m << "Done";
    }

    void initializeParticles(){
        Inform m("Initialize Particles");

        auto *mesh = &this->fcontainer_m->getMesh();
        auto *FL = &this->fcontainer_m->getFL();
        using DistR_t = ippl::random::Distribution<double, Dim, 2 * Dim, CustomDistributionFunctions>;
        //using DistR_t = ippl::random::NormalDistribution<double, Dim>;
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
    void externalForceField(){

        auto &F = this->pcontainer_m->F.getView();
        auto &R = this->pcontainer_m->R.getView();

        Vector_t params = this->params_m;
        Vector_t dparams = this->dparams_m;

        // compute F given external force field
        Kokkos::parallel_for(this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
            for(unsigned int d=0; d<Dim; d++){
                F(i)[d] = ExternalForceField<Dim>()(R(i), d, params);
            }
	});

	Kokkos::fence();
        ippl::Comm->barrier();
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
        static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");

        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        // external force on particle
        externalForceField();

        IpplTimings::startTimer(PTimer);
        pc->P = pc->P - 0.5 * dt * ( pc->E + pc->F );
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
                auto* FL = &fc->getFL();
                this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
                IpplTimings::stopTimer(domainDecomposition);
        }

	// scatter the charge onto the underlying grid
        this->par2grid();

        delFdelparm();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        this->grid2par();

        // recomputed external force on particle since they have moved
        this->externalForceField();

        // kick
        IpplTimings::startTimer(PTimer);
        pc->P = pc->P - 0.5 * dt * ( pc->E + pc->F );
        IpplTimings::stopTimer(PTimer);
    }

    // TODO: Implement the perturbation of particles at the final time according to the merit, to be called at the end of forward run
    void PerturbParticles(){
        // compute Dl

        auto &P = this->pcontainer_m->P.getView();
        auto &R = this->pcontainer_m->R.getView();

        double gD = costFunction();

        Kokkos::parallel_for(this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
            for (unsigned int j = 0; j < 3; ++j) {
                double tempR = R(i)[j];
                double tempP = P(i)[j];
                P(i)[j] += -2.0 * gD * tempR * 1.e-3;
                R(i)[j] +=  2.0 * gD * tempP * 1.e-3;
            }
        });
    }

    // computation of dC/dalpha
    void delFdelparm(){
        //Inform m("delF: ");
        auto *FL = &this->fcontainer_m->getFL();

        Field_t<Dim> *rho     = &this->fcontainer_m->getRho();

        auto rhoview         = this->fcontainer_m->getRho().getView();

        Vector_t<double, Dim> hr     = this->hr_m;
        Vector_t<double, Dim> origin = this->origin_m;

        Vector_t params = this->params_m;
        unsigned int nparams = this->nparams_m;

        const int nghost = (*rho).getNghost();
        const ippl::NDIndex<Dim>& lDom = FL->getLocalNDIndex();
        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        for (unsigned idparams = 0; idparams < nparams; ++idparams) {
            T temp = 0.0;
            ippl::parallel_reduce(
                "rho*dphi_dalpha0 reduce", (*rho).getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args, T& valL) {
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr + origin;
                    T myVal = ippl::apply(rhoview, args) * DExternalForceFieldDparms<Dim>()(xvec, idparams, params);
                    valL += myVal;
              }, Kokkos::Sum<T>(temp));
            Kokkos::fence();

            T globaltemp          = 0.0;
            ippl::Comm->reduce(temp, globaltemp, 1, std::plus<double>());
            ippl::Comm->barrier();

            this->dparams_m[idparams] += this->dt_m*globaltemp*this->hr_m[0]*this->hr_m[1]*this->hr_m[2];
        }
    }

    void post_step() override {

        //this->par2grid();
        //delFdelparm();

        int sign = (this->dt_m >= 0) ? 1 : -1;

        this->time_m += this->dt_m;

        this->it_m += sign;

        this->dump();

        Inform m("Post-step:");
        m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
    }

   double costFunction(){
        // compute Dl
        auto Pview = this->pcontainer_m->P.getView();
        auto Rview = this->pcontainer_m->R.getView();
        double D   = 0.0;
        double gD  = 0.0;

        Kokkos::parallel_reduce(
            "Dl", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(Pview(i), Pview(i)).apply();
                myVal       += dot(Rview(i), Rview(i)).apply();
                valL        += myVal;
            },
            Kokkos::Sum<double>(D));
        ippl::Comm->reduce(D, gD, 1, std::plus<double>());
        gD = gD/this->totalP_m;
        gD = gD - 160;// global D - final time D
        if(this->it_m==this->nt_m){
            Inform csvout(NULL, "data/D.csv", Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            csvout << gD << " " << D/this->totalP_m << endl;
            ippl::Comm->barrier();

            Inform m("Estimate moment:");
            m << " -------    res: " << gD << " " << D/this->totalP_m << endl;
        }
        return gD;
   }
   void runOptimization(int optIt){
        Inform m("Optimization");

        this->nparams_m = 9;
        this->params_m = 1.e-16;
        this->dparams_m = 0.0;

        this->pre_run();

        double dt0 = this->dt_m;

        for(int iter = 0; iter < optIt; iter++) {
             m << "iter " << iter << endl << endl;

             this->dparams_m = 0.0;

             this->pre_run();

             // Forward run, umpurturbed potential
             this->dt_m = dt0;
             this->run(this->nt_m); // compute unpertubed terms of df/dalpha during forward simulation

             PerturbParticles();

             // Backward
             this->dt_m = -this->dt_m;
             this->run(this->nt_m); // compute pertubed terms of df/dalpha during backward simulation

             this->params_m = this->params_m + 0.001*this->dparams_m;

             m << this->params_m << " " << this->dparams_m << endl;
         }
   }
};

/*
  void delFdelparm{
     Field_t<Dim> *rhoA     = &pertSim->fcontainer_m->getRho();
     Field_t<Dim> *rho     = &unpertSim->fcontainer_m->getRho();

     const int nghostrhoA = pertSim->fcontainer_m->getRho().getNghost();
     auto rhoAview        = pertSim->fcontainer_m->getRho().getView();
     const ippl::NDIndex<Dim>& lDomA = pertSim->fcontainer_m->getFL()->getLocalNDIndex();

     const int nghostrho = unpertSim->fcontainer_m->getRho().getNghost();
     auto rhoview        = unpertSim->fcontainer_m->getRho().getView();

     Vector_t<double, Dim> hr     = unpertSim->hr_m;
     Vector_t<double, Dim> origin = unpertSim->origin_m;

     using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
     for (unsigned idparams = 0; idparams < unpertSim.nparams; ++idparams) {
         T temp = 0.0;
         ippl::parallel_reduce(
                "rhoA*dphi_dalpha0 reduce", rhoA.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args, T& valL) {
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr + origin;

                    T myVal = ippl::apply(rhoAview, args) * dphi_dparams(xvec, idparams);
                    valL += myVal;
              }, Kokkos::Sum<T>(temp));
          Kokkos::fence();

          T globaltemp          = 0.0;
          ippl::Comm->reduce(temp, globaltemp, 1, std::plus<double>());
          ippl::Comm->barrier();

          pertSim.dparams[iparams] += globaltemp;
  }
*/


/*
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
*/
//};
#endif
